"""Detect disc flights as small fast objects on a static camera.

The camera never moves, so a rolling median of the frames is a clean
background and anything that differs from it is moving. People are removed
with their detection boxes; what remains inside the playing area (or in the
sky above it) is linked frame to frame into trajectories. A disc flight is a
trajectory that is long, fast, smooth, and starts and ends near a person.

This runs at the full frame rate because a flight lasts a second or two and
moves 20 to 120 pixels per frame; the 5 Hz player analysis is far too coarse.
Person boxes come from that analysis, interpolated in time, so no second
detector pass is needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import subprocess

import cv2
import numpy as np

from .models import DetectionSet, ProcessingConfig

# Bump whenever a change alters which flights are found, so saved flights.json
# from older runs are rescanned instead of reused. Unversioned files predate 2.
DETECTOR_VERSION = 2

SCAN_SCALE = 2                 # frames are decoded at 1/SCAN_SCALE; a disc is still 8 to 12 px across
DIFFERENCE_THRESHOLD = 30      # gray levels against the rolling median
CHUNK_FRAMES = 40              # frames per background median
MIN_AREA, MAX_AREA = 6, 700    # candidate blob size in pixels
MAX_SIDE = 70                  # candidate bounding box side
FAR_EDGE_BAND = 60             # pixels below the field's far edge where horizontal movers are not discs
FAR_EDGE_STRIP = 120           # pixels above the far edge covered by crowd, tents and road
CLUTTER_RISE = 0.15            # a track confined to the strip above the far edge must rise this share of its chord
STRIP_BOW, STRIP_BOW_SHARE = 40.0, 0.06  # ... or bow this far (pixels, share of chord) above the line joining its ends
MAX_DURATION = 9.0             # seconds; a high pull can hang seven, clouds drift far longer
SKY_MIN_SPEED = 15.0           # pixels per frame for a track entirely above the field
SEED_MIN, SEED_MAX = 8.0, 160.0  # per-frame displacement for a new trajectory
MAX_MISSES = 4                 # frames a trajectory may go unseen (occlusion by a player)
MIN_FRAMES = 8
MIN_CHORD = 250.0              # pixels end to end
MIN_STRAIGHTNESS = 0.8         # chord / path length
NEAR_PERSON = 2.5              # body heights from a person box at launch or landing
CLUTTER_CELL = 64              # pixels; cells busy with movers for much of the time are clutter
CLUTTER_WINDOW = 100           # frames of history for the clutter map
CLUTTER_FRACTION = 0.15        # share of recent frames with a candidate that makes a cell clutter
AREA_RATIO = 3.0               # frame-to-frame blob area may change by at most this factor
INTENSITY_STEP = 50            # frame-to-frame blob brightness change allowed
STEP_RATIO = 1.5               # consecutive step lengths may differ by at most this factor (90th pct)
HEADING_CHANGE = 30.0          # degrees, 90th percentile of turn between steps
DISC_DIAMETER_M = 0.27
MAX_DISC_SPEED = 30.0          # m/s; with the blob diameter this bounds pixels per frame at any depth
MINIMUM_BODY_SCALE = 30.0      # pixels; floor for the person size used to scale a flight's length


@dataclass
class Flight:
    start: float               # seconds from the configured start
    end: float
    x0: float
    y0: float
    x1: float
    y1: float
    direction: int             # +1 rightwards, -1 leftwards
    frames: int
    chord: float
    speed: float               # pixels per frame
    chord_bodies: float = 0.0  # chord in body heights at the flight's depth: a scale-free length
    launch_seen: bool = True   # the track starts at a person; otherwise it was found mid-air (a high pull)
    points: list = field(default_factory=list)  # [time, x, y]

    def to_json(self):
        data = asdict(self)
        data['points'] = [[round(t, 3), round(x, 1), round(y, 1)] for t, x, y in self.points]
        return data


def _sample_times(detections: DetectionSet) -> np.ndarray:
    return np.array([s.time for s in detections.samples])


def _boxes_at(detections: DetectionSet, times: np.ndarray, relative_time: float):
    """Person boxes near a moment, taken from the two nearest analysis samples."""
    index = int(np.searchsorted(times, relative_time))
    chosen = []
    for i in (index - 1, index):
        if 0 <= i < len(detections.samples) and abs(times[i] - relative_time) <= 1.0 / detections.sample_rate + 1e-6:
            chosen.extend(d.box for d in detections.samples[i].detections)
    return chosen


def person_mask(detections: DetectionSet, times: np.ndarray, relative_time: float, height: int, width: int, scale: int) -> np.ndarray:
    """Dilated person boxes on a (height, width) mask whose pixels are `scale` source pixels."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in _boxes_at(detections, times, relative_time):
        w, h = x2 - x1, y2 - y1
        cv2.rectangle(mask, (int((x1 - 0.6 * w) / scale), int((y1 - 0.4 * h) / scale)),
                      (int((x2 + 0.6 * w) / scale), int((y2 + 0.3 * h) / scale)), 255, -1)
    return mask


def allowed_region(play_area: np.ndarray | None, height: int, width: int):
    """The field plus everything above it (a throw arcs over the far sideline).

    Returns the region and the far edge row per column. The crowd, tents and
    road above the far edge are not excluded here; their movers are rejected
    later by shape, because they travel horizontally while a throw rises.
    """
    tops = np.full(width, height, dtype=np.int32)
    if play_area is None:
        return np.full((height, width), 255, dtype=np.uint8), tops
    allowed = (play_area > 0).astype(np.uint8) * 255
    columns = np.where(play_area.max(axis=0) > 0)[0]
    for column in columns:
        top = int(np.argmax(play_area[:, column] > 0))
        tops[column] = top
        allowed[:top, column] = 255
    return allowed, tops


def _candidates(gray: np.ndarray, median: np.ndarray, blocked: np.ndarray, scale: int = 1):
    """Small moving blobs as (x, y, area, intensity) in source-pixel units.

    The frame may be a 1/scale decode; thresholds shrink accordingly and the
    results are scaled back so every later stage works in source pixels.
    """
    moving = (np.abs(gray.astype(np.int16) - median) > DIFFERENCE_THRESHOLD).astype(np.uint8)
    moving[blocked == 0] = 0
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(moving, 8)
    min_area, max_area, max_side = MIN_AREA / scale ** 2, MAX_AREA / scale ** 2, MAX_SIDE / scale
    found = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area and max(w, h) <= max_side and max(w, h) <= 6 * max(1, min(w, h)):
            cx, cy = centroids[i]
            found.append((float(cx) * scale, float(cy) * scale, int(area * scale ** 2), int(gray[int(cy), int(cx)])))
    return found


class ClutterMap:
    """Cells that hold a moving candidate in a large share of recent frames.

    Crowds, flags, trees in the wind and traffic produce candidates all the
    time in the same places; a disc passes a cell once. Candidates in clutter
    cells may not start or extend a trajectory, which stops the linker from
    chaining unrelated movers into a straight line.
    """

    def __init__(self, height: int, width: int):
        self.rows, self.columns = height // CLUTTER_CELL + 1, width // CLUTTER_CELL + 1
        self.counts = np.zeros((self.rows, self.columns), dtype=np.int32)
        self.history: list[np.ndarray] = []

    def cell(self, x: float, y: float):
        return min(self.rows - 1, int(y // CLUTTER_CELL)), min(self.columns - 1, int(x // CLUTTER_CELL))

    def update(self, found):
        cells = np.zeros((self.rows, self.columns), dtype=bool)
        for x, y, *_ in found:
            cells[self.cell(x, y)] = True
        self.history.append(cells)
        self.counts += cells
        if len(self.history) > CLUTTER_WINDOW:
            self.counts -= self.history.pop(0)

    def split(self, found):
        """(clean candidates, candidates inside clutter cells)."""
        limit = CLUTTER_FRACTION * len(self.history)
        clean, cluttered = [], []
        for c in found:
            (clean if self.counts[self.cell(c[0], c[1])] <= limit else cluttered).append(c)
        return clean, cluttered


class Linker:
    """Greedy nearest-prediction linking with a short memory for occlusions."""

    def __init__(self):
        self.active: list[dict] = []
        self.done: list[dict] = []
        self.previous: list[tuple[float, float, int, int]] = []

    def step(self, frame: int, found, cluttered=()):
        """Extend tracks with clean candidates, or through clutter under a tight gate; seed only from clean ones."""
        used = set()
        kept = []
        pool = list(found) + list(cluttered)
        clean_count = len(found)
        for track in self.active:
            gap = frame - track['frame']
            vx, vy = track['v']
            px, py = track['x'] + vx * gap, track['y'] + vy * gap
            speed = math.hypot(vx, vy)
            best = None
            for j, (cx, cy, area, intensity) in enumerate(pool):
                if j in used:
                    continue
                if not (track['area'] / AREA_RATIO <= area <= track['area'] * AREA_RATIO) or abs(intensity - track['intensity']) > INTENSITY_STEP:
                    continue
                gate = (25.0 + 0.35 * speed * gap) if j < clean_count else (10.0 + 0.15 * speed * gap)
                distance = math.hypot(cx - px, cy - py)
                if distance < gate and (best is None or distance < best[0]):
                    best = (distance, j)
            if best is not None:
                j = best[1]
                used.add(j)
                cx, cy, area, intensity = pool[j]
                ovx, ovy = (cx - track['x']) / gap, (cy - track['y']) / gap
                track['v'] = (0.5 * vx + 0.5 * ovx, 0.5 * vy + 0.5 * ovy)
                track['x'], track['y'], track['frame'] = cx, cy, frame
                track['area'], track['intensity'] = area, intensity
                track['points'].append((frame, cx, cy, area))
                kept.append(track)
            elif frame - track['frame'] <= MAX_MISSES:
                kept.append(track)
            elif len(track['points']) >= MIN_FRAMES:
                self.done.append(track)
        for j, (cx, cy, area, intensity) in enumerate(found):
            if j in used:
                continue
            for (lx, ly, larea, lintensity) in self.previous:
                displacement = math.hypot(cx - lx, cy - ly)
                if SEED_MIN <= displacement <= SEED_MAX and larea / AREA_RATIO <= area <= larea * AREA_RATIO \
                        and abs(intensity - lintensity) <= INTENSITY_STEP:
                    kept.append({'x': cx, 'y': cy, 'v': (cx - lx, cy - ly), 'frame': frame, 'area': area,
                                 'intensity': intensity, 'points': [(frame - 1, lx, ly, larea), (frame, cx, cy, area)]})
                    break
        self.previous = [c for j, c in enumerate(found) if j not in used]
        self.active = kept

    def finish(self):
        self.done.extend(track for track in self.active if len(track['points']) >= MIN_FRAMES)
        self.active = []
        return self.done


def _body_height_model(detections: DetectionSet):
    """Median box height as a linear function of foot y, for speed plausibility."""
    feet, heights = [], []
    for sample in detections.samples[::max(1, len(detections.samples) // 400)]:
        for d in sample.detections:
            feet.append(d.box[3])
            heights.append(d.box[3] - d.box[1])
    if len(feet) < 50:
        return lambda y: 100.0
    slope, intercept = np.polyfit(feet, heights, 1)
    return lambda y: max(30.0, slope * y + intercept)


def _near_person(detections: DetectionSet, times: np.ndarray, t: float, x: float, y: float, body: float):
    """Height of the nearest person box within reach of (x, y), or None."""
    best = None
    for x1, y1, x2, y2 in _boxes_at(detections, times, t):
        cx, cy = (x1 + x2) / 2, y2
        distance = math.hypot(x - cx, y - cy)
        if distance <= NEAR_PERSON * max(body, y2 - y1) and (best is None or distance < best[0]):
            best = (distance, float(y2 - y1))
    return None if best is None else best[1]


def _bow(points: np.ndarray) -> float:
    """Greatest height of the path above the straight line joining its ends.

    A throw along the far sideline seen from the near side barely rises in the
    image, and when it ends lower than it started its total rise is mostly
    the drop. It still arches above its own chord; traffic and walkers do not.
    """
    start, chord = points[0], points[-1] - points[0]
    along = (points - start) @ chord / max(float(chord @ chord), 1e-9)
    return float(np.max(start[1] + along * chord[1] - points[:, 1]))


def _qualify(tracks, detections: DetectionSet, fps: float, play_area: np.ndarray | None = None,
             tops: np.ndarray | None = None) -> list[Flight]:
    body_at = _body_height_model(detections)
    times = _sample_times(detections)
    if play_area is not None and play_area.any():
        rows = np.where(play_area.max(axis=1) > 0)[0]
        far_edge, near_edge = float(rows.min()), float(rows.max())
    else:
        far_edge, near_edge = 0.0, float(max(1, max(d.box[3] for s in detections.samples for d in s.detections) if any(s.detections for s in detections.samples) else 1))
    # A disc is somewhere between the far and near sidelines, so its apparent
    # speed lies between a slow throw at the far edge and a fast one up close.
    slowest = 0.2 * body_at(far_edge)
    flights = []
    for track in tracks:
        points = np.array([(x, y) for _, x, y, _ in track['points']])
        frames = np.array([f for f, _, _, _ in track['points']])
        areas = np.array([a for _, _, _, a in track['points']], dtype=float)
        chord = float(np.linalg.norm(points[-1] - points[0]))
        path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        span = int(frames[-1] - frames[0])
        if chord < MIN_CHORD or span < MIN_FRAMES or path <= 0 or chord / path < MIN_STRAIGHTNESS:
            continue
        speed = path / span
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.maximum(1, np.diff(frames))
        # Apparent speed changes smoothly along a flight (perspective makes it
        # peak nearest the camera); chained clutter jumps around.
        ratios = np.abs(np.log(np.maximum(steps[1:], 1e-3) / np.maximum(steps[:-1], 1e-3)))
        if len(ratios) and np.percentile(ratios, 90) > math.log(STEP_RATIO):
            continue
        headings = np.degrees(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])))
        turns = np.abs((np.diff(headings) + 180) % 360 - 180)
        if len(turns) and np.percentile(turns, 90) > HEADING_CHANGE:
            continue
        lowest = float(points[:, 1].max())
        edge = far_edge
        if tops is not None:
            edge = float(np.median(tops[np.clip(points[:, 0].astype(int), 0, len(tops) - 1)]))
            rise = float(lowest - points[:, 1].min())
            in_strip = points[:, 1].min() > edge - FAR_EDGE_STRIP and lowest < edge + FAR_EDGE_BAND
            if in_strip and rise < max(60.0, CLUTTER_RISE * chord) and _bow(points) < max(STRIP_BOW, STRIP_BOW_SHARE * chord):
                continue  # traffic on the road or walkers along the far sideline
        normalized = speed * 20 / fps
        # The blob's size says how far away it is: a disc of apparent
        # diameter d pixels cannot move faster than MAX_DISC_SPEED at that
        # scale, whatever its height above the field. On the field the body
        # size at that depth bounds it as well.
        diameter = 2.0 * math.sqrt(float(np.median(areas)) / math.pi)
        by_size = MAX_DISC_SPEED / fps * diameter / DISC_DIAMETER_M
        depth = lowest if lowest >= edge else (edge + near_edge) / 2
        body = body_at(depth)
        fastest = min(by_size, 1.4 * body) if lowest >= edge else by_size
        if span / fps > MAX_DURATION or not slowest <= normalized <= fastest:
            continue
        if lowest < edge and normalized < SKY_MIN_SPEED:
            continue
        t0, t1 = frames[0] / fps, frames[-1] / fps
        # Every flight leaves a hand or arrives in one; clouds, birds and
        # traffic do neither. The people involved also give the flight's
        # length a scale, which the disc's own height above the field cannot.
        thrower = _near_person(detections, times, t0, *points[0], body)
        receiver = _near_person(detections, times, t1, *points[-1], body)
        if thrower is None and receiver is None:
            continue
        people = [h for h in (thrower, receiver) if h is not None]
        length_scale = max(MINIMUM_BODY_SCALE, float(np.mean(people)))
        flights.append(Flight(round(t0, 3), round(t1, 3), *map(float, points[0]), *map(float, points[-1]),
                             1 if points[-1][0] >= points[0][0] else -1, len(points), chord, speed,
                             round(chord / length_scale, 2), thrower is not None,
                             [(f / fps, x, y) for f, x, y, _ in track['points']]))
    return _merge(flights)


def _merge(flights: list[Flight]) -> list[Flight]:
    """Drop duplicates: trajectories that overlap in time and run along the same path."""
    flights.sort(key=lambda f: -f.frames)
    kept: list[Flight] = []
    for flight in flights:
        duplicate = False
        for other in kept:
            overlap = min(flight.end, other.end) - max(flight.start, other.start)
            if overlap <= 0:
                continue
            if math.hypot(flight.x0 - other.x0, flight.y0 - other.y0) < 120 or math.hypot(flight.x1 - other.x1, flight.y1 - other.y1) < 120:
                duplicate = True
                break
        if not duplicate:
            kept.append(flight)
    kept.sort(key=lambda f: f.start)
    return kept


def detect_flights(config: ProcessingConfig, video, detections: DetectionSet, play_area: np.ndarray | None,
                   progress=None, scale: int = SCAN_SCALE) -> list[Flight]:
    """Scan the configured range at full frame rate, decoding at 1/scale."""
    width, height, fps = video.width, video.height, video.fps
    total = int(round((config.end - config.start) * fps))
    allowed, tops = allowed_region(play_area, height, width)
    low_width, low_height = width // scale, height // scale
    allowed_low = cv2.resize(allowed, (low_width, low_height), interpolation=cv2.INTER_NEAREST)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(config.start), '-i', str(config.input),
               '-frames:v', str(total), '-an', '-sn', '-dn', '-vf', f'scale={low_width}:{low_height}',
               '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    linker = Linker()
    clutter = ClutterMap(height, width)
    times = _sample_times(detections)
    frame_bytes = low_width * low_height
    frame_index = 0
    try:
        while frame_index < total:
            chunk = []
            for _ in range(min(CHUNK_FRAMES, total - frame_index)):
                data = process.stdout.read(frame_bytes)
                if len(data) < frame_bytes:
                    break
                chunk.append(np.frombuffer(data, dtype=np.uint8).reshape(low_height, low_width))
            if not chunk:
                break
            median = np.median(np.stack(chunk[::3]), axis=0).astype(np.int16)
            for gray in chunk:
                t = frame_index / fps
                blocked = allowed_low.copy()
                blocked[person_mask(detections, times, t, low_height, low_width, scale) > 0] = 0
                found = _candidates(gray, median, blocked, scale)
                clutter.update(found)
                linker.step(frame_index, *clutter.split(found))
                frame_index += 1
            if progress:
                progress(frame_index, total)
    finally:
        process.stdout.close()
        process.wait()
    return _qualify(linker.finish(), detections, fps, play_area, tops)
