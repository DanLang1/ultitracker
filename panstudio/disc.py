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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import islice
import math
import os
import subprocess

import cv2
import numpy as np

from .media import hwaccel_args, prefetched
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
    if median.dtype != np.uint8:
        median = np.clip(median, 0, 255).astype(np.uint8)
    moving = cv2.compare(cv2.absdiff(gray, median), DIFFERENCE_THRESHOLD, cv2.CMP_GT)
    moving = cv2.bitwise_and(moving, moving, mask=blocked)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(moving, 8)
    min_area, max_area, max_side = MIN_AREA / scale ** 2, MAX_AREA / scale ** 2, MAX_SIDE / scale
    w, h, area = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], stats[1:, cv2.CC_STAT_AREA]
    longest = np.maximum(w, h)
    keep = (min_area <= area) & (area <= max_area) & (longest <= max_side) & (longest <= 6 * np.maximum(1, np.minimum(w, h)))
    cx, cy = centroids[1:][keep].T
    intensity = gray[cy.astype(int), cx.astype(int)]
    return list(zip((cx * scale).tolist(), (cy * scale).tolist(), (area[keep] * scale ** 2).tolist(), intensity.tolist()))


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

    def cells(self, found):
        points = np.array([c[:2] for c in found], dtype=float).reshape(-1, 2)
        rows = np.minimum(self.rows - 1, (points[:, 1] // CLUTTER_CELL).astype(int))
        columns = np.minimum(self.columns - 1, (points[:, 0] // CLUTTER_CELL).astype(int))
        return rows, columns

    def update(self, found):
        cells = np.zeros((self.rows, self.columns), dtype=bool)
        cells[self.cells(found)] = True
        self.history.append(cells)
        self.counts += cells
        if len(self.history) > CLUTTER_WINDOW:
            self.counts -= self.history.pop(0)

    def split(self, found):
        """(clean candidates, candidates inside clutter cells)."""
        limit = CLUTTER_FRACTION * len(self.history)
        is_clean = self.counts[self.cells(found)] <= limit
        clean, cluttered = [], []
        for c, ok in zip(found, is_clean.tolist()):
            (clean if ok else cluttered).append(c)
        return clean, cluttered


class Linker:
    """Greedy nearest-prediction linking with a short memory for occlusions."""

    def __init__(self):
        self.active: list[dict] = []
        self.done: list[dict] = []
        self.previous: list[tuple[float, float, int, int]] = []

    def step(self, frame: int, found, cluttered=()):
        """Extend tracks with clean candidates, or through clutter under a tight gate; seed only from clean ones."""
        kept = []
        pool = list(found) + list(cluttered)
        clean_count = len(found)
        candidates = np.array(pool, dtype=float).reshape(-1, 4)
        xs, ys, areas, intensities = candidates.T
        is_clean = np.arange(len(pool)) < clean_count
        used = np.zeros(len(pool), dtype=bool)
        for track in self.active:
            gap = frame - track['frame']
            vx, vy = track['v']
            px, py = track['x'] + vx * gap, track['y'] + vy * gap
            speed = math.hypot(vx, vy)
            best = None
            if len(pool):
                gate = np.where(is_clean, 25.0 + 0.35 * speed * gap, 10.0 + 0.15 * speed * gap)
                distance = np.hypot(xs - px, ys - py)
                eligible = (~used & (track['area'] / AREA_RATIO <= areas) & (areas <= track['area'] * AREA_RATIO)
                            & (np.abs(intensities - track['intensity']) <= INTENSITY_STEP) & (distance < gate))
                if eligible.any():
                    best = int(np.flatnonzero(eligible)[np.argmin(distance[eligible])])
            if best is not None:
                j = best
                used[j] = True
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
        if clean_count and self.previous:
            # Each unused clean candidate seeds from the first previous candidate that fits.
            previous = np.array(self.previous, dtype=float)
            fresh = candidates[:clean_count]
            displacement = np.hypot(fresh[:, None, 0] - previous[None, :, 0], fresh[:, None, 1] - previous[None, :, 1])
            fits = ((SEED_MIN <= displacement) & (displacement <= SEED_MAX)
                    & (previous[None, :, 2] / AREA_RATIO <= fresh[:, None, 2]) & (fresh[:, None, 2] <= previous[None, :, 2] * AREA_RATIO)
                    & (np.abs(fresh[:, None, 3] - previous[None, :, 3]) <= INTENSITY_STEP))
            fits[used[:clean_count]] = False
            for j in np.flatnonzero(fits.any(axis=1)).tolist():
                cx, cy, area, intensity = found[j]
                lx, ly, larea, _ = self.previous[int(np.argmax(fits[j]))]
                kept.append({'x': cx, 'y': cy, 'v': (cx - lx, cy - ly), 'frame': frame, 'area': area,
                             'intensity': intensity, 'points': [(frame - 1, lx, ly, larea), (frame, cx, cy, area)]})
        self.previous = [c for j, c in enumerate(found) if not used[j]]
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


def _chunk_median(chunk: list[np.ndarray], pool: ThreadPoolExecutor, strips: int) -> np.ndarray:
    """Per-pixel median of every third frame, computed in row strips across the pool."""
    stack = np.stack(chunk[::3])
    bounds = np.linspace(0, stack.shape[1], strips + 1).astype(int)
    medians = pool.map(lambda rows: np.median(stack[:, rows[0]:rows[1]], axis=0), zip(bounds[:-1], bounds[1:]))
    return np.concatenate(list(medians)).astype(np.uint8)


def _gray_frames(command: list[str], frame_bytes: int, shape: tuple[int, int]):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        while len(data := process.stdout.read(frame_bytes)) == frame_bytes:
            yield np.frombuffer(data, dtype=np.uint8).reshape(shape)
    finally:
        process.stdout.close()
        process.wait()


def detect_flights(config: ProcessingConfig, video, detections: DetectionSet, play_area: np.ndarray | None,
                   progress=None, scale: int = SCAN_SCALE) -> list[Flight]:
    """Scan the configured range at full frame rate, decoding at 1/scale."""
    width, height, fps = video.width, video.height, video.fps
    total = int(round((config.end - config.start) * fps))
    allowed, tops = allowed_region(play_area, height, width)
    low_width, low_height = width // scale, height // scale
    allowed_low = cv2.resize(allowed, (low_width, low_height), interpolation=cv2.INTER_NEAREST)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *hwaccel_args(), '-ss', str(config.start), '-i', str(config.input),
               '-frames:v', str(total), '-an', '-sn', '-dn', '-vf', f'scale={low_width}:{low_height}',
               '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    frames = prefetched(_gray_frames(command, low_width * low_height, (low_height, low_width)), depth=CHUNK_FRAMES)
    linker = Linker()
    clutter = ClutterMap(height, width)
    times = _sample_times(detections)
    frame_index = 0
    # Candidates depend only on their own frame, so a chunk's frames are
    # searched in parallel; linking them into tracks stays sequential.
    workers = min(8, os.cpu_count() or 1)
    pool = ThreadPoolExecutor(workers)

    def frame_candidates(index, gray, median):
        blocked = allowed_low.copy()
        blocked[person_mask(detections, times, index / fps, low_height, low_width, scale) > 0] = 0
        return _candidates(gray, median, blocked, scale)

    try:
        while frame_index < total:
            chunk = list(islice(frames, min(CHUNK_FRAMES, total - frame_index)))
            if not chunk:
                break
            median = _chunk_median(chunk, pool, workers)
            indices = range(frame_index, frame_index + len(chunk))
            for found in pool.map(frame_candidates, indices, chunk, [median] * len(chunk)):
                clutter.update(found)
                linker.step(frame_index, *clutter.split(found))
                frame_index += 1
            if progress:
                progress(frame_index, total)
    finally:
        pool.shutdown(cancel_futures=True)
        frames.close()
    return _qualify(linker.finish(), detections, fps, play_area, tops)
