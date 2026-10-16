from __future__ import annotations

import csv
import math
from pathlib import Path
import numpy as np
from .models import ProcessingConfig, Sample
from .calibration import Calibration

PROTECTION_PERSISTENCE = 0.4    # seconds a formation-protection bound must hold before it steers the camera
EASE_HALF_WIDTH = 0.4           # seconds; zero-phase rounding of the final path, like an operator's hands
FLIGHT_MIN_BODIES = 12.0        # chord in body heights (about 20 m); shorter throws never move the camera
FLIGHT_MIN_CHORD_SHARE = 0.2    # fallback, of the frame width, for flights without a body-scaled length
FLIGHT_HOLD_SECONDS = 2.0       # stay on the landing after the catch
FLIGHT_FOLLOW_MARGIN = 0.1      # crop-width share kept ahead of receivers still running the throw's way
FLIGHT_ESCORT_SHARE = 0.6       # a "flight" with a player alongside at its speed this much of the time is that player
FLIGHT_ESCORT_REACH = 1.5       # body heights between the flight and the player running with it
FLIGHT_ESCORT_MIN_SPEED = 150.0 # px/s across the frame; slower stretches say nothing (standing players match them)


def apply_deep_coverage(args: ProcessingConfig, records: list[Sample], centers: np.ndarray, source_width: int) -> np.ndarray:
    """Keep a persistent runner/defender pair safely visible.

    Deep tracking is an optional second objective.  The formation visible to
    the ordinary (median) center remains a hard constraint whenever it can be
    represented in the crop.  This matters when a deep pair and a nearby
    field player are on opposite sides of the source frame.
    """
    if args.no_deep_tracking:
        return centers
    if not any(record.get("deep_active") for record in records):
        return centers

    calibration = args.calibration
    desired = centers.copy()
    safe_half = args.crop_width / 2 - calibration.deep_margin
    lookahead = args.deep_lookahead
    crop_half = args.crop_width / 2
    lower_bounds = np.full(len(records), crop_half)
    upper_bounds = np.full(len(records), float(source_width) - crop_half)
    for index, record in enumerate(records):
        deep_ids = {int(track_id) for track_id in record.get("deep_track_ids", [])}
        # Protect the extents of credible formation players.  Suppressed,
        # stationary tracks are deliberately omitted: they are often crowd or
        # sideline detections and should not freeze the camera.
        formation = [] if args.no_formation_protection else [
            player for player in record.get("players", [])
            if int(player.get("track_id", -1)) not in deep_ids
            and float(player.get("stationary_factor", 1.0)) >= 0.5
        ]
        if formation:
            left = []
            right = []
            baseline_left = float(centers[index] - crop_half)
            baseline_right = float(centers[index] + crop_half)
            for player in formation:
                box = player.get("box")
                if box and len(box) >= 4:
                    player_left = float(box[0])
                    player_right = float(box[2])
                else:
                    x = float(player.get("center_x", centers[index]))
                    player_left = player_right = x
                # Only preserve players the formation-only crop actually
                # showed. This avoids freezing on spectators or stale tracks
                # outside the baseline's current window.
                if player_right < baseline_left or player_left > baseline_right:
                    continue
                left.append(max(player_left, baseline_left))
                right.append(min(player_right, baseline_right))
            if left:
                lower_bounds[index] = max(lower_bounds[index], max(right) - crop_half)
                upper_bounds[index] = min(upper_bounds[index], min(left) + crop_half)
                # The formation-only center is always a valid fallback, even when
                # detector boxes are slightly inconsistent with their weighted median.
                lower_bounds[index] = min(lower_bounds[index], float(centers[index]))
                upper_bounds[index] = max(upper_bounds[index], float(centers[index]))
                if lower_bounds[index] > upper_bounds[index]:
                    lower_bounds[index] = upper_bounds[index] = float(centers[index])
                    record["deep_baseline_infeasible"] = True
                record["deep_protected_baseline_min"] = round(float(lower_bounds[index]), 2)
                record["deep_protected_baseline_max"] = round(float(upper_bounds[index]), 2)

        if record.get("deep_active"):
            pair_min = record.get("deep_pair_min_x", record["deep_center_x"])
            pair_max = record.get("deep_pair_max_x", record["deep_center_x"])
            actual_pair_min, actual_pair_max = pair_min, pair_max
            direction = record.get("deep_direction", 1)
            leading = record.get("deep_leading_x", record["deep_center_x"])
            velocity = record.get("deep_velocity_x", 0.0)
            projected_lead = leading + direction * min(
                args.crop_width * 0.28,
                max(0.0, direction * velocity * lookahead),
            )
            if direction > 0:
                pair_max = max(pair_max, projected_lead)
            else:
                pair_min = min(pair_min, projected_lead)
            minimum_center = pair_max - safe_half
            maximum_center = pair_min + safe_half
            if minimum_center > maximum_center:
                # If the entire envelope cannot fit, favor the leading edge.
                target = minimum_center if direction > 0 else maximum_center
                minimum_center = maximum_center = target
            deep_target = float(np.clip(desired[index], minimum_center, maximum_center))
            record["deep_coverage_target_x"] = round(deep_target, 2)
            # Formation coverage wins when the two objectives conflict.  This
            # also naturally gives up the optional 310px deep margin first.
            feasible_min = max(minimum_center, lower_bounds[index])
            feasible_max = min(maximum_center, upper_bounds[index])
            actual_min = max(actual_pair_max - crop_half, lower_bounds[index])
            actual_max = min(actual_pair_min + crop_half, upper_bounds[index])
            record["deep_actual_pair_feasible"] = bool(actual_min <= actual_max)
            if feasible_min <= feasible_max:
                desired[index] = np.clip(deep_target, feasible_min, feasible_max)
                record["deep_pair_feasible"] = True
            else:
                desired[index] = np.clip(deep_target, lower_bounds[index], upper_bounds[index])
                record["deep_pair_feasible"] = False

    # A protection bound that holds for a single sample is a detector flicker,
    # not a player who needs keeping in frame. A bound counts from its first
    # sample only if it persists for PROTECTION_PERSISTENCE seconds beyond it:
    # the loosest value in that forward window wins. The future is known, so
    # looking ahead costs nothing and keeps the camera anticipating.
    persistence = max(1, round(PROTECTION_PERSISTENCE * args.sample_rate))
    lower_bounds = np.array([lower_bounds[i:i + persistence + 1].min() for i in range(len(records))])
    upper_bounds = np.array([upper_bounds[i:i + persistence + 1].max() for i in range(len(records))])
    for index in range(len(records)):
        if records[index].get("deep_active"):
            desired[index] = float(np.clip(desired[index], lower_bounds[index], upper_bounds[index]))
    # Limit the extra camera movement and soften mode entry/exit with a 1 s
    # window. Smooth only the deep correction: re-smoothing absolute formation
    # centers would alter long stretches that have no deep signal.
    correction = smooth_series(desired - centers, 1.0, args.sample_rate)
    desired = centers + correction
    max_step = calibration.deep_pan_speed / args.sample_rate
    # Work backwards to remove targets that cannot reach a future protected
    # interval at the permitted speed. The formation center is retained when
    # an interval is intrinsically infeasible.
    corridor_min = lower_bounds.copy()
    corridor_max = upper_bounds.copy()
    for index in range(len(records) - 2, -1, -1):
        next_min, next_max = corridor_min[index + 1], corridor_max[index + 1]
        reachable_min = next_min - max_step
        reachable_max = next_max + max_step
        new_min = max(corridor_min[index], reachable_min)
        new_max = min(corridor_max[index], reachable_max)
        if new_min <= new_max:
            corridor_min[index], corridor_max[index] = new_min, new_max
    result = np.empty_like(desired)
    result[0] = np.clip(desired[0], corridor_min[0], corridor_max[0])
    for index in range(1, len(desired)):
        reachable_min = result[index - 1] - max_step
        reachable_max = result[index - 1] + max_step
        allowed_min = max(corridor_min[index], reachable_min)
        allowed_max = min(corridor_max[index], reachable_max)
        if allowed_min <= allowed_max:
            result[index] = np.clip(desired[index], allowed_min, allowed_max)
        else:
            # A sudden, impossible constraint should not create a camera jump;
            # retain the speed-limited path and report that protection lost.
            result[index] = np.clip(desired[index], reachable_min, reachable_max)
            records[index]["deep_protection_speed_limited"] = True
    return result


def _escorted(records: list[Sample], times: np.ndarray, flight) -> bool:
    """Whether a player runs alongside the flight at its speed for most of it.

    A disc outruns everyone until it is caught. A sprinter's bobbing head or
    hand, left outside a lagging person box, moves with the sprinter: a long,
    smooth, plausible "flight" that would drag the camera after one runner.
    """
    points = np.asarray(flight.get("points") or [], dtype=float)
    if len(points) < 2:
        return False
    first, last = np.searchsorted(times, points[0, 0]), np.searchsorted(times, points[-1, 0], side="right")
    escorted = checked = 0
    for record in records[first:last]:
        t = record["time"]
        x, y = np.interp(t, points[:, 0], points[:, 1]), np.interp(t, points[:, 0], points[:, 2])
        k = int(np.searchsorted(points[:, 0], t))
        a, b = max(0, k - 3), min(len(points) - 1, k + 3)
        velocity = (points[b, 1] - points[a, 1]) / max(1e-3, points[b, 0] - points[a, 0])
        if abs(velocity) < FLIGHT_ESCORT_MIN_SPEED:
            continue
        checked += 1
        for player in record.get("players", []):
            reach = FLIGHT_ESCORT_REACH * float(player.get("height", 100.0))
            if abs(player["center_x"] - x) <= reach and abs(player["center_y"] - y) <= reach \
                    and abs(player.get("velocity_x", 0.0) - velocity) <= max(60.0, 0.35 * abs(velocity)):
                escorted += 1
                break
    return checked > 0 and escorted / checked >= FLIGHT_ESCORT_SHARE


def annotate_flights(records: list[Sample], flights, crop_width: float, frame_width: int, look_room: float) -> list:
    """Mark the samples each long disc flight governs and what they should show.

    A flight is a dictionary with start, end, x0, y0, x1, y1, chord and
    direction. The camera target is the landing point plus look room in the
    throw direction. Later flights override earlier overlapping ones.
    Returns the long flights ignored because a runner moved with them.
    """
    half = crop_width / 2
    times = np.array([record["time"] for record in records])
    for record in records:
        record["flight_active"] = False
    escorted = []
    for flight in sorted(flights, key=lambda f: f["start"]):
        bodies = flight.get("chord_bodies")
        if (bodies is not None and bodies > 0 and bodies < FLIGHT_MIN_BODIES) or (not bodies and flight["chord"] < FLIGHT_MIN_CHORD_SHARE * frame_width):
            continue
        if _escorted(records, times, flight):
            escorted.append(flight)
            continue
        direction = 1 if flight["direction"] >= 0 else -1
        target = float(np.clip(flight["x1"] + direction * look_room * crop_width, half, frame_width - half))
        low, high = flight["start"], flight["end"] + FLIGHT_HOLD_SECONDS
        for record in records[np.searchsorted(times, low):np.searchsorted(times, high, side="right")]:
            record.update({
                "flight_active": True, "flight_direction": direction, "flight_target_x": round(target, 2),
                "flight_min_x": round(min(flight["x0"], flight["x1"]), 2), "flight_max_x": round(max(flight["x0"], flight["x1"]), 2),
                "flight_x0": round(float(flight["x0"]), 1), "flight_y0": round(float(flight["y0"]), 1),
                "flight_x1": round(float(flight["x1"]), 1), "flight_y1": round(float(flight["y1"]), 1),
                "flight_start": float(flight["start"]), "flight_launch_seen": bool(flight.get("launch_seen", True)),
            })
    return escorted


def apply_flight_targets(args: ProcessingConfig, records: list[Sample], centers: np.ndarray, source_width: int) -> np.ndarray:
    """Pan to where a long throw lands, starting at the release.

    When the release was seen, the camera holds its path until the disc
    leaves the hand so the throw itself is on screen, then follows it at the
    deep pan speed and stays on the landing for the whole hold. A flight
    first found mid-air (a high pull coming down) had its release out of
    view, so the camera may start early to arrive with the disc; the whole
    game is known in advance. Trailing players are
    allowed to leave the frame: a completed long throw is where the play is.
    Afterwards, while the receivers keep running the throw's way, the camera
    does not fall back behind them.
    """
    if not any(record.get("flight_active") for record in records):
        return centers
    calibration = args.calibration
    half = args.crop_width / 2
    desired = centers.copy()
    unseen = np.zeros(len(records), dtype=bool)
    following = 0
    for index, record in enumerate(records):
        if record.get("flight_active"):
            desired[index] = record["flight_target_x"]
            unseen[index] = not record.get("flight_launch_seen", True)
            following = record["flight_direction"]
        elif following and record.get("deep_active") and record.get("deep_direction") == following:
            leading = float(record.get("deep_leading_x", record.get("deep_center_x", desired[index])))
            floor = leading - following * (half - FLIGHT_FOLLOW_MARGIN * args.crop_width)
            desired[index] = max(desired[index], floor) if following > 0 else min(desired[index], floor)
        else:
            following = 0
    step = calibration.deep_pan_speed / args.sample_rate
    result = desired.copy()
    # Only a flight whose release was out of view starts its pan early,
    # walking back from its first sample until the early path meets the
    # planned one. Nothing else is anticipated, including the return after
    # the hold.
    for start in np.where(unseen & ~np.r_[False, unseen[:-1]])[0]:
        for index in range(start - 1, -1, -1):
            early = np.clip(result[index], result[index + 1] - step, result[index + 1] + step)
            if early == result[index]:
                break
            result[index] = early
    for index in range(1, len(result)):
        result[index] = np.clip(result[index], result[index - 1] - step, result[index - 1] + step)
    return np.clip(result, half, source_width - half)


def plan_crop_widths(args: ProcessingConfig, records: list[Sample], centers: np.ndarray, source_width: int, source_height: int) -> np.ndarray:
    """Plan restrained variable widths for active deep pairs and disc flights."""
    calibration = args.calibration
    base = float(args.crop_width)
    widths = np.full(len(records), base)
    if not records:
        return widths
    if args.no_auto_widen or args.no_deep_tracking:
        return widths
    # Widen only while the original aspect ratio fits inside the footage.
    # A width-only cap can exhaust the source height and introduce black bars.
    cap = min(base * 1.15, float(source_width), source_height * base / args.crop_height)
    for i, record in enumerate(records):
        spans = []
        if record.get("deep_active"):
            # A cutter who has beaten their defender gets room ahead: that is
            # where the throw to them lands.
            ahead = calibration.deep_margin if record.get("deep_breakaway") else calibration.widening_margin
            direction = record.get("deep_direction", 0)
            spans.append((float(record.get("deep_pair_min_x", record.get("deep_center_x", 0))),
                          float(record.get("deep_pair_max_x", record.get("deep_center_x", 0))),
                          ahead if direction < 0 else calibration.widening_margin,
                          ahead if direction > 0 else calibration.widening_margin))
        if record.get("flight_active"):
            spans.append((float(record["flight_min_x"]), float(record["flight_max_x"]), calibration.widening_margin, calibration.widening_margin))
        for span_min, span_max, left_margin, right_margin in spans:
            pair_min = np.clip(span_min - left_margin, 0, source_width)
            pair_max = np.clip(span_max + right_margin, 0, source_width)
            center = float(np.clip(centers[i], 0, source_width))
            # Expansion stays centered until it reaches a source edge. After
            # that, the opposite edge advances by the full added width.
            left_width = min(2 * (center - pair_min), source_width - pair_min)
            right_width = min(2 * (pair_max - center), pair_max)
            widths[i] = min(cap, max(widths[i], left_width, right_width))
    hold = max(1, round(1.5 * args.sample_rate))
    raw_widths = widths.copy()
    for i in range(len(widths)):
        widths[i] = max(raw_widths[max(0, i - hold):i + 1])
    result = np.empty_like(widths)
    expand = calibration.widening_speed / args.sample_rate
    contract = calibration.contraction_speed / args.sample_rate
    result[-1] = widths[-1]
    for i in range(len(widths) - 2, -1, -1):
        result[i] = max(widths[i], result[i + 1] - expand)
    for i in range(1, len(widths)):
        result[i] = max(result[i], result[i - 1] - contract)
    return np.clip(result, base, cap)


def apply_crop_geometry(args: ProcessingConfig, records: list[Sample], centers: np.ndarray, source_width: int, source_height: int):
    widths = plan_crop_widths(args, records, centers, source_width, source_height)
    base_ratio = args.crop_height / args.crop_width
    # Ease after clipping to the width-aware frame bounds, then clip again:
    # a pan that would hit the edge of the source now decelerates into it
    # instead of stopping dead, and the second clip removes the tiny
    # overshoot easing can introduce where the bounds themselves move.
    halves = widths / 2
    centers = np.clip(centers, halves, source_width - halves)
    centers = np.clip(ease_path(centers, args.sample_rate), halves, source_width - halves)
    for record, center, width in zip(records, centers, widths):
        crop_height = min(float(source_height), width * base_ratio)
        half = width / 2
        center = float(np.clip(center, half, source_width - half))
        record["crop_width"] = round(width, 2)
        record["crop_height"] = round(crop_height, 2)
        record["center_x"] = round(center, 2)
        record["crop_x"] = round(center - half, 2)
        record["crop_y"] = round(source_height - crop_height, 2)
    return widths


def triangular_kernel(half_width_seconds: float, sample_rate: float) -> np.ndarray:
    """A normalized triangular smoothing kernel spanning a fixed duration.

    Windows are specified in seconds so the camera behaves the same at any
    analysis sample rate: at 2 Hz a 1.5 s half-width is [1, 2, 3, 4, 3, 2, 1].
    """
    half = max(1, round(half_width_seconds * sample_rate))
    kernel = np.array([half + 1 - abs(i) for i in range(-half, half + 1)], dtype=float)
    return kernel / kernel.sum()


def smooth_series(values: np.ndarray, half_width_seconds: float, sample_rate: float) -> np.ndarray:
    kernel = triangular_kernel(half_width_seconds, sample_rate)
    half = len(kernel) // 2
    return np.convolve(np.pad(values, (half, half), mode="edge"), kernel, mode="valid")


def smooth_centers(raw: np.ndarray, crop_width: int, frame_width: int, sample_rate: float, calibration: Calibration):
    # Suppress brief detector mistakes (a 1 s median half-width), then soften
    # the operator's aim with a 1.5 s triangular window.
    median_half = max(1, round(1.0 * sample_rate))
    median = raw.copy()
    for i in range(len(raw)):
        lo, hi = max(0, i - median_half), min(len(raw), i + median_half + 1)
        median[i] = np.median(raw[lo:hi])
    desired = smooth_series(median, 1.5, sample_rate)

    dead_zone = calibration.pan_dead_zone
    max_speed = calibration.pan_speed  # source pixels per second
    centers = np.empty_like(desired)
    centers[0] = desired[0]
    max_step = max_speed / sample_rate
    for i in range(1, len(desired)):
        delta = desired[i] - centers[i - 1]
        if abs(delta) <= dead_zone:
            centers[i] = centers[i - 1]
            continue
        requested = delta - math.copysign(dead_zone, delta)
        centers[i] = centers[i - 1] + np.clip(requested, -max_step, max_step)

    half = crop_width / 2
    return np.clip(centers, half, frame_width - half)


def read_anchor_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if any(value.strip() for value in row.values() if value)]


def apply_manual_corrections(args: ProcessingConfig, times: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Blend hand-authored centers after automatic movement limiting.

    These are explicit operator overrides and may move faster than the normal
    automatic pan-speed cap.
    """
    if not args.corrections:
        return centers
    result = centers.copy()
    for row in read_anchor_csv(args.corrections):
        source_time = float(row["source_time_seconds"])
        relative_time = source_time - args.start
        target = float(row["center_x"])
        influence = np.maximum(0.0, 1.0 - np.abs(times - relative_time) / args.correction_radius)
        result = result * (1.0 - influence**2) + target * influence**2
    return result


def ease_path(centers: np.ndarray, sample_rate: float) -> np.ndarray:
    """Round the corners of a planned path without delaying it.

    Every earlier stage limits speed but not acceleration, so a path can
    change speed abruptly from one sample to the next. A zero-phase
    triangular window over EASE_HALF_WIDTH seconds on each side turns those
    corners into eases. It never increases a step, because every output is a
    convex combination of nearby inputs.
    """
    if len(centers) < 3:
        return centers
    return smooth_series(centers, EASE_HALF_WIDTH, sample_rate)
