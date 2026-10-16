from __future__ import annotations

import math
import numpy as np
from .models import Player, Sample
from .calibration import Calibration

# Calibration.track_distance is the displacement a player may make between
# samples taken this far apart; shorter gaps get proportionally tighter gates.
TRACK_GATE_SECONDS = 0.5
# Velocities are estimated over observations within this many seconds on
# each side, long enough to average detector jitter but short enough to
# notice a cut within a stride or two.
VELOCITY_WINDOW_SECONDS = 0.8
# Bystander suppression. A player who travels MOBILITY_FULL_CREDIT body
# heights within MOBILITY_WINDOW_SECONDS on either side keeps full weight;
# less mobile people scale down to MOBILITY_FLOOR.
MOBILITY_WINDOW_SECONDS = 5.0
MOBILITY_FULL_CREDIT = 2.0
MOBILITY_FLOOR = 0.1
MINIMUM_BODY_SCALE = 30.0
# A cell occupied in more than OCCUPANCY_THRESHOLD of the samples in a rolling
# OCCUPANCY_WINDOW_SECONDS window is treated as sideline or furniture.
OCCUPANCY_CELLS_ACROSS = 48
OCCUPANCY_WINDOW_SECONDS = 600.0
OCCUPANCY_MINIMUM_SECONDS = 120.0
OCCUPANCY_THRESHOLD = 0.5
OCCUPANCY_FACTOR = 0.2
# Boxes shorter than SEATED_ASPECT times their width are likely seated.
SEATED_ASPECT = 1.1
SEATED_FACTOR = 0.5


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.size == 0:
        raise ValueError("weighted_median requires at least one value and weight")
    if values.shape != weights.shape:
        raise ValueError("weighted_median values and weights must have matching shapes")
    if np.any(weights < 0):
        raise ValueError("weighted_median weights cannot be negative")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("weighted_median weights must have a positive total")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), total_weight / 2, side="left")])


def field_bounds(foot_y: float, width: int, calibration: Calibration) -> tuple[float, float]:
    """Approximate the two sidelines from the calibrated camera perspective."""
    depth = max(0.0, foot_y - calibration.sideline_origin_y)
    return calibration.sideline_left_slope * depth, width - calibration.sideline_right_slope * depth


def select_field_players(
    boxes: np.ndarray,
    confidences: np.ndarray,
    width: int,
    play_area_mask: np.ndarray | None,
    calibration: Calibration,
) -> list[Player]:
    selected = []
    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = map(float, box)
        height = y2 - y1
        center_x = (x1 + x2) / 2
        if play_area_mask is None:
            left, right = field_bounds(y2, width, calibration)
            if height < calibration.minimum_player_height or y2 < calibration.minimum_foot_y or not (left <= center_x <= right):
                continue
        elif height < calibration.masked_minimum_player_height:
            # The operator-drawn mask is more reliable than camera-specific
            # perspective rules, especially for small players on deep cuts.
            continue
        foot_x = int(np.clip(center_x, 0, width - 1))
        foot_y = int(np.clip(y2 - calibration.foot_inset, 0, play_area_mask.shape[0] - 1)) if play_area_mask is not None else 0
        if play_area_mask is not None and play_area_mask[foot_y, foot_x] == 0:
            continue
        size_ratio = height / calibration.typical_player_height
        if play_area_mask is None:
            # Without a mask, nearby players must outweigh tiny detections from
            # games beyond the far sideline.
            weight = float(confidence) * max(0.45, min(3.2, size_ratio)) ** 1.35
        else:
            # The mask already removes off-field people. A sideline camera makes
            # near-sideline players several times taller than far-sideline
            # players, but not more important, so keep the size influence mild.
            weight = float(confidence) * max(0.6, min(1.6, size_ratio)) ** 0.5
        selected.append(
            {
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(float(confidence), 4),
                "weight": round(weight, 4),
                "center_x": center_x,
                "center_y": (y1 + y2) / 2,
                "height": height,
                "stationary_factor": 1.0,
                "velocity_x": 0.0,
                "velocity_y": 0.0,
                "speed_body": 0.0,
            }
        )
    return selected


def suppress_stationary_tracks(records: list[Sample], sample_rate: float, calibration: Calibration):
    """Down-weight people who remain in one patch for most of the point."""
    tracks: dict[int, list[tuple[int, Player]]] = {}
    track_state: dict[int, dict] = {}
    next_track_id = 0
    maximum_gap = max(1, round(1.5 * sample_rate))
    base_distance = calibration.track_distance
    velocity_half_window = max(1, round(VELOCITY_WINDOW_SECONDS * sample_rate))

    for frame_index, record in enumerate(records):
        track_state = {
            track_id: state
            for track_id, state in track_state.items()
            if frame_index - state["frame"] <= maximum_gap
        }
        players = record["players"]
        pairs = []
        for player_index, player in enumerate(players):
            for track_id, state in track_state.items():
                gap = frame_index - state["frame"]
                if gap > maximum_gap:
                    continue
                predicted_x = state["x"] + state["vx"] * gap
                predicted_y = state["y"] + state["vy"] * gap
                distance = math.hypot(player["center_x"] - predicted_x, player["center_y"] - predicted_y)
                # The calibrated distance is the allowed displacement over
                # TRACK_GATE_SECONDS, so it scales with the actual gap and
                # stays meaningful at any sample rate. A player can also
                # always move about one body length between nearby samples,
                # which keeps near-sideline sprinters attached when a sprint
                # starts from rest and the predicted velocity is still zero.
                gap_seconds = gap / sample_rate
                gate = max(base_distance * gap_seconds / TRACK_GATE_SECONDS, 1.2 * state["height"])
                height_ratio = player["height"] / max(1.0, state["height"])
                if distance <= gate and 0.5 <= height_ratio <= 2.0:
                    size_penalty = abs(math.log(height_ratio)) * 0.35
                    gap_penalty = 0.15 * (gap - 1)
                    score = distance / base_distance + size_penalty + gap_penalty
                    pairs.append((score, player_index, track_id, gap))
        assigned_players, assigned_tracks = set(), set()
        for _, player_index, track_id, gap in sorted(pairs):
            if player_index in assigned_players or track_id in assigned_tracks:
                continue
            player = players[player_index]
            state = track_state[track_id]
            observed_vx = (player["center_x"] - state["x"]) / gap
            observed_vy = (player["center_y"] - state["y"]) / gap
            player["track_id"] = track_id
            tracks[track_id].append((frame_index, player))
            track_state[track_id] = {
                "x": player["center_x"],
                "y": player["center_y"],
                "vx": 0.55 * state["vx"] + 0.45 * observed_vx,
                "vy": 0.55 * state["vy"] + 0.45 * observed_vy,
                "height": player["height"],
                "frame": frame_index,
            }
            assigned_players.add(player_index)
            assigned_tracks.add(track_id)
        for player_index, player in enumerate(players):
            if player_index in assigned_players:
                continue
            track_id = next_track_id
            next_track_id += 1
            player["track_id"] = track_id
            tracks[track_id] = [(frame_index, player)]
            track_state[track_id] = {
                "x": player["center_x"],
                "y": player["center_y"],
                "vx": 0.0,
                "vy": 0.0,
                "height": player["height"],
                "frame": frame_index,
            }

    minimum_samples = round(10.0 * sample_rate)
    for observations in tracks.values():
        if len(observations) >= 2:
            points = np.array([[p["center_x"], p["center_y"]] for _, p in observations])
            frames = np.array([frame for frame, _ in observations], dtype=float)
            heights = np.array([p["height"] for _, p in observations])
            for index, (_, player) in enumerate(observations):
                lo = max(0, index - velocity_half_window)
                hi = min(len(observations) - 1, index + velocity_half_window)
                elapsed = (frames[hi] - frames[lo]) / sample_rate
                if elapsed <= 0:
                    continue
                delta = points[hi] - points[lo]
                velocity = delta / elapsed
                body_height = max(calibration.minimum_body_height, float(np.median(heights[lo:hi + 1])))
                player["velocity_x"] = round(float(velocity[0]), 2)
                player["velocity_y"] = round(float(velocity[1]), 2)
                player["speed_body"] = round(float(np.linalg.norm(velocity) / body_height), 3)

        if len(observations) < minimum_samples:
            continue
        points = np.array([[p["center_x"], p["center_y"]] for _, p in observations])
        center = np.median(points, axis=0)
        radius_90 = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
        if radius_90 <= calibration.stationary_radius:
            for _, player in observations:
                player["stationary_factor"] = 0.2
                player["weight"] = round(player["weight"] * 0.2, 4)
    return tracks


def suppress_bystanders(records: list[Sample], sample_rate: float, frame_width: int, frame_height: int):
    """Down-weight people who do not behave like players, whatever the mask says.

    Two signals that need no field geometry:

    * Mobility: how far a track travels within a window around each sample,
      in body heights. Players run several body lengths every few seconds;
      sideline people shuffle. The weight scales down smoothly toward a floor.
    * Occupancy: a cell of the frame that contains somebody in most samples
      of a rolling window is a bench, a tent line, or a fixed object, never
      a spot on the field. Everything detected there is suppressed. This is
      independent of track continuity, which fails inside dense crowds.

    A seated or crouching person also makes a squat box; those get a mild
    penalty so a bench inside the outline does not pull the camera.
    """
    if not records:
        return
    half_window = max(1, round(MOBILITY_WINDOW_SECONDS * sample_rate))
    by_track: dict[int, list[tuple[int, Player]]] = {}
    for frame_index, record in enumerate(records):
        for player in record["players"]:
            by_track.setdefault(int(player.get("track_id", -1)), []).append((frame_index, player))
    for observations in by_track.values():
        frames = np.array([frame for frame, _ in observations])
        points = np.array([[player["center_x"], player["box"][3]] for _, player in observations])
        body = max(MINIMUM_BODY_SCALE, float(np.median([player["height"] for _, player in observations])))
        for index, (frame, player) in enumerate(observations):
            nearby = np.abs(frames - frame) <= half_window
            travelled = float(np.linalg.norm(points[nearby] - points[index], axis=1).max())
            player["mobility"] = round(travelled / body, 2)

    # Occupancy windows are two fine cells wide and overlap by one, so a crowd
    # straddling a boundary is still seen whole. Each detection marks every
    # window that contains it and is scored by the busiest of them.
    fine = max(8, round(frame_width / (2 * OCCUPANCY_CELLS_ACROSS)))
    rows, columns = frame_height // fine + 2, frame_width // fine + 2
    windows_per_record = []
    for record in records:
        groups = []  # one tuple of window ids per player, in player order
        for player in record["players"]:
            row = min(rows - 2, int(player["box"][3] // fine))
            column = min(columns - 2, int(player["center_x"] // fine))
            groups.append(tuple((row - dr) * columns + (column - dc) for dr in (0, 1) for dc in (0, 1)))
        windows_per_record.append(groups)
    presence = [np.unique(np.array([w for group in marks for w in group], dtype=np.int64)) for marks in windows_per_record]
    occupancy_half = max(1, round(OCCUPANCY_WINDOW_SECONDS / 2 * sample_rate))
    minimum_samples = round(OCCUPANCY_MINIMUM_SECONDS * sample_rate)
    window = np.zeros(rows * columns, dtype=np.int32)
    lo = hi = 0  # the rolling window covers records[lo:hi]

    for frame_index, record in enumerate(records):
        target_lo, target_hi = max(0, frame_index - occupancy_half), min(len(records), frame_index + occupancy_half + 1)
        while hi < target_hi:
            window[presence[hi]] += 1
            hi += 1
        while lo < target_lo:
            window[presence[lo]] -= 1
            lo += 1
        count = hi - lo
        for player, group in zip(record["players"], windows_per_record[frame_index]):
            occupancy = float(window[list(group)].max()) / count if count >= minimum_samples else 0.0
            player["occupancy"] = round(occupancy, 3)
            factor = min(1.0, max(MOBILITY_FLOOR, player.get("mobility", 0.0) / MOBILITY_FULL_CREDIT))
            if occupancy > OCCUPANCY_THRESHOLD:
                factor = min(factor, OCCUPANCY_FACTOR)
            box = player["box"]
            if (box[3] - box[1]) < SEATED_ASPECT * max(1.0, box[2] - box[0]):
                factor *= SEATED_FACTOR
            player["weight"] = round(player["weight"] * factor, 4)
            player["stationary_factor"] = round(min(float(player.get("stationary_factor", 1.0)), factor), 3)


def calculate_raw_centers(records: list[Sample], width: int):
    last_center = width / 2
    for record in records:
        players = record["players"]
        if players:
            centers = np.array([p["center_x"] for p in players])
            weights = np.array([p["weight"] for p in players])
            last_center = weighted_median(centers, weights)
        record["raw_center_x"] = round(last_center, 2)


def identify_deep_pairs(records: list[Sample], sample_rate: float, calibration: Calibration):
    """Find persistent, fast breakaway tracks without chasing every deep cut."""
    # Confirm a pair after about 0.6 s of agreement, never fewer than two samples.
    confirmation_samples = max(2, round(0.6 * sample_rate))
    pending_pair: tuple[int, int] | None = None
    pending_direction = 0
    pending_count = 0
    active_track_ids: list[int] = []
    breakaway: list[int] = []
    active_direction = 0
    hold_samples = 0
    last_members: dict[int, dict] = {}

    for record_index, record in enumerate(records):
        baseline = record["raw_center_x"]
        groups = {-1: [], 1: []}
        for player in record["players"]:
            track_id = int(player["track_id"])
            last_members[track_id] = {
                "x": float(player["center_x"]),
                "velocity_x": float(player.get("velocity_x", 0.0)),
                "record_index": record_index,
            }
            velocity_x = player.get("velocity_x", 0.0)
            direction = 1 if velocity_x > 0 else -1
            lead = direction * (player["center_x"] - baseline)
            speed_body = player.get("speed_body", 0.0)
            if abs(velocity_x) < calibration.deep_minimum_velocity or speed_body < 0.52 or lead < calibration.deep_minimum_lead:
                continue
            score = speed_body * (1.0 + min(2.0, lead / calibration.deep_score_distance))
            groups[direction].append((score, lead, player))

        candidate = None
        for direction, runners in groups.items():
            runners.sort(key=lambda item: item[0], reverse=True)
            # A useful deep signal is normally an offensive runner and the
            # defender traveling with them, not merely the two fastest people
            # anywhere on the field.
            for first_index, first in enumerate(runners):
                for second in runners[first_index + 1:]:
                    first_player, second_player = first[2], second[2]
                    horizontal_gap = abs(first_player["center_x"] - second_player["center_x"])
                    vertical_gap = abs(first_player["center_y"] - second_player["center_y"])
                    if horizontal_gap > calibration.pair_horizontal_gap or vertical_gap > calibration.pair_vertical_gap:
                        continue
                    velocity_gap = abs(first_player["velocity_x"] - second_player["velocity_x"])
                    if velocity_gap > calibration.pair_velocity_gap:
                        continue
                    proximity_bonus = 1.2 * (1.0 - horizontal_gap / calibration.pair_horizontal_gap)
                    score = first[0] + second[0] + proximity_bonus
                    if candidate is None or score > candidate[0]:
                        candidate = (score, direction, [first_player, second_player])

        candidate_confirmed = False
        if candidate:
            _, direction, players = candidate
            pair = tuple(sorted(int(player["track_id"]) for player in players))
            if pair == pending_pair and direction == pending_direction:
                pending_count += 1
            else:
                pending_pair = pair
                pending_direction = direction
                pending_count = 1
            if pending_count >= confirmation_samples:
                # A cutter who beats their defender outruns the pair limit and
                # a new pair forms behind them, around that defender. Keep the
                # cutter while they are still sprinting the same way, ahead of
                # it; an unrelated pair elsewhere does not inherit them.
                runners = {int(player["track_id"]): float(player["center_x"]) for _, _, player in groups[direction]}
                front = max(direction * runners[track_id] for track_id in pair)
                same_play = direction == active_direction and bool(set(pair) & set(active_track_ids))
                breakaway = [track_id for track_id in active_track_ids
                             if same_play and track_id not in pair
                             and track_id in runners and direction * runners[track_id] > front]
                active_track_ids = list(pair) + breakaway
                active_direction = direction
                hold_samples = round(2.5 * sample_rate)
                candidate_confirmed = True
        else:
            pending_pair = None
            pending_direction = 0
            pending_count = 0

        member_positions = []
        member_velocities = []
        for track_id in active_track_ids:
            member = last_members.get(track_id)
            if member is None:
                continue
            elapsed = (record_index - member["record_index"]) / sample_rate
            if elapsed > 1.5:
                continue
            member_positions.append(member["x"] + member["velocity_x"] * elapsed)
            member_velocities.append(member["velocity_x"])

        if hold_samples > 0 and member_positions:
            if not candidate_confirmed:
                hold_samples -= 1
            record["deep_track_ids"] = active_track_ids.copy()
            record["deep_direction"] = active_direction
            record["deep_pair_min_x"] = round(float(min(member_positions)), 2)
            record["deep_pair_max_x"] = round(float(max(member_positions)), 2)
            record["deep_center_x"] = round(float(np.mean(member_positions)), 2)
            leading_index = int(np.argmax(np.array(member_positions) * active_direction))
            record["deep_leading_x"] = round(float(member_positions[leading_index]), 2)
            record["deep_velocity_x"] = round(float(member_velocities[leading_index]), 2)
            record["deep_active"] = True
            record["deep_breakaway"] = bool(breakaway)
        else:
            active_track_ids = []
            breakaway = []
            active_direction = 0
            record["deep_track_ids"] = []
            record["deep_direction"] = 0
            record["deep_active"] = False

    # Detection uses future frames already, so make that offline advantage
    # explicit: begin a little before confirmation without committing too early.
    pre_roll_samples = max(1, round(0.5 * sample_rate))
    for index, record in enumerate(records):
        if not record.get("deep_active") or (index and records[index - 1].get("deep_active")):
            continue
        anchor_players = {
            int(player["track_id"]): player
            for player in record["players"]
            if int(player.get("track_id", -1)) in record["deep_track_ids"]
        }
        for earlier_index in range(max(0, index - pre_roll_samples), index):
            earlier = records[earlier_index]
            if earlier.get("deep_active"):
                continue
            elapsed = (index - earlier_index) / sample_rate
            positions = []
            velocities = []
            earlier_players = {int(player.get("track_id", -1)): player for player in earlier["players"]}
            for track_id in record["deep_track_ids"]:
                player = earlier_players.get(track_id) or anchor_players.get(track_id)
                if player is None:
                    continue
                velocity = float(player.get("velocity_x", 0.0))
                position = float(player["center_x"])
                if track_id not in earlier_players:
                    position -= velocity * elapsed
                positions.append(position)
                velocities.append(velocity)
            if not positions:
                continue
            direction = int(record["deep_direction"])
            leading_index = int(np.argmax(np.array(positions) * direction))
            earlier.update(
                {
                    "deep_track_ids": record["deep_track_ids"].copy(),
                    "deep_direction": direction,
                    "deep_pair_min_x": round(float(min(positions)), 2),
                    "deep_pair_max_x": round(float(max(positions)), 2),
                    "deep_center_x": round(float(np.mean(positions)), 2),
                    "deep_leading_x": round(float(positions[leading_index]), 2),
                    "deep_velocity_x": round(float(velocities[leading_index]), 2),
                    "deep_active": True,
                    "deep_preroll": True,
                }
            )
