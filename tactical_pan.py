#!/usr/bin/env python3
"""Create a smooth tactical crop from fixed, ultra-wide game footage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


def field_bounds(foot_y: float, width: int) -> tuple[float, float]:
    """Approximate the two blue sidelines visible in this camera view."""
    depth = max(0.0, foot_y - 900.0)
    return 0.62 * depth, width - 1.20 * depth


def load_play_area_mask(path: Path, width: int, height: int) -> np.ndarray:
    """Load a direct binary mask or turn a closed red outline into one."""
    annotated = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if annotated is None:
        raise ValueError(f"Could not read play-area image: {path}")
    if annotated.ndim == 2:
        return cv2.resize(
            (annotated > 127).astype(np.uint8) * 255,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    if annotated.shape[2] == 4:
        annotated = annotated[:, :, :3]
    annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_AREA)
    blue, green, red = cv2.split(annotated)
    red_ink = (red > 155) & (red.astype(np.int16) - green.astype(np.int16) > 55) & (
        red.astype(np.int16) - blue.astype(np.int16) > 55
    )
    candidates = []
    ink = (red_ink.astype(np.uint8) * 255)
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=2,
    )
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        area = abs(cv2.contourArea(contour))
        if w > width * 0.65 and h > height * 0.18 and area > width * height * 0.12:
            candidates.append(contour)
    if not candidates:
        raise ValueError("No large closed red play-area outline was found")
    outline = max(candidates, key=cv2.contourArea)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [outline], -1, 255, cv2.FILLED)
    return mask


def save_mask_preview(mask: np.ndarray, clean_frame: np.ndarray, path: Path):
    preview = clean_frame.copy()
    excluded = mask == 0
    tint = np.zeros_like(preview)
    tint[:, :] = (20, 20, 210)
    preview[excluded] = cv2.addWeighted(preview[excluded], 0.38, tint[excluded], 0.62, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (40, 230, 40), 14)
    cv2.imwrite(str(path), preview)


def select_field_players(
    boxes: np.ndarray,
    confidences: np.ndarray,
    width: int,
    play_area_mask: np.ndarray | None,
):
    selected = []
    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = map(float, box)
        height = y2 - y1
        center_x = (x1 + x2) / 2
        left, right = field_bounds(y2, width)
        if height < 45 or y2 < 930 or not (left <= center_x <= right):
            continue
        foot_x = int(np.clip(center_x, 0, width - 1))
        foot_y = int(np.clip(y2 - 2, 0, play_area_mask.shape[0] - 1)) if play_area_mask is not None else 0
        if play_area_mask is not None and play_area_mask[foot_y, foot_x] == 0:
            continue
        # Nearby players should influence the tactical center more than tiny
        # detections from games beyond the far sideline.
        weight = float(confidence) * max(0.45, min(3.2, height / 105.0)) ** 1.35
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


def suppress_stationary_tracks(records: list[dict], sample_rate: float):
    """Down-weight people who remain in one patch for most of the point."""
    tracks: dict[int, list[tuple[int, dict]]] = {}
    previous: dict[int, tuple[float, float]] = {}
    next_track_id = 0
    max_distance = 280.0

    for frame_index, record in enumerate(records):
        players = record["players"]
        pairs = []
        for player_index, player in enumerate(players):
            for track_id, position in previous.items():
                distance = math.hypot(player["center_x"] - position[0], player["center_y"] - position[1])
                if distance <= max_distance:
                    pairs.append((distance, player_index, track_id))
        assigned_players, assigned_tracks = set(), set()
        current = {}
        for _, player_index, track_id in sorted(pairs):
            if player_index in assigned_players or track_id in assigned_tracks:
                continue
            player = players[player_index]
            player["track_id"] = track_id
            tracks[track_id].append((frame_index, player))
            current[track_id] = (player["center_x"], player["center_y"])
            assigned_players.add(player_index)
            assigned_tracks.add(track_id)
        for player_index, player in enumerate(players):
            if player_index in assigned_players:
                continue
            track_id = next_track_id
            next_track_id += 1
            player["track_id"] = track_id
            tracks[track_id] = [(frame_index, player)]
            current[track_id] = (player["center_x"], player["center_y"])
        previous = current

    minimum_samples = round(10.0 * sample_rate)
    for observations in tracks.values():
        if len(observations) >= 2:
            points = np.array([[p["center_x"], p["center_y"]] for _, p in observations])
            frames = np.array([frame for frame, _ in observations], dtype=float)
            heights = np.array([p["height"] for _, p in observations])
            for index, (_, player) in enumerate(observations):
                lo = max(0, index - 2)
                hi = min(len(observations) - 1, index + 2)
                elapsed = (frames[hi] - frames[lo]) / sample_rate
                if elapsed <= 0:
                    continue
                delta = points[hi] - points[lo]
                velocity = delta / elapsed
                body_height = max(45.0, float(np.median(heights[lo:hi + 1])))
                player["velocity_x"] = round(float(velocity[0]), 2)
                player["velocity_y"] = round(float(velocity[1]), 2)
                player["speed_body"] = round(float(np.linalg.norm(velocity) / body_height), 3)

        if len(observations) < minimum_samples:
            continue
        points = np.array([[p["center_x"], p["center_y"]] for _, p in observations])
        center = np.median(points, axis=0)
        radius_90 = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
        if radius_90 <= 75.0:
            for _, player in observations:
                player["stationary_factor"] = 0.2
                player["weight"] = round(player["weight"] * 0.2, 4)
    return tracks


def calculate_raw_centers(records: list[dict], width: int):
    last_center = width / 2
    for record in records:
        players = record["players"]
        if players:
            centers = np.array([p["center_x"] for p in players])
            weights = np.array([p["weight"] for p in players])
            last_center = weighted_median(centers, weights)
        record["raw_center_x"] = round(last_center, 2)


def identify_deep_pairs(records: list[dict], sample_rate: float):
    """Find persistent, fast breakaway tracks without chasing every deep cut."""
    pending_direction = 0
    pending_count = 0
    active_track_ids: list[int] = []
    active_direction = 0
    active_deep_x = None
    hold_samples = 0

    for record in records:
        baseline = record["raw_center_x"]
        groups = {-1: [], 1: []}
        for player in record["players"]:
            velocity_x = player.get("velocity_x", 0.0)
            direction = 1 if velocity_x > 0 else -1
            lead = direction * (player["center_x"] - baseline)
            speed_body = player.get("speed_body", 0.0)
            if abs(velocity_x) < 85 or speed_body < 0.52 or lead < 180:
                continue
            score = speed_body * (1.0 + min(2.0, lead / 650.0))
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
                    if horizontal_gap > 520 or vertical_gap > 260:
                        continue
                    velocity_gap = abs(first_player["velocity_x"] - second_player["velocity_x"])
                    if velocity_gap > 260:
                        continue
                    proximity_bonus = 1.2 * (1.0 - horizontal_gap / 520.0)
                    score = first[0] + second[0] + proximity_bonus
                    if candidate is None or score > candidate[0]:
                        candidate = (score, direction, [first_player, second_player])

        candidate_confirmed = False
        if candidate:
            _, direction, players = candidate
            if direction == pending_direction:
                pending_count += 1
            else:
                pending_direction = direction
                pending_count = 1
            if pending_count >= 2:
                active_track_ids = [int(player["track_id"]) for player in players]
                active_direction = direction
                hold_samples = round(3.0 * sample_rate)
                candidate_confirmed = True
        else:
            pending_count = max(0, pending_count - 1)

        active_players = [
            player for player in record["players"] if int(player.get("track_id", -1)) in active_track_ids
        ]
        if active_players:
            deep_x = float(np.mean([player["center_x"] for player in active_players]))
            active_deep_x = deep_x
        if hold_samples > 0 and active_deep_x is not None:
            if not candidate_confirmed:
                hold_samples -= 1
            record["deep_track_ids"] = active_track_ids.copy()
            record["deep_direction"] = active_direction
            record["deep_center_x"] = round(float(active_deep_x), 2)
            record["deep_active"] = True
        else:
            active_track_ids = []
            active_direction = 0
            active_deep_x = None
            record["deep_track_ids"] = []
            record["deep_direction"] = 0
            record["deep_active"] = False


def apply_deep_coverage(args, records: list[dict], centers: np.ndarray) -> np.ndarray:
    """Keep a persistent runner/defender pair safely visible."""
    if args.no_deep_tracking:
        return centers
    desired = centers.copy()
    safe_half = args.crop_width / 2 - 310
    for index, record in enumerate(records):
        if record.get("deep_active"):
            deep_x = record["deep_center_x"]
            minimum_center = deep_x - safe_half
            maximum_center = deep_x + safe_half
            desired[index] = np.clip(desired[index], minimum_center, maximum_center)

    # Limit the extra camera movement and soften mode entry/exit.
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    desired = np.convolve(np.pad(desired, (2, 2), mode="edge"), kernel, mode="valid")
    result = np.empty_like(desired)
    result[0] = desired[0]
    max_step = 620.0 / args.sample_rate
    for index in range(1, len(desired)):
        result[index] = result[index - 1] + np.clip(desired[index] - result[index - 1], -max_step, max_step)
    return result


def smooth_centers(raw: np.ndarray, crop_width: int, frame_width: int, sample_rate: float):
    # Suppress one-sample detector mistakes, then soften the operator's aim.
    median = raw.copy()
    for i in range(len(raw)):
        lo, hi = max(0, i - 2), min(len(raw), i + 3)
        median[i] = np.median(raw[lo:hi])
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(median, (3, 3), mode="edge")
    desired = np.convolve(padded, kernel, mode="valid")

    dead_zone = 120.0
    max_speed = 520.0  # source pixels per second
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


def apply_disc_anchors(args, times: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Weakly bias framing toward manually identified disc positions plus look room.

    Manual anchors intentionally run after automatic movement limiting so an
    operator can override the generated camera path when necessary.
    """
    if not args.disc_points:
        return centers
    result = centers.copy()
    trend = np.gradient(centers, times, edge_order=1)
    for row in read_anchor_csv(args.disc_points):
        source_time = float(row["source_time_seconds"])
        relative_time = source_time - args.start
        disc_x = float(row["disc_x"])
        nearest = int(np.argmin(np.abs(times - relative_time)))
        direction_text = (row.get("direction") or "").strip()
        direction = float(direction_text) if direction_text else float(np.sign(trend[nearest]) or 1)
        target = disc_x + direction * args.crop_width * args.look_room
        influence = np.maximum(0.0, 1.0 - np.abs(times - relative_time) / args.disc_radius)
        blend = args.disc_weight * influence**2
        result = result * (1.0 - blend) + target * blend
    return result


def apply_manual_corrections(args, times: np.ndarray, centers: np.ndarray) -> np.ndarray:
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


def analyze(args, model: YOLO, cap: cv2.VideoCapture, width: int, fps: float, play_area_mask):
    sample_times = np.arange(args.start, args.end + 1e-6, 1 / args.sample_rate)
    records = []
    for index, timestamp in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = cap.read()
        if not ok:
            if records and timestamp >= args.end - (1 / args.sample_rate):
                print(f"Reached the end of the video at {timestamp:.2f}s; using {len(records)} completed samples.")
                break
            raise RuntimeError(f"Could not decode frame at {timestamp:.2f}s")
        result = model.predict(
            frame,
            imgsz=args.inference_size,
            conf=args.confidence,
            iou=0.45,
            classes=[0],
            device="cpu",
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        players = select_field_players(boxes, confidences, width, play_area_mask)
        records.append(
            {
                "time": round(float(timestamp - args.start), 3),
                "source_time": round(float(timestamp), 3),
                "players": players,
            }
        )
        print(f"Analyzed {index + 1:3d}/{len(sample_times)} at {timestamp:6.2f}s: {len(players):2d} field players")

    suppress_stationary_tracks(records, args.sample_rate)
    calculate_raw_centers(records, width)
    identify_deep_pairs(records, args.sample_rate)
    raw = np.array([record["raw_center_x"] for record in records])
    smooth = smooth_centers(raw, args.crop_width, width, args.sample_rate)
    smooth = apply_deep_coverage(args, records, smooth)
    relative_times = np.array([record["time"] for record in records])
    smooth = apply_disc_anchors(args, relative_times, smooth)
    smooth = apply_manual_corrections(args, relative_times, smooth)
    smooth = np.clip(smooth, args.crop_width / 2, width - args.crop_width / 2)
    for record, center in zip(records, smooth):
        record["center_x"] = round(float(center), 2)
        record["crop_x"] = int(round(center - args.crop_width / 2))
    return records


def encoder_command(args, output: Path, width: int, height: int, fps: float):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "pipe:0",
        "-ss", str(args.start), "-t", str(args.end - args.start), "-i", str(args.input),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(output),
    ]


def render(args, records, source_width: int, source_height: int, fps: float, output: Path, debug: bool):
    if debug:
        out_width, out_height = source_width // 2, source_height // 2
    else:
        out_width, out_height = args.output_width, args.output_height
    command = encoder_command(args, output, out_width, out_height, fps)
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(args.input))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    total_frames = int(round((args.end - args.start) * fps))
    sample_times = np.array([r["time"] for r in records])
    crop_xs = np.array([r["crop_x"] for r in records])
    crop_y = source_height - args.crop_height

    try:
        for frame_index in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break
            relative_time = frame_index / fps
            crop_x = int(round(np.interp(relative_time, sample_times, crop_xs)))
            if debug:
                sample_index = int(np.argmin(np.abs(sample_times - relative_time)))
                deep_ids = set(records[sample_index].get("deep_track_ids", []))
                for player in records[sample_index]["players"]:
                    x1, y1, x2, y2 = map(int, player["box"])
                    if int(player.get("track_id", -1)) in deep_ids:
                        color = (255, 220, 30)
                    elif player.get("stationary_factor", 1.0) < 1:
                        color = (30, 140, 255)
                    else:
                        color = (40, 220, 40)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                if records[sample_index].get("deep_active"):
                    cv2.putText(
                        frame,
                        "DEEP PAIR",
                        (60, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.7,
                        (255, 220, 30),
                        5,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(
                    frame,
                    (crop_x, crop_y),
                    (crop_x + args.crop_width - 1, crop_y + args.crop_height - 1),
                    (0, 220, 255),
                    9,
                )
                cv2.putText(
                    frame,
                    f"tactical crop  t={relative_time:05.1f}s",
                    (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,
                    (0, 220, 255),
                    6,
                    cv2.LINE_AA,
                )
                rendered = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            else:
                rendered = frame[crop_y:crop_y + args.crop_height, crop_x:crop_x + args.crop_width]
                if rendered.shape[1] != out_width or rendered.shape[0] != out_height:
                    rendered = cv2.resize(rendered, (out_width, out_height), interpolation=cv2.INTER_AREA)
            assert encoder.stdin is not None
            encoder.stdin.write(rendered.tobytes())
            if frame_index % max(1, round(fps * 5)) == 0:
                print(f"Rendering {output.name}: {frame_index}/{total_frames}")
    finally:
        cap.release()
        if encoder.stdin:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg encoder exited with status {return_code}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prototype/output"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--sample-rate", type=float, default=2.0)
    parser.add_argument("--inference-size", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--crop-width", type=int, default=2560)
    parser.add_argument("--crop-height", type=int, default=1440)
    parser.add_argument("--output-width", type=int, default=2560)
    parser.add_argument("--output-height", type=int, default=1440)
    parser.add_argument("--play-area", type=Path, help="Image with a closed red outline around valid play")
    parser.add_argument("--preview-mask-only", action="store_true", help="Write the mask preview and stop")
    parser.add_argument("--corrections", type=Path, help="CSV with source_time_seconds,center_x anchors")
    parser.add_argument("--correction-radius", type=float, default=4.0)
    parser.add_argument("--disc-points", type=Path, help="CSV with source_time_seconds,disc_x,direction anchors")
    parser.add_argument("--disc-weight", type=float, default=0.22)
    parser.add_argument("--disc-radius", type=float, default=2.0)
    parser.add_argument("--look-room", type=float, default=0.15, help="Crop-width fraction shown ahead of the disc")
    parser.add_argument("--no-deep-tracking", action="store_true", help="Disable fast breakaway-pair coverage")
    parser.add_argument("--debug", action="store_true", help="Also render the full-width detection review video")
    parser.add_argument("--analysis-only", action="store_true", help="Save detection and pan data without rendering video")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.input}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if args.crop_width > width or args.crop_height > height:
        raise ValueError("Crop is larger than the source video")

    play_area_mask = None
    if args.play_area:
        play_area_mask = load_play_area_mask(args.play_area, width, height)
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
        ok, clean_frame = cap.read()
        if not ok:
            raise RuntimeError("Could not decode a frame for the mask preview")
        save_mask_preview(play_area_mask, clean_frame, args.output_dir / "play-area-preview.jpg")
        if args.preview_mask_only:
            print(f"Wrote {args.output_dir / 'play-area-preview.jpg'}")
            cap.release()
            return

    model = YOLO(args.model)
    records = analyze(args, model, cap, width, fps, play_area_mask)
    cap.release()

    with (args.output_dir / "analysis.json").open("w") as handle:
        json.dump(records, handle, indent=2)
    with (args.output_dir / "pan-path.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_seconds", "source_time_seconds", "raw_center_x", "center_x", "crop_x"])
        for record in records:
            writer.writerow([record[key] for key in ("time", "source_time", "raw_center_x", "center_x", "crop_x")])
    with (args.output_dir / "manual-corrections-template.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(["source_time_seconds", "center_x"])
    with (args.output_dir / "disc-points-template.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(["source_time_seconds", "disc_x", "direction"])

    if args.analysis_only:
        print(f"Finished analysis. Data is in {args.output_dir}")
        return

    if args.debug:
        render(args, records, width, height, fps, args.output_dir / "debug-wide.mp4", debug=True)
    render(args, records, width, height, fps, args.output_dir / "tactical-crop.mp4", debug=False)
    print(f"Finished. Outputs are in {args.output_dir}")


if __name__ == "__main__":
    main()
