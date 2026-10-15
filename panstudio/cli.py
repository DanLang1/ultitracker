"""Command-line adapter for the reusable processing stages."""
import argparse
import csv
import json
import os
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path
import cv2

from .artifacts import atomic_json, flights_version, load_detections, load_flights, load_plan, save_detections, save_flights, save_plan
from .events import report
from .masking import load_play_area_mask, save_mask_preview
from .models import DEVICES, ProcessingConfig
from .pipeline import detect, plan_camera, probe_video, render_plan, validate_reuse, validate_video


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prototype/output"))
    parser.add_argument("--output-name", default="tactical-crop", help="File name for the tactical video (.mp4 is added)")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--sample-rate", type=float, default=5.0)
    parser.add_argument("--inference-size", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--crop-width", type=int, default=2560)
    parser.add_argument("--crop-height", type=int, default=1440)
    parser.add_argument("--output-width", type=int, default=2560)
    parser.add_argument("--output-height", type=int, default=1440)
    parser.add_argument("--play-area", type=Path, help="Image with a closed red outline around valid play")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview-mask-only", action="store_true", help="Write the mask preview and stop")
    parser.add_argument("--corrections", type=Path, help="CSV with source_time_seconds,center_x anchors")
    parser.add_argument("--correction-radius", type=float, default=4.0)
    parser.add_argument("--disc-points", type=Path, help="CSV with source_time_seconds,disc_x,direction anchors")
    parser.add_argument("--disc-weight", type=float, default=0.22)
    parser.add_argument("--disc-radius", type=float, default=2.0)
    parser.add_argument("--look-room", type=float, default=0.15, help="Crop-width fraction shown ahead of the disc")
    parser.add_argument("--deep-lookahead", type=float, default=0.6, help="Seconds to project a deep pair's leading edge")
    parser.add_argument("--no-deep-tracking", action="store_true", help="Disable fast breakaway-pair coverage")
    parser.add_argument("--no-disc-tracking", action="store_true", help="Do not detect disc flights or pan to where they land")
    parser.add_argument("--flights-in", type=Path, help="Reuse saved disc flights instead of scanning the video again")
    parser.add_argument(
        "--no-formation-protection", action="store_true",
        help="Allow deep tracking to ignore formation visibility (A/B comparison)",
    )
    parser.add_argument("--no-auto-widen", action="store_true", help="Keep the tactical crop at its fixed width")
    parser.add_argument("--debug", action="store_true", help="Also render the full-width detection review video")
    modes.add_argument("--analysis-only", action="store_true", help="Save detection and pan data without rendering video")
    modes.add_argument("--detect-only", action="store_true", help="Save raw detections without planning or rendering")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--detections-in", type=Path, help="Reuse raw detections, including with a new field mask")
    reuse.add_argument("--plan-in", type=Path, help="Render an existing camera plan without detection or planning")
    parser.add_argument("--calibration", type=Path, help="JSON camera calibration overrides")
    parser.add_argument("--device", choices=DEVICES, default="auto",
                        help="Inference device; auto uses the Apple GPU (mps) when available")
    parser.add_argument("--supervised", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def export_plan(directory, plan):
    save_plan(directory / 'camera-plan.json', plan)
    atomic_json(directory / 'analysis.json', plan.samples)
    keys = ('time', 'source_time', 'raw_center_x', 'center_x', 'crop_x', 'crop_width', 'crop_height', 'crop_y')
    with (directory / 'pan-path.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_seconds', 'source_time_seconds', *keys[2:]])
        writer.writerows([record[key] for key in keys] for record in plan.samples)
    for filename, columns in (
        ('manual-corrections-template.csv', ['source_time_seconds', 'center_x']),
        ('disc-points-template.csv', ['source_time_seconds', 'disc_x', 'direction']),
    ):
        path = directory / filename
        if not path.exists():
            with path.open('w', newline='') as handle:
                csv.writer(handle).writerow(columns)


def suggestion_mask(directory, width, height):
    """Rasterize a saved outline suggestion so flight scanning can use it before the operator draws."""
    path = directory / 'suggested-outline.json'
    if not path.is_file():
        return None
    try:
        points = json.loads(path.read_text()).get('points')
    except (ValueError, TypeError):
        return None
    if not points or len(points) < 3:
        return None
    import numpy as np
    polygon = np.array([[p['x'] * (width - 1), p['y'] * (height - 1)] for p in points], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def export_flights(config, video, detections, mask, directory):
    """Scan for disc flights and save them; advisory, so failures are reported, not fatal."""
    from .disc import DETECTOR_VERSION, detect_flights
    report('disc', 0, 1, 'Scanning for disc flights…')
    try:
        flights = detect_flights(config, video, detections, mask, progress=lambda done, total: report('disc', done, total))
    except Exception as error:  # noqa: BLE001 - advisory output
        report('disc', 1, 1, f'Disc flight scan failed: {error}')
        return []
    save_flights(directory / 'flights.json', flights, config.start, config.end, DETECTOR_VERSION)
    report('disc', 1, 1, f'Found {len(flights)} disc flights')
    return [flight.to_json() for flight in flights]


def export_suggestion(directory, detections):
    """Write the suggested outline and sideline heat map next to the detections.

    A suggestion is advisory, so a failure here is reported but never fails
    the detection run that produced the reusable artifact.
    """
    report('planning', 0, 1, 'Suggesting the playing area…')
    try:
        from .suggest import suggest_outline
        points, heat = suggest_outline(detections)
        minutes = (detections.end - detections.start) / 60
        atomic_json(directory / 'suggested-outline.json', {'schema': 'suggested-outline/v1', 'points': points, 'minutes': round(minutes, 1)})
        if heat is not None:
            cv2.imwrite(str(directory / 'sideline-heat.png'), heat)
    except Exception as error:  # noqa: BLE001 - advisory output
        report('planning', 1, 1, f'No outline suggestion: {error}')
        return
    report('planning', 1, 1, 'Suggested the playing area' if points else 'Recording too short for an outline suggestion')


def watch_parent():
    """Stop this worker and its encoders if the owning server disappears."""
    if os.getpgrp() != os.getpid():
        raise ValueError('Supervised workers must run in their own process group')
    def watch():
        while os.read(sys.stdin.fileno(), 4096):
            pass
        os.killpg(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch, daemon=True, name='parent-watchdog').start()


def main(argv=None):
    options = parse_args(argv)
    try:
        if options.supervised:
            watch_parent()
        config = ProcessingConfig.from_namespace(options)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        video = probe_video(config.input)
        if options.plan_in:
            if options.detect_only or options.preview_mask_only or options.analysis_only:
                raise ValueError('--plan-in is a render-only operation')
            plan = load_plan(options.plan_in)
            validate_reuse(config, plan, video)
            validate_video(replace(config, crop_width=plan.crop_width, crop_height=plan.crop_height), video)
            render_plan(config, plan)
        else:
            if config.end > video.duration + 1 / video.fps:
                raise ValueError('Requested end exceeds the source duration')
            if options.detect_only:
                detections = load_detections(options.detections_in) if options.detections_in else detect(config, video)
                validate_reuse(config, detections, video)
                save_detections(config.output_dir / 'detections.json', detections)
                export_suggestion(config.output_dir, detections)
                if not config.no_disc_tracking:
                    export_flights(config, video, detections, suggestion_mask(config.output_dir, video.width, video.height), config.output_dir)
                report('complete', 1, 1, 'Saved raw detections')
                return
            validate_video(config, video)
            mask = None
            if config.play_area:
                mask = load_play_area_mask(config.play_area, video.width, video.height)
                cap = cv2.VideoCapture(str(config.input))
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, config.start * 1000)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('Could not decode a frame for the mask preview')
                    save_mask_preview(mask, frame, config.output_dir / 'play-area-preview.jpg')
                finally:
                    cap.release()
            if options.preview_mask_only:
                if mask is None:
                    raise ValueError('--preview-mask-only requires --play-area')
                report('complete', 1, 1, 'Saved mask preview')
                return
            if options.detections_in:
                detections = load_detections(options.detections_in)
                validate_reuse(config, detections, video)
                report('detection', len(detections.samples), len(detections.samples), 'Reusing saved detections')
            else:
                detections = detect(config, video)
            save_detections(config.output_dir / 'detections.json', detections)
            flights = []
            if not config.no_disc_tracking:
                from .disc import DETECTOR_VERSION
                # Flights saved by an older disc detector are scanned again rather than reused.
                if options.flights_in and flights_version(options.flights_in) == DETECTOR_VERSION:
                    flights = load_flights(options.flights_in)
                    save_flights(config.output_dir / 'flights.json', flights, config.start, config.end, DETECTOR_VERSION)
                    report('disc', 1, 1, f'Reusing {len(flights)} saved disc flights')
                else:
                    flights = export_flights(config, video, detections, mask, config.output_dir)
            plan = plan_camera(config, detections, mask, flights)
            export_plan(config.output_dir, plan)
            if not options.analysis_only:
                render_plan(config, plan)
        report('complete', 1, 1, f'Outputs are in {config.output_dir}')
    except Exception as error:
        report('error', message=str(error))
        raise
