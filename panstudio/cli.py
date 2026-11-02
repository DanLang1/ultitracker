"""Command-line adapter for the reusable processing stages."""
import argparse
import csv
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
from .models import DEFAULT_MODEL, DEVICES, ProcessingConfig
from .pipeline import detect, plan_camera, probe_video, render_plan, select_device, validate_reuse, validate_video
from .timing import StageTimer, machine_info


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prototype/output"))
    parser.add_argument("--output-name", default="tactical-crop", help="File name for the tactical video (.mp4 is added)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-rate", type=float, default=5.0)
    parser.add_argument("--inference-size", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--crop-width", type=int, default=2560)
    parser.add_argument("--crop-height", type=int, default=1440)
    parser.add_argument("--output-width", type=int, default=2560)
    parser.add_argument("--output-height", type=int, default=1440)
    parser.add_argument("--play-area", type=Path, help="Binary mask, or an image with a closed red outline around valid play")
    parser.add_argument("--corrections", type=Path, help="CSV with source_time_seconds,center_x anchors")
    parser.add_argument("--correction-radius", type=float, default=4.0)
    parser.add_argument("--look-room", type=float, default=0.15, help="Crop-width fraction shown ahead of a disc landing")
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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview-mask-only", action="store_true", help="Write the mask preview and stop")
    modes.add_argument("--analysis-only", action="store_true", help="Save detection and pan data without rendering video")
    modes.add_argument("--detect-only", action="store_true", help="Save raw detections without planning or rendering")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--detections-in", type=Path, help="Reuse raw detections, including with a new field mask")
    reuse.add_argument("--plan-in", type=Path, help="Render an existing camera plan without detection or planning")
    parser.add_argument("--calibration", type=Path, help="JSON camera calibration overrides")
    parser.add_argument("--device", choices=DEVICES, default="auto",
                        help="Inference device; auto uses an NVIDIA (cuda) or Apple (mps) GPU when available")
    parser.add_argument("--supervised", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.plan_in and (options.detect_only or options.preview_mask_only or options.analysis_only):
        parser.error("--plan-in is a render-only operation")
    if options.detect_only and options.detections_in:
        parser.error("--detect-only runs detection; it cannot reuse --detections-in")
    return options


def export_plan(directory, plan):
    save_plan(directory / 'camera-plan.json', plan)
    atomic_json(directory / 'analysis.json', plan.samples)
    keys = ('time', 'source_time', 'raw_center_x', 'center_x', 'crop_x', 'crop_width', 'crop_height', 'crop_y')
    with (directory / 'pan-path.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_seconds', 'source_time_seconds', *keys[2:]])
        writer.writerows([record[key] for key in keys] for record in plan.samples)


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


def flights_for(config, options, video, detections, mask):
    if config.no_disc_tracking:
        return []
    from .disc import DETECTOR_VERSION
    # Flights saved by an older disc detector are scanned again rather than reused.
    if options.flights_in and flights_version(options.flights_in) == DETECTOR_VERSION:
        flights = load_flights(options.flights_in)
        save_flights(config.output_dir / 'flights.json', flights, config.start, config.end, DETECTOR_VERSION)
        report('disc', 1, 1, f'Reusing {len(flights)} saved disc flights')
        return flights
    return export_flights(config, video, detections, mask, config.output_dir)


def play_area_mask(config, video):
    """Load the field mask and save a preview of it over the first analysed frame."""
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
    return mask


def watch_parent():
    """Stop this worker and its encoders if the owning server disappears."""
    if os.getpgrp() != os.getpid():
        raise ValueError('Supervised workers must run in their own process group')
    def watch():
        while os.read(sys.stdin.fileno(), 4096):
            pass
        os.killpg(os.getpid(), signal.SIGTERM)
    threading.Thread(target=watch, daemon=True, name='parent-watchdog').start()


def run(config, options, video, timer=None) -> str:
    """Run the requested stages and return the completion message."""
    timer = timer or StageTimer()
    if options.plan_in:
        plan = load_plan(options.plan_in)
        validate_reuse(config, plan, video)
        validate_video(replace(config, crop_width=plan.crop_width, crop_height=plan.crop_height), video)
        with timer.stage('render'):
            render_plan(config, plan)
        return f'Outputs are in {config.output_dir}'
    if options.detect_only:
        validate_video(config, video, check_crop=False)
        with timer.stage('detection'):
            save_detections(config.output_dir / 'detections.json', detect(config, video))
        return 'Saved raw detections'
    validate_video(config, video)
    with timer.stage('mask'):
        mask = play_area_mask(config, video) if config.play_area else None
    if options.preview_mask_only:
        if mask is None:
            raise ValueError('--preview-mask-only requires --play-area')
        return 'Saved mask preview'
    with timer.stage('detection'):
        if options.detections_in:
            detections = load_detections(options.detections_in)
            validate_reuse(config, detections, video)
            report('detection', len(detections.samples), len(detections.samples), 'Reusing saved detections')
        else:
            detections = detect(config, video)
        save_detections(config.output_dir / 'detections.json', detections)
    with timer.stage('disc'):
        flights = flights_for(config, options, video, detections, mask)
    with timer.stage('planning'):
        plan = plan_camera(config, detections, mask, flights)
        export_plan(config.output_dir, plan)
    if not options.analysis_only:
        with timer.stage('render'):
            render_plan(config, plan)
    return f'Outputs are in {config.output_dir}'


def save_timings(config, options, video, timer):
    """Record how long each stage took, on what hardware, and what was reused rather than recomputed."""
    atomic_json(config.output_dir / 'timings.json', {
        **timer.summary(),
        'footage_seconds': round(config.end - config.start, 3),
        'video': {'width': video.width, 'height': video.height, 'fps': video.fps},
        'start': config.start, 'end': config.end, 'sample_rate': config.sample_rate, 'model': config.model,
        'play_area': config.play_area is not None, 'debug': config.debug,
        'reused': {'detections': bool(options.detections_in), 'flights': bool(options.flights_in), 'plan': bool(options.plan_in)},
        'machine': machine_info(select_device(config.device)),
    })


def main(argv=None):
    options = parse_args(argv)
    try:
        if options.supervised:
            watch_parent()
        config = ProcessingConfig.from_namespace(options)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timer = StageTimer()
        video = probe_video(config.input)
        message = run(config, options, video, timer)
        save_timings(config, options, video, timer)
        report('complete', 1, 1, message)
    except Exception as error:
        report('error', message=str(error))
        raise
