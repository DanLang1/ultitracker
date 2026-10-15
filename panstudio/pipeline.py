"""Composable detection, planning, and rendering stages. No CLI or web state."""
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from .artifacts import source_identity
from .events import report
from .media import render, sampled_analysis_frames
from .models import CameraPlan, Detection, DetectionSample, DetectionSet, DetectorSettings, ProcessingConfig, Sample, VideoInfo
from .planning import annotate_flights, apply_crop_geometry, apply_deep_coverage, apply_disc_anchors, apply_flight_targets, apply_manual_corrections, smooth_centers
from .tracking import calculate_raw_centers, identify_deep_pairs, select_field_players, suppress_bystanders, suppress_stationary_tracks


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Could not open {path}')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS)), float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if min(width, height, fps, frames) <= 0:
            raise ValueError('Video has invalid dimensions, frame rate, or duration')
        return VideoInfo(width, height, fps, frames / fps)
    finally:
        cap.release()


def validate_video(config: ProcessingConfig, video: VideoInfo):
    if config.end > video.duration + 1 / video.fps:
        raise ValueError('Requested end exceeds the source duration')
    if config.crop_width > video.width or config.crop_height > video.height:
        raise ValueError('Crop is larger than the source video')


def detector_settings(config: ProcessingConfig) -> DetectorSettings:
    weights = Path(config.model)
    return {'model': config.model, 'inference_size': config.inference_size, 'confidence': config.confidence,
            'weights': source_identity(weights) if weights.is_file() else None}


def select_device(requested: str = 'auto') -> str:
    """Resolve 'auto' to the Apple GPU when PyTorch can use it, otherwise the CPU.

    The device only changes inference speed, not the detector settings that
    decide whether saved detections can be reused.
    """
    if requested != 'auto':
        return requested
    import torch
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def detect(config: ProcessingConfig, video: VideoInfo, model=None) -> DetectionSet:
    """Keep all person detections; masks and camera policy belong to planning."""
    if model is None:
        from ultralytics import YOLO
        model = YOLO(config.model)
    device = select_device(config.device)
    report('detection', 0, 1, f'Running inference on {device}')
    sample_times = np.arange(config.start, config.end + 1e-6, 1 / config.sample_rate)
    samples = []
    for index, timestamp, frame in sampled_analysis_frames(config, video.width, video.height, sample_times):
        result = model.predict(frame, imgsz=config.inference_size, conf=config.confidence,
                               iou=0.45, classes=[0], device=device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        samples.append(DetectionSample(
            round(float(timestamp - config.start), 3), round(float(timestamp), 3),
            tuple(Detection(tuple(map(float, box)), float(confidence)) for box, confidence in zip(boxes, confidences)),
        ))
        report('detection', index + 1, len(sample_times))
    if len(samples) < len(sample_times):
        missing = float(sample_times[len(samples)])
        if not samples or missing < config.end - 1 / config.sample_rate:
            raise RuntimeError(f'Could not decode frame at {missing:.2f}s')
    return DetectionSet(video, source_identity(config.input), config.start, config.end, config.sample_rate,
                        detector_settings(config), tuple(samples))


def validate_reuse(config: ProcessingConfig, artifact: DetectionSet | CameraPlan, video: VideoInfo):
    if artifact.source != source_identity(config.input) or artifact.video != video:
        raise ValueError('Saved artifact belongs to a different or changed source video')
    if (artifact.start, artifact.end) != (config.start, config.end):
        raise ValueError('Saved artifact has a different start/end range')
    if isinstance(artifact, DetectionSet):
        if artifact.sample_rate != config.sample_rate:
            raise ValueError('Saved detections have a different sample rate')
        expected = detector_settings(config)
        if artifact.detector != expected:
            raise ValueError('Saved detections have different detector settings; rerun detection')


def plan_camera(config: ProcessingConfig, detections: DetectionSet, mask=None, flights=()) -> CameraPlan:
    """Build fresh working records on every call so repeated planning is deterministic."""
    video = detections.video
    calibration = config.calibration.scaled(video.width, video.height)
    args = replace(config, frame_width=video.width, calibration=calibration)
    records: list[Sample] = []
    for sample in detections.samples:
        players = select_field_players(
            np.array([d.box for d in sample.detections]),
            np.array([d.confidence for d in sample.detections]), video.width, mask, calibration,
        )
        records.append({'time': sample.time, 'source_time': sample.source_time, 'players': players})
    if not records:
        raise ValueError('Cannot plan an empty detection set')
    suppress_stationary_tracks(records, args.sample_rate, calibration)
    suppress_bystanders(records, args.sample_rate, video.width, video.height)
    calculate_raw_centers(records, video.width)
    identify_deep_pairs(records, args.sample_rate, calibration)
    raw = np.array([record['raw_center_x'] for record in records])
    centers = smooth_centers(raw, args.crop_width, video.width, args.sample_rate, calibration)
    centers = apply_deep_coverage(args, records, centers)
    if flights and not config.no_disc_tracking:
        ignored = annotate_flights(records, flights, args.crop_width, video.width, args.look_room)
        if ignored:
            report('planning', 0, 1, f'Ignored {len(ignored)} long flights that moved with a runner: '
                   + ', '.join(f"{int(f['start'] // 60)}:{f['start'] % 60:04.1f}" for f in ignored))
        centers = apply_flight_targets(args, records, centers)
    times = np.array([record['time'] for record in records])
    centers = apply_disc_anchors(args, times, centers)
    centers = apply_manual_corrections(args, times, centers)
    centers = np.clip(centers, args.crop_width / 2, video.width - args.crop_width / 2)
    apply_crop_geometry(args, records, centers, video.width, video.height)
    report('planning', 1, 1)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    return CameraPlan(video, detections.source, args.start, args.end, args.crop_width, args.crop_height, tuple(records), settings)


def render_plan(config: ProcessingConfig, plan: CameraPlan):
    video = plan.video
    args = replace(config, crop_width=plan.crop_width, crop_height=plan.crop_height)
    if args.debug:
        render(args, list(plan.samples), video.width, video.height, video.fps, args.output_dir / 'debug-wide.mp4', True)
    render(args, list(plan.samples), video.width, video.height, video.fps, args.output_dir / f'{args.output_name}.mp4', False)
