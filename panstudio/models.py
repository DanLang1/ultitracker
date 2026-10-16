"""Explicit contracts at pipeline boundaries and typed algorithm working records."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import NotRequired, TypedDict
import math

from .calibration import Calibration


DEVICES = ('auto', 'cpu', 'mps')
DEFAULT_MODEL = 'yolo11s.pt'


class SourceIdentity(TypedDict):
    path: str
    size: int
    mtime_ns: int


class DetectorSettings(TypedDict):
    model: str
    inference_size: int
    confidence: float
    weights: SourceIdentity | None


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class DetectionSample:
    time: float
    source_time: float
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float


class Player(TypedDict):
    box: list[float]
    confidence: float
    center_x: float
    center_y: float
    height: float
    weight: float
    stationary_factor: float
    velocity_x: float
    velocity_y: float
    speed_body: float
    track_id: NotRequired[int]
    mobility: NotRequired[float]
    occupancy: NotRequired[float]


class Sample(TypedDict):
    time: float
    source_time: float
    players: list[Player]
    raw_center_x: NotRequired[float]
    deep_track_ids: NotRequired[list[int]]
    deep_direction: NotRequired[int]
    deep_pair_min_x: NotRequired[float]
    deep_pair_max_x: NotRequired[float]
    deep_center_x: NotRequired[float]
    deep_leading_x: NotRequired[float]
    deep_velocity_x: NotRequired[float]
    deep_active: NotRequired[bool]
    deep_breakaway: NotRequired[bool]
    deep_preroll: NotRequired[bool]
    deep_baseline_infeasible: NotRequired[bool]
    deep_protected_baseline_min: NotRequired[float]
    deep_protected_baseline_max: NotRequired[float]
    deep_coverage_target_x: NotRequired[float]
    deep_actual_pair_feasible: NotRequired[bool]
    deep_pair_feasible: NotRequired[bool]
    deep_protection_speed_limited: NotRequired[bool]
    flight_active: NotRequired[bool]
    flight_direction: NotRequired[int]
    flight_target_x: NotRequired[float]
    flight_min_x: NotRequired[float]
    flight_max_x: NotRequired[float]
    flight_x0: NotRequired[float]
    flight_y0: NotRequired[float]
    flight_x1: NotRequired[float]
    flight_y1: NotRequired[float]
    flight_start: NotRequired[float]
    flight_launch_seen: NotRequired[bool]
    crop_width: NotRequired[float]
    crop_height: NotRequired[float]
    center_x: NotRequired[float]
    crop_x: NotRequired[float]
    crop_y: NotRequired[float]


@dataclass(frozen=True)
class ProcessingConfig:
    input: Path
    start: float
    end: float
    output_dir: Path = Path('prototype/output')
    output_name: str = 'tactical-crop'
    model: str = DEFAULT_MODEL
    sample_rate: float = 5.0
    inference_size: int = 1920
    confidence: float = 0.18
    crop_width: int = 2560
    crop_height: int = 1440
    output_width: int = 2560
    output_height: int = 1440
    play_area: Path | None = None
    corrections: Path | None = None
    correction_radius: float = 4.0
    look_room: float = 0.15
    deep_lookahead: float = 0.6
    no_deep_tracking: bool = False
    no_disc_tracking: bool = False
    no_formation_protection: bool = False
    no_auto_widen: bool = False
    debug: bool = False
    device: str = 'auto'
    calibration: Calibration = field(default_factory=Calibration)

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"Device must be one of {', '.join(DEVICES)}")
        numeric = ('start', 'end', 'sample_rate', 'confidence', 'correction_radius', 'look_room', 'deep_lookahead')
        if any(not math.isfinite(getattr(self, name)) for name in numeric):
            raise ValueError('Processing settings must be finite')
        if self.start < 0 or self.end <= self.start:
            raise ValueError('Require 0 <= start < end')
        name = self.output_name.removesuffix('.mp4')
        if not name or name in ('.', '..') or '/' in name or '\\' in name or name == 'debug-wide':
            raise ValueError('Output name must be a plain file name')
        object.__setattr__(self, 'output_name', name)
        if min(self.sample_rate, self.correction_radius) <= 0:
            raise ValueError('Sample rate and correction radius must be positive')
        if not 0 <= self.confidence <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        if self.look_room < 0 or self.deep_lookahead < 0:
            raise ValueError('Look room and lookahead cannot be negative')
        if min(self.inference_size, self.crop_width, self.crop_height, self.output_width, self.output_height) <= 0:
            raise ValueError('Image dimensions must be positive')
        if self.output_width % 2 or self.output_height % 2:
            raise ValueError('Output dimensions must be even for H.264')

    @classmethod
    def from_namespace(cls, namespace):
        values = {f.name: getattr(namespace, f.name) for f in fields(cls) if hasattr(namespace, f.name)}
        values['calibration'] = Calibration.load(getattr(namespace, 'calibration', None))
        return cls(**values)


@dataclass(frozen=True)
class DetectionSet:
    video: VideoInfo
    source: SourceIdentity
    start: float
    end: float
    sample_rate: float
    detector: DetectorSettings
    samples: tuple[DetectionSample, ...]


@dataclass(frozen=True)
class CameraPlan:
    video: VideoInfo
    source: SourceIdentity
    start: float
    end: float
    crop_width: int
    crop_height: int
    samples: tuple[Sample, ...]
    settings: dict = field(default_factory=dict)
