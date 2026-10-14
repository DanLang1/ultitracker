"""Camera-specific distances in reference pixels; durations remain in seconds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    reference_width: float = 4608.0
    reference_height: float = 1728.0
    sideline_origin_y: float = 900.0
    sideline_left_slope: float = 0.62
    sideline_right_slope: float = 1.20
    minimum_foot_y: float = 930.0
    minimum_player_height: float = 45.0
    masked_minimum_player_height: float = 26.0
    typical_player_height: float = 105.0
    foot_inset: float = 2.0
    track_distance: float = 420.0
    stationary_radius: float = 75.0
    minimum_body_height: float = 45.0
    deep_minimum_velocity: float = 85.0
    deep_minimum_lead: float = 180.0
    deep_score_distance: float = 650.0
    pair_horizontal_gap: float = 520.0
    pair_vertical_gap: float = 260.0
    pair_velocity_gap: float = 260.0
    deep_margin: float = 310.0
    deep_pan_speed: float = 570.0
    widening_margin: float = 40.0
    widening_speed: float = 280.0
    contraction_speed: float = 220.0
    pan_dead_zone: float = 120.0
    pan_speed: float = 520.0

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Calibration {item.name} must be a finite nonnegative number")
        for name in ('reference_width', 'reference_height', 'typical_player_height', 'track_distance', 'deep_score_distance', 'pair_horizontal_gap'):
            if getattr(self, name) <= 0:
                raise ValueError(f"Calibration {name} must be positive")

    @classmethod
    def load(cls, path: Path | None):
        return cls(**json.loads(path.read_text())) if path else cls()

    def scaled(self, width: int, height: int) -> Calibration:
        """Scale distances to source pixels; body sizes use the smaller axis scale."""
        sx, sy = width / self.reference_width, height / self.reference_height
        size = min(sx, sy)
        vertical = {'sideline_origin_y', 'minimum_foot_y', 'pair_vertical_gap'}
        body = {'minimum_player_height', 'masked_minimum_player_height', 'typical_player_height', 'foot_inset', 'track_distance', 'stationary_radius', 'minimum_body_height'}
        values = {}
        for name, value in asdict(self).items():
            if name.startswith('reference_'):
                continue
            if name.endswith('_slope'):
                values[name] = value * sx / sy
            else:
                values[name] = value * (sy if name in vertical else size if name in body else sx)
        return replace(self, reference_width=width, reference_height=height, **values)
