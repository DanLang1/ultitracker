"""Suggest a playing-area outline and a sideline heat map from raw detections.

Both come from behaviour rather than appearance, so they need no field
geometry and no camera model:

* People run on the field and walk or stand beside it. Cells of the frame
  where running happens in a large share of the recording's minutes form the
  playing area. Minutes rather than raw counts, so one substitute jogging
  the sideline for a while does not outweigh a quiet end zone.
* Cells that contain somebody for a large share of all samples are benches,
  tent lines and fixed objects. They are carved out of the suggestion and
  returned as a warning overlay for the operator.

Cells scale with the local body size, so a cell near the camera covers the
same stretch of ground as a cell at the far sideline and densities are
comparable at every depth. Boxes touching the bottom of the frame are
ignored: their feet are out of view and their position is unreliable.
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibration import Calibration
from .models import DetectionSet, Sample
from .tracking import select_field_players, suppress_stationary_tracks

RUNNING_SPEED = 1.2          # body heights per second; walking is about 0.8
RUNNING_SHARE = 0.3          # share of minutes with running that seeds the field
RUNNING_SHARE_WEAK = 0.15    # cells above this join the field when connected to the seed
MARGIN_BODIES = 0.25         # outward margin so players on the boundary stay inside
OCCUPANCY_HOT = 0.3          # share of samples with somebody present
CELLS_PER_BODY = 1.5         # grid cells per body height, in both directions
MINIMUM_BODY = 40.0          # pixels; keeps far cells finite
MINIMUM_MINUTES = 5          # shorter recordings give no suggestion
PREVIEW_SCALE = 4            # pixel maps are built at 1/4 resolution


def tracked_records(detections: DetectionSet) -> list[Sample]:
    """Build tracked records over the full frame, so no mask shapes the result."""
    video = detections.video
    calibration = Calibration().scaled(video.width, video.height)
    everywhere = np.full((video.height, video.width), 255, dtype=np.uint8)
    records: list[Sample] = []
    for sample in detections.samples:
        boxes = np.array([d.box for d in sample.detections]).reshape(-1, 4)
        confidences = np.array([d.confidence for d in sample.detections])
        records.append({'time': sample.time, 'source_time': sample.source_time,
                        'players': select_field_players(boxes, confidences, video.width, everywhere, calibration)})
    suppress_stationary_tracks(records, detections.sample_rate, calibration)
    return records


class BodyGrid:
    """A grid whose cells are a fixed fraction of the local body height."""

    def __init__(self, foot_y: np.ndarray, heights: np.ndarray, width: int, height: int):
        self.width, self.height = width, height
        edges = np.linspace(foot_y.min(), foot_y.max() + 1, 25)
        centers, medians = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = (foot_y >= low) & (foot_y < high)
            if chosen.sum() >= 30:
                centers.append((low + high) / 2)
                medians.append(np.median(heights[chosen]))
        if len(centers) >= 2:
            self.slope, self.intercept = np.polyfit(centers, medians, 1)
        else:
            self.slope, self.intercept = 0.0, float(np.median(heights))
        y_axis = np.arange(height + 1, dtype=float)
        self.depth = np.cumsum(1.0 / self.body(y_axis)) * CELLS_PER_BODY
        self.rows = int(self.depth[-1]) + 2
        self.columns = int(width / self.body(np.array([height - 1.0]))[0] * CELLS_PER_BODY) + 2
        far_body = self.body(np.array([0.0]))[0]
        self.columns = max(self.columns, int(width / far_body * CELLS_PER_BODY) + 2)

    def body(self, y: np.ndarray) -> np.ndarray:
        return np.maximum(MINIMUM_BODY, self.slope * y + self.intercept)

    def cell(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_index = np.clip(y, 0, self.height).astype(int)
        rows = np.clip(self.depth[y_index].astype(int), 0, self.rows - 1)
        columns = np.clip((x / self.body(y.astype(float)) * CELLS_PER_BODY).astype(int), 0, self.columns - 1)
        return rows, columns

    def pixel_lookup(self, scale: int) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices for every pixel of a 1/scale frame."""
        yy, xx = np.mgrid[0:self.height:scale, 0:self.width:scale]
        return self.cell(xx.astype(float), yy.astype(float))


def suggest_outline(detections: DetectionSet):
    """Return (normalized polygon points or None, heat overlay BGRA at 1/PREVIEW_SCALE or None)."""
    video = detections.video
    records = tracked_records(detections)
    observations = [(p['center_x'], p['box'][3], p['height'], p.get('speed_body', 0.0), r['time'])
                    for r in records for p in r['players'] if p['box'][3] < video.height - 3]
    if len(records) < 2 or not observations:
        return None, None
    xs, ys, heights, speeds, times = (np.array(column, dtype=float) for column in zip(*observations))
    fast = speeds >= RUNNING_SPEED
    grid = BodyGrid(ys[fast] if fast.sum() >= 200 else ys, heights[fast] if fast.sum() >= 200 else heights, video.width, video.height)
    rows, columns = grid.cell(xs, ys)
    minutes = int(times.max() // 60) + 1

    occupancy = np.zeros((grid.rows, grid.columns), dtype=np.int32)
    sample_index = np.searchsorted([r['time'] for r in records], times)
    for index in np.unique(sample_index):
        chosen = sample_index == index
        occupancy[np.unique(rows[chosen] * grid.columns + columns[chosen]) // grid.columns,
                  np.unique(rows[chosen] * grid.columns + columns[chosen]) % grid.columns] += 1
    occupancy = occupancy / len(records)
    lookup = grid.pixel_lookup(PREVIEW_SCALE)
    heat = heat_overlay(occupancy[lookup])

    if minutes < MINIMUM_MINUTES:
        return None, heat
    running = np.zeros((minutes, grid.rows, grid.columns), dtype=bool)
    running[(times[fast] // 60).astype(int), rows[fast], columns[fast]] = True
    share = running.mean(axis=0)
    hot = occupancy > OCCUPANCY_HOT
    strong = ((share >= RUNNING_SHARE) & ~hot).astype(np.uint8)
    weak = ((share >= RUNNING_SHARE_WEAK) & ~hot).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    if count < 2:
        return None, heat
    seed = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # Hysteresis: the busiest region grows into connected, less busy cells,
    # which is how a quiet end zone attaches to the field core.
    count, labels = cv2.connectedComponents(weak | strong, connectivity=8)
    field = np.isin(labels, np.unique(labels[seed])) & (labels > 0)
    field = cv2.morphologyEx(field.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    pixels = field[lookup]
    # A modest outward margin that follows the local body size.
    scaled_height = pixels.shape[0]
    bands = 6
    margin = np.zeros_like(pixels)
    for band in range(bands):
        top, bottom = band * scaled_height // bands, (band + 1) * scaled_height // bands
        body = float(grid.body(np.array([(top + bottom) / 2 * PREVIEW_SCALE]))[0])
        radius = max(1, int(MARGIN_BODIES * body / PREVIEW_SCALE))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        margin[top:bottom] = cv2.dilate(pixels, kernel)[top:bottom]
    pixels = margin & ~hot[lookup]
    contours, _ = cv2.findContours(pixels.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, heat
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(pixels, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
    # Smooth the fan-shaped cell edges before simplifying to a polygon.
    smooth = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 5) >= 0.5
    contours, _ = cv2.findContours(smooth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 5.0, True).reshape(-1, 2).astype(float) * PREVIEW_SCALE
    points = [{'x': round(float(np.clip(x / video.width, 0, 1)), 4), 'y': round(float(np.clip(y / video.height, 0, 1)), 4)}
              for x, y in polygon]
    return (points if len(points) >= 3 else None), heat


def heat_overlay(occupancy_pixels: np.ndarray) -> np.ndarray:
    """A BGRA image: translucent red where people are usually present."""
    alpha = np.clip((occupancy_pixels - OCCUPANCY_HOT * 0.5) / OCCUPANCY_HOT, 0, 1) * 170
    overlay = np.zeros((*occupancy_pixels.shape, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255
    overlay[:, :, 3] = alpha.astype(np.uint8)
    return overlay
