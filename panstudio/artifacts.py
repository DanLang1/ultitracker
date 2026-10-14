"""Versioned reusable artifacts, validated before entering the pipeline."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile

from .models import CameraPlan, Detection, DetectionSample, DetectionSet, SourceIdentity, VideoInfo


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def source_identity(path: Path) -> SourceIdentity:
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def save_detections(path: Path, detections: DetectionSet):
    atomic_json(path, {'schema': 'detections/v1', **asdict(detections)})


def save_plan(path: Path, plan: CameraPlan):
    atomic_json(path, {'schema': 'camera-plan/v1', **asdict(plan)})


def _read(path: Path, schema: str) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.pop('schema', None) != schema:
        raise ValueError(f'Unsupported artifact schema in {path}')
    video = VideoInfo(**data['video'])
    if min(video.width, video.height, video.fps, video.duration) <= 0 or not all(math.isfinite(v) for v in asdict(video).values()):
        raise ValueError('Invalid video metadata')
    data['video'] = video
    start, end = data['start'], data['end']
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
        raise ValueError('Invalid artifact time range')
    if not data['samples']:
        raise ValueError('Artifact contains no samples')
    previous = -1.0
    for sample in data['samples']:
        t, source_t = sample['time'], sample['source_time']
        if not math.isfinite(t) or not math.isfinite(source_t) or t < 0 or t <= previous or abs(source_t - start - t) > 0.002 or source_t > end + 0.002:
            raise ValueError('Invalid sample timestamps')
        previous = t
    if data['samples'][0]['time'] > 0.002:
        raise ValueError('Artifact does not start at the requested time')
    return data


def load_detections(path: Path) -> DetectionSet:
    data = _read(path, 'detections/v1')
    if not math.isfinite(data['sample_rate']) or data['sample_rate'] <= 0:
        raise ValueError('Invalid sample rate')
    expected = math.floor((data['end'] - data['start']) * data['sample_rate'] + 1e-6) + 1
    if not max(1, expected - 1) <= len(data['samples']) <= expected:
        raise ValueError('Detection artifact is incomplete')
    samples = []
    for index, sample in enumerate(data['samples']):
        if abs(sample['time'] - index / data['sample_rate']) > 0.002:
            raise ValueError('Detection samples must follow the declared sample rate')
        detections = []
        for item in sample['detections']:
            box, confidence = item['box'], item['confidence']
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1] or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid detection')
            detections.append(Detection(tuple(box), confidence))
        samples.append(DetectionSample(sample['time'], sample['source_time'], tuple(detections)))
    data['samples'] = tuple(samples)
    return DetectionSet(**data)


def load_plan(path: Path) -> CameraPlan:
    data = _read(path, 'camera-plan/v1')
    video = data['video']
    for sample in data['samples']:
        x, y, width, height = (sample[key] for key in ('crop_x', 'crop_y', 'crop_width', 'crop_height'))
        if not all(math.isfinite(v) for v in (x, y, width, height)) or min(x, y) < -0.02 or min(width, height) <= 0 or x + width > video.width + 0.02 or y + height > video.height + 0.02:
            raise ValueError('Camera plan contains an out-of-bounds crop')
        if abs(y + height - video.height) > 0.02:
            raise ValueError('Camera plan crops must be bottom-aligned')
    data['samples'] = tuple(data['samples'])
    return CameraPlan(**data)


def save_flights(path: Path, flights, start: float, end: float, detector_version: int | None):
    """Flights are advisory framing input: a list of {start, end, x0, y0, x1, y1, direction, chord, frames, speed, points}."""
    atomic_json(path, {'schema': 'flights/v1', 'detector_version': detector_version, 'start': start, 'end': end,
                       'flights': [f.to_json() if hasattr(f, 'to_json') else dict(f) for f in flights]})


def flights_version(path: Path) -> int | None:
    """The disc detector version that produced a flights file; None when unversioned or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    version = data.get('detector_version') if isinstance(data, dict) else None
    return version if isinstance(version, int) and not isinstance(version, bool) else None


def load_flights(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('schema') != 'flights/v1' or not isinstance(data.get('flights'), list):
        raise ValueError(f'Unsupported flights schema in {path}')
    flights = []
    for item in data['flights']:
        try:
            values = {key: float(item[key]) for key in ('start', 'end', 'x0', 'y0', 'x1', 'y1', 'chord')}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Invalid flight record') from error
        if not all(math.isfinite(v) for v in values.values()) or values['end'] < values['start'] or values['chord'] < 0:
            raise ValueError('Invalid flight record')
        flights.append({**item, **values, 'direction': 1 if float(item.get('direction', 1)) >= 0 else -1})
    return flights
