"""One processing attempt for a job: the request a run directory is processed with, and the result read back from it.

Owns the artifact layout of a run, the search for reusable work in earlier attempts,
and the rules for when saved detections and flights may be reused.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from .artifacts import DEBUG_VIDEO_FILE, DETECTIONS_FILE, FLIGHTS_FILE, TIMINGS_FILE, flights_reusable, load_detections
from .models import ProcessingConfig
from .pipeline import probe_video, validate_reuse


@dataclass(frozen=True)
class AttemptRequest:
    source: Path
    end: float
    output_dir: Path
    model: str
    play_area: Path
    debug: bool = False
    disc_tracking: bool = True
    calibration: Path | None = None
    detections_in: Path | None = None
    flights_in: Path | None = None

    def args(self) -> list[str]:
        """Command-line arguments for the processor."""
        args = [str(self.source), '--start', '0', '--end', str(self.end), '--output-dir', str(self.output_dir),
                '--model', self.model, '--play-area', str(self.play_area)]
        if self.detections_in:
            args.extend(['--detections-in', str(self.detections_in)])
        if self.flights_in:
            args.extend(['--flights-in', str(self.flights_in)])
        if self.debug:
            args.append('--debug')
        if not self.disc_tracking:
            args.append('--no-disc-tracking')
        if self.calibration:
            args.extend(['--calibration', str(self.calibration)])
        return args

    def config(self) -> ProcessingConfig:
        """The exact configuration the processor will build from these arguments, defaults included."""
        from .cli import parse_args
        return ProcessingConfig.from_namespace(parse_args(self.args()))

    @property
    def output(self) -> Path:
        return self.output_dir / f'{ProcessingConfig.output_name}.mp4'


@dataclass(frozen=True)
class AttemptResult:
    output: Path
    debug: Path | None
    timings: dict | None


def prepare(job_directory: Path, run_id: str, source: Path, duration: float, play_area: Path, *, model: str,
            debug: bool = False, disc_tracking: bool = True, calibration: Path | None = None) -> AttemptRequest:
    """A request for a new attempt that processes the whole video, reusing earlier work where it is still valid."""
    output_dir = job_directory / 'runs' / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    # The whole video, stopping just short of the reported duration, which
    # containers often overstate by a frame.
    end = max(0.1, duration - 0.05)
    request = AttemptRequest(Path(source), end, output_dir, model, play_area, debug, disc_tracking, calibration)
    return reuse_earlier_work(request, job_directory / 'runs')


def reuse_earlier_work(request: AttemptRequest, runs: Path) -> AttemptRequest:
    """A failed retry may not have produced detections, so look back through all attempts
    and reuse the newest valid cache rather than losing earlier work."""
    caches = sorted(runs.glob(f'*/{DETECTIONS_FILE}'), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    if not caches:
        return request
    config = request.config()
    video = probe_video(config.input)
    for cached in caches:
        try:
            validate_reuse(config, load_detections(cached), video)
        except (ValueError, TypeError, KeyError, OSError):
            continue
        flights = cached.parent / FLIGHTS_FILE
        return replace(request, detections_in=cached,
                       flights_in=flights if request.disc_tracking and flights_reusable(flights) else None)
    return request


def collect(request: AttemptRequest) -> AttemptResult:
    """Read a finished attempt back, raising if an output the request asked for is missing."""
    if not request.output.is_file():
        raise RuntimeError('Processor completed without an output video.')
    debug = request.output_dir / DEBUG_VIDEO_FILE
    if request.debug and not debug.is_file():
        raise RuntimeError('Processor completed without the requested review video.')
    try:
        data = json.loads((request.output_dir / TIMINGS_FILE).read_text())
        timings = {'total_seconds': data['total_seconds'], 'stages': data['stages']}
    except (OSError, ValueError, KeyError, TypeError):
        timings = None
    return AttemptResult(request.output, debug if request.debug else None, timings)
