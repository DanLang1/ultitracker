"""Time every processing stage on a fixed clip and keep the results for comparison across machines and commits."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import processor
from .artifacts import TIMINGS_FILE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks' / 'results.jsonl'
STAGES = ('mask', 'detection', 'disc', 'planning', 'render')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Time each processing stage on a fixed clip and record the result')
    parser.add_argument('input', type=Path, help='Source video; use the same file on every machine you compare')
    parser.add_argument('--start', type=float, default=600.0, help='Clip start in seconds (default 600)')
    parser.add_argument('--duration', type=float, default=300.0, help='Clip length in seconds (default 300)')
    parser.add_argument('--play-area', type=Path, help='Field mask or outline image, as in the GUI')
    parser.add_argument('--label', default=platform.node(), help='Name for this machine in the results (default: hostname)')
    parser.add_argument('--device', default='auto', help='Inference device passed to the processor')
    parser.add_argument('--results', type=Path, default=RESULTS, help='JSON-lines file the result is appended to')
    parser.add_argument('--keep', type=Path, help='Keep the outputs in this directory instead of deleting them')
    return parser.parse_args(argv)


def git_commit() -> str | None:
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


def clock(seconds: float) -> str:
    seconds = round(seconds)
    return f'{seconds // 60}:{seconds % 60:02d}'


def run_processor(options, output: Path) -> dict:
    """Run the normal command-line processor in a fresh output folder so nothing is reused."""
    args = [str(options.input), '--start', str(options.start), '--end', str(options.start + options.duration),
            '--output-dir', str(output), '--device', options.device]
    if options.play_area:
        args.extend(['--play-area', str(options.play_area)])
    run, log, stage = processor.start(args, cwd=ROOT, supervised=False), [], None
    try:
        for event in run.events(on_line=log.append):
            if event.stage != stage:
                stage = event.stage
                print(f'  {time.strftime("%H:%M:%S")} {stage}', flush=True)
        run.finish()
    except processor.ProcessorFailed as error:
        sys.stdout.writelines(log[-20:])
        raise SystemExit(f'Benchmark run failed: {error}') from None
    finally:
        run.close()
    return json.loads((output / TIMINGS_FILE).read_text())


def record(options, timings: dict) -> dict:
    minutes = timings['footage_seconds'] / 60
    return {
        'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'label': options.label, 'commit': git_commit(),
        'clip': {'file': options.input.name, 'size': options.input.stat().st_size, 'start': options.start,
                 'duration': options.duration, 'play_area': options.play_area is not None},
        'total_seconds': timings['total_seconds'], 'stages': timings['stages'],
        # Seconds of processing per minute of footage, so clips of different lengths compare.
        'seconds_per_footage_minute': {name: round(seconds / minutes, 2) for name, seconds in timings['stages'].items()},
        'video': timings['video'], 'machine': timings['machine'],
    }


def same_clip(a: dict, b: dict) -> bool:
    return all(a['clip'][key] == b['clip'][key] for key in ('file', 'size', 'start', 'duration', 'play_area'))


def comparison_table(results: list[dict]) -> str:
    """Every recorded run of one clip, one row each, stage times as m:ss."""
    header = ['date', 'label', 'commit', 'gpu', 'decode', 'encode', *STAGES, 'total']
    rows = [header]
    for result in results:
        machine = result['machine']
        rows.append([result['date'][:16].replace('T', ' '), result['label'], result['commit'] or '?',
                     machine['gpu'] or machine['cpu'], machine['decoder'], machine['encoder'],
                     *(clock(result['stages'][s]) if s in result['stages'] else '-' for s in STAGES), clock(result['total_seconds'])])
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    return '\n'.join('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows)


def main(argv=None):
    options = parse_args(argv)
    if not options.input.is_file():
        raise SystemExit(f'No such video: {options.input}')
    print(f'Benchmarking {options.input.name} from {clock(options.start)} for {clock(options.duration)} on {options.label}', flush=True)
    output = options.keep or Path(tempfile.mkdtemp(prefix='panstudio-benchmark-'))
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = record(options, run_processor(options, output))
    finally:
        if not options.keep:
            shutil.rmtree(output, ignore_errors=True)
    options.results.parent.mkdir(parents=True, exist_ok=True)
    with options.results.open('a') as results:
        results.write(json.dumps(result) + '\n')
    history = [json.loads(line) for line in options.results.read_text().splitlines() if line.strip()]
    print(f'\nRecorded in {options.results}. All runs of this clip:\n')
    print(comparison_table([entry for entry in history if same_clip(entry, result)]))
