"""Client for the processor: start one run, read its typed progress events, and get a single outcome.

The subprocess, its process group, the parent-watchdog pairing with ``cli.watch_parent``,
line parsing and terminate/kill escalation all live here, so the job worker and the
benchmark share one completion rule.
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Callable, Iterator

from .events import ProgressEvent

ROOT = Path(__file__).resolve().parents[1]


class ProcessorFailed(RuntimeError):
    pass


class ProcessorRun:
    """One running processor. Accepts any Popen-like object, so tests can script its output."""

    def __init__(self, process):
        self.process = process
        self.error = ''
        self.completed = False

    def events(self, on_line: Callable[[str], None] | None = None) -> Iterator[ProgressEvent]:
        """Yield progress events until the processor closes its output; every raw line goes to on_line."""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if on_line:
                on_line(line)
            event = ProgressEvent.parse(line)
            if event is None:
                continue
            if event.stage == 'error':
                self.error = event.message
            elif event.stage == 'complete':
                self.completed = True
            else:
                yield event

    def finish(self):
        """Raise ProcessorFailed unless the processor reported completion and exited cleanly."""
        return_code = self.process.wait()
        if return_code or not self.completed:
            raise ProcessorFailed(self.error or f'Processor did not complete (exit status {return_code}). See the job log.')

    def terminate(self):
        if self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def close(self):
        self.terminate()
        for stream in (self.process.stdout, getattr(self.process, 'stdin', None)):
            if stream is not None:
                stream.close()


def start(args: list[str], *, cwd: Path = ROOT, env: dict | None = None, supervised: bool = True) -> ProcessorRun:
    """Start ``tactical_pan.py`` with the given arguments.

    A supervised run gets its own process group and a stdin pipe; the processor exits
    when that pipe closes, so it never outlives the server that started it.
    """
    command = [sys.executable, str(Path(cwd) / 'tactical_pan.py'), *args]
    if supervised:
        command.append('--supervised')
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1', **(env or {})}
    process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.PIPE if supervised else None, text=True, bufsize=1,
                               start_new_session=supervised)
    return ProcessorRun(process)


def overall_progress(event: ProgressEvent, debug: bool) -> tuple[int, str] | None:
    """Map a stage event onto the job's 0–100 progress bar and a default message."""
    fraction = event.completed / max(1, event.total)
    if event.stage == 'detection':
        return 5 + round(40 * fraction), 'Analyzing players…'
    if event.stage == 'disc':
        return 45 + round(24 * fraction), 'Scanning for disc flights…'
    if event.stage == 'planning':
        return 70, 'Planning the camera path…'
    if event.stage == 'debug':
        return 70 + round(15 * fraction), 'Rendering the review video…'
    if event.stage == 'render':
        return (85 if debug else 70) + round((14 if debug else 29) * fraction), 'Rendering the tactical crop…'
    return None
