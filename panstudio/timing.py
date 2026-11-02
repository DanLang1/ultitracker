"""Wall-clock stage timings and the machine they were measured on."""
from __future__ import annotations

from contextlib import contextmanager
import os
import platform
import subprocess
import time


class StageTimer:
    """Accumulates wall-clock seconds per named stage, in the order stages first run."""

    def __init__(self):
        self.started = time.monotonic()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        began = time.monotonic()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.monotonic() - began

    def summary(self) -> dict:
        return {'total_seconds': round(time.monotonic() - self.started, 2),
                'stages': {name: round(seconds, 2) for name, seconds in self.stages.items()}}


def _cpu_name() -> str:
    if platform.system() == 'Darwin':
        try:
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        with open('/proc/cpuinfo') as cpuinfo:
            for line in cpuinfo:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def machine_info(device: str) -> dict:
    """What the timings depend on: the processors used and how video was decoded and encoded."""
    from .media import hwaccel_args, nvenc_available
    gpu = None
    if device == 'cuda':
        import torch
        gpu = torch.cuda.get_device_name(0)
    elif device == 'mps':
        gpu = 'Apple GPU (MPS)'
    release = platform.release()
    return {
        'system': f'{platform.system()} {release}' + (' (WSL)' if 'microsoft' in release.lower() else ''),
        'cpu': _cpu_name(), 'cpu_count': os.cpu_count(), 'gpu': gpu, 'inference_device': device,
        'decoder': 'cuda' if hwaccel_args() else 'cpu', 'encoder': 'h264_nvenc' if nvenc_available() else 'libx264',
    }
