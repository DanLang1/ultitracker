"""Versioned JSON-lines worker protocol, independent of human log wording."""
from dataclasses import asdict, dataclass
import json
import time

# Stages the processor reports progress for, in the order they run.
STAGES = ('detection', 'disc', 'planning', 'debug', 'render')


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int = 0
    total: int = 0
    message: str = ''
    protocol: str = 'tactical-pan/v1'
    time: float = 0.0  # Unix seconds when emitted; 0 for events from older workers

    @classmethod
    def parse(cls, line: str):
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or data.get('protocol') != 'tactical-pan/v1':
                return None
            event = cls(**data)
            if event.stage not in (*STAGES, 'complete', 'error'):
                return None
            if type(event.completed) is not int or type(event.total) is not int or not isinstance(event.message, str) \
                    or type(event.time) not in (int, float):
                return None
            if not 0 <= event.completed <= event.total:
                return None
            return event
        except (ValueError, TypeError):
            return None


def report(stage: str, completed: int = 0, total: int = 0, message: str = ''):
    print(json.dumps(asdict(ProgressEvent(stage, completed, total, message, time=round(time.time(), 3)))), flush=True)
