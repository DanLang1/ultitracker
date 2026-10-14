"""Versioned JSON-lines worker protocol, independent of human log wording."""
from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int = 0
    total: int = 0
    message: str = ''
    protocol: str = 'tactical-pan/v1'

    @classmethod
    def parse(cls, line: str):
        try:
            data = json.loads(line)
            if not isinstance(data, dict) or data.get('protocol') != 'tactical-pan/v1':
                return None
            event = cls(**data)
            if event.stage not in {'detection', 'disc', 'planning', 'debug', 'render', 'complete', 'error'}:
                return None
            if type(event.completed) is not int or type(event.total) is not int or not isinstance(event.message, str):
                return None
            if not 0 <= event.completed <= event.total:
                return None
            return event
        except (ValueError, TypeError):
            return None


def report(stage: str, completed: int = 0, total: int = 0, message: str = ''):
    print(json.dumps(asdict(ProgressEvent(stage, completed, total, message))), flush=True)
