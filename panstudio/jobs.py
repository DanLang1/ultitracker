"""Persistent local jobs with transactional state transitions and restart recovery."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from typing import Callable, Literal

JobState = Literal['ready', 'queued', 'processing', 'complete', 'failed', 'interrupted']
BUSY = {'queued', 'processing'}


# Files in a job's directory.
SOURCE_FILE = 'source.mp4'
FRAME_FILE = 'active-area-frame.jpg'
MASK_FILE = 'active-area-mask.png'
PREVIEW_FILE = 'play-area-preview.jpg'


class JobConflict(ValueError):
    pass


@dataclass(frozen=True)
class ProcessingSettings:
    """The options an operator chooses when processing a job."""
    generate_debug: bool = False
    disc_tracking: bool = True

    @classmethod
    def from_request(cls, payload) -> 'ProcessingSettings':
        if not isinstance(payload, dict):
            raise ValueError('Expected an object containing processing options.')
        settings = cls(payload.get('debug', cls.generate_debug), payload.get('disc_tracking', cls.disc_tracking))
        if type(settings.generate_debug) is not bool or type(settings.disc_tracking) is not bool:
            raise ValueError('Processing options must be booleans.')
        return settings

    @classmethod
    def from_metadata(cls, metadata: dict) -> 'ProcessingSettings':
        # Jobs saved before an option existed get its default.
        return cls(metadata.get('generate_debug', cls.generate_debug) is True,
                   metadata.get('disc_tracking', cls.disc_tracking) is not False)


@dataclass(frozen=True)
class Job:
    id: str
    metadata: dict
    status: dict
    updated: float

    @property
    def busy(self) -> bool:
        return self.status['state'] in BUSY

    @property
    def settings(self) -> ProcessingSettings:
        return ProcessingSettings.from_metadata(self.metadata)

    @property
    def source(self) -> Path:
        return Path(self.metadata['video_path'])


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'jobs.sqlite3'
        self.lock = threading.RLock()
        with self.connection() as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, metadata TEXT NOT NULL, status TEXT NOT NULL, updated REAL NOT NULL
            )''')

    @contextmanager
    def connection(self):
        with self.lock:
            connection = sqlite3.connect(self.database, timeout=30)
            try:
                with connection:
                    yield connection
            finally:
                connection.close()

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise ValueError('Invalid job id')
        directory = self.root / job_id
        if not directory.is_dir():
            raise FileNotFoundError(job_id)
        return directory

    def create(self, job_id: str, metadata: dict):
        self.directory(job_id)
        status = {'state': 'ready', 'progress': 0, 'message': 'Draw the active area.'}
        with self.connection() as connection:
            connection.execute('INSERT INTO jobs VALUES (?, ?, ?, ?)',
                               (job_id, json.dumps(metadata), json.dumps(status), time.time()))

    def get(self, job_id: str) -> Job:
        self.directory(job_id)
        with self.connection() as connection:
            row = connection.execute('SELECT metadata, status, updated FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(job_id)
        return Job(job_id, json.loads(row[0]), json.loads(row[1]), row[2])

    def list(self) -> list[Job]:
        with self.connection() as connection:
            rows = connection.execute('SELECT id, metadata, status, updated FROM jobs ORDER BY updated DESC').fetchall()
        return [Job(row[0], json.loads(row[1]), json.loads(row[2]), row[3]) for row in rows if (self.root / row[0]).is_dir()]

    def has_mask(self, job_id: str) -> bool:
        return (self.directory(job_id) / MASK_FILE).is_file()

    def _save(self, job_id: str, *, metadata=None, status=None):
        """Internal write; lifecycle operations below hold the store lock."""
        with self.lock:
            job = self.get(job_id)
            with self.connection() as connection:
                connection.execute('UPDATE jobs SET metadata=?, status=?, updated=? WHERE id=?',
                                   (json.dumps(metadata if metadata is not None else job.metadata),
                                    json.dumps(status if status is not None else job.status), time.time(), job_id))

    def set_active_area(self, job_id: str, points: list, write_files: Callable[[Path], None]):
        """Store a new field outline. write_files renders the mask and preview into the job directory."""
        with self.lock:
            job = self.get(job_id)
            if job.busy:
                raise JobConflict('Wait for processing to finish before changing the active area.')
            if len(points) < 3:
                raise ValueError('Draw one closed outline around the active field.')
            write_files(self.directory(job_id))
            self._save(job_id, metadata={**job.metadata, 'points': points},
                       status={'state': 'ready', 'progress': 0, 'message': 'Review the active area, then process.'})

    def claim(self, job_id: str, settings: ProcessingSettings, run_id: str) -> bool:
        with self.lock:
            job = self.get(job_id)
            if job.busy:
                return False
            metadata = {**job.metadata, **asdict(settings), 'run_id': run_id}
            self._save(job_id, metadata=metadata, status={'state': 'queued', 'progress': 0,
                                                         'message': 'Waiting for the processor…'})
            return True

    def transition(self, job_id: str, run_id: str, state: JobState, **changes):
        with self.lock:
            job = self.get(job_id)
            if job.metadata.get('run_id') != run_id:
                return False
            allowed = {'queued': {'processing', 'failed', 'interrupted'},
                       'processing': {'processing', 'complete', 'failed', 'interrupted'}}
            if state not in allowed.get(job.status['state'], set()):
                return False
            self._save(job_id, status={**job.status, **changes, 'state': state})
            return True

    def delete(self, job_id: str):
        with self.lock:
            job = self.get(job_id)
            if job.busy:
                raise JobConflict('Wait for processing to finish before deleting this job.')
            shutil.rmtree(self.directory(job_id))
            with self.connection() as connection:
                connection.execute('DELETE FROM jobs WHERE id=?', (job_id,))

    def recover(self):
        """Called once by the server owner. Never restart work without user action."""
        with self.lock:
            for job in self.list():
                if job.busy:
                    self._save(job.id, status={'state': 'interrupted', 'progress': 0,
                                             'message': 'Processing was interrupted. You can retry this job.'})
