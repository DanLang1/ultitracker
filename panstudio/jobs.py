"""Persistent local jobs with transactional state transitions and restart recovery."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sqlite3
import threading
import time
from typing import Literal

JobState = Literal['ready', 'queued', 'processing', 'analyzed', 'complete', 'failed', 'interrupted']
BUSY = {'queued', 'processing'}
# A run is either the player analysis that follows an upload (detections plus
# an outline suggestion) or the full processing of a tactical video.
MODES = ('analyze', 'process')


class JobConflict(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    id: str
    metadata: dict
    status: dict
    updated: float


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

    def save(self, job_id: str, *, metadata=None, status=None):
        """Internal write; lifecycle operations below hold the store lock."""
        with self.lock:
            job = self.get(job_id)
            with self.connection() as connection:
                connection.execute('UPDATE jobs SET metadata=?, status=?, updated=? WHERE id=?',
                                   (json.dumps(metadata if metadata is not None else job.metadata),
                                    json.dumps(status if status is not None else job.status), time.time(), job_id))

    def claim(self, job_id: str, settings: dict, run_id: str) -> bool:
        with self.lock:
            job = self.get(job_id)
            if job.status['state'] in BUSY:
                return False
            mode = settings.get('mode', 'process')
            if mode not in MODES:
                raise ValueError('Unknown run mode')
            metadata = {**job.metadata, **settings, 'previous_run_id': job.metadata.get('run_id'), 'run_id': run_id}
            self.save(job_id, metadata=metadata, status={'state': 'queued', 'progress': 0, 'mode': mode,
                                                         'message': 'Waiting for the processor…'})
            return True

    def analysis_busy(self, job_id: str) -> bool:
        """True while a player-analysis run owns the job; the mask may still change."""
        status = self.get(job_id).status
        return status['state'] in BUSY and status.get('mode') == 'analyze'

    def transition(self, job_id: str, run_id: str, state: JobState, **changes):
        with self.lock:
            job = self.get(job_id)
            if job.metadata.get('run_id') != run_id:
                return False
            allowed = {'queued': {'processing', 'failed', 'interrupted'},
                       'processing': {'processing', 'analyzed', 'complete', 'failed', 'interrupted'}}
            if state not in allowed.get(job.status['state'], set()):
                return False
            self.save(job_id, status={**job.status, **changes, 'state': state})
            return True

    def delete(self, job_id: str):
        with self.lock:
            job = self.get(job_id)
            if job.status['state'] in BUSY:
                raise JobConflict('Wait for processing to finish before deleting this job.')
            shutil.rmtree(self.directory(job_id))
            with self.connection() as connection:
                connection.execute('DELETE FROM jobs WHERE id=?', (job_id,))

    def recover(self):
        """Called once by the server owner. Never restart work without user action."""
        with self.lock:
            # Import jobs made by the prototype without inferring success from a possibly partial video.
            for path in self.root.glob('*/job.json'):
                if not re.fullmatch(r'[a-f0-9]{32}', path.parent.name):
                    continue
                try:
                    self.get(path.parent.name)
                except FileNotFoundError:
                    try:
                        metadata = json.loads(path.read_text())
                        self.create(path.parent.name, metadata)
                    except (ValueError, TypeError):
                        continue
            for job in self.list():
                if job.status['state'] in BUSY:
                    self.save(job.id, status={'state': 'interrupted', 'progress': 0,
                                             'message': 'Processing was interrupted. You can retry this job.'})
