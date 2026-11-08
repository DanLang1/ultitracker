"""A bounded single-worker queue, retaining subprocess isolation for inference."""
from pathlib import Path
import queue
import threading
import time
import uuid

from . import attempt, processor
from .jobs import MASK_FILE, JobConflict, JobStore, ProcessingSettings
from .models import DEFAULT_MODEL


class JobRunner:
    def __init__(self, store: JobStore, root: Path, capacity: int = 8, calibration: Path | None = None,
                 start_processor=processor.start):
        self.store = store
        self.start_processor = start_processor
        self.root = Path(root).resolve()
        self.calibration = Path(calibration).resolve() if calibration else None
        self.pending = queue.Queue()
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.RLock()
        self.thread = None
        self.running = None
        self.closed = False

    def submit(self, job_id: str, settings: ProcessingSettings):
        with self.lock, self.store.lock:
            job = self.store.get(job_id)
            if job.busy:
                return job.status
            if self.closed:
                raise JobConflict('The processor is shutting down.')
            if not self.slots.acquire(blocking=False):
                raise JobConflict('The processing queue is full. Try again when a job finishes.')
            run_id = uuid.uuid4().hex
            try:
                self.store.claim(job_id, settings, run_id)
                if self.thread is None:
                    self.thread = threading.Thread(target=self._work, daemon=True, name='video-worker')
                    self.thread.start()
                self.pending.put((job_id, run_id))
            except Exception:
                self.thread = None if self.thread is not None and not self.thread.is_alive() else self.thread
                self.slots.release()
                self.store.transition(job_id, run_id, 'failed', message='Could not queue processing.')
                raise
            return self.store.get(job_id).status

    def _work(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                job_id, run_id = item
                if self.closed:
                    self.store.transition(job_id, run_id, 'interrupted', message='Processing was interrupted.')
                else:
                    self.run(job_id, run_id)
            finally:
                if item is not None:
                    self.slots.release()
                self.pending.task_done()

    def run(self, job_id: str, run_id: str):
        running = None
        try:
            job = self.store.get(job_id)
            directory = self.store.directory(job_id)
            settings = job.settings
            request = attempt.prepare(directory, run_id, job.source, job.metadata['duration'], directory / MASK_FILE,
                                      model=str(self.root / DEFAULT_MODEL), debug=settings.generate_debug,
                                      disc_tracking=settings.disc_tracking, calibration=self.calibration)
            started = time.time()
            if not self.store.transition(job_id, run_id, 'processing', progress=1, message='Starting player analysis…',
                                         started_at=started, stage=None, stage_started_at=started, stage_fraction=0.0):
                return
            with (request.output_dir / 'process.log').open('w') as log:
                def write(line):
                    log.write(line)
                    log.flush()
                with self.lock:
                    if self.closed:
                        self.store.transition(job_id, run_id, 'interrupted', message='Processing was interrupted.')
                        return
                    running = self.start_processor(request.args(), cwd=self.root, env={
                        'YOLO_CONFIG_DIR': str(directory / '.ultralytics'), 'MPLCONFIGDIR': str(directory / '.matplotlib')})
                    self.running = running
                for event in running.events(on_line=write):
                    self._progress(job_id, run_id, event, request.debug)
                running.finish()
            result = attempt.collect(request)
            status = {'progress': 100, 'message': 'Your tactical video is ready.', 'finished_at': time.time(),
                      'output_url': self._url(job_id, run_id, result.output)}
            if result.timings:
                status['timings'] = result.timings
            if result.debug:
                status['debug_url'] = self._url(job_id, run_id, result.debug)
            self.store.transition(job_id, run_id, 'complete', **status)
        except Exception as error:
            self.store.transition(job_id, run_id, 'interrupted' if self.closed else 'failed', message=str(error), error=str(error))
        finally:
            if running is not None:
                running.close()
            with self.lock:
                self.running = None

    @staticmethod
    def _url(job_id, run_id, path: Path):
        return f'/jobs/{job_id}/runs/{run_id}/{path.name}'

    def _progress(self, job_id, run_id, event, debug):
        overall = processor.overall_progress(event, debug)
        if overall is None:
            return
        progress, message = overall
        timing = {'stage': event.stage, 'stage_fraction': round(event.completed / max(1, event.total), 4)}
        if event.stage != self.store.get(job_id).status.get('stage'):
            timing['stage_started_at'] = event.time or time.time()
        self.store.transition(job_id, run_id, 'processing', progress=progress, message=event.message or message, **timing)

    def close(self):
        with self.lock:
            self.closed = True
            if self.running is not None:
                self.running.terminate()
            if self.thread is not None:
                self.pending.put(None)
        if self.thread is not None:
            self.thread.join(timeout=10)
