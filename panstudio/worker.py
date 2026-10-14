"""A bounded single-worker queue, retaining subprocess isolation for inference."""
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import uuid

from .artifacts import flights_version, load_detections
from .disc import DETECTOR_VERSION
from .models import ProcessingConfig
from .pipeline import probe_video, validate_reuse
from .events import ProgressEvent
from .jobs import BUSY, JobConflict, JobStore


class JobRunner:
    def __init__(self, store: JobStore, root: Path, capacity: int = 8, calibration: Path | None = None):
        self.store = store
        self.root = Path(root).resolve()
        self.calibration = Path(calibration).resolve() if calibration else None
        self.pending = queue.Queue()
        self.slots = threading.BoundedSemaphore(capacity)
        self.lock = threading.RLock()
        self.thread = None
        self.process = None
        self.closed = False

    def submit(self, job_id: str, settings: dict):
        with self.lock, self.store.lock:
            job = self.store.get(job_id)
            if job.status['state'] in BUSY:
                if job.status.get('mode') == 'analyze' and settings.get('mode', 'process') == 'process':
                    # Processing requested during the player analysis: start it
                    # automatically as soon as the analysis finishes.
                    self.store.save(job_id, metadata={**job.metadata, 'process_after_analysis': settings})
                return self.store.get(job_id).status
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
        process = None
        try:
            job = self.store.get(job_id)
            directory = self.store.directory(job_id)
            output = directory / 'runs' / run_id
            output.mkdir(parents=True, exist_ok=True)
            metadata = job.metadata
            mode = metadata.get('mode', 'process')
            command = [sys.executable, str(self.root / 'tactical_pan.py'), metadata['video_path'],
                       '--start', '0', '--end', str(max(0.1, metadata['duration'] - 0.05)),
                       '--output-dir', str(output), '--model', str(self.root / 'yolo11s.pt'), '--supervised']
            if mode == 'analyze':
                # Detection only, plus the outline suggestion; no mask is needed yet.
                command.append('--detect-only')
            else:
                command.extend(['--play-area', str(directory / 'active-area-mask.png')])
            # A failed retry may not have produced detections. Look back through all
            # attempts and reuse the newest valid cache, rather than losing earlier work.
            caches = sorted((directory / 'runs').glob('*/detections.json'), key=lambda path: path.stat().st_mtime_ns, reverse=True)
            if caches:
                config = ProcessingConfig(Path(metadata['video_path']), 0, max(0.1, metadata['duration'] - 0.05), model=str(self.root / 'yolo11s.pt'))
                video = probe_video(config.input)
                for cached in caches:
                    try:
                        validate_reuse(config, load_detections(cached), video)
                    except (ValueError, TypeError, KeyError, OSError):
                        continue
                    command.extend(['--detections-in', str(cached)])
                    # Flights from an older disc detector are rescanned so detector fixes take effect.
                    if mode == 'process' and flights_version(cached.parent / 'flights.json') == DETECTOR_VERSION:
                        command.extend(['--flights-in', str(cached.parent / 'flights.json')])
                    break
            if mode == 'process' and metadata.get('generate_debug'):
                command.append('--debug')
            if mode == 'process' and not metadata.get('auto_widen', True):
                command.append('--no-auto-widen')
            if self.calibration:
                command.extend(['--calibration', str(self.calibration)])
            environment = os.environ.copy()
            environment.update(PYTHONUNBUFFERED='1', YOLO_CONFIG_DIR=str(directory / '.ultralytics'),
                               MPLCONFIGDIR=str(directory / '.matplotlib'))
            if not self.store.transition(job_id, run_id, 'processing', progress=1, message='Starting player analysis…', mode=mode):
                return
            with (output / 'process.log').open('w') as log:
                with self.lock:
                    if self.closed:
                        self.store.transition(job_id, run_id, 'interrupted', message='Processing was interrupted.')
                        return
                    process = subprocess.Popen(command, cwd=self.root, env=environment, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
                    self.process = process
                detail = ''
                completed = False
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    event = ProgressEvent.parse(line)
                    if event:
                        if event.stage == 'error':
                            detail = event.message
                        elif event.stage == 'complete':
                            completed = True
                        else:
                            self._progress(job_id, run_id, event, bool(metadata.get('generate_debug')), mode)
                return_code = process.wait()
                if return_code or not completed:
                    raise RuntimeError(detail or f'Processor did not complete (exit status {return_code}). See the job log.')
            if mode == 'analyze':
                if not (output / 'detections.json').is_file():
                    raise RuntimeError('Analysis completed without saving detections.')
                with self.store.lock:
                    fresh = self.store.get(job_id)
                    follow_up = fresh.metadata.get('process_after_analysis')
                    cleaned = {key: value for key, value in fresh.metadata.items() if key != 'process_after_analysis'}
                    self.store.save(job_id, metadata={**cleaned, 'analysis_run_id': run_id})
                    self.store.transition(job_id, run_id, 'analyzed', progress=100, message='Player analysis complete.')
                if follow_up:
                    self.submit(job_id, follow_up)
                return
            if not (output / 'tactical-crop.mp4').is_file():
                raise RuntimeError('Processor completed without an output video.')
            result = {'progress': 100, 'message': 'Your tactical video is ready.',
                      'output_url': f'/jobs/{job_id}/runs/{run_id}/tactical-crop.mp4'}
            if metadata.get('generate_debug'):
                if not (output / 'debug-wide.mp4').is_file():
                    raise RuntimeError('Processor completed without the requested review video.')
                result['debug_url'] = f'/jobs/{job_id}/runs/{run_id}/debug-wide.mp4'
            self.store.transition(job_id, run_id, 'complete', **result)
        except Exception as error:
            self.store.transition(job_id, run_id, 'interrupted' if self.closed else 'failed', message=str(error), error=str(error))
        finally:
            if process is not None:
                if process.poll() is None:
                    self._terminate(process)
                if process.stdout is not None:
                    process.stdout.close()
                if getattr(process, 'stdin', None) is not None:
                    process.stdin.close()
            with self.lock:
                self.process = None

    def _progress(self, job_id, run_id, event, debug, mode='process'):
        fraction = event.completed / max(1, event.total)
        if mode == 'analyze':
            if event.stage == 'detection':
                progress, message = 2 + round(58 * fraction), 'Analyzing players…'
            elif event.stage == 'planning':
                progress, message = 60, 'Suggesting the playing area…'
            elif event.stage == 'disc':
                progress, message = 62 + round(36 * fraction), 'Scanning for disc flights…'
            else:
                return
            self.store.transition(job_id, run_id, 'processing', progress=progress, message=event.message or message)
            return
        if event.stage == 'detection':
            progress, message = 5 + round(40 * fraction), 'Analyzing players…'
        elif event.stage == 'disc':
            progress, message = 45 + round(24 * fraction), 'Scanning for disc flights…'
        elif event.stage == 'planning':
            progress, message = 70, 'Planning the camera path…'
        elif event.stage == 'debug':
            progress, message = 70 + round(15 * fraction), 'Rendering the review video…'
        elif event.stage == 'render':
            progress = (85 if debug else 70) + round((14 if debug else 29) * fraction)
            message = 'Rendering the tactical crop…'
        else:
            return
        self.store.transition(job_id, run_id, 'processing', progress=progress, message=event.message or message)

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    def close(self):
        with self.lock:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                self._terminate(self.process)
            if self.thread is not None:
                self.pending.put(None)
        if self.thread is not None:
            self.thread.join(timeout=10)
