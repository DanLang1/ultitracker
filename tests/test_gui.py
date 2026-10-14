import io
import json
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from gui import create_app
from panstudio.events import ProgressEvent
from panstudio.jobs import JobConflict, JobStore
from panstudio.worker import JobRunner


class IdleRunner:
    def __init__(self, store, root, **kwargs):
        self.store = store
        self.calls = 0

    def submit(self, job_id, settings):
        self.calls += int(self.store.claim(job_id, settings, 'b' * 32))
        return self.store.get(job_id).status


def create_job(store, job_id='a' * 32):
    directory = store.root / job_id
    directory.mkdir()
    (directory / 'source.mp4').write_bytes(b'video')
    cv2.imwrite(str(directory / 'active-area-frame.jpg'), np.full((48, 64, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(directory / 'active-area-mask.png'), np.full((48, 64), 255, dtype=np.uint8))
    store.create(job_id, {'job_id': job_id, 'filename': 'source.mp4', 'video_path': str(directory / 'source.mp4'),
                         'duration': 10., 'width': 64, 'height': 48, 'size': 5})
    return job_id


@pytest.fixture
def application(tmp_path):
    app = create_app(tmp_path, runner_factory=IdleRunner)
    create_job(app.extensions['job_store'])
    return app


def test_process_claims_once_and_persists_options(application):
    client = application.test_client()
    job_id = 'a' * 32
    first = client.post(f'/api/process/{job_id}', json={'debug': False, 'auto_widen': False})
    second = client.post(f'/api/process/{job_id}', json={'debug': True})
    assert first.json['state'] == second.json['state'] == 'queued'
    assert application.extensions['job_runner'].calls == 1
    metadata = application.extensions['job_store'].get(job_id).metadata
    assert metadata['auto_widen'] is False
    assert metadata['generate_debug'] is False


def test_restart_recovers_active_jobs_and_preserves_completed_jobs(application):
    store = application.extensions['job_store']
    active = 'a' * 32
    complete = create_job(store, 'c' * 32)
    for job_id in (active, complete):
        store.claim(job_id, {}, 'b' * 32)
        store.transition(job_id, 'b' * 32, 'processing')
    store.transition(complete, 'b' * 32, 'complete', progress=100, output_url='/saved.mp4')
    restarted = create_app(store.root, runner_factory=IdleRunner).test_client()
    assert restarted.get(f'/api/status/{active}').json['state'] == 'interrupted'
    assert restarted.get(f'/api/status/{complete}').json['output_url'] == '/saved.mp4'
    assert len(restarted.get('/api/jobs').json) == 2
    assert restarted.post(f'/api/process/{active}', json={}).json['state'] == 'queued'


def test_delete_and_mask_changes_reject_busy_jobs(application):
    client = application.test_client()
    job_id = 'a' * 32
    client.post(f'/api/process/{job_id}', json={})
    assert client.delete(f'/api/jobs/{job_id}').status_code == 409
    assert client.post(f'/api/active-area/{job_id}', json={'points': []}).status_code == 409


def test_delete_removes_persisted_job_and_files(application):
    client = application.test_client()
    job_id = 'a' * 32
    assert client.delete(f'/api/jobs/{job_id}').json == {'deleted': True}
    assert client.get('/api/jobs').json == []
    assert client.get(f'/api/status/{job_id}').status_code == 404


def test_mask_and_reopen_preserve_outline(application):
    client = application.test_client()
    job_id = 'a' * 32
    points = [{'x': .1, 'y': .1}, {'x': .9, 'y': .1}, {'x': .9, 'y': .9}, {'x': .1, 'y': .9}]
    assert client.post(f'/api/active-area/{job_id}', json={'points': points}).status_code == 200
    restored = client.get(f'/api/jobs/{job_id}').json
    assert restored['points'] == points
    assert restored['has_mask']
    assert 'video_path' not in restored


def test_queue_is_bounded_serial_and_duplicate_safe(tmp_path):
    store = JobStore(tmp_path)
    ids = [create_job(store, letter * 32) for letter in 'abc']
    runner = JobRunner(store, tmp_path, capacity=2)
    started, release = threading.Event(), threading.Event()
    seen = []
    def run(job_id, run_id):
        store.transition(job_id, run_id, 'processing')
        seen.append(job_id)
        started.set()
        assert release.wait(5)
        store.transition(job_id, run_id, 'complete')
    runner.run = run
    try:
        runner.submit(ids[0], {})
        assert started.wait(5)
        runner.submit(ids[0], {})
        runner.submit(ids[1], {})
        with pytest.raises(JobConflict, match='full'):
            runner.submit(ids[2], {})
        assert seen == [ids[0]]
        release.set()
        runner.pending.join()
        assert seen == ids[:2]
        assert all(store.get(job_id).status['state'] == 'complete' for job_id in ids[:2])
    finally:
        release.set()
        runner.close()


def test_worker_uses_structured_events_and_requires_output(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, {'auto_widen': False}, run_id)
    runner = JobRunner(store, Path(__file__).resolve().parents[1])
    class Process:
        def __init__(self, command, **kwargs):
            assert '--no-auto-widen' in command
            output = Path(command[command.index('--output-dir') + 1])
            (output / 'tactical-crop.mp4').write_bytes(b'output')
            self.stdout = io.StringIO('Human wording can change completely\n' + json.dumps(ProgressEvent('render', 1, 2).__dict__) + '\n' + json.dumps(ProgressEvent('complete', 1, 1).__dict__) + '\n')
        def wait(self):
            return 0
        def poll(self):
            return 0
    with patch('panstudio.worker.subprocess.Popen', Process):
        runner.run(job_id, run_id)
    result = store.get(job_id).status
    assert result['state'] == 'complete'
    assert result['progress'] == 100
    assert f'/runs/{run_id}/' in result['output_url']


@pytest.mark.parametrize('version, reused', [(None, False), ('current', True), (-1, False)])
def test_worker_reuses_flights_only_from_current_detector(application, version, reused):
    from panstudio.artifacts import save_flights
    from panstudio.disc import DETECTOR_VERSION
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    previous = store.directory(job_id) / 'runs' / ('c' * 32)
    previous.mkdir(parents=True)
    (previous / 'detections.json').write_text('{}')  # load_detections and validate_reuse are stubbed below
    flights = previous / 'flights.json'
    if version is None:
        flights.write_text(json.dumps({'schema': 'flights/v1', 'start': 0, 'end': 10, 'flights': []}))
    else:
        save_flights(flights, [], 0, 10, DETECTOR_VERSION if version == 'current' else version)
    store.claim(job_id, {}, run_id)
    runner = JobRunner(store, Path(__file__).resolve().parents[1])
    commands = []
    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            output = Path(command[command.index('--output-dir') + 1])
            (output / 'tactical-crop.mp4').write_bytes(b'output')
            self.stdout = io.StringIO(json.dumps(ProgressEvent('complete', 1, 1).__dict__) + '\n')
        def wait(self): return 0
        def poll(self): return 0
    with patch('panstudio.worker.subprocess.Popen', Process), \
         patch('panstudio.worker.probe_video'), patch('panstudio.worker.load_detections'), \
         patch('panstudio.worker.validate_reuse'):
        runner.run(job_id, run_id)
    command = commands[0]
    assert '--detections-in' in command
    assert ('--flights-in' in command) == reused
    if reused:
        assert command[command.index('--flights-in') + 1] == str(flights)


def test_worker_reports_structured_error(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, {}, run_id)
    runner = JobRunner(store, Path(__file__).resolve().parents[1])
    class Process:
        stdout = io.StringIO(json.dumps(ProgressEvent('error', message='Bad calibration').__dict__) + '\nTraceback noise\n')
        def wait(self): return 1
        def poll(self): return 1
    with patch('panstudio.worker.subprocess.Popen', return_value=Process()):
        runner.run(job_id, run_id)
    assert store.get(job_id).status['error'] == 'Bad calibration'


def test_legacy_jobs_are_imported(tmp_path):
    directory = tmp_path / ('a' * 32)
    directory.mkdir()
    (directory / 'job.json').write_text(json.dumps({'filename': 'old.mp4'}))
    store = JobStore(tmp_path)
    store.recover()
    assert store.get('a' * 32).metadata['filename'] == 'old.mp4'


def test_invalid_boolean_does_not_start_job(application):
    client = application.test_client()
    assert client.post('/api/process/' + 'a' * 32, json={'debug': 'false'}).status_code == 400
    assert application.extensions['job_runner'].calls == 0


def test_full_local_workflow_with_generated_video(tmp_path):
    """Exercise real HTTP handlers, inference subprocess, artifacts, and encoding."""
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[1]
    if not (root / 'yolo11s.pt').exists() or not shutil.which('ffmpeg'):
        pytest.skip('Requires FFmpeg and locally downloaded yolo11s.pt weights')
    source = tmp_path / 'synthetic.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'color=gray:size=2560x1440:rate=10:duration=0.6', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
    app = create_app(tmp_path / 'jobs')
    runner = app.extensions['job_runner']
    client = app.test_client()
    try:
        with source.open('rb') as handle:
            response = client.post('/api/upload', data={'video': (handle, 'synthetic.mp4')}, content_type='multipart/form-data')
        assert response.status_code == 200, response.json
        job_id = response.json['job_id']
        points = [{'x': .05, 'y': .05}, {'x': .95, 'y': .05}, {'x': .95, 'y': .95}, {'x': .05, 'y': .95}]
        assert client.post(f'/api/active-area/{job_id}', json={'points': points}).status_code == 200
        assert client.post(f'/api/process/{job_id}', json={'debug': False}).status_code == 200
        wait_for_job(app, job_id)
        result = client.get(f'/api/status/{job_id}').json
        assert result['state'] == 'complete', result
        assert client.get(result['output_url']).status_code == 200
        job = app.extensions['job_store'].get(job_id)
        directory = app.extensions['job_store'].directory(job_id)
        first_run = job.metadata['run_id']
        assert (directory / 'runs' / first_run / 'camera-plan.json').exists()
        # A second processing attempt exercises automatic cache reuse and debug rendering.
        client.post(f'/api/process/{job_id}', json={'debug': True, 'auto_widen': False})
        wait_for_job(app, job_id)
        result = client.get(f'/api/status/{job_id}').json
        assert result['state'] == 'complete', result
        assert client.get(result['debug_url']).status_code == 200
        second_run = app.extensions['job_store'].get(job_id).metadata['run_id']
        assert second_run != first_run
        log = (directory / 'runs' / second_run / 'process.log').read_text()
        assert 'Reusing saved detections' in log
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(directory / 'runs' / second_run / 'tactical-crop.mp4'), '-f', 'null', '-'], check=True)
    finally:
        runner.close()


def wait_for_job(app, job_id):
    import time
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if app.extensions['job_store'].get(job_id).status['state'] not in {'queued', 'processing'}:
            return
        time.sleep(.1)
    pytest.fail('Worker did not finish within 90 seconds')


def test_relative_workspace_paths_are_resolved_before_worker_changes_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = JobStore(Path('jobs'))
    runner = JobRunner(store, Path('project'), calibration=Path('camera.json'))
    assert store.root == tmp_path / 'jobs'
    assert runner.root == tmp_path / 'project'
    assert runner.calibration == tmp_path / 'camera.json'


def test_upload_starts_player_analysis_and_mask_edits_stay_allowed(tmp_path):
    class RecordingRunner(IdleRunner):
        submissions = []
        def submit(self, job_id, settings):
            RecordingRunner.submissions.append(settings)
            return super().submit(job_id, settings)
    app = create_app(tmp_path, runner_factory=RecordingRunner)
    client = app.test_client()
    source = tmp_path / 'source.mp4'
    import shutil
    import subprocess
    if not shutil.which('ffmpeg'):
        pytest.skip('Requires FFmpeg')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=gray:size=192x108:rate=10:duration=1',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
    with source.open('rb') as handle:
        response = client.post('/api/upload', data={'video': (handle, 'game.mp4')}, content_type='multipart/form-data')
    assert response.status_code == 200, response.json
    job_id = response.json['job_id']
    assert RecordingRunner.submissions == [{'mode': 'analyze'}]
    assert response.json['status']['state'] == 'queued' and response.json['status']['mode'] == 'analyze'
    assert response.json['analysis']['state'] == 'running'
    # Drawing the outline must not be blocked by the analysis run.
    points = [{'x': .1, 'y': .1}, {'x': .9, 'y': .1}, {'x': .9, 'y': .9}, {'x': .1, 'y': .9}]
    assert client.post(f'/api/active-area/{job_id}', json={'points': points}).status_code == 200
    # Deleting still waits for the run to finish.
    assert client.delete(f'/api/jobs/{job_id}').status_code == 409


def test_worker_analysis_run_records_suggestion_and_starts_queued_processing(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, {'mode': 'analyze'}, run_id)
    assert store.analysis_busy(job_id)
    runner = JobRunner(store, Path(__file__).resolve().parents[1])
    # Processing requested mid-analysis is remembered rather than dropped.
    assert runner.submit(job_id, {'mode': 'process', 'auto_widen': True})['state'] == 'queued'
    assert store.get(job_id).metadata['process_after_analysis'] == {'mode': 'process', 'auto_widen': True}
    class Process:
        def __init__(self, command, **kwargs):
            assert '--detect-only' in command and '--play-area' not in command
            output = Path(command[command.index('--output-dir') + 1])
            (output / 'detections.json').write_bytes(b'{}')
            (output / 'suggested-outline.json').write_text(json.dumps({'points': [{'x': .1, 'y': .2}, {'x': .9, 'y': .2}, {'x': .5, 'y': .9}], 'minutes': 12}))
            (output / 'sideline-heat.png').write_bytes(b'png')
            self.stdout = io.StringIO(json.dumps(ProgressEvent('detection', 1, 2).__dict__) + '\n' + json.dumps(ProgressEvent('complete', 1, 1).__dict__) + '\n')
        def wait(self): return 0
        def poll(self): return 0
    followed = []
    with patch('panstudio.worker.subprocess.Popen', Process), patch.object(runner, 'submit', lambda job, settings: followed.append(settings)):
        runner.run(job_id, run_id)
    job = store.get(job_id)
    assert job.status['state'] == 'analyzed'
    assert job.metadata['analysis_run_id'] == run_id and 'process_after_analysis' not in job.metadata
    assert followed == [{'mode': 'process', 'auto_widen': True}]
    summary = application.test_client().get(f'/api/jobs/{job_id}').json['analysis']
    assert summary['state'] == 'ready' and len(summary['points']) == 3 and summary['heat_url'].endswith('sideline-heat.png')
