import io
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from gui import create_app
from panstudio import attempt
from panstudio.events import ProgressEvent
from panstudio.jobs import JobConflict, JobStore, ProcessingSettings
from panstudio.processor import ProcessorRun
from panstudio.worker import JobRunner

ROOT = Path(__file__).resolve().parents[1]


class IdleRunner:
    def __init__(self, store, root, **kwargs):
        self.store = store
        self.calls = 0

    def submit(self, job_id, settings):
        self.calls += int(self.store.claim(job_id, settings, 'b' * 32))
        return self.store.get(job_id).status


class ScriptedProcess:
    """A Popen stand-in that replays processor output."""
    def __init__(self, events, exit_code=0, noise=''):
        self.stdout = io.StringIO(noise + ''.join(json.dumps(event.__dict__) + '\n' for event in events))
        self.exit_code = exit_code
    def wait(self, timeout=None): return self.exit_code
    def poll(self): return self.exit_code


COMPLETE = ProgressEvent('complete', 1, 1)


class ScriptedProcessor:
    """Test adapter for the processor seam: records each request and replays scripted output."""
    def __init__(self, events=(COMPLETE,), exit_code=0, files=('tactical-crop.mp4',), timings=None, noise=''):
        self.events, self.exit_code, self.files, self.timings, self.noise = events, exit_code, files, timings, noise
        self.commands = []
    def __call__(self, args, **kwargs):
        self.commands.append(args)
        output = Path(args[args.index('--output-dir') + 1])
        for name in self.files:
            (output / name).write_bytes(b'output')
        if self.timings is not None:
            (output / 'timings.json').write_text(json.dumps(self.timings))
        return ProcessorRun(ScriptedProcess(self.events, self.exit_code, self.noise))


def create_job(store, job_id='a' * 32, **metadata):
    directory = store.root / job_id
    directory.mkdir()
    (directory / 'source.mp4').write_bytes(b'video')
    cv2.imwrite(str(directory / 'active-area-frame.jpg'), np.full((48, 64, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(directory / 'active-area-mask.png'), np.full((48, 64), 255, dtype=np.uint8))
    store.create(job_id, {'job_id': job_id, 'filename': 'source.mp4', 'video_path': str(directory / 'source.mp4'),
                         'duration': 10., 'width': 64, 'height': 48, 'size': 5, **metadata})
    return job_id


@pytest.fixture
def application(tmp_path):
    app = create_app(tmp_path, runner_factory=IdleRunner)
    create_job(app.extensions['job_store'])
    return app


def test_process_claims_once_and_persists_options(application):
    client = application.test_client()
    job_id = 'a' * 32
    first = client.post(f'/api/process/{job_id}', json={'debug': False})
    second = client.post(f'/api/process/{job_id}', json={'debug': True})
    assert first.json['state'] == second.json['state'] == 'queued'
    assert application.extensions['job_runner'].calls == 1
    metadata = application.extensions['job_store'].get(job_id).metadata
    assert metadata['generate_debug'] is False
    assert metadata['disc_tracking'] is True  # on unless the request turns it off


def test_process_persists_disc_tracking_off(application):
    client = application.test_client()
    job_id = 'a' * 32
    client.post(f'/api/process/{job_id}', json={'debug': False, 'disc_tracking': False})
    assert application.extensions['job_store'].get(job_id).metadata['disc_tracking'] is False
    assert client.get(f'/api/jobs/{job_id}').json['disc_tracking'] is False


def test_restart_recovers_active_jobs_and_preserves_completed_jobs(application):
    store = application.extensions['job_store']
    active = 'a' * 32
    complete = create_job(store, 'c' * 32)
    for job_id in (active, complete):
        store.claim(job_id, ProcessingSettings(), 'b' * 32)
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
        runner.submit(ids[0], ProcessingSettings())
        assert started.wait(5)
        runner.submit(ids[0], ProcessingSettings())
        runner.submit(ids[1], ProcessingSettings())
        with pytest.raises(JobConflict, match='full'):
            runner.submit(ids[2], ProcessingSettings())
        assert seen == [ids[0]]
        release.set()
        runner.pending.join()
        assert seen == ids[:2]
        assert all(store.get(job_id).status['state'] == 'complete' for job_id in ids[:2])
    finally:
        release.set()
        runner.close()


def test_worker_uses_structured_events_and_requires_output(tmp_path):
    store = JobStore(tmp_path)
    job_id, run_id = create_job(store, auto_widen=False), 'b' * 32  # saved before widening became mandatory
    store.claim(job_id, ProcessingSettings(), run_id)
    processor = ScriptedProcessor([ProgressEvent('render', 1, 2), ProgressEvent('complete', 1, 1)],
                                  noise='Human wording can change completely\n')
    JobRunner(store, ROOT, start_processor=processor).run(job_id, run_id)
    assert '--no-auto-widen' not in processor.commands[0]
    result = store.get(job_id).status
    assert result['state'] == 'complete'
    assert result['progress'] == 100
    assert f'/runs/{run_id}/' in result['output_url']


def test_worker_fails_when_output_is_missing(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, ProcessingSettings(generate_debug=True), run_id)
    JobRunner(store, ROOT, start_processor=ScriptedProcessor()).run(job_id, run_id)
    assert store.get(job_id).status['error'] == 'Processor completed without the requested review video.'


def test_worker_skips_disc_scan_when_tracking_is_off(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, ProcessingSettings(disc_tracking=False), run_id)
    processor = ScriptedProcessor()
    JobRunner(store, ROOT, start_processor=processor).run(job_id, run_id)
    assert '--no-disc-tracking' in processor.commands[0] and '--flights-in' not in processor.commands[0]
    assert store.get(job_id).status['state'] == 'complete'


def test_worker_tracks_stage_timing_and_attaches_final_timings(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, ProcessingSettings(), run_id)
    seen = []
    original = store.transition
    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(dict(store.get(job_id).status))
        return result
    processor = ScriptedProcessor(
        [ProgressEvent('detection', 1, 4, time=1000.0), ProgressEvent('detection', 2, 4, time=1010.0),
         ProgressEvent('render', 1, 2, time=1050.0), ProgressEvent('complete', 1, 1, time=1060.0)],
        timings={'total_seconds': 90.5, 'stages': {'detection': 60.0, 'render': 30.0}, 'machine': {}})
    with patch.object(store, 'transition', recording):
        JobRunner(store, ROOT, start_processor=processor).run(job_id, run_id)
    progress = [status for status in seen if status['state'] == 'processing' and status.get('stage')]
    assert [(s['stage'], s['stage_started_at'], s['stage_fraction']) for s in progress] == \
        [('detection', 1000.0, 0.25), ('detection', 1000.0, 0.5), ('render', 1050.0, 0.5)]
    result = store.get(job_id).status
    assert result['timings'] == {'total_seconds': 90.5, 'stages': {'detection': 60.0, 'render': 30.0}}
    assert result['finished_at'] >= result['started_at']


def test_status_reports_server_time(application):
    job_id = 'a' * 32
    application.extensions['job_store'].claim(job_id, ProcessingSettings(), 'b' * 32)
    before = time.time()
    status = application.test_client().get(f'/api/status/{job_id}').get_json()
    assert status['state'] == 'queued' and before <= status['server_time'] <= time.time()


@pytest.mark.parametrize('version, reused', [(None, False), ('current', True), (-1, False)])
def test_attempt_reuses_flights_only_from_current_detector(tmp_path, version, reused):
    from panstudio.artifacts import save_flights
    from panstudio.disc import DETECTOR_VERSION
    previous = tmp_path / 'runs' / ('c' * 32)
    previous.mkdir(parents=True)
    (previous / 'detections.json').write_text('{}')  # load_detections and validate_reuse are stubbed below
    flights = previous / 'flights.json'
    if version is None:
        flights.write_text(json.dumps({'schema': 'flights/v1', 'start': 0, 'end': 10, 'flights': []}))
    else:
        save_flights(flights, [], 0, 10, DETECTOR_VERSION if version == 'current' else version)
    with patch('panstudio.attempt.probe_video'), patch('panstudio.attempt.load_detections'), \
         patch('panstudio.attempt.validate_reuse'):
        request = attempt.prepare(tmp_path, 'b' * 32, tmp_path / 'source.mp4', 10.0, tmp_path / 'mask.png', model='m.pt')
    assert request.detections_in == previous / 'detections.json'
    assert (request.flights_in == flights) == reused
    assert ('--flights-in' in request.args()) == reused


def test_attempt_skips_invalid_caches_and_validates_with_processor_defaults(tmp_path):
    runs = tmp_path / 'runs'
    for name in ('old', 'new'):
        (runs / name).mkdir(parents=True)
        (runs / name / 'detections.json').write_text('{}')
    seen = []
    def validate(config, artifact, video):
        seen.append(config)
        if artifact == 'new':
            raise ValueError('different detector settings')
    with patch('panstudio.attempt.probe_video'), patch('panstudio.attempt.validate_reuse', validate), \
         patch('panstudio.attempt.load_detections', lambda path: path.parent.name):
        import os
        os.utime(runs / 'old' / 'detections.json', ns=(1, 1))
        request = attempt.prepare(tmp_path, 'b' * 32, tmp_path / 'source.mp4', 10.0, tmp_path / 'mask.png',
                                  model='m.pt', disc_tracking=False)
    assert request.detections_in == runs / 'old' / 'detections.json' and request.flights_in is None
    # The cache is checked against the configuration the processor itself will parse.
    from panstudio.cli import parse_args
    from panstudio.models import ProcessingConfig
    assert seen[0] == ProcessingConfig.from_namespace(parse_args(request.args()))
    assert seen[0].end == 9.95 and seen[0].no_disc_tracking


def test_processor_requires_completion_event_and_clean_exit():
    from panstudio.processor import ProcessorFailed
    run = ProcessorRun(ScriptedProcess([ProgressEvent('render', 1, 2)]))
    assert [event.stage for event in run.events()] == ['render']
    with pytest.raises(ProcessorFailed, match='did not complete'):
        run.finish()
    run = ProcessorRun(ScriptedProcess([ProgressEvent('complete', 1, 1)], exit_code=1))
    list(run.events())
    with pytest.raises(ProcessorFailed, match='exit status 1'):
        run.finish()


def test_processing_settings_default_and_validate():
    assert ProcessingSettings.from_request({}) == ProcessingSettings(False, True)
    assert ProcessingSettings.from_metadata({}) == ProcessingSettings(False, True)  # saved before the options existed
    assert ProcessingSettings.from_metadata({'disc_tracking': False}).disc_tracking is False
    for bad in ({'debug': 'false'}, {'disc_tracking': 0}, []):
        with pytest.raises(ValueError):
            ProcessingSettings.from_request(bad)


def test_worker_reports_structured_error(application):
    store = application.extensions['job_store']
    job_id, run_id = 'a' * 32, 'b' * 32
    store.claim(job_id, ProcessingSettings(), run_id)
    processor = ScriptedProcessor([ProgressEvent('error', message='Bad calibration')], exit_code=1, files=())
    JobRunner(store, ROOT, start_processor=processor).run(job_id, run_id)
    assert store.get(job_id).status['error'] == 'Bad calibration'


def test_invalid_boolean_does_not_start_job(application):
    client = application.test_client()
    assert client.post('/api/process/' + 'a' * 32, json={'debug': 'false'}).status_code == 400
    assert client.post('/api/process/' + 'a' * 32, json={'disc_tracking': 0}).status_code == 400
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
        client.post(f'/api/process/{job_id}', json={'debug': True})
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


def test_upload_starts_no_work_until_process(tmp_path):
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
    assert RecordingRunner.submissions == []
    assert response.json['status']['state'] == 'ready'


def test_saved_video_downloads_under_the_chosen_name(application):
    client = application.test_client()
    job_id = 'a' * 32
    run = application.extensions['job_store'].root / job_id / 'runs' / ('b' * 32)
    run.mkdir(parents=True)
    (run / 'tactical-crop.mp4').write_bytes(b'video')
    url = f'/jobs/{job_id}/runs/{"b" * 32}/tactical-crop.mp4'
    named = client.get(f'{url}?download=1&name=Semifinal%20game%202')
    assert 'filename=Semifinal_game_2.mp4' in named.headers['Content-Disposition']
    traversal = client.get(f'{url}?download=1&name=../../etc/passwd')
    assert '/' not in traversal.headers['Content-Disposition'].split('filename=')[1]
    plain = client.get(f'{url}?download=1')
    assert 'filename=tactical-crop.mp4' in plain.headers['Content-Disposition']
    assert client.get(f'{url}?name=ignored').headers['Content-Disposition'].startswith('inline')


def test_jobs_whose_source_was_deleted_are_hidden_and_cannot_run(application):
    client = application.test_client()
    job_id = 'a' * 32
    (application.extensions['job_store'].root / job_id / 'source.mp4').unlink()
    assert client.get('/api/jobs').json == []          # nothing left to open, so not listed
    assert client.get(f'/api/jobs/{job_id}').json['source_missing'] is True
    response = client.post(f'/api/process/{job_id}', json={})
    assert response.status_code == 400 and 'deleted' in response.json['error']
    assert application.extensions['job_runner'].calls == 0
