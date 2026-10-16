#!/usr/bin/env python3
"""Local HTTP adapter. Job persistence and processing live in panstudio."""
from __future__ import annotations

import argparse
import atexit
import fcntl
from functools import wraps
import shutil
import threading
import uuid
import webbrowser
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from panstudio.jobs import BUSY, JobConflict, JobStore
from panstudio.masking import polygon_mask, save_mask_preview
from panstudio.media import extract_frame
from panstudio.pipeline import probe_video
from panstudio.worker import JobRunner

ROOT = Path(__file__).resolve().parent
JOBS_ROOT = ROOT / 'gui_jobs'


def create_app(jobs_root: Path = JOBS_ROOT, *, runner_factory=JobRunner, calibration: Path | None = None):
    app = Flask(__name__)
    store = JobStore(jobs_root)
    store.recover()
    runner = runner_factory(store, ROOT, calibration=calibration)
    app.extensions.update(job_store=store, job_runner=runner)

    def api_errors(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except JobConflict as error:
                return jsonify(error=str(error)), 409
            except FileNotFoundError:
                return jsonify(error='Unknown job or missing working file'), 404
            except (ValueError, TypeError, KeyError) as error:
                return jsonify(error=str(error)), 400
        return wrapped

    def public_job(job):
        directory = store.directory(job.id)
        metadata = job.metadata
        return {
            'job_id': job.id, 'filename': metadata['filename'],
            'width': metadata['width'], 'height': metadata['height'],
            'duration': metadata['duration'], 'size': metadata['size'],
            'auto_widen': metadata.get('auto_widen', True),
            'generate_debug': metadata.get('generate_debug', False),
            'has_mask': (directory / 'active-area-mask.png').is_file(),
            'frame_url': f'/jobs/{job.id}/active-area-frame.jpg',
            'preview_url': f'/jobs/{job.id}/play-area-preview.jpg',
            'status': job.status, 'points': metadata.get('points'),
            'source_missing': not Path(metadata['video_path']).is_file(),
        }

    def require_source(job_id):
        """Analysis and processing read the source video; a job whose file was deleted cannot run."""
        if not Path(store.get(job_id).metadata['video_path']).is_file():
            raise ValueError("This job's source video has been deleted. Upload the video again to process it.")

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/jobs')
    @api_errors
    def list_jobs():
        # A job whose source video was deleted can only still show a finished output.
        jobs = [public_job(job) for job in store.list()]
        return jsonify([job for job in jobs if not job['source_missing'] or job['status'].get('output_url')])

    @app.get('/api/jobs/<job_id>')
    @api_errors
    def get_job(job_id):
        return jsonify(public_job(store.get(job_id)))

    @app.post('/api/upload')
    @api_errors
    def upload_video():
        uploaded = request.files.get('video')
        if not uploaded or not uploaded.filename:
            return jsonify(error='Choose a video first.'), 400
        job_id = uuid.uuid4().hex
        directory = store.root / job_id
        directory.mkdir()
        filename = secure_filename(uploaded.filename) or 'source-video.mp4'
        video_path = directory / 'source.mp4'
        try:
            uploaded.save(video_path)
            video = probe_video(video_path)
            preview_time = min(40.0, max(0.0, video.duration * 0.2))
            extract_frame(video_path, preview_time, directory / 'active-area-frame.jpg', width=1600)
            metadata = {'width': video.width, 'height': video.height, 'duration': video.duration,
                        'size': video_path.stat().st_size, 'job_id': job_id, 'filename': filename,
                        'video_path': str(video_path.resolve()), 'auto_widen': True}
            store.create(job_id, metadata)
        except Exception as error:
            shutil.rmtree(directory, ignore_errors=True)
            return jsonify(error=f'Could not prepare that video: {error}'), 400
        return jsonify(public_job(store.get(job_id)))

    @app.post('/api/active-area/<job_id>')
    @api_errors
    def create_active_area(job_id):
        with store.lock:
            job = store.get(job_id)
            if job.status['state'] in BUSY:
                raise JobConflict('Wait for processing to finish before changing the active area.')
            directory = store.directory(job_id)
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                raise ValueError('Expected an object containing outline points.')
            points = payload.get('points', [])
            if len(points) < 3:
                raise ValueError('Draw one closed outline around the active field.')
            clean = cv2.imread(str(directory / 'active-area-frame.jpg'))
            if clean is None:
                raise ValueError('The preview frame is missing')
            height, width = clean.shape[:2]
            mask = polygon_mask(points, width, height)
            cv2.imwrite(str(directory / 'active-area-mask.png'), mask)
            save_mask_preview(mask, clean, directory / 'play-area-preview.jpg')
            store.save(job_id, metadata={**job.metadata, 'points': points},
                       status={'state': 'ready', 'progress': 0, 'message': 'Review the active area, then process.'})
            return jsonify(preview_url=f'/jobs/{job_id}/play-area-preview.jpg?v={uuid.uuid4().hex[:8]}')

    @app.post('/api/process/<job_id>')
    @api_errors
    def process_video(job_id):
        directory = store.directory(job_id)
        require_source(job_id)
        if not (directory / 'active-area-mask.png').is_file():
            raise ValueError('Generate and approve an active-area preview first.')
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError('Expected an object containing processing options.')
        debug, auto_widen = payload.get('debug', False), payload.get('auto_widen', True)
        if type(debug) is not bool or type(auto_widen) is not bool:
            raise ValueError('Processing options must be booleans.')
        return jsonify(runner.submit(job_id, {'generate_debug': debug, 'auto_widen': auto_widen}))

    @app.get('/api/status/<job_id>')
    @api_errors
    def processing_status(job_id):
        return jsonify(store.get(job_id).status)

    @app.delete('/api/jobs/<job_id>')
    @api_errors
    def delete_job(job_id):
        store.delete(job_id)
        return jsonify(deleted=True)

    @app.get('/jobs/<job_id>/<path:filename>')
    @api_errors
    def job_file(job_id, filename):
        download = request.args.get('download') == '1'
        # Saved under the operator's chosen name; the file on disk keeps its run name.
        name = secure_filename(request.args.get('name', '')) if download else ''
        if name and not name.lower().endswith(Path(filename).suffix.lower()):
            name += Path(filename).suffix
        return send_from_directory(store.directory(job_id), filename, as_attachment=download,
                                   download_name=name or None, conditional=True)

    return app


def parse_args():
    parser = argparse.ArgumentParser(description='Run the local Tactical Pan Studio')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--jobs-root', type=Path, default=JOBS_ROOT)
    parser.add_argument('--calibration', type=Path, help='Camera calibration JSON profile')
    return parser.parse_args()


def main():
    arguments = parse_args()
    arguments.jobs_root.mkdir(parents=True, exist_ok=True)
    # One server owns recovery and the queue for this workspace.
    with (arguments.jobs_root / '.server.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another Tactical Pan Studio server is using this jobs folder.') from None
        app = create_app(arguments.jobs_root, calibration=arguments.calibration)
        runner = app.extensions['job_runner']
        atexit.register(runner.close)
        url = f'http://127.0.0.1:{arguments.port}'
        if not arguments.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        print(f'Tactical Pan Studio is running at {url}')
        try:
            app.run(host='127.0.0.1', port=arguments.port, debug=False, threaded=True)
        finally:
            runner.close()
            atexit.unregister(runner.close)


if __name__ == '__main__':
    main()
