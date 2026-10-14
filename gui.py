#!/usr/bin/env python3
"""Local HTTP adapter. Job persistence and processing live in panstudio."""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
from functools import wraps
import shutil
import subprocess
import threading
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from panstudio.jobs import BUSY, JobConflict, JobStore
from panstudio.masking import save_mask_preview
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

    def analysis_summary(job, directory):
        """What the automatic player analysis has produced for the outline step."""
        status, metadata = job.status, job.metadata
        if status['state'] in BUSY and status.get('mode') == 'analyze':
            return {'state': 'running', 'progress': status.get('progress', 0), 'message': status.get('message', '')}
        run_id = metadata.get('analysis_run_id')
        if not run_id:
            failed = status['state'] in {'failed', 'interrupted'} and status.get('mode') == 'analyze'
            return {'state': 'failed' if failed else 'none', 'message': status.get('message', '') if failed else ''}
        run_dir = directory / 'runs' / run_id
        summary = {'state': 'ready', 'points': None, 'minutes': None, 'heat_url': None}
        suggestion = run_dir / 'suggested-outline.json'
        if suggestion.is_file():
            try:
                data = json.loads(suggestion.read_text())
                summary['points'], summary['minutes'] = data.get('points'), data.get('minutes')
            except (ValueError, TypeError):
                pass
        if (run_dir / 'sideline-heat.png').is_file():
            summary['heat_url'] = f'/jobs/{job.id}/runs/{run_id}/sideline-heat.png'
        return summary

    def public_job(job):
        directory = store.directory(job.id)
        metadata = job.metadata
        return {
            'analysis': analysis_summary(job, directory),
            'awaiting_process': bool(metadata.get('process_after_analysis')),
            'job_id': job.id, 'filename': metadata['filename'],
            'width': metadata['width'], 'height': metadata['height'],
            'duration': metadata['duration'], 'size': metadata['size'],
            'auto_widen': metadata.get('auto_widen', True),
            'generate_debug': metadata.get('generate_debug', False),
            'has_mask': (directory / 'active-area-mask.png').is_file(),
            'frame_url': f'/jobs/{job.id}/active-area-frame.jpg',
            'preview_url': f'/jobs/{job.id}/play-area-preview.jpg',
            'status': job.status, 'points': metadata.get('points'),
        }

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/jobs')
    @api_errors
    def list_jobs():
        return jsonify([public_job(job) for job in store.list()])

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
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(preview_time),
                '-i', str(video_path), '-frames:v', '1', '-vf', 'scale=1600:-2', '-q:v', '2',
                str(directory / 'active-area-frame.jpg'),
            ], check=True, capture_output=True)
            metadata = {'width': video.width, 'height': video.height, 'duration': video.duration,
                        'size': video_path.stat().st_size, 'job_id': job_id, 'filename': filename,
                        'video_path': str(video_path.resolve()), 'preview_time': preview_time, 'auto_widen': True}
            store.create(job_id, metadata)
        except Exception as error:
            shutil.rmtree(directory, ignore_errors=True)
            return jsonify(error=f'Could not prepare that video: {error}'), 400
        # Player analysis starts immediately so the outline step can suggest an
        # outline. If the queue is full the user can still draw and process.
        try:
            runner.submit(job_id, {'mode': 'analyze'})
        except JobConflict:
            pass
        return jsonify(public_job(store.get(job_id)))

    @app.post('/api/analyze/<job_id>')
    @api_errors
    def analyze_video(job_id):
        store.directory(job_id)
        return jsonify(runner.submit(job_id, {'mode': 'analyze'}))

    @app.post('/api/active-area/<job_id>')
    @api_errors
    def create_active_area(job_id):
        with store.lock:
            job = store.get(job_id)
            if job.status['state'] in BUSY and not store.analysis_busy(job_id):
                raise JobConflict('Wait for processing to finish before changing the active area.')
            directory = store.directory(job_id)
            payload = request.get_json(force=True)
            if not isinstance(payload, dict):
                raise ValueError('Expected an object containing outline points.')
            points = payload.get('points', [])
            if len(points) < 3:
                raise ValueError('Draw one closed outline around the active field.')
            coordinates = np.array([[point['x'], point['y']] for point in points], dtype=float)
            if not np.isfinite(coordinates).all():
                raise ValueError('Outline coordinates must be finite.')
            clean = cv2.imread(str(directory / 'active-area-frame.jpg'))
            if clean is None:
                raise ValueError('The preview frame is missing')
            height, width = clean.shape[:2]
            polygon = (np.clip(coordinates, 0, 1) * [width - 1, height - 1]).astype(np.int32)
            if abs(cv2.contourArea(polygon)) < 1:
                raise ValueError('Draw an outline with a nonempty interior.')
            annotated = clean.copy()
            cv2.polylines(annotated, [polygon], True, (0, 0, 255), max(8, width // 180), cv2.LINE_AA)
            cv2.imwrite(str(directory / 'active-area.jpg'), annotated, [cv2.IMWRITE_JPEG_QUALITY, 96])
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            cv2.imwrite(str(directory / 'active-area-mask.png'), mask)
            save_mask_preview(mask, clean, directory / 'play-area-preview.jpg')
            # A running player analysis keeps its lifecycle; only the outline changes.
            status = job.status if store.analysis_busy(job_id) else {'state': 'ready', 'progress': 0, 'message': 'Review the active area, then process.'}
            store.save(job_id, metadata={**job.metadata, 'points': points}, status=status)
            return jsonify(preview_url=f'/jobs/{job_id}/play-area-preview.jpg?v={uuid.uuid4().hex[:8]}',
                           annotated_url=f'/jobs/{job_id}/active-area.jpg')

    @app.post('/api/process/<job_id>')
    @api_errors
    def process_video(job_id):
        directory = store.directory(job_id)
        if not (directory / 'active-area-mask.png').is_file():
            raise ValueError('Generate and approve an active-area preview first.')
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError('Expected an object containing processing options.')
        debug, auto_widen = payload.get('debug', False), payload.get('auto_widen', True)
        if type(debug) is not bool or type(auto_widen) is not bool:
            raise ValueError('Processing options must be booleans.')
        return jsonify(runner.submit(job_id, {'mode': 'process', 'generate_debug': debug, 'auto_widen': auto_widen}))

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
        return send_from_directory(store.directory(job_id), filename,
                                   as_attachment=request.args.get('download') == '1', conditional=True)

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
            raise SystemExit('Another Tactical Pan Studio server is using this jobs folder.')
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
