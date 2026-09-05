#!/usr/bin/env python3
"""Local browser GUI for the tactical auto-pan workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from tactical_pan import save_mask_preview


ROOT = Path(__file__).resolve().parent
JOBS_ROOT = ROOT / "gui_jobs"
JOBS_ROOT.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = None

job_status: dict[str, dict] = {}
status_lock = threading.Lock()


def job_directory(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("Invalid job id")
    directory = JOBS_ROOT / job_id
    if not directory.is_dir():
        raise FileNotFoundError(job_id)
    return directory


def metadata_path(job_id: str) -> Path:
    return job_directory(job_id) / "job.json"


def read_metadata(job_id: str) -> dict:
    return json.loads(metadata_path(job_id).read_text())


def write_metadata(job_id: str, metadata: dict):
    metadata_path(job_id).write_text(json.dumps(metadata, indent=2))


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration,size",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(data["format"]["duration"]),
        "size": int(data["format"]["size"]),
    }


def update_status(job_id: str, **changes):
    with status_lock:
        current = job_status.setdefault(job_id, {"state": "ready", "progress": 0, "message": "Ready"})
        current.update(changes)


def run_processor(job_id: str):
    metadata = read_metadata(job_id)
    generate_debug = bool(metadata.get("generate_debug", False))
    auto_widen = bool(metadata.get("auto_widen", True))
    directory = job_directory(job_id)
    output_dir = directory / "output"
    output_dir.mkdir(exist_ok=True)
    log_path = directory / "process.log"
    command = [
        sys.executable,
        str(ROOT / "tactical_pan.py"),
        metadata["video_path"],
        "--start", "0",
        "--end", str(max(0.1, metadata["duration"] - 0.05)),
        "--play-area", str(directory / "active-area-mask.png"),
        "--output-dir", str(output_dir),
    ]
    if generate_debug:
        command.append("--debug")
    if not auto_widen:
        command.append("--no-auto-widen")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "YOLO_CONFIG_DIR": str(directory / ".ultralytics"),
            "MPLCONFIGDIR": str(directory / ".matplotlib"),
        }
    )
    try:
        update_status(job_id, state="processing", progress=1, message="Starting player analysis…")
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                analysis = re.search(r"Analyzed\s+(\d+)/(\d+)", line)
                render = re.search(r"Rendering (debug-wide|tactical-crop)\.mp4: (\d+)/(\d+)", line)
                if analysis:
                    current, total = map(int, analysis.groups())
                    progress = 5 + round(65 * current / max(1, total))
                    update_status(job_id, progress=progress, message=f"Analyzing players — {current} of {total}")
                elif render:
                    name, current_text, total_text = render.groups()
                    current, total = int(current_text), int(total_text)
                    if name == "debug-wide":
                        progress = 70 + round(15 * current / max(1, total))
                        message = "Rendering the review video…"
                    else:
                        render_start = 85 if generate_debug else 70
                        render_span = 14 if generate_debug else 29
                        progress = render_start + round(render_span * current / max(1, total))
                        message = "Rendering the tactical crop…"
                    update_status(job_id, progress=progress, message=message)
            return_code = process.wait()
            process.stdout.close()
        if return_code:
            lines = [line.strip() for line in log_path.read_text().splitlines() if line.strip()]
            detail = lines[-1] if lines else f"Processor exited with status {return_code}"
            raise RuntimeError(detail)
        completion = {
            "state": "complete",
            "progress": 100,
            "message": "Your tactical video is ready.",
            "output_url": f"/jobs/{job_id}/output/tactical-crop.mp4",
        }
        if generate_debug:
            completion["debug_url"] = f"/jobs/{job_id}/output/debug-wide.mp4"
        update_status(job_id, **completion)
    except Exception as error:
        update_status(job_id, state="failed", message=str(error), error=str(error))


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload_video():
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose a video first."}), 400
    job_id = uuid.uuid4().hex
    directory = JOBS_ROOT / job_id
    directory.mkdir(parents=True)
    filename = secure_filename(uploaded.filename) or "source-video.mp4"
    video_path = directory / filename
    try:
        uploaded.save(video_path)
        details = probe_video(video_path)
        preview_time = min(40.0, max(0.0, details["duration"] * 0.2))
        frame_path = directory / "active-area-frame.jpg"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(preview_time), "-i", str(video_path),
                "-frames:v", "1", "-vf", "scale=1600:-2", "-q:v", "2", str(frame_path),
            ],
            check=True,
        )
        metadata = {
            **details,
            "job_id": job_id,
            "filename": filename,
            "video_path": str(video_path),
            "preview_time": preview_time,
            "auto_widen": True,
        }
        write_metadata(job_id, metadata)
        update_status(job_id, state="ready", progress=0, message="Draw the active area.")
        return jsonify(
            {
                **metadata,
                "frame_url": f"/jobs/{job_id}/active-area-frame.jpg",
            }
        )
    except Exception as error:
        shutil.rmtree(directory, ignore_errors=True)
        return jsonify({"error": f"Could not prepare that video: {error}"}), 400


@app.post("/api/active-area/<job_id>")
def create_active_area(job_id: str):
    try:
        directory = job_directory(job_id)
        metadata = read_metadata(job_id)
        payload = request.get_json(force=True)
        points = payload.get("points", [])
        if len(points) < 3:
            return jsonify({"error": "Draw one closed outline around the active field."}), 400
        clean = cv2.imread(str(directory / "active-area-frame.jpg"))
        if clean is None:
            raise ValueError("The preview frame is missing")
        height, width = clean.shape[:2]
        polygon = np.array(
            [
                [int(np.clip(point["x"], 0, 1) * (width - 1)), int(np.clip(point["y"], 0, 1) * (height - 1))]
                for point in points
            ],
            dtype=np.int32,
        )
        annotated = clean.copy()
        stroke_width = max(8, width // 180)
        cv2.polylines(annotated, [polygon], True, (0, 0, 255), stroke_width, cv2.LINE_AA)
        cv2.imwrite(str(directory / "active-area.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 96])

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        cv2.imwrite(str(directory / "active-area-mask.png"), mask)
        save_mask_preview(mask, clean, directory / "play-area-preview.jpg")
        write_metadata(job_id, metadata)
        return jsonify(
            {
                "preview_url": f"/jobs/{job_id}/play-area-preview.jpg?v={uuid.uuid4().hex[:8]}",
                "annotated_url": f"/jobs/{job_id}/active-area.jpg",
            }
        )
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/process/<job_id>")
def process_video(job_id: str):
    try:
        directory = job_directory(job_id)
        if not (directory / "active-area-mask.png").exists():
            return jsonify({"error": "Generate and approve an active-area preview first."}), 400
        payload = request.get_json(silent=True) or {}
        thread = threading.Thread(target=run_processor, args=(job_id,), daemon=True)
        with status_lock:
            if job_status.get(job_id, {}).get("state") == "processing":
                return jsonify(job_status[job_id])
            metadata = read_metadata(job_id)
            metadata["generate_debug"] = bool(payload.get("debug", False))
            metadata["auto_widen"] = bool(payload.get("auto_widen", True))
            write_metadata(job_id, metadata)
            job_status[job_id] = {
                "state": "processing",
                "progress": 1,
                "message": "Starting player analysis…",
            }
            try:
                thread.start()
            except Exception:
                job_status[job_id] = {
                    "state": "failed",
                    "progress": 0,
                    "message": "Could not start processing.",
                }
                raise
        return jsonify({"state": "processing"})
    except (ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 404


@app.get("/api/status/<job_id>")
def processing_status(job_id: str):
    try:
        job_directory(job_id)
    except (ValueError, FileNotFoundError):
        return jsonify({"error": "Unknown job"}), 404
    with status_lock:
        return jsonify(job_status.get(job_id, {"state": "ready", "progress": 0, "message": "Ready"}))


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    """Delete one job's source, working files, and outputs on explicit request."""
    try:
        directory = job_directory(job_id)
        with status_lock:
            if job_status.get(job_id, {}).get("state") == "processing":
                return jsonify({"error": "Wait for processing to finish before deleting this job."}), 409
            shutil.rmtree(directory)
            job_status.pop(job_id, None)
        return jsonify({"deleted": True})
    except (ValueError, FileNotFoundError):
        return jsonify({"error": "Unknown job"}), 404


@app.get("/jobs/<job_id>/<path:filename>")
def job_file(job_id: str, filename: str):
    try:
        directory = job_directory(job_id)
    except (ValueError, FileNotFoundError):
        return jsonify({"error": "Unknown job"}), 404
    return send_from_directory(
        directory,
        filename,
        as_attachment=request.args.get("download") == "1",
        conditional=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local Tactical Pan Studio")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    url = f"http://127.0.0.1:{arguments.port}"
    if not arguments.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Tactical Pan Studio is running at {url}")
    app.run(host="127.0.0.1", port=arguments.port, debug=False, threaded=True)
