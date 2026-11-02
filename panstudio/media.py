from __future__ import annotations

import functools
import queue
import subprocess
import tempfile
import threading
from pathlib import Path
import cv2
import numpy as np
from .models import ProcessingConfig, Sample
from .events import report


def extract_frame(source: Path, time: float, output: Path, width: int):
    """Save one frame as a JPEG scaled to the given width."""
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(time), "-i", str(source),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "2", str(output),
    ], check=True, capture_output=True)


@functools.cache
def hwaccel_args() -> list[str]:
    """Decode on an NVIDIA GPU when one is reachable; FFmpeg falls back to software per stream."""
    try:
        usable = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-init_hw_device", "cuda",
            "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1", "-f", "null", "-",
        ], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        usable = False
    return ["-hwaccel", "cuda"] if usable else []


def letterbox_crop(frame: np.ndarray, crop_x: int, crop_y: int, crop_width: int, crop_height: int, out_width: int, out_height: int) -> np.ndarray:
    cropped = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    # Independently rounded crop dimensions can differ from the target aspect
    # by one source pixel. Don't turn that rounding error into a black border.
    if abs(cropped.shape[0] - cropped.shape[1] * out_height / out_width) <= 1.0:
        return cv2.resize(cropped, (out_width, out_height), interpolation=cv2.INTER_AREA)
    scale = min(out_width / cropped.shape[1], out_height / cropped.shape[0])
    resized = cv2.resize(cropped, (round(cropped.shape[1] * scale), round(cropped.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    rendered = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    oy = (out_height - resized.shape[0]) // 2
    ox = (out_width - resized.shape[1]) // 2
    rendered[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return rendered


def analysis_decoder_command(args: ProcessingConfig, sample_count: int) -> list[str]:
    """Decode analysis samples in one pass instead of seeking for every frame."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", *hwaccel_args(),
        "-ss", str(args.start), "-i", str(args.input),
        "-vf", f"fps={args.sample_rate}:start_time=0",
        "-frames:v", str(sample_count),
        "-an", "-sn", "-dn",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


def sampled_analysis_frames(args: ProcessingConfig, width: int, height: int, sample_times: np.ndarray):
    """Yield fixed-rate, full-resolution BGR frames from one FFmpeg decoder."""
    error_file = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(
            analysis_decoder_command(args, len(sample_times)),
            stdout=subprocess.PIPE, stderr=error_file,
        )
    except BaseException:
        error_file.close()
        raise
    assert process.stdout is not None
    frame_size = width * height * 3
    completed = False
    try:
        for index, timestamp in enumerate(sample_times):
            data = bytearray()
            while len(data) < frame_size:
                chunk = process.stdout.read(frame_size - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                break
            if len(data) != frame_size:
                raise RuntimeError(
                    f"FFmpeg returned a partial analysis frame ({len(data)} of {frame_size} bytes)"
                )
            frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
            yield index, timestamp, frame
        completed = True
    finally:
        process.stdout.close()
        if not completed and process.poll() is None:
            process.terminate()
        return_code = process.wait()
        error_file.seek(0)
        error = error_file.read().decode(errors="replace").strip()
        error_file.close()
        if completed and return_code:
            raise RuntimeError(error or f"FFmpeg analysis decoder exited with status {return_code}")


def prefetched(items, depth: int = 4):
    """Iterate on a background thread so decoding overlaps with the consumer's work."""
    buffer = queue.Queue(depth)
    stop = threading.Event()
    finished = object()

    def put(entry):
        while not stop.is_set():
            try:
                buffer.put(entry, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            for item in items:
                if not put((None, item)):
                    break
        except BaseException as error:
            put((error, None))
        finally:
            if hasattr(items, 'close'):
                items.close()
            put((None, finished))

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            error, item = buffer.get()
            if error is not None:
                raise error
            if item is finished:
                return
            yield item
    finally:
        stop.set()
        thread.join()


def frame_rate_path(records: list[Sample], sample_times: np.ndarray, frame_times: np.ndarray,
                    default_width: float, default_height: float):
    """Resample the planned crop to frame rate without velocity kinks.

    Linear interpolation between planning samples changes pan speed abruptly
    at every sample. Averaging the interpolated path over one sample interval
    rounds those corners into smooth accelerations. The average of values on a
    line segment is a point on that segment, so the path never overshoots the
    plan and stays inside the source like the plan does.
    """
    series = []
    for key, default in (("crop_x", None), ("crop_width", default_width), ("crop_height", default_height)):
        values = np.array([r[key] if default is None else r.get(key, default) for r in records], dtype=float)
        series.append(np.interp(frame_times, sample_times, values))
    if len(sample_times) < 2 or len(frame_times) < 2:
        return tuple(series)
    sample_interval = float(np.median(np.diff(sample_times)))
    frame_interval = float(frame_times[1] - frame_times[0])
    half = int(round(sample_interval / frame_interval / 2))
    if half < 1:
        return tuple(series)
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return tuple(np.convolve(np.pad(s, (half, half), mode="edge"), kernel, mode="valid") for s in series)


@functools.cache
def nvenc_available() -> bool:
    """FFmpeg can list h264_nvenc without a usable NVIDIA driver, so try a tiny encode."""
    try:
        return subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
            "-c:v", "h264_nvenc", "-f", "null", "-",
        ], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def video_codec_args(width: int, height: int) -> list[str]:
    # NVENC rejects frames below 145x49; tiny renders are cheap on the CPU anyway.
    if width >= 256 and height >= 256 and nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19"]


def encoder_command(args: ProcessingConfig, output: Path, width: int, height: int, fps: float):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "pipe:0",
        "-ss", str(args.start), "-t", str(args.end - args.start), "-i", str(args.input),
        "-map", "0:v:0", "-map", "1:a:0?",
        *video_codec_args(width, height),
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(output),
    ]


def render(args: ProcessingConfig, records: list[Sample], source_width: int, source_height: int, fps: float, output: Path, debug: bool):
    if debug:
        out_width, out_height = max(2, source_width // 4 * 2), max(2, source_height // 4 * 2)
    else:
        out_width, out_height = args.output_width, args.output_height
    command = encoder_command(args, output, out_width, out_height, fps)
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(args.input))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    total_frames = int(round((args.end - args.start) * fps))
    sample_times = np.array([r["time"] for r in records])
    frame_times = np.arange(total_frames) / fps
    frame_crop_xs, frame_crop_widths, frame_crop_heights = frame_rate_path(
        records, sample_times, frame_times, args.crop_width, args.crop_height)

    try:
        for frame_index in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                # Containers often report a few more frames than can be decoded.
                # Tolerate up to a second missing at the very end.
                if frame_index < total_frames - max(1, round(fps)):
                    raise RuntimeError(f"Source ended after {frame_index} of {total_frames} render frames")
                break
            relative_time = frame_index / fps
            crop_x = int(round(frame_crop_xs[frame_index]))
            crop_width = int(round(frame_crop_widths[frame_index]))
            crop_height = int(round(frame_crop_heights[frame_index]))
            crop_x = int(np.clip(crop_x, 0, source_width - crop_width))
            crop_y = source_height - crop_height
            if debug:
                sample_index = int(np.argmin(np.abs(sample_times - relative_time)))
                deep_ids = set(records[sample_index].get("deep_track_ids", []))
                for player in records[sample_index]["players"]:
                    x1, y1, x2, y2 = map(int, player["box"])
                    if int(player.get("track_id", -1)) in deep_ids:
                        color = (255, 220, 30)
                    elif player.get("stationary_factor", 1.0) < 0.5:
                        color = (30, 140, 255)
                    else:
                        color = (40, 220, 40)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                if records[sample_index].get("flight_active"):
                    flight = records[sample_index]
                    cv2.line(frame, (int(flight["flight_x0"]), int(flight["flight_y0"])), (int(flight["flight_x1"]), int(flight["flight_y1"])), (255, 0, 255), 4)
                    cv2.circle(frame, (int(flight["flight_x1"]), int(flight["flight_y1"])), 40, (255, 0, 255), 6)
                    cv2.putText(frame, "DISC FLIGHT", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 0, 255), 5, cv2.LINE_AA)
                if records[sample_index].get("deep_active"):
                    cv2.putText(
                        frame,
                        "DEEP PAIR",
                        (60, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.7,
                        (255, 220, 30),
                        5,
                        cv2.LINE_AA,
                    )
                cv2.rectangle(
                    frame,
                    (crop_x, crop_y),
                    (crop_x + crop_width - 1, crop_y + crop_height - 1),
                    (0, 220, 255),
                    9,
                )
                cv2.putText(
                    frame,
                    f"tactical crop  t={relative_time:05.1f}s",
                    (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.2,
                    (0, 220, 255),
                    6,
                    cv2.LINE_AA,
                )
                rendered = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            else:
                rendered = letterbox_crop(frame, crop_x, crop_y, crop_width, crop_height, out_width, out_height)
            assert encoder.stdin is not None
            encoder.stdin.write(rendered.tobytes())
            if frame_index % max(1, round(fps * 5)) == 0:
                report("debug" if debug else "render", frame_index, total_frames)
    finally:
        cap.release()
        try:
            if encoder.stdin:
                encoder.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg encoder exited with status {return_code}")
    report("debug" if debug else "render", total_frames, total_frames)
