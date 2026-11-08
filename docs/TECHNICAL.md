# Technical details

Command-line usage and tuning. For setup and the GUI, see the [README](../README.md). Run `uv run python tactical_pan.py --help` for every option.

## Hardware

Detection runs on an NVIDIA GPU (CUDA) or Apple GPU (MPS) when available, several times faster than the CPU. Rendering uses NVENC when FFmpeg can reach an NVIDIA GPU and falls back to libx264. Pass `--device cpu` to force CPU inference; the GUI always picks automatically.

The default model is `yolo11s.pt`, downloaded on first run. `--model yolo11n.pt` is about twice as fast but misses more of the small far-sideline players.

Under WSL2, `gui.py` opens the GUI in your Windows browser at the same localhost URL. The GUI server owns its jobs folder exclusively; use `--jobs-root` to run a separate one.

## Command line

```bash
uv run python tactical_pan.py game.mp4 \
  --start 21.8 --end 74.2 \
  --play-area active-area.jpg \
  --debug \
  --output-dir prototype/point-2
```

This writes `tactical-crop.mp4`. With `--debug` it also writes `debug-wide.mp4`: the full frame with detections, ignored bystanders (orange), disc flights (magenta), and the crop.

## Field outline

`--play-area` takes an image with a thick, closed, bright-red outline drawn on an uncropped frame (or a binary mask). To check it without running detection:

```bash
uv run python tactical_pan.py game.mp4 --start 21.8 --end 74.2 \
  --play-area active-area.jpg --preview-mask-only --output-dir prototype/mask-check
```

`play-area-preview.jpg` tints the excluded area red. People who stand still or sit inside the outline are mostly ignored, but keeping spectators outside it still helps.

## Framing

The camera follows the main group of players. Fast pairs running deep pull the camera ahead and can widen the crop by up to 15%; if both can't fit, the main group stays in frame. The output stays 2560×1440.

Long throws (about 20 m or more) that the disc detector picks up pan the camera to the landing point and hold there for two seconds after the catch. Short passes never move the camera. The disc scan is about a quarter of the processing time; turn off **Follow long throws** in the GUI, or pass `--no-disc-tracking`, to frame from players alone.

For occasional fixes, pass `--corrections` a CSV with a `source_time_seconds,center_x` header and one row per fix. Each fix blends into the automatic path over four seconds.

## Reusing work

Detection is the slow stage, and each stage saves its result, so you can replan or rerender without running it again:

```bash
# Detection only
uv run python tactical_pan.py game.mp4 --start 0 --end 60 --detect-only --output-dir prototype/detected

# Replan with a new mask, corrections, calibration, or framing settings
uv run python tactical_pan.py game.mp4 --start 0 --end 60 \
  --detections-in prototype/detected/detections.json \
  --play-area active-area.jpg --analysis-only --output-dir prototype/replanned

# Render a saved plan
uv run python tactical_pan.py game.mp4 --start 0 --end 60 \
  --plan-in prototype/replanned/camera-plan.json --output-dir prototype/rendered
```

Reuse needs the same source file and time range. The GUI reuses earlier work between attempts on a job automatically.

## Camera calibration

[`calibration/fixed-wide.json`](../calibration/fixed-wide.json) holds the thresholds for the original 4608×1728 camera. For another camera, copy it, keep only the fields you want to change, and pass it in:

```bash
uv run python tactical_pan.py game.mp4 --start 0 --end 60 --play-area active-area.jpg --calibration calibration/my-camera.json
uv run python gui.py --calibration calibration/my-camera.json
```

Distances are scaled to the source resolution, so a different resolution of the same view works without changes. A different perspective needs its own outline or sideline values. The source must be at least 2560×1440.

## Benchmarks

```bash
uv run python benchmark.py /path/to/vsfuse.mp4 --play-area benchmarks/vsfuse-field.png --label macbook-air
```

This processes five minutes from 10:00 (`--start`, `--duration`) in a fresh temporary folder, appends the result to `benchmarks/results.jsonl`, and prints every recorded run of that clip side by side. Use the same video and mask on each machine.

## Tests

```bash
uv run pytest
uv run ruff check .
```

The full GUI test runs real inference on a synthetic video when `yolo11s.pt` and FFmpeg are present, and is skipped otherwise.
