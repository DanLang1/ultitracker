# Tactical auto-pan prototype

This prototype turns fixed, ultra-wide ultimate footage into a smooth 2560x1440
"virtual camera" view. It detects people, filters detections to the foreground
field, estimates the tactical center of the player group, smooths that path, and
renders the final crop. A full-width review overlay is available as an option.

## Local GUI

Launch the guided workflow with:

```bash
uv run python gui.py
```

The browser opens Tactical Pan Studio on this computer. The four steps are:

1. Select the source game video.
2. Draw one closed red outline around the active playing area.
3. Approve the red/green mask preview.
4. Process, preview, and save the finished 2560x1440 video.

Uploaded working copies and generated files are stored under `gui_jobs/`, which
is excluded from git. Processing continues locally while the page remains open.
After saving an output, use **Delete current job's working files** to remove its
uploaded source, mask, logs, and generated videos.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and FFmpeg. `uv` installs the
project's Python interpreter and packages into a local `.venv` automatically.

```bash
uv sync
```

The first run downloads the small YOLO model weights into this directory.

## Run one point

```bash
uv run python tactical_pan.py vssdynasty.mp4 \
  --start 21.8 \
  --end 74.2 \
  --play-area active-area.jpg \
  --debug \
  --output-dir prototype/point-2
```

When `--debug` is supplied, `debug-wide.mp4` shows retained detections and the
smoothed crop. Without it, only the faster `tactical-crop.mp4` render is created.
The crop decisions are also saved as `pan-path.csv` and `analysis.json`.

The default tracking window and final deliverable are both 2560x1440, retaining
the full detail of the tactical crop. To define the field, draw one thick, closed bright-red outline around
the valid playing area on any uncropped frame and pass it with `--play-area`. The
generated `play-area-preview.jpg` tints excluded areas red so the mask can be
checked before relying on it.

To inspect a mask without rerunning detection or video rendering:

```bash
uv run python tactical_pan.py vssdynasty.mp4 \
  --start 21.8 --end 74.2 \
  --play-area active-area.jpg \
  --preview-mask-only \
  --output-dir prototype/mask-check
```

The red outline is the exact tracking boundary. Redraw the outline when an edge
needs adjustment.

## Lightweight manual guidance

For occasional framing fixes, copy `manual-corrections-template.csv`, add a few
rows containing `source_time_seconds,center_x`, and pass it with `--corrections`.
Each correction blends smoothly into the automatic path over four seconds.

Automatic disc scanning is currently disabled. A few known disc positions can
still provide a weak, safe signal: copy
`disc-points-template.csv`, add `source_time_seconds,disc_x,direction` rows, and
pass it with `--disc-points`. Direction is `1` for right and `-1` for left. The
camera gives 15% of the crop width as look room in that direction, while the
player formation remains the dominant signal.
