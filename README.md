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

Tracking is currently calibrated for the project's fixed wide-angle 4K source.
The normal tracking window and final deliverable are both 2560x1440. Temporary
widening includes more of the source at reduced scale, while the deliverable
stays 2560x1440. To define the field, draw one thick, closed bright-red outline around
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

Fast deep pairs are tracked automatically. The tracker tolerates brief missed
detections, begins the pan about half a second before a pair is fully confirmed,
and reserves space ahead of the leading runner. Deep coverage is bounded by the
credible formation players visible in the same sample, so an anticipatory pan
does not cut those players out; when both goals cannot fit, formation coverage
is preferred and the optional deep margin is reduced. Use `--deep-lookahead` to
tune the forward projection (the default is 0.6 seconds), or
`--no-deep-tracking` to compare against formation-only framing.
For an A/B run that favors deep-pair coverage when the two objectives compete,
use `--no-formation-protection`. A mask that includes sideline spectators can
cause formation protection to favor those spectators too; exclude them when
defining the active area.

When a deep pair approaches the crop edge, automatic widening gives it more
room without abandoning the protected formation. The source window expands
by at most 35%, remains inside the source, and preserves image proportions.
Very wide views use letterboxing. Offline planning starts expansion early,
holds the needed width for 1.5 seconds, and returns slowly to avoid repeated
zooming. Expansion is limited to 700 source pixels per second and contraction
to 220. These transitions can extend before and after a deep signal; sequences
without deep signals stay at the normal width.

**Allow gentle widening** is enabled by default in the processing screen.
Uncheck it, or use `--no-auto-widen`, to keep the normal crop width.
`pan-path.csv` and `analysis.json` include each sample's crop width, height,
and position, and the debug video shows the actual variable crop.

The tracker still cannot tell whether a deep cut was thrown to. The width cap
also means extremely spread-out action may not fit; in that case the protected
formation takes priority. Manual center corrections remain available.
