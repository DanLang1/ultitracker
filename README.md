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
2. Draw an outline around the active playing area.
3. Approve the red/green mask preview.
4. Process, preview, and save the finished 2560x1440 video. Player detection
   runs first (about twenty minutes per hour of footage on an Apple GPU) and is
   cached, so processing the same job again with a new outline skips it.

Uploaded working copies and generated files are stored under `gui_jobs/`, which
is excluded from git. Processing continues locally while the server is running, even if the browser tab closes.
Reopen a job from **Saved jobs** or its bookmarked URL. Job status and settings
are stored in `gui_jobs/jobs.sqlite3`. One video processes at a time, with up to
eight jobs admitted (including the active job). After a server restart, queued
or running jobs are marked interrupted and can be retried; completed jobs remain
available. Each attempt has its own output directory, and retries automatically
reuse valid raw detections from previous attempts.
After saving an output, use **Delete current job's working files** to remove its
uploaded source, mask, logs, and generated videos.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and FFmpeg. `uv` installs the
project's Python interpreter and packages into a local `.venv` automatically.

```bash
uv sync
```

The first run downloads the YOLO model weights (`yolo11s.pt` by default) into
this directory. The small model (`--model yolo11n.pt`) is about twice as fast
but misses more of the smallest far-sideline players.
Detection runs on the Apple GPU (PyTorch MPS) when it is available, which is
several times faster than the CPU on Apple silicon. Pass `--device cpu` to the
CLI to force CPU inference; the GUI always uses automatic selection. The device
does not affect detection reuse.

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
smoothed crop. Without it, only the faster `tactical-crop.mp4` render is created;
`--output-name vs-dynasty` names it `vs-dynasty.mp4` instead. In the GUI, the
**Save as** field next to **Save tactical video** sets the downloaded file's name
(it defaults to the source video's name plus `-tactical`).
The crop decisions are also saved as `pan-path.csv` and `analysis.json`.

Player analysis samples the source at 5 Hz by default (`--sample-rate`). Every
smoothing window, confirmation delay, and tracking gate is defined in seconds,
so lower rates behave the same way with coarser timing. The final camera path is
resampled to the source frame rate and rounded over one sample interval so pan
speed changes smoothly instead of kinking at each analysis sample.

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

The CLI equivalent of the analysis step is `--detect-only`, which also writes
`suggested-outline.json` (normalized polygon points) and `sideline-heat.png`
next to `detections.json`. Recordings shorter than five minutes get a heat map
but no outline suggestion. The suggestion is behaviour-based: cells of the frame
where running happens in a large share of the recording's minutes form the
field, cells occupied for most of the recording are carved out, and grid cells
scale with the local body size so near and far depths are comparable. Its weak
spot is the near sideline where substitutes jog, so check that edge.

The outline is a coarse boundary, not a precision instrument. Inside it, the
planner separates players from bystanders on its own: each detection is
weighted by how far its track travels within a ten-second window (players run
several body lengths, sideline people shuffle), anyone standing in a patch of
the frame that is occupied for most of a rolling ten-minute window (a bench, a
tent line, a fixed object) is suppressed regardless of track continuity, and
squat boxes typical of seated people get a mild penalty. Suppressed people are
drawn orange in the debug video and never anchor the camera. An outline that
hugs the far sideline through a crowd therefore no longer drags the camera
toward that crowd, although keeping obvious spectator areas outside the outline
still helps.

## Lightweight manual guidance

For occasional framing fixes, copy `manual-corrections-template.csv`, add a few
rows containing `source_time_seconds,center_x`, and pass it with `--corrections`.
Each correction blends smoothly into the automatic path over four seconds.

## Disc flights

Long throws are detected automatically and drive the camera. The camera is
fixed, so a rolling median of the frames is a clean background; small moving
blobs outside the player boxes, inside the playing area or in the sky above it,
are linked frame to frame at the full frame rate. A trajectory qualifies as a
flight when it is long, smooth in speed and heading, physically plausible for a
disc of its apparent size, and either starts or ends near a player. Crowds, flags, trees and traffic are
rejected by a rolling clutter map of places that move all the time and by the
plausibility rules. Along the far sideline, where road traffic and walkers
move, a track must also rise or arch above the line joining its ends, as a
throw does even when it is seen nearly edge-on. Short passes are deliberately ignored: they never move the
camera.

For every flight longer than about twelve body heights (roughly 20 m), the camera
targets the landing point plus look room in the throw direction. The pan starts
at the release, not before it, so the throw itself is on screen. A flight first
found mid-air, such as a high pull coming down, had its release out of view, so
there the camera starts early to arrive with the disc. The camera holds for
two seconds after the catch. Trailing players may leave the frame during that
time. If the receivers are still running the throw's way when the hold ends,
the camera keeps their leading edge in frame instead of easing back.
The crop also widens, within the usual cap, to keep both ends of the throw
visible when it can. The debug video draws the flight and its landing point in
magenta. Use `--no-disc-tracking` to compare against formation-only framing.

Flights are written to `flights.json` (schema `flights/v1`) next to
`detections.json`. The analysis step that follows an upload computes them
using the suggested outline as its playing area; a processing run without a
saved file scans the video again, and `--flights-in` reuses one. Scanning runs
at roughly twice real time on this machine.

Manual disc anchors remain available: copy `disc-points-template.csv`, add
`source_time_seconds,disc_x,direction` rows, and pass it with `--disc-points`.
Direction is `1` for right and `-1` for left. The camera gives 15% of the crop
width as look room in that direction.

Fast deep pairs are tracked automatically. The tracker tolerates brief missed
detections, begins the pan about half a second before a pair is fully confirmed,
and reserves space ahead of the leading runner. Deep coverage is bounded by the
credible formation players visible in the same sample, so an anticipatory pan
does not cut those players out; when both goals cannot fit, formation coverage
is preferred and the optional deep margin is reduced. Use `--deep-lookahead` to
tune the forward projection (the default is 0.6 seconds), or
`--no-deep-tracking` to compare against formation-only framing.
When a cutter beats their defender and pulls clear of the pair, a new pair
forms behind them; the cutter stays tracked for as long as they keep sprinting
ahead of it, and the crop widens to leave room in front of them, which is where
a throw to them lands.
For an A/B run that favors deep-pair coverage when the two objectives compete,
use `--no-formation-protection`. A mask that includes sideline spectators can
cause formation protection to favor those spectators too; exclude them when
defining the active area.

When a deep pair approaches the crop edge, automatic widening gives it more
room without abandoning the protected formation. The source window expands
by at most 15%, remains inside the source in both dimensions, and preserves
image proportions without introducing letterboxing. Sources with no spare
height stay at the normal width. Offline planning starts expansion early,
holds the needed width for 1.5 seconds, and returns slowly to avoid repeated
zooming. Expansion is limited to 280 source pixels per second and contraction
to 220. These transitions can extend before and after a deep signal; sequences
without deep signals stay at the normal width.

**Allow gentle widening** is enabled by default in the processing screen.
Uncheck it, or use `--no-auto-widen`, to keep the normal crop width.
`pan-path.csv` and `analysis.json` include each sample's crop width, height,
and position, and the debug video shows the actual variable crop.

Without a detected flight the tracker cannot tell whether a deep cut was thrown
to; with one, the landing point takes over. The width cap also means extremely
spread-out action may not fit; in that case the protected formation takes
priority. Manual center corrections remain available.


## Reuse detection and camera plans

The pipeline has three independent stages: detection, camera planning, and
rendering. A normal run writes `detections.json` (raw person boxes before field
filtering), `camera-plan.json` (renderable geometry and tracking diagnostics),
and the existing `analysis.json` and `pan-path.csv` exports.

Run only the expensive detection stage:

```bash
uv run python tactical_pan.py game.mp4 --start 0 --end 60 \
  --detect-only --output-dir prototype/detected
```

Replan with a different field mask, manual correction, calibration, or framing
setting without running YOLO again:

```bash
uv run python tactical_pan.py game.mp4 --start 0 --end 60 \
  --detections-in prototype/detected/detections.json \
  --play-area active-area.jpg --no-auto-widen \
  --analysis-only --output-dir prototype/replanned
```

Render that plan independently:

```bash
uv run python tactical_pan.py game.mp4 --start 0 --end 60 \
  --plan-in prototype/replanned/camera-plan.json \
  --output-dir prototype/rendered
```

Omit `--analysis-only` to replan and render together. Source identity (resolved
path, size, modification time), video metadata, and time range must match the
saved artifact. Detection reuse also checks sample rate, detector settings,
and local model-weight identity. Changing those inputs requires fresh detection.
These are local cache checks, not content hashes; moving a source invalidates
reuse. Artifacts have versioned schemas and are written atomically.

## Camera calibration

[`calibration/fixed-wide.json`](calibration/fixed-wide.json) records the original
4608×1728 camera assumptions. Copy it and override values for another camera:

```bash
uv run python tactical_pan.py game.mp4 --start 0 --end 60 \
  --play-area active-area.jpg --calibration calibration/my-camera.json
uv run python gui.py --calibration calibration/my-camera.json
```

A profile can contain only the fields you want to override. Distances and speeds
are expressed in reference-image pixels and pixels per second; the pipeline
scales them to the actual source dimensions. Body-size thresholds use the smaller
axis scale. Timing and body-relative speed rules remain independent of resolution.
The default profile preserves the original thresholds at its reference size.
Resizing a camera view is supported by this scaling; a different perspective
still needs an operator mask or a calibrated sideline profile. Crop and output
sizes remain explicit CLI options, with 2560×1440 defaults.

## Code structure and tests

- `panstudio/models.py`: immutable processing configuration and artifact contracts;
  typed dictionaries describe the mutable working records used by the algorithms.
- `panstudio/pipeline.py`: independently callable `detect`, `plan_camera`, and
  `render_plan` stages. Planning constructs fresh records and does not mutate detections.
- `panstudio/tracking.py`, `planning.py`, `masking.py`, `media.py`: algorithms and
  video I/O. YOLO loads only when fresh inference is requested.
- `panstudio/artifacts.py`, `events.py`: validated, versioned JSON artifacts and
  JSON-lines progress/error events, separate from human logs.
- `panstudio/jobs.py`, `worker.py`: SQLite persistence, lifecycle transitions,
  restart recovery, and a bounded local subprocess queue.
- `gui.py` and `panstudio/cli.py`: HTTP and command-line adapters. The web app uses
  an application factory; tests can supply an isolated job store and runner.
- `tactical_pan.py`: existing CLI entry point and compatibility exports for the
  original algorithm functions.

The local GUI server owns its jobs folder exclusively; a second server using the
same folder is rejected. Use a different `--jobs-root` to run a separate workspace.
Queued/running jobs cannot be deleted or have their field mask changed. Existing
prototype `job.json` files are imported on startup without treating old output
files as proof of successful completion.

```bash
uv run pytest
```

Tests include artifact reuse, calibration scaling, framing regressions, queue
limits, restart recovery, and real FFmpeg rendering. The complete GUI processing
test generates a short synthetic video and runs real inference when local
`yolo11s.pt` weights and FFmpeg are available; otherwise that test is skipped.
