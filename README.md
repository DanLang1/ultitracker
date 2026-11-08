# Auto Pan Frisbee Tracker

Vibe coded python program that runs over wide view footage of Ultimate Frisbee and does its best to follow the players automatically. Previously I was doing this all by hand in this repo: https://github.com/DanLang1/videopan. Currently this only is setup for my Reolink Duo V2 camera dimensions (4608 x 1728). I know Insta 360 is a pretty popular wide view setup as well, if you have footage like this, I'm sure you could use this as a base and have Claude/Codex adjust it to roughly work for that as well.

## How It Works
Tried a few already existing yolo models to try out disc tracking, but it doesn't really work with the low quality footage and the fact that players often obscure the disc when they are holding it.
Instead the tracker mostly looks at the groups of players on the field and does its best to follow them. Main weaknesses with this approach are some deep shots happen so quickly that 2 people going deep sometimes don't get picked up right away. To help mitigate this I tried several different things with varying success. There's some tracking for players running deep that will sometimes zoom out if it gets detected. There's also some disc flight detection that I've been messing around with on longer throws, since that's the ones most likely to be picked up. However I've had pretty mild results with tracking a disc. For now I consider the footage good enough and it saves me having to pan manually over the footage myself. There's probably other approaches or ways to do this, but given I mostly just want footage for tactical purposes it's good enough for now.

## Hardware
I've run this on my Macbook and Windows PC via WSL2 with a Nvidia 3080 (so really linux). Pretty sure this would crash if you tried running natively on Windows atm, so I'd recommend using WSL, otherwise maybe eventually I'll add support for Windows.

## Setup

Works on macOS and Linux. On Windows, run it inside [WSL2](https://learn.microsoft.com/windows/wsl/install) (GPU acceleration works there with the normal NVIDIA Windows driver).

1. Install [uv](https://docs.astral.sh/uv/) and FFmpeg (`brew install ffmpeg` on Mac, `sudo apt install ffmpeg` on Ubuntu/WSL).
2. Install dependencies from the project folder:

```bash
uv sync
```

The first time you process a video it downloads the YOLO model weights (~20 MB), so you'll need an internet connection.

## Local GUI

Runs both as a CLI or a GUI. To launch the GUI:

```bash
uv run python gui.py
```

1. Select the source game video.
2. Draw an outline around the active playing area. This is probably the most important part of getting good results. Sideline players are the things most likely to throw off the tracking, so make sure you give a little buffer for them so they don't get picked up.
3. Process / download. Optionally turn on **Generate debug review video** (shows what the tracker saw) or turn off **Follow long throws** to skip disc detection, which saves about a quarter of the processing time.

## More

Command-line usage, tuning options, camera calibration, and how the tracking works are in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## License

[AGPL-3.0](LICENSE), matching the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library and model weights it depends on.
