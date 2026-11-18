# clip916 — Twitter/X clips → 9:16 TikTok videos

Local app: paste X links (or drop video files), get 1080×1920 MP4s ready to upload to TikTok / Reels / Shorts.

## Setup (one time)
1. **ffmpeg**: macOS `brew install ffmpeg` · Windows `winget install ffmpeg`
2. **yt-dlp**: `pip3 install -U yt-dlp` (update it again right before a big drop; X changes things often)

## Run
- Mac: double-click `run.command` (first time, run `chmod +x run.command`)
- Windows: double-click `run.bat`
- Or: `python3 clip916.py`, which opens http://localhost:8916

Finished videos go to `clip916/output/`. Preview and download them right in the page.

## Options
| Option | What it does |
|---|---|
| **Blur fill** (default) | Full clip centered over a blurred, zoomed copy. Nothing gets cut off. |
| **Crop** | Fills the screen by cutting the sides. Best when the action is in the middle. |
| **Bars** | Black bars above and below. |
| **Zoom** | Makes the clip bigger in blur and bars modes (it crops the sides a little). 1.3–1.5 works well for 16:9 gameplay. |
| **Caption** | White-box headline on top or bottom, e.g. "GTA 6 IS OUT 🔥". |
| **Start/End** | Trim in seconds. |
| **Browser cookies** | If a tweet needs login or is age-restricted, pick the browser you're logged into X with. |
| **Turbo** | Faster encode, larger files. |

Tip: paste many links at once (one per line). Two convert in parallel.

## CLI
```
python3 clip916.py https://x.com/.../status/123 --mode crop --caption "GTA 6 IS HERE"
python3 clip916.py clip.mp4 --zoom 1.4 --start 2 --end 15
```
