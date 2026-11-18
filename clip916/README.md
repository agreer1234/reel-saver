# clip916: copy an X link, get a TikTok-ready video

1. Start the app.
2. Copy a tweet/X post link (Share → Copy link).
3. It downloads and converts the clip **automatically**. You'll hear a ding when it's ready.
4. Hit **Post ↗**. It opens TikTok's upload page and shows the file in Finder/Explorer, so you can drag it in.

## Setup (once)
```
brew install ffmpeg                  # Windows: winget install ffmpeg
pip3 install -U -r requirements.txt  # yt-dlp + OpenCV (facecam detection)
```
Run `pip3 install -U yt-dlp` again right before a big drop. X changes things often.

## Run
- Mac: double-click `run.command` (first time: `chmod +x run.command`)
- Windows: double-click `run.bat`
- Or: `python3 clip916.py` → http://localhost:8916

## What Auto does
| Clip | Result |
|---|---|
| Stream with a **facecam overlay** | Facecam on top (snapped to the cam's border), gameplay below |
| **Talking head** / big face | Vertical crop that follows the face |
| Plain gameplay / trailer | Whole clip over a blurred background, with nothing cut off |
| Already vertical | Scaled to 1080×1920 |

Didn't like the layout? Each finished clip has **stack / blur / crop / bars** buttons to redo it instantly, with no re-download.

## Quality
- 1080×1920, x264 CRF 15 (near-lossless), source frame rate kept (60fps stays 60), original AAC audio copied untouched.
- yt-dlp grabs the highest resolution X offers. The final quality can't beat what X serves (often 720p or 1080p).
- **Turbo** (under More options) encodes much faster at slightly lower quality, for when speed matters more.

## Other options (More options)
Caption (white-box headline; on stacked clips it sits on the seam), start/end trim, gameplay zoom, and
**Browser cookies** if a post needs login (pick the browser you're logged into X with).

## CLI
```
python3 clip916.py https://x.com/.../status/123
python3 clip916.py clip.mp4 --mode stack --caption "GTA 6 IS HERE" --start 2 --end 15
```
