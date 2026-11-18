#!/usr/bin/env python3
"""clip916 - copy an X/Twitter link, get a max-quality 9:16 TikTok video.

Run:   python3 clip916.py            (opens the local web UI + watches your clipboard)
CLI:   python3 clip916.py URL_OR_FILE [...] [--mode auto|stack|blur|crop|bars]

Needs: ffmpeg/ffprobe on PATH, yt-dlp, and (for facecam detection) opencv-python-headless.
  pip3 install -U yt-dlp "opencv-python-headless<5"
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

W, H = 1080, 1920
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
CACHE_DIR = HERE / ".cache"
MODES = ("auto", "stack", "blur", "crop", "bars")
TWEET_RE = re.compile(r"https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/\w+/status/(\d+)[^\s]*", re.I)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def find_font():
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    if shutil.which("fc-match"):
        try:
            p = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return None


def ytdlp_cmd():
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        exe = shutil.which("yt-dlp")
        return [exe] if exe else None


def have_cv2():
    try:
        import cv2
        return hasattr(cv2, "CascadeClassifier")
    except ImportError:
        return False


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True).stdout
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = {}
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    try:
        num, den = (v.get("r_frame_rate") or "30/1").split("/")
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "w": int(v.get("width") or 0), "h": int(v.get("height") or 0), "fps": fps,
        "has_audio": a is not None, "audio_codec": (a or {}).get("codec_name"),
    }


# ---------------------------------------------------------------- facecam detection

def detect_face(src, samples=14):
    """Find a face that stays in the same spot across the clip (a facecam overlay or a
    talking head). Returns (x, y, w, h) in source pixels, or None."""
    try:
        return _detect_face(src, samples)
    except Exception as e:  # noqa: BLE001 - detection must never break a convert
        print(f"! facecam detection skipped: {e}")
        return None


def _detect_face(src, samples):
    try:
        import cv2
    except ImportError:
        return None
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    cap = cv2.VideoCapture(str(src))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes, valid, grays, s = [], 0, [], 1.0
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (0.04 + 0.92 * i / (samples - 1))))
        ok, fr = cap.read()
        if not ok:
            continue
        valid += 1
        s = 960 / fr.shape[1] if fr.shape[1] > 960 else 1.0
        g = cv2.cvtColor(cv2.resize(fr, None, fx=s, fy=s) if s != 1 else fr, cv2.COLOR_BGR2GRAY)
        grays.append(g)
        g = cv2.equalizeHist(g)
        mn = max(20, int(g.shape[0] * 0.04))
        for (x, y, w, h) in cascade.detectMultiScale(g, 1.1, 6, minSize=(mn, mn)):
            boxes.append((i, x / s, y / s, w / s, h / s))
    cap.release()
    if valid < 3 or not boxes:
        return None

    def near(a, b):
        return (abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < 0.5 * a[3]
                and abs((a[2] + a[4] / 2) - (b[2] + b[4] / 2)) < 0.5 * a[4]
                and 0.7 < b[3] / a[3] < 1.43)

    best = []
    for a in boxes:
        members = {}
        for b in boxes:
            if near(a, b) and b[0] not in members:
                members[b[0]] = b
        if len(members) > len(best):
            best = list(members.values())
    if len(best) < max(3, round(valid * 0.4)):
        return None

    def med(k):
        vals = sorted(b[k] for b in best)
        return vals[len(vals) // 2]
    face = (med(1), med(2), med(3), med(4))
    sh, sw = grays[0].shape[0] / s, grays[0].shape[1] / s
    return {"face": face, "cam": find_cam_box(face, grays, s, sw, sh)}


def find_cam_box(face, grays, s, sw, sh):
    """Snap to the facecam overlay's rectangle. Its borders are straight lines that stay put
    while the gameplay around them changes, so they survive averaging edges over time."""
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    # fallback: a generous box around the face
    fb = [cx - 1.8 * fw, cy - 1.1 * fw, cx + 1.8 * fw, cy - 1.1 * fw + 2.4 * fw]
    try:
        import numpy as np
        G = np.stack(grays).astype(np.float32)
        dx = np.abs(np.diff(G, axis=2)).mean(0)        # [y, x] edge between x and x+1
        dy = np.abs(np.diff(G, axis=1)).mean(0)        # [y, x] edge between y and y+1
        hh, ww = G.shape[1:]
        f = [v * s for v in face]
        X, Y, FW, FH = f
        r0, r1 = int(max(0, Y - 0.4 * FH)), int(min(hh - 1, Y + 1.5 * FH))
        c0, c1 = int(max(0, X - 0.2 * FW)), int(min(ww - 1, X + 1.2 * FW))

        def pick(scores, idx):
            if len(scores) < 3:
                return None
            sc = np.array(scores)
            b = int(sc.argmax())
            if sc[b] >= 9 and sc[b] >= 2.5 * max(np.median(sc), 1.0):
                return idx[b]
            return None
        lx = list(range(int(max(0, X - 3 * FW)), int(X - 0.15 * FW)))
        rx = list(range(int(X + 1.15 * FW), int(min(ww - 1, X + 4 * FW))))
        ty = list(range(int(max(0, Y - 2.5 * FH)), int(Y - 0.15 * FH)))
        by = list(range(int(Y + 1.15 * FH), int(min(hh - 1, Y + 4.5 * FH))))
        L = pick([np.median(dx[r0:r1, x]) for x in lx], lx)
        R = pick([np.median(dx[r0:r1, x]) for x in rx], rx)
        T = pick([np.median(dy[y, c0:c1]) for y in ty], ty)
        B = pick([np.median(dy[y, c0:c1]) for y in by], by)
        pad = max(2, 0.004 * ww)                          # step inside any border line
        found = [None if L is None else (L + 1 + pad) / s, None if T is None else (T + 1 + pad) / s,
                 None if R is None else (R + 1 - pad) / s, None if B is None else (B + 1 - pad) / s]
        box = [found[i] if found[i] is not None else fb[i] for i in range(4)]
        # sides with no border: the cam must be touching the frame edge if the other side was found
        if found[0] is None and found[2] is not None and fx < sw / 2:
            box[0] = 0
        if found[2] is None and found[0] is not None and fx > sw / 2:
            box[2] = sw
        if found[1] is None and found[3] is not None and fy < sh / 2:
            box[1] = 0
        if found[3] is None and found[1] is not None and fy > sh / 2:
            box[3] = sh
    except Exception:  # noqa: BLE001
        box = fb
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(sw, box[2]), min(sh, box[3])
    w, h = x1 - x0, y1 - y0
    if w < fw * 1.2 or h < fh * 1.2 or not 0.5 < w / max(h, 1) < 3:
        x0, y0 = max(0, fb[0]), max(0, fb[1])
        w, h = min(sw, fb[2]) - x0, min(sh, fb[3]) - y0
    return x0, y0, w, h


def even(v):
    return max(2, int(v) // 2 * 2)


def plan_layout(mode, info, face):
    """Pick the final layout. Returns (layout, details) where layout is stack/facecrop/blur/crop/bars."""
    sw, sh = info["w"], info["h"]
    if mode == "auto":
        if not sw or sw / max(sh, 1) <= 0.7:     # already vertical
            return "blur", None
        if face:
            return ("stack" if face["face"][2] / sw < 0.09 else "facecrop"), face
        return "blur", None
    if mode == "stack":
        return ("stack", face) if face else ("blur", None)
    return mode, None


def build_filter(layout, info, face, zoom, caption, cap_pos, fps_cap):
    sw, sh = info["w"], info["h"]
    zoom = max(1.0, min(float(zoom or 1.0), 3.0))
    S = "flags=lanczos"
    if layout == "stack":
        fx, fy, fw, fh = face["face"]
        bx, by, bw, bh = face["cam"]
        cam_h = even(min(max(W * bh / bw, 540), 900))      # panel follows the cam's own shape
        game_h = H - cam_h
        target = W / cam_h
        cw, ch = bw, bh
        if cw / ch > target:
            cw = ch * target
        else:
            ch = cw / target
        cx = min(max(fx + fw / 2 - cw / 2, bx), bx + bw - cw)
        cy = min(max(fy + fh / 2 - ch * 0.45, by), by + bh - ch)
        gh = sh / zoom
        gw = gh * (W / game_h)
        if gw > sw:
            gw, gh = sw, sw / (W / game_h)
        gx, gy = (sw - gw) / 2, (sh - gh) / 2
        if by < gy + gh and by + bh > gy:                 # nudge gameplay away from the cam
            if bx + bw / 2 > sw / 2 and bx < gx + gw:
                gx -= min(gx + gw - bx, 0.2 * sw, gx)
            elif bx + bw / 2 <= sw / 2 and bx + bw > gx:
                gx += min(bx + bw - gx, 0.2 * sw, sw - gw - gx)
        v = (f"[0:v]split=2[c][g];"
             f"[c]crop={even(cw)}:{even(ch)}:{int(cx)}:{int(cy)},scale={W}:{cam_h}:{S}[top];"
             f"[g]crop={even(gw)}:{even(gh)}:{int(gx)}:{int(gy)},scale={W}:{game_h}:{S}[bot];"
             f"[top][bot]vstack,setsar=1[v0]")
    elif layout == "facecrop":
        fx, fy, fw, fh = face["face"]
        cw = min(sh * W / H, sw)
        cx = min(max(fx + fw / 2 - cw / 2, 0), sw - cw)
        v = f"[0:v]crop={even(cw)}:{even(sh)}:{int(cx)}:0,scale={W}:{H}:{S},setsar=1[v0]"
    elif layout == "crop":
        v = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase:{S},"
             f"crop={W}:{H},setsar=1[v0]")
    else:
        fw_ = even(W * zoom)
        fg = f"scale={fw_}:-2:{S},crop='min(iw,{W})':'min(ih,{H})'"
        fit = f"scale='min(iw,{W})':'min(ih,{H})':force_original_aspect_ratio=decrease:{S}"
        if layout == "bars":
            v = f"[0:v]{fg},{fit},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v0]"
        else:  # blur
            v = (f"[0:v]split=2[a][b];"
                 f"[a]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,"
                 f"crop={W // 4}:{H // 4},boxblur=20:3,eq=brightness=-0.12:saturation=1.2,"
                 f"scale={W}:{H}[bg];[b]{fg},{fit}[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    last = "v0"
    if fps_cap:
        v += f";[{last}]fps=60[vf]"
        last = "vf"
    if caption:
        if layout == "stack" and cap_pos == "top":
            y = f"{cam_h}-th/2"                     # sits on the seam between cam and gameplay
        else:
            y = {"top": "h*0.16-th/2", "bottom": "h*0.82-th/2"}[cap_pos]
        v += (f";[{last}]drawtext=fontfile=font.ttf:textfile=caption.txt:fontsize=68:"
              f"fontcolor=black:box=1:boxcolor=white:boxborderw=28:line_spacing=14:"
              f"x=(w-tw)/2:y={y}[v]")
    else:
        v += f";[{last}]null[v]"
    return v


# ---------------------------------------------------------------- jobs

JOBS = {}
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=2)
SETTINGS = {"mode": "auto", "zoom": 1.0, "cookies": "", "turbo": False, "watch": True}


def new_job(label, opts):
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "label": label, "status": "queued", "progress": 0, "msg": "",
           "outputs": [], "layout": "", "opts": opts, "sources": [], "created": time.time()}
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def upd(job, **kw):
    with JOBS_LOCK:
        job.update(kw)


def safe_name(s):
    return re.sub(r"[^\w\-]+", "_", s).strip("_")[:60] or "clip"


def download(url, workdir, opts, job):
    cmd = ytdlp_cmd()
    if not cmd:
        raise RuntimeError("yt-dlp not found - run: pip3 install -U yt-dlp")
    args = cmd + [
        "-f", "bv*+ba/b", "-S", "res,fps,br", "--merge-output-format", "mp4",
        "-o", str(Path(workdir) / "%(uploader_id)s_%(id)s_%(autonumber)s.%(ext)s"),
        "--no-progress", "--print", "after_move:filepath", "--no-simulate",
    ]
    if opts.get("cookies"):
        args += ["--cookies-from-browser", opts["cookies"]]
    args.append(url)
    upd(job, status="downloading", msg="Downloading best quality...")
    p = subprocess.run(args, capture_output=True, text=True)
    files = [ln.strip() for ln in p.stdout.splitlines() if ln.strip() and os.path.exists(ln.strip())]
    if not files:
        err = (p.stderr or p.stdout).strip().splitlines()
        hint = ""
        if any(k in (p.stderr or "") for k in ("login", "authenticat", "NSFW", "age", "No video")):
            hint = " (set Browser cookies to the browser you're logged into X with)"
        raise RuntimeError((err[-1] if err else "download failed") + hint)
    return files


def convert(src, dst, opts, job, idx, total):
    info = probe(src)
    start, end = float(opts.get("start") or 0), float(opts.get("end") or 0)
    clip_len = (end if end > 0 else info["duration"]) - start
    face = None
    if opts["mode"] in ("auto", "stack"):
        upd(job, msg="Looking for facecam...")
        face = detect_face(src)
    layout, face = plan_layout(opts["mode"], info, face)
    workdir = Path(tempfile.mkdtemp(prefix="c916_"))
    try:
        caption = (opts.get("caption") or "").strip()
        if caption:
            font = find_font()
            if font:
                shutil.copy(font, workdir / "font.ttf")
                (workdir / "caption.txt").write_text(
                    "\n".join(textwrap.wrap(caption, width=22)[:4]), encoding="utf-8")
            else:
                caption = ""
        graph = build_filter(layout, info, face, opts.get("zoom", 1.0), caption,
                             opts.get("cap_pos", "top"), info["fps"] > 61)
        (workdir / "graph.txt").write_text(graph, encoding="utf-8")
        cmd = ["ffmpeg", "-hide_banner", "-y", "-nostats", "-progress", "pipe:1"]
        if start > 0:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(Path(src).resolve())]
        if end > 0:
            cmd += ["-t", str(max(0.1, end - start))]
        cmd += ["-filter_complex_script", "graph.txt", "-map", "[v]"]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0"]
            if info["audio_codec"] == "aac" and not start and not end:
                cmd += ["-c:a", "copy"]                       # lossless audio passthrough
            else:
                cmd += ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"]
        if opts.get("turbo"):
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "15", "-tune", "film"]
        cmd += ["-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-movflags", "+faststart", str(Path(dst).resolve())]
        names = {"stack": "facecam stack", "facecrop": "face-tracked crop", "blur": "blur fill",
                 "crop": "center crop", "bars": "bars"}
        upd(job, status="converting", layout=names[layout],
            msg=f"Encoding ({names[layout]}) {idx + 1}/{total}")
        p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        err_tail = []

        def drain():
            for line in p.stderr:
                err_tail.append(line)
                del err_tail[:-15]
        t = threading.Thread(target=drain, daemon=True)
        t.start()
        for line in p.stdout:
            if line.startswith("out_time_us=") and clip_len > 0:
                try:
                    frac = min(int(line.split("=")[1]) / 1e6 / clip_len, 1.0)
                    upd(job, progress=round(((idx + frac) / total) * 100))
                except ValueError:
                    pass
        p.wait()
        t.join(timeout=2)
        if p.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + "".join(err_tail[-4:]).strip())
        return names[layout]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_job(job, source=None, is_file=False, sources=None):
    opts = job["opts"]
    cache = CACHE_DIR / job["id"]
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if sources:
            files = sources
        elif is_file:
            files = [shutil.move(source, cache / Path(source).name)]
        else:
            files = download(source, cache, opts, job)
        upd(job, sources=[str(f) for f in files])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        outs, layouts = [], []
        for i, f in enumerate(files):
            base = safe_name(Path(f).stem)
            dst = OUT_DIR / f"{base}_916.mp4"
            n = 2
            while dst.exists():
                dst = OUT_DIR / f"{base}_916_{n}.mp4"
                n += 1
            layouts.append(convert(f, dst, opts, job, i, len(files)))
            outs.append(dst.name)
        upd(job, status="done", progress=100, msg="Ready to post", outputs=outs,
            layout=", ".join(dict.fromkeys(layouts)))
    except Exception as e:  # noqa: BLE001
        upd(job, status="error", msg=str(e)[:400])
    finally:
        if is_file and source and Path(source).parent.name.startswith("c916up_"):
            shutil.rmtree(Path(source).parent, ignore_errors=True)


def clean_opts(d, base=None):
    d = {**(base or {}), **{k: v for k, v in d.items() if v not in (None,)}}
    return {
        "mode": d.get("mode") if d.get("mode") in MODES else "auto",
        "zoom": float(d.get("zoom") or 1.0),
        "caption": str(d.get("caption") or "")[:200],
        "cap_pos": "bottom" if d.get("cap_pos") == "bottom" else "top",
        "start": float(d.get("start") or 0),
        "end": float(d.get("end") or 0),
        "cookies": d.get("cookies") if d.get("cookies") in
        ("chrome", "firefox", "safari", "edge", "brave", "opera", "vivaldi", "chromium") else "",
        "turbo": bool(d.get("turbo")),
    }


def submit_url(url, opts):
    job = new_job(url, opts)
    POOL.submit(run_job, job, url)
    return job


def prune_cache():
    if not CACHE_DIR.exists():
        return
    for d in CACHE_DIR.iterdir():
        try:
            if time.time() - d.stat().st_mtime > 2 * 86400:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------- clipboard watcher

def read_clipboard():
    try:
        if sys.platform == "darwin":
            cmd = ["pbpaste"]
        elif os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command", "Get-Clipboard"]
        elif shutil.which("wl-paste"):
            cmd = ["wl-paste", "-n"]
        elif shutil.which("xclip"):
            cmd = ["xclip", "-o", "-selection", "clipboard"]
        else:
            return None
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3, **kw).stdout
    except Exception:
        return None


def clipboard_watcher():
    seen = set()
    first = read_clipboard()
    if first is None:
        print("! clipboard not readable on this system - paste links into the page instead")
        return
    seen.update(m.group(1) for m in TWEET_RE.finditer(first))   # ignore what's already copied
    while True:
        time.sleep(1.0 if os.name != "nt" else 1.5)
        if not SETTINGS["watch"]:
            continue
        txt = read_clipboard() or ""
        for m in TWEET_RE.finditer(txt[:5000]):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                url = m.group(0).split("?")[0]
                print(f"clipboard -> {url}")
                submit_url(url, clean_opts({}, SETTINGS))


# ---------------------------------------------------------------- web UI

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>clip916</title>
<style>
:root{--bg:#0b0b10;--card:#15151d;--line:#262633;--txt:#ececf3;--mut:#8b8ba0;--acc:#ff2d6f;--acc2:#25f4ee}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:24px 16px}
h1{margin:0;font-size:28px;letter-spacing:-.5px}h1 b{color:var(--acc)}h1 i{color:var(--acc2);font-style:normal}
.head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px}
.sub{color:var(--mut);margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
textarea,input,select{width:100%;background:#0e0e15;color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit}
textarea{min-height:56px;resize:vertical}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}
label{font-size:12px;color:var(--mut);display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.seg{display:flex;gap:6px;flex-wrap:wrap}.seg button{flex:1}
button{background:#22222e;color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:9px 12px;font:inherit;cursor:pointer}
button.on{border-color:var(--acc);background:#2a1420}
.go{background:var(--acc);border:0;font-weight:700;padding:10px 18px}
.watch{display:flex;align-items:center;gap:10px;padding:8px 14px;border-radius:999px;border:1px solid var(--line);background:var(--card);cursor:pointer;user-select:none}
.dot{width:10px;height:10px;border-radius:50%;background:#555}.watch.on .dot{background:#2ee66b;box-shadow:0 0 10px #2ee66b;animation:p 1.6s infinite}
@keyframes p{50%{opacity:.4}}
.drop{border:2px dashed var(--line);border-radius:12px;padding:12px;text-align:center;color:var(--mut);margin-top:10px;cursor:pointer;font-size:14px}
.drop.hot{border-color:var(--acc2);color:var(--acc2)}
details summary{cursor:pointer;color:var(--mut);margin-top:12px}
.jobs{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.job{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px;font-size:13px}
.job.fresh{border-color:var(--acc2)}
.job video{width:100%;aspect-ratio:9/16;background:#000;border-radius:8px;display:block;margin-top:6px}
.bar{height:6px;background:#0e0e15;border-radius:3px;overflow:hidden;margin:8px 0}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .3s}
.lbl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--mut)}
.err{color:#ff6b6b;word-break:break-word}.ok{color:var(--acc2)}
.acts{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}.acts button,.acts a{flex:1;font-size:12px;padding:7px 6px;text-align:center;text-decoration:none;color:var(--txt);background:#22222e;border:1px solid var(--line);border-radius:8px}
.acts a.post{background:var(--acc);border:0;font-weight:700}
.redo{display:flex;gap:4px;margin-top:6px}.redo button{flex:1;font-size:11px;padding:5px 2px;color:var(--mut)}
.tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:999px;background:#1f2b2b;color:var(--acc2);margin-top:4px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:8px 0 12px}
.chk{display:flex;align-items:center;gap:8px;margin-top:22px}.chk input{width:auto}
.warn{background:#2b2210;border:1px solid #5c4511;color:#f3c969;padding:10px 12px;border-radius:10px;margin-bottom:12px;font-size:14px}
</style></head><body><div class="wrap">
<div class="head"><h1>clip<b>9</b>:<i>16</i></h1>
<div class="watch" id="watch"><span class="dot"></span><span id="wtxt">Watching clipboard</span></div></div>
<div class="sub">Copy an X post link anywhere and it converts automatically. Facecam clips get stacked (cam on top, gameplay below).</div>
<div id="warns"></div>
<div class="card">
 <div style="display:flex;gap:8px"><textarea id="urls" placeholder="…or paste links here (one per line)"></textarea><button class="go" id="go">Convert</button></div>
 <div class="drop" id="drop">drop / pick video files<input type="file" id="file" accept="video/*" multiple hidden></div>
 <div class="row">
  <div style="grid-column:span 2"><label>Layout</label><div class="seg" id="modes">
   <button data-m="auto">Auto</button><button data-m="stack">Facecam stack</button><button data-m="blur">Blur fill</button><button data-m="crop">Crop</button><button data-m="bars">Bars</button></div></div>
 </div>
 <details><summary>More options (caption, trim, zoom, cookies)</summary>
 <div class="row">
  <div style="grid-column:span 2"><label>Caption for next convert</label><input id="caption" placeholder="GTA 6 IS FINALLY OUT 🔥"></div>
  <div><label>Caption position</label><select id="cap_pos"><option value="top">Top / seam</option><option value="bottom">Bottom</option></select></div>
 </div>
 <div class="row">
  <div><label>Start (sec)</label><input id="start" type="number" min="0" step="0.1" placeholder="0"></div>
  <div><label>End (sec)</label><input id="end" type="number" min="0" step="0.1" placeholder="full"></div>
  <div><label>Gameplay zoom <span id="zv">1.00×</span></label><input type="range" id="zoom" min="1" max="2" step="0.05" value="1"></div>
  <div><label>Browser cookies</label><select id="cookies"><option value="">none</option><option>chrome</option><option>safari</option><option>firefox</option><option>edge</option><option>brave</option></select></div>
  <div class="chk"><input type="checkbox" id="turbo"><span>Turbo (faster, lower quality)</span></div>
 </div></details>
</div>
<div class="top"><b>Clips</b><span><button id="clear">Clear finished</button> <button id="open">Open folder</button></span></div>
<div class="jobs" id="jobs"></div>
</div>
<script>
const $=id=>document.getElementById(id);let S={};
async function loadSettings(){S=await (await fetch('/api/settings')).json();
 $('zoom').value=S.zoom;$('cookies').value=S.cookies;$('turbo').checked=S.turbo;paint();
 $('warns').innerHTML=(S.warnings||[]).map(w=>`<div class="warn">${esc(w)}</div>`).join('')}
function paint(){document.querySelectorAll('#modes button').forEach(b=>b.classList.toggle('on',b.dataset.m===S.mode));
 $('zv').textContent=(+$('zoom').value).toFixed(2)+'×';$('watch').classList.toggle('on',S.watch);
 $('wtxt').textContent=S.watch?(S.clip_ok?'Watching clipboard':'Clipboard unavailable'):'Clipboard paused'}
async function save(p){S={...S,...p};paint();await fetch('/api/settings',{method:'POST',body:JSON.stringify(p)})}
$('modes').onclick=e=>{if(e.target.dataset.m)save({mode:e.target.dataset.m})};
$('zoom').oninput=()=>save({zoom:+$('zoom').value});$('cookies').onchange=()=>save({cookies:$('cookies').value});
$('turbo').onchange=()=>save({turbo:$('turbo').checked});$('watch').onclick=()=>save({watch:!S.watch});
function once(){const o={caption:$('caption').value,cap_pos:$('cap_pos').value,start:$('start').value,end:$('end').value};
 $('caption').value='';$('start').value='';$('end').value='';return o}
async function sendUrls(urls){if(!urls.length)return;await fetch('/api/jobs',{method:'POST',body:JSON.stringify({urls,opts:once()})});poll()}
$('go').onclick=()=>{sendUrls($('urls').value.split(/\s+/).filter(u=>/^https?:\/\//.test(u)));$('urls').value=''};
document.addEventListener('paste',e=>{const t=e.clipboardData.getData('text');if(document.activeElement===$('urls')||document.activeElement===$('caption'))return;
 const urls=t.split(/\s+/).filter(u=>/^https?:\/\//.test(u));if(urls.length&&!S.watch){e.preventDefault();sendUrls(urls)}});
async function upload(files){const o=once();for(const f of files){await fetch('/api/upload?name='+encodeURIComponent(f.name)+'&opts='+encodeURIComponent(JSON.stringify(o)),{method:'POST',body:f})}poll()}
const d=$('drop');d.onclick=()=>$('file').click();$('file').onchange=e=>upload(e.target.files);
d.ondragover=e=>{e.preventDefault();d.classList.add('hot')};d.ondragleave=()=>d.classList.remove('hot');
d.ondrop=e=>{e.preventDefault();d.classList.remove('hot');upload(e.dataTransfer.files)};
$('open').onclick=()=>fetch('/api/reveal',{method:'POST',body:'{}'});
$('clear').onclick=async()=>{await fetch('/api/clear',{method:'POST'});poll()};
const seen={};let timer=null,first=true;
function ding(){try{const a=new AudioContext(),o=a.createOscillator(),g=a.createGain();o.connect(g);g.connect(a.destination);o.frequency.value=880;g.gain.setValueAtTime(.15,a.currentTime);g.gain.exponentialRampToValueAtTime(.001,a.currentTime+.4);o.start();o.stop(a.currentTime+.4)}catch(e){}}
async function poll(){clearTimeout(timer);let jobs=[];try{jobs=await (await fetch('/api/jobs')).json()}catch(e){}
 const box=$('jobs');const ids=new Set(jobs.map(j=>j.id));
 for(const id in seen){if(!ids.has(id)){seen[id].el.remove();delete seen[id]}}
 for(const j of jobs){let s=seen[j.id];if(!s){const el=document.createElement('div');el.className='job';box.prepend(el);s=seen[j.id]={el,key:'',st:j.status}}
  const key=j.status+j.progress+j.msg+j.outputs.join();if(key===s.key)continue;
  if(!first&&j.status==='done'&&s.st!=='done'){ding();s.el.classList.add('fresh');document.title='✅ clip ready';setTimeout(()=>document.title='clip916',4000)}
  s.key=key;s.st=j.status;
  let h=`<div class="lbl" title="${esc(j.label)}">${esc(j.label)}</div>`;
  if(j.status==='done'){h+=`<span class="tag">${esc(j.layout)}</span>`+j.outputs.map(o=>`<video src="/out/${encodeURIComponent(o)}" controls muted loop playsinline preload="metadata"></video>
   <div class="acts"><a class="post" href="https://www.tiktok.com/tiktokstudio/upload" target="_blank" data-f="${esc(o)}">Post ↗</a><button data-reveal="${esc(o)}">Show file</button><a href="/out/${encodeURIComponent(o)}" download>⬇</a></div>`).join('')
   +`<div class="redo">${['stack','blur','crop','bars'].map(m=>`<button data-redo="${j.id}" data-m="${m}">${m}</button>`).join('')}</div>`}
  else if(j.status==='error'){h+=`<div class="err">✕ ${esc(j.msg)}</div>`}
  else{h+=`<div class="bar"><i style="width:${j.progress}%"></i></div><div class="ok">${esc(j.msg||j.status)} ${j.progress?j.progress+'%':''}</div>`}
  s.el.innerHTML=h}
 first=false;timer=setTimeout(poll,jobs.some(j=>j.status!=='done'&&j.status!=='error')?600:1500)}
$('jobs').onclick=e=>{const t=e.target;
 if(t.dataset.reveal){fetch('/api/reveal',{method:'POST',body:JSON.stringify({file:t.dataset.reveal})})}
 if(t.dataset.f){fetch('/api/reveal',{method:'POST',body:JSON.stringify({file:t.dataset.f})})}
 if(t.dataset.redo){fetch('/api/redo',{method:'POST',body:JSON.stringify({id:t.dataset.redo,mode:t.dataset.m})}).then(poll)}};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
loadSettings();poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _local_only(self):
        if (self.headers.get("Host") or "").split(":")[0] not in ("127.0.0.1", "localhost"):
            self.send_error(403)
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if not self._local_only():
            return
        u = urlparse(self.path)
        if u.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/jobs":
            with JOBS_LOCK:
                jobs = sorted(JOBS.values(), key=lambda j: j["created"])
                self._json([{k: v for k, v in j.items() if k not in ("opts", "sources")} for j in jobs])
        elif u.path == "/api/settings":
            self._json({**SETTINGS, "clip_ok": CLIP_OK, "warnings": WARNINGS})
        elif u.path.startswith("/out/"):
            self._serve_file(OUT_DIR / Path(unquote(u.path[5:])).name)
        else:
            self.send_error(404)

    def _serve_file(self, p):
        if not p.is_file():
            return self.send_error(404)
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if (a and b) else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        if not self._local_only():
            return
        u = urlparse(self.path)
        if u.path == "/api/jobs":
            data = self._body()
            opts = clean_opts(data.get("opts") or {}, SETTINGS)
            ids = [submit_url(url, opts)["id"] for url in data.get("urls", [])[:50]]
            self._json({"ids": ids})
        elif u.path == "/api/upload":
            n = int(self.headers.get("Content-Length") or 0)
            q = parse_qs(u.query)
            name = Path(q.get("name", ["upload.mp4"])[0]).name
            opts = clean_opts(json.loads(q.get("opts", ["{}"])[0]), SETTINGS)
            d = Path(tempfile.mkdtemp(prefix="c916up_"))
            dst = d / name
            with open(dst, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            job = new_job(name, opts)
            POOL.submit(run_job, job, str(dst), True)
            self._json({"id": job["id"]})
        elif u.path == "/api/redo":
            data = self._body()
            with JOBS_LOCK:
                old = JOBS.get(data.get("id"))
            if not old or not old["sources"]:
                return self._json({"error": "nothing to redo"}, 400)
            opts = {**old["opts"], "mode": data.get("mode") if data.get("mode") in MODES else "auto"}
            job = new_job(f"{old['label']}  ({opts['mode']})", opts)
            POOL.submit(run_job, job, None, False, old["sources"])
            self._json({"id": job["id"]})
        elif u.path == "/api/settings":
            data = self._body()
            for k in ("mode", "zoom", "cookies", "turbo", "watch"):
                if k in data:
                    SETTINGS[k] = data[k]
            SETTINGS.update({k: v for k, v in clean_opts(SETTINGS).items()
                             if k in ("mode", "zoom", "cookies", "turbo")})
            SETTINGS["watch"] = bool(SETTINGS["watch"])
            self._json(SETTINGS)
        elif u.path == "/api/clear":
            with JOBS_LOCK:
                for k in [k for k, j in JOBS.items() if j["status"] in ("done", "error")]:
                    shutil.rmtree(CACHE_DIR / k, ignore_errors=True)
                    del JOBS[k]
            self._json({"ok": True})
        elif u.path == "/api/reveal":
            f = self._body().get("file")
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            reveal(OUT_DIR / Path(f).name if f else None)
            self._json({"ok": True})
        else:
            self.send_error(404)


def reveal(p):
    """Open the output folder, with the file selected when possible (for drag-to-upload)."""
    try:
        if p and p.exists():
            if sys.platform == "darwin":
                return subprocess.Popen(["open", "-R", str(p)])
            if os.name == "nt":
                return subprocess.Popen(["explorer", "/select,", str(p)])
        target = str(OUT_DIR)
        if sys.platform == "darwin":
            subprocess.Popen(["open", target])
        elif os.name == "nt":
            os.startfile(target)  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception:
        pass


CLIP_OK = False
WARNINGS = []


def check_deps():
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Install ffmpeg (mac: brew install ffmpeg, "
                 f"win: winget install ffmpeg) and try again.")
    if not ytdlp_cmd():
        WARNINGS.append("yt-dlp isn't installed, so links won't download. Run: pip3 install -U yt-dlp")
    if not have_cv2():
        WARNINGS.append("Facecam detection is off. Run: pip3 install \"opencv-python-headless<5\" "
                        "(until then Auto uses blur fill)")
    for w in WARNINGS:
        print("! " + w)


def main():
    global OUT_DIR, CLIP_OK
    ap = argparse.ArgumentParser(description="Convert Twitter/X clips to 9:16 TikTok videos")
    ap.add_argument("inputs", nargs="*", help="URLs or video files (CLI mode). Omit for the web UI.")
    ap.add_argument("--mode", default="auto", choices=MODES)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--caption", default="")
    ap.add_argument("--cap-pos", default="top", choices=["top", "bottom"])
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=0)
    ap.add_argument("--cookies", default="", help="browser to borrow X login cookies from")
    ap.add_argument("--turbo", action="store_true", help="faster encode, lower quality")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--port", type=int, default=8916)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-watch", action="store_true", help="don't auto-convert copied links")
    a = ap.parse_args()
    OUT_DIR = Path(a.out).expanduser().resolve()
    check_deps()
    prune_cache()

    if a.inputs:
        opts = clean_opts({**vars(a), "cap_pos": a.cap_pos})
        for src in a.inputs:
            is_file = os.path.exists(src)
            job = new_job(src, opts)
            print(f"-> {src}")
            if is_file:      # don't move the user's own file into the cache
                run_job(job, None, False, [src])
            else:
                run_job(job, src)
            if job["status"] == "done":
                for o in job["outputs"]:
                    print(f"   ok  [{job['layout']}] {OUT_DIR / o}")
            else:
                print(f"   ERR {job['msg']}")
        return

    SETTINGS.update({"mode": a.mode, "zoom": a.zoom, "cookies": a.cookies, "turbo": a.turbo,
                     "watch": not a.no_watch})
    CLIP_OK = read_clipboard() is not None
    threading.Thread(target=clipboard_watcher, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"clip916 running at {url}  (output -> {OUT_DIR})")
    print("Copy any x.com/.../status/... link and it converts automatically. Ctrl+C to quit.")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
