#!/usr/bin/env python3
"""clip916 - turn Twitter/X clips into 9:16 TikTok-ready MP4s, fast.

Run:   python3 clip916.py            (opens the local web UI)
CLI:   python3 clip916.py URL_OR_FILE [...] [--mode blur|crop|bars] [--caption "TEXT"]

Needs: ffmpeg + ffprobe on PATH, and yt-dlp (pip install -U yt-dlp).
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

FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    # Windows
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    # Linux
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


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
         "-of", "json", str(path)], capture_output=True, text=True).stdout
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"duration": 0, "has_audio": False}
    streams = data.get("streams", [])
    return {
        "duration": float(data.get("format", {}).get("duration") or 0),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


# ---------------------------------------------------------------- jobs

JOBS = {}           # id -> dict
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=2)


def new_job(label, opts):
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "label": label, "status": "queued", "progress": 0,
           "msg": "", "outputs": [], "opts": opts, "created": time.time()}
    with JOBS_LOCK:
        JOBS[jid] = job
    return job


def upd(job, **kw):
    with JOBS_LOCK:
        job.update(kw)


def safe_name(s):
    s = re.sub(r"[^\w\-]+", "_", s).strip("_")
    return s[:60] or "clip"


def download(url, workdir, opts, job):
    cmd = ytdlp_cmd()
    if not cmd:
        raise RuntimeError("yt-dlp not found - run: pip install -U yt-dlp")
    args = cmd + [
        "-f", "bv*+ba/b", "--merge-output-format", "mp4",
        "-o", str(Path(workdir) / "%(uploader_id)s_%(id)s_%(autonumber)s.%(ext)s"),
        "--no-progress", "--print", "after_move:filepath", "--no-simulate",
    ]
    if opts.get("cookies"):
        args += ["--cookies-from-browser", opts["cookies"]]
    args.append(url)
    upd(job, status="downloading", msg="Downloading...")
    p = subprocess.run(args, capture_output=True, text=True)
    files = [l.strip() for l in p.stdout.splitlines() if l.strip() and os.path.exists(l.strip())]
    if not files:
        err = (p.stderr or p.stdout).strip().splitlines()
        hint = ""
        if any(k in (p.stderr or "") for k in ("login", "authenticat", "NSFW", "age", "No video")):
            hint = " (try setting Browser cookies to the browser you're logged into X with)"
        raise RuntimeError((err[-1] if err else "download failed") + hint)
    return files


def build_filter(mode, zoom, caption_file, has_font, cap_pos):
    zoom = max(1.0, min(float(zoom or 1.0), 3.0))
    fw = int(W * zoom) // 2 * 2
    fg = (f"scale={fw}:-2:flags=lanczos,"
          f"crop='min(iw,{W})':'min(ih,{H})'")
    if mode == "crop":
        v = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
             f"crop={W}:{H},setsar=1[v0]")
    elif mode == "bars":
        v = (f"[0:v]{fg},scale='min(iw,{W})':'min(ih,{H})':force_original_aspect_ratio=decrease,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v0]")
    else:  # blur
        v = (f"[0:v]split=2[a][b];"
             f"[a]scale={W//4}:{H//4}:force_original_aspect_ratio=increase,crop={W//4}:{H//4},"
             f"boxblur=20:3,eq=brightness=-0.12:saturation=1.2,scale={W}:{H}[bg];"
             f"[b]{fg},scale='min(iw,{W})':'min(ih,{H})':force_original_aspect_ratio=decrease[fg];"
             f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]")
    if caption_file and has_font:
        y = {"top": "h*0.16-th/2", "bottom": "h*0.82-th/2"}.get(cap_pos, "h*0.16-th/2")
        v += (f";[v0]drawtext=fontfile=font.ttf:textfile={caption_file}:fontsize=68:"
              f"fontcolor=black:box=1:boxcolor=white:boxborderw=28:line_spacing=14:"
              f"x=(w-tw)/2:y={y}[v]")
    else:
        v += ";[v0]null[v]"
    return v


def convert(src, dst, opts, job, idx, total):
    info = probe(src)
    start = float(opts.get("start") or 0)
    end = float(opts.get("end") or 0)
    dur = info["duration"]
    clip_len = (end if end > 0 else dur) - start
    workdir = Path(tempfile.mkdtemp(prefix="c916_"))
    try:
        caption = (opts.get("caption") or "").strip()
        has_font = False
        if caption:
            font = find_font()
            if font:
                shutil.copy(font, workdir / "font.ttf")
                has_font = True
            wrapped = "\n".join(textwrap.wrap(caption, width=22)[:4])
            (workdir / "caption.txt").write_text(wrapped, encoding="utf-8")
        fg = build_filter(opts.get("mode", "blur"), opts.get("zoom", 1.0),
                          "caption.txt" if caption else None, has_font, opts.get("cap_pos", "top"))
        (workdir / "graph.txt").write_text(fg, encoding="utf-8")
        cmd = ["ffmpeg", "-hide_banner", "-y", "-nostats", "-progress", "pipe:1"]
        if start > 0:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(Path(src).resolve())]
        if end > 0:
            cmd += ["-t", str(max(0.1, end - start))]
        cmd += ["-filter_complex_script", "graph.txt", "-map", "[v]"]
        if info["has_audio"]:
            cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        cmd += ["-c:v", "libx264", "-preset", opts.get("preset", "veryfast"), "-crf", "19",
                "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "30",
                "-movflags", "+faststart", str(Path(dst).resolve())]
        upd(job, status="converting", msg=f"Converting {idx + 1}/{total}...")
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
                    secs = int(line.split("=")[1]) / 1e6
                    frac = min(secs / clip_len, 1.0)
                    upd(job, progress=round(((idx + frac) / total) * 100))
                except ValueError:
                    pass
        p.wait()
        t.join(timeout=2)
        if p.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + "".join(err_tail[-4:]).strip())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_job(job, source, is_file=False):
    opts = job["opts"]
    tmp = Path(tempfile.mkdtemp(prefix="c916dl_"))
    try:
        if is_file:
            files = [source]
        else:
            files = download(source, tmp, opts, job)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        outs = []
        for i, f in enumerate(files):
            base = safe_name(Path(f).stem)
            dst = OUT_DIR / f"{base}_916.mp4"
            n = 2
            while dst.exists():
                dst = OUT_DIR / f"{base}_916_{n}.mp4"
                n += 1
            convert(f, dst, opts, job, i, len(files))
            outs.append(dst.name)
        upd(job, status="done", progress=100, msg="Done", outputs=outs)
    except Exception as e:  # noqa: BLE001
        upd(job, status="error", msg=str(e)[:400])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if is_file and Path(source).parent.name.startswith("c916up_"):
            shutil.rmtree(Path(source).parent, ignore_errors=True)


def clean_opts(d):
    return {
        "mode": d.get("mode") if d.get("mode") in ("blur", "crop", "bars") else "blur",
        "zoom": float(d.get("zoom") or 1.0),
        "caption": str(d.get("caption") or "")[:200],
        "cap_pos": "bottom" if d.get("cap_pos") == "bottom" else "top",
        "start": float(d.get("start") or 0),
        "end": float(d.get("end") or 0),
        "cookies": d.get("cookies") if d.get("cookies") in
        ("chrome", "firefox", "safari", "edge", "brave", "opera", "vivaldi", "chromium") else "",
        "preset": "ultrafast" if d.get("turbo") else "veryfast",
    }


# ---------------------------------------------------------------- web UI

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>clip916</title>
<style>
:root{--bg:#0b0b10;--card:#15151d;--line:#262633;--txt:#ececf3;--mut:#8b8ba0;--acc:#ff2d6f;--acc2:#25f4ee}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:24px 16px}
h1{margin:0 0 4px;font-size:28px;letter-spacing:-.5px}h1 b{color:var(--acc)}h1 i{color:var(--acc2);font-style:normal}
.sub{color:var(--mut);margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
textarea,input,select{width:100%;background:#0e0e15;color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:10px;font:inherit}
textarea{min-height:90px;resize:vertical}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}
label{font-size:12px;color:var(--mut);display:block;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.modes{display:flex;gap:8px}.modes button{flex:1}
button{background:#22222e;color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font:inherit;cursor:pointer}
button.on{border-color:var(--acc);background:#2a1420}
.go{background:var(--acc);border:0;font-weight:700;width:100%;margin-top:14px;padding:14px;font-size:16px}
.go:hover{filter:brightness(1.1)}
.drop{border:2px dashed var(--line);border-radius:12px;padding:16px;text-align:center;color:var(--mut);margin-top:12px;cursor:pointer}
.drop.hot{border-color:var(--acc2);color:var(--acc2)}
.jobs{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.job{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px;font-size:13px}
.job video{width:100%;aspect-ratio:9/16;background:#000;border-radius:8px;display:block}
.bar{height:6px;background:#0e0e15;border-radius:3px;overflow:hidden;margin:8px 0}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--acc2));transition:width .3s}
.lbl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--mut)}
.err{color:#ff6b6b;word-break:break-word}.ok{color:var(--acc2)}
a.dl{display:inline-block;margin-top:6px;color:var(--acc2)}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:8px 0 12px}
.chk{display:flex;align-items:center;gap:8px;margin-top:22px}.chk input{width:auto}
</style></head><body><div class="wrap">
<h1>clip<b>9</b>:<i>16</i></h1>
<div class="sub">Paste X / Twitter links (one per line) or drop video files. Output: 1080×1920 MP4, TikTok-ready.</div>
<div class="card">
 <textarea id="urls" placeholder="https://x.com/RockstarGames/status/...&#10;https://twitter.com/.../status/..."></textarea>
 <div class="drop" id="drop">or drop / click to pick video files<input type="file" id="file" accept="video/*" multiple hidden></div>
 <div class="row">
  <div><label>Layout</label><div class="modes" id="modes">
   <button data-m="blur" class="on">Blur fill</button><button data-m="crop">Crop</button><button data-m="bars">Bars</button></div></div>
  <div><label>Zoom (blur/bars) <span id="zv">1.00×</span></label><input type="range" id="zoom" min="1" max="2" step="0.05" value="1"></div>
 </div>
 <div class="row">
  <div style="grid-column:span 2"><label>Caption (optional)</label><input id="caption" placeholder="GTA 6 IS FINALLY OUT 🔥"></div>
  <div><label>Caption position</label><select id="cap_pos"><option value="top">Top</option><option value="bottom">Bottom</option></select></div>
 </div>
 <div class="row">
  <div><label>Start (sec)</label><input id="start" type="number" min="0" step="0.1" placeholder="0"></div>
  <div><label>End (sec)</label><input id="end" type="number" min="0" step="0.1" placeholder="full"></div>
  <div><label>Browser cookies</label><select id="cookies"><option value="">none</option><option>chrome</option><option>safari</option><option>firefox</option><option>edge</option><option>brave</option></select></div>
  <div class="chk"><input type="checkbox" id="turbo"><span>Turbo encode</span></div>
 </div>
 <button class="go" id="go">Convert →</button>
</div>
<div class="top"><b>Queue</b><span><button id="clear">Clear finished</button> <button id="open">Open output folder</button></span></div>
<div class="jobs" id="jobs"></div>
</div>
<script>
let mode='blur';const $=id=>document.getElementById(id);
try{const s=JSON.parse(localStorage.getItem('c916')||'{}');if(s.mode)mode=s.mode;if(s.zoom)$('zoom').value=s.zoom;if(s.cookies)$('cookies').value=s.cookies;if(s.cap_pos)$('cap_pos').value=s.cap_pos;$('turbo').checked=!!s.turbo}catch(e){}
function paintModes(){document.querySelectorAll('#modes button').forEach(b=>b.classList.toggle('on',b.dataset.m===mode))}
paintModes();$('zv').textContent=(+$('zoom').value).toFixed(2)+'×';
$('modes').onclick=e=>{if(e.target.dataset.m){mode=e.target.dataset.m;paintModes();save()}};
$('zoom').oninput=()=>{$('zv').textContent=(+$('zoom').value).toFixed(2)+'×';save()};
['cookies','cap_pos','turbo'].forEach(i=>$(i).onchange=save);
function opts(){return{mode,zoom:$('zoom').value,caption:$('caption').value,cap_pos:$('cap_pos').value,start:$('start').value,end:$('end').value,cookies:$('cookies').value,turbo:$('turbo').checked}}
function save(){try{const o=opts();delete o.caption;delete o.start;delete o.end;localStorage.setItem('c916',JSON.stringify(o))}catch(e){}}
$('go').onclick=async()=>{const urls=$('urls').value.split(/\s+/).filter(u=>/^https?:\/\//.test(u));
 if(!urls.length){$('urls').focus();return}
 await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({urls,opts:opts()})});
 $('urls').value='';poll()};
async function upload(files){for(const f of files){await fetch('/api/upload?name='+encodeURIComponent(f.name)+'&opts='+encodeURIComponent(JSON.stringify(opts())),{method:'POST',body:f})}poll()}
const d=$('drop');d.onclick=()=>$('file').click();$('file').onchange=e=>upload(e.target.files);
d.ondragover=e=>{e.preventDefault();d.classList.add('hot')};d.ondragleave=()=>d.classList.remove('hot');
d.ondrop=e=>{e.preventDefault();d.classList.remove('hot');upload(e.dataTransfer.files)};
document.addEventListener('paste',e=>{if(document.activeElement!==$('urls')){const t=e.clipboardData.getData('text');if(/^https?:\/\//.test(t)){$('urls').value+=($('urls').value?'\n':'')+t.trim();}}});
$('open').onclick=()=>fetch('/api/open',{method:'POST'});
$('clear').onclick=async()=>{await fetch('/api/clear',{method:'POST'});poll()};
const seen={};
async function poll(){const jobs=await (await fetch('/api/jobs')).json();const box=$('jobs');
 const ids=new Set(jobs.map(j=>j.id));for(const id in seen){if(!ids.has(id)){seen[id].el.remove();delete seen[id]}}
 for(const j of jobs){let s=seen[j.id];if(!s){const el=document.createElement('div');el.className='job';box.prepend(el);s=seen[j.id]={el,key:''}}
  const key=j.status+j.progress+j.msg+j.outputs.join();if(key===s.key)continue;s.key=key;
  let h=`<div class="lbl" title="${esc(j.label)}">${esc(j.label)}</div>`;
  if(j.status==='done'){h+=j.outputs.map(o=>`<video src="/out/${encodeURIComponent(o)}" controls muted loop playsinline></video><a class="dl" href="/out/${encodeURIComponent(o)}" download>⬇ ${esc(o)}</a>`).join('')}
  else if(j.status==='error'){h+=`<div class="err">✕ ${esc(j.msg)}</div>`}
  else{h+=`<div class="bar"><i style="width:${j.progress}%"></i></div><div class="ok">${esc(j.msg||j.status)} ${j.progress?j.progress+'%':''}</div>`}
  s.el.innerHTML=h}
 if(jobs.some(j=>j.status!=='done'&&j.status!=='error'))setTimeout(poll,700)}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
poll();
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
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403)
            return False
        return True

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
                self._json([{k: v for k, v in j.items() if k != "opts"} for j in jobs])
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
            end = int(b) if (a and b) else size - 1
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
        n = int(self.headers.get("Content-Length") or 0)
        if u.path == "/api/jobs":
            data = json.loads(self.rfile.read(n) or b"{}")
            opts = clean_opts(data.get("opts") or {})
            ids = []
            for url in data.get("urls", [])[:50]:
                job = new_job(url, opts)
                POOL.submit(run_job, job, url)
                ids.append(job["id"])
            self._json({"ids": ids})
        elif u.path == "/api/upload":
            q = parse_qs(u.query)
            name = Path(q.get("name", ["upload.mp4"])[0]).name
            opts = clean_opts(json.loads(q.get("opts", ["{}"])[0]))
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
        elif u.path == "/api/clear":
            with JOBS_LOCK:
                for k in [k for k, j in JOBS.items() if j["status"] in ("done", "error")]:
                    del JOBS[k]
            self._json({"ok": True})
        elif u.path == "/api/open":
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            open_folder(OUT_DIR)
            self._json({"ok": True})
        else:
            self.send_error(404)


def open_folder(p):
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        elif os.name == "nt":
            os.startfile(str(p))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception:
        pass


def check_deps():
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}. Install ffmpeg (mac: brew install ffmpeg, "
                 f"win: winget install ffmpeg) and try again.")
    if not ytdlp_cmd():
        print("! yt-dlp not found - links won't work until you run: pip install -U yt-dlp "
              "(dropping local files still works)")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description="Convert Twitter/X clips to 9:16 TikTok videos")
    ap.add_argument("inputs", nargs="*", help="URLs or video files (CLI mode). Omit for the web UI.")
    ap.add_argument("--mode", default="blur", choices=["blur", "crop", "bars"])
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--caption", default="")
    ap.add_argument("--cap-pos", default="top", choices=["top", "bottom"])
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=0)
    ap.add_argument("--cookies", default="", help="browser to borrow X login cookies from (chrome, safari, ...)")
    ap.add_argument("--turbo", action="store_true", help="faster encode, slightly bigger files")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--port", type=int, default=8916)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    OUT_DIR = Path(a.out).expanduser().resolve()
    check_deps()

    if a.inputs:
        opts = clean_opts({**vars(a), "cap_pos": a.cap_pos})
        for src in a.inputs:
            is_file = os.path.exists(src)
            job = new_job(src, opts)
            print(f"-> {src}")
            run_job(job, src, is_file)
            if job["status"] == "done":
                for o in job["outputs"]:
                    print(f"   ok  {OUT_DIR / o}")
            else:
                print(f"   ERR {job['msg']}")
        return

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"clip916 running at {url}  (output -> {OUT_DIR})  Ctrl+C to quit")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
