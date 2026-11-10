# download a slice of a youtube video's audio and work out its start time
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .timeutil import parse_time


def ytdlp():
    if os.environ.get("YTDLP"):
        return [os.environ["YTDLP"]]
    if importlib.util.find_spec("yt_dlp"):
        return [sys.executable, "-m", "yt_dlp"]
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    raise RuntimeError("yt-dlp not found: pip install yt-dlp, or set YTDLP=/path/to/yt-dlp")


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


# what to try when yt-dlp fails; only the newest yt-dlp keeps up with youtube
def _hint():
    try:
        v = subprocess.run(ytdlp() + ["--version"], capture_output=True, text=True).stdout.strip()
    except Exception:
        v = "?"
    py = sys.version_info
    if ytdlp()[0] == sys.executable and py < (3, 10):
        return (f"yt-dlp {v} on Python {py.major}.{py.minor}: yt-dlp no longer updates on Python < 3.10, "
                "so run the app from a 3.10+ env with a current yt-dlp, or set YTDLP=/path/to/yt-dlp")
    return f"yt-dlp {v} is probably out of date: pip install -U yt-dlp"


def stream(cmd, log=print, on_proc=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if on_proc:
        on_proc(p)
    for line in p.stdout:
        log(line.rstrip("\n"))
    if p.wait() != 0:
        raise RuntimeError(f"yt-dlp failed. {_hint()}")


# [(start unix s, duration s, url)] from a live hls media playlist with program date-times
def _playlist(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        lines = r.read().decode().splitlines()
    segs, t, dur = [], None, None
    for ln in lines:
        if ln.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            t = parse_time(ln.split(":", 1)[1]).timestamp()
        elif ln.startswith("#EXTINF:"):
            dur = float(ln[8:].split(",")[0])
        elif ln and not ln.startswith("#") and dur is not None:
            if t is None:
                raise RuntimeError("this live stream gives no segment times")
            segs.append((t, dur, ln))
            t, dur = t + dur, None
    return segs


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


# saves the last `length` s of a live stream, ending `offset` s before now (youtube keeps ~1 h)
# returns the unix time of the first sample, from the segments' program date-times
def record_live(url, out_dir, offset, length, log=print, cancelled=lambda: False):
    out = subprocess.run(ytdlp() + ["-g", "-f", "bestaudio", "--no-playlist", url], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"yt-dlp failed. {_hint()}")
    segs = _playlist(out.stdout.strip().splitlines()[0])
    i = len(segs) - 1  # newest unbroken run; a reconnect shows up as a jump in the times
    while i > 0 and abs(segs[i - 1][0] + segs[i - 1][1] - segs[i][0]) < 1.0:
        i -= 1
    segs = segs[i:]
    end = segs[-1][0] + segs[-1][1] - offset
    pick = [s for s in segs if s[0] >= end - length - 0.01 and s[0] + s[1] <= end + 0.01]
    if not pick:
        raise RuntimeError(f"only the last {segs[-1][0] + segs[-1][1] - segs[0][0]:.0f} s of this stream are available")
    log(f"live: {sum(s[1] for s in pick):.0f} s from "
        f"{datetime.fromtimestamp(pick[0][0], timezone.utc):%H:%M:%S} UTC")
    data = [None] * len(pick)
    with ThreadPoolExecutor(8) as ex:
        jobs = {ex.submit(_get, s[2]): k for k, s in enumerate(pick)}
        for n, f in enumerate(as_completed(jobs)):
            if cancelled():
                raise RuntimeError("cancelled")
            data[jobs[f]] = f.result()
            log(f"[download] {100 * (n + 1) / len(pick):.1f}% {n + 1} of {len(pick)} segments")
    raw = os.path.join(out_dir, "live.part")
    with open(raw, "wb") as fh:
        fh.write(b"".join(data))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", raw, "-vn", "-c:a", "aac", "-b:a", "128k",
                    os.path.join(out_dir, "clip.m4a")], check=True)
    os.remove(raw)
    return pick[0][0]


# saves out_dir/clip.m4a; returns meta with start set when youtube knows it
# for a stream still live, offset counts back from now (see record_live)
def fetch(url, out_dir, offset=0.0, length=900.0, log=print, on_proc=None, cancelled=lambda: False):
    out = subprocess.run(ytdlp() + ["-J", "--skip-download", "--no-playlist", url],
                         capture_output=True, text=True)
    if out.returncode != 0:
        err = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "yt-dlp failed"
        raise RuntimeError(f"{err}\n{_hint()}")
    info = json.loads(out.stdout)
    live = info.get("live_status")
    if live == "is_live":
        os.makedirs(out_dir, exist_ok=True)
        log(f"{info.get('title')}")
        t0 = record_live(url, out_dir, offset, length, log, cancelled)
        return dict(title=info.get("title") or url, source=info.get("webpage_url") or url, offset=0.0, live=True,
                    start=datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z",
                    clock_tol=10, note="recorded from the live stream; start from its segment times")
    dur = float(info.get("duration") or 0)
    if dur:
        offset = min(offset, max(0.0, dur - 10))
        end = min(dur, offset + length)
    else:
        end = offset + length
    log(f"{info.get('title')}: {offset:.0f}-{end:.0f} s of {dur:.0f} s")
    os.makedirs(out_dir, exist_ok=True)
    stream(ytdlp() + ["--no-playlist", "-f", "bestaudio", "-x", "--audio-format", "m4a",
                      "--download-sections", f"*{offset}-{end}", "--force-keyframes-at-cuts",
                      "--newline", "-o", os.path.join(out_dir, "clip.%(ext)s"), url], log, on_proc)
    t0 = info.get("release_timestamp") if live in ("was_live", "post_live") else None
    start = datetime.fromtimestamp(t0 + offset, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if t0 else None
    return dict(title=info.get("title") or url, source=info.get("webpage_url") or url, start=start, offset=offset,
                clock_tol=30 if start else 2,
                note="start from livestream start; expect tens of seconds of clock error" if start
                else "YouTube gives no recording time for normal uploads; enter it below")
