# local web app: list clips in ./clips, edit settings, run the pipeline, view results
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import adsb, audio
from .timeutil import parse_time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "clips")
JOBS = {}
mimetypes.add_type("audio/mp4", ".m4a")


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "-", name).strip(".-") or "clip"


def clip_file(d):
    for f in sorted(os.listdir(d)):
        if f.startswith("clip.") and not f.endswith(".json"):
            return os.path.join(d, f)
    return None


def load_meta(d):
    p = os.path.join(d, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def bbox(meta):
    lat, lon = meta["center"]
    r = float(meta.get("radius_km", 20))
    dlat, dlon = r / 111.32, r / (111.32 * math.cos(math.radians(lat)))
    return [lat - dlat, lat + dlat, lon - dlon, lon + dlon]


def day_ts(day):
    return datetime(day.year, day.month, day.day, 12, tzinfo=timezone.utc).timestamp()


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def overall(job):
    steps = job.get("steps") or []
    return round(sum(s["frac"] for s in steps) / len(steps), 3) if steps else None


def clips():
    out = []
    if not os.path.isdir(CLIPS):
        return out
    for name in sorted(os.listdir(CLIPS)):
        d = os.path.join(CLIPS, name)
        if not os.path.isdir(d) or not (clip_file(d) or name in JOBS):
            continue
        out.append(dict(name=name, meta=load_meta(d),
                        has={k: os.path.exists(os.path.join(d, f)) for k, f in
                             (("tracks", "tracks.csv"), ("heat", "find.json"), ("viz", "viz/index.html"))},
                        job=JOBS.get(name, {}).get("state"), kind=JOBS.get(name, {}).get("kind"),
                        progress=overall(JOBS.get(name, {})),
                        ready=bool(clip_file(d))))
    return out


def run_job(name, opts):
    job = JOBS[name]
    log = job["log"]
    d = os.path.join(CLIPS, name)
    try:
        meta = load_meta(d)
        clip = clip_file(d)
        if not clip:
            raise ValueError("no audio yet")
        if not meta.get("start") or not meta.get("center"):
            raise ValueError("set a start time and a region first")
        S = parse_time(meta["start"]).timestamp()
        dur = audio.probe(clip)["duration"]
        bb = [round(v, 5) for v in bbox(meta)]
        py = [sys.executable, "-u", "-m", "doppler"]
        steps = []
        day = datetime.fromtimestamp(S, timezone.utc).strftime("%Y-%m-%d")
        if meta.get("unknown_time"):  # the whole UTC day: every flight that day is a candidate
            S = parse_time(day + "T00:00:00Z").timestamp()
            want = dict(start=iso(S), end=iso(S + 86399), bbox=bb)
        else:
            want = dict(start=iso(S - 900), end=iso(S + dur + 900), bbox=bb)
        from . import adsbindex
        tj = os.path.join(d, "tracks.json")
        have = json.load(open(tj)) if os.path.exists(tj) else {}
        tracks_ok = os.path.exists(os.path.join(d, "tracks.csv")) and (have.get("synthetic") or have == want)
        # a big area, or a whole day over more than a town, is searched tile by tile straight from the day's index
        side = max((bb[1] - bb[0]) * 111.32, (bb[3] - bb[2]) * 111.32 * math.cos(math.radians((bb[0] + bb[1]) / 2)))
        days = adsbindex.days_between(parse_time(want["start"]).timestamp(), parse_time(want["end"]).timestamp())
        tiled = not have.get("synthetic") and (side > 150 or (meta.get("unknown_time") and side > 50)) and \
            not any(adsb.recent(day_ts(d0)) and not adsbindex.has_days(day_ts(d0), day_ts(d0)) for d0 in days)
        if tiled:
            for d0 in days:
                if not adsbindex.has_days(day_ts(d0), day_ts(d0)):
                    steps.append((f"ADS-B index for {d0}", py + ["adsb-index", "--date", d0.isoformat()]))
        elif not tracks_ok:
            for d0 in days:
                # recent days come from adsb.lol's replay files instead (see adsb.fetch_recent)
                if not adsbindex.has_days(day_ts(d0), day_ts(d0)) and not adsb.recent(day_ts(d0)):
                    steps.append((f"ADS-B index for {d0}", py + ["adsb-index", "--date", d0.isoformat()]))
            steps.append(("Flight tracks", py + ["adsb", "--start", want["start"], "--end", want["end"],
                                                 "--bbox", *map(str, bb), "-o", os.path.join(d, "tracks.csv")]))
        # event-based search (find): terrain from elevation tiles, so the page's elevation field isn't used
        find = py + ["find", clip, *([] if tiled else ["--tracks", os.path.join(d, "tracks.csv")]), "--bbox", *map(str, bb),
                     *(["--date", day] if meta.get("unknown_time") else
                       ["--start", meta["start"], "--clock-tol", str(opts.get("clock_tol", meta.get("clock_tol", 30)))]),
                     "-o", os.path.join(d, "find.json"), "--viz", os.path.join(d, "viz")]
        if meta.get("source") == "synthetic":  # simulated aircraft fly over flat ground at 0 m in uniform 15 C air
            find += ["--elev", "0", "--temp", "15", "--lapse", "0"]
        if meta.get("truth"):
            find += ["--truth", *map(str, meta["truth"])]
        if opts.get("jets"):
            find.append("--jets")
        steps.append(("Search", find))
        job["steps"] = [new_step(label) for label, _ in steps]
        for st, (label, cmd) in zip(job["steps"], steps):
            log.append(f"== {label}\n")
            st.update(state="running", t0=time.time())
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            job["proc"] = p
            for line in p.stdout:
                if line.startswith("@progress "):
                    pr = json.loads(line[10:])
                    st.update(frac=pr["frac"], msg=pr["msg"])
                else:
                    log.append(line)
            rc = p.wait()
            st["t1"] = time.time()
            if job.get("cancel"):
                st["state"] = "cancelled"
                raise Cancelled()
            if rc != 0:
                st["state"] = "error"
                raise RuntimeError(f"{label} failed")
            st.update(state="done", frac=1.0)
            if label == "Flight tracks":
                json.dump(want, open(tj, "w"))
        job["state"] = "done"
    except Cancelled:
        log.append("cancelled\n")
        job["state"] = "cancelled"
    except Exception as e:
        log.append(f"error: {e}\n")
        job["state"] = "error"


class Cancelled(Exception):
    pass


def new_step(label):
    return dict(label=label, state="pending", frac=0.0, msg="", t0=None, t1=None)


def youtube_job(name, url, offset, length):
    from . import youtube
    job = JOBS[name]
    d = os.path.join(CLIPS, name)
    st = new_step("Download")
    st.update(state="running", t0=time.time())
    job["steps"] = [st]

    def log(line):
        m = re.search(r"\[download\]\s+([\d.]+)%(.*)", line)
        if m:
            st.update(frac=float(m[1]) / 100, msg=" ".join(m[2].split())[:60])
        else:
            job["log"].append(line + "\n")

    try:
        meta = youtube.fetch(url, d, offset, length, log=log, on_proc=lambda p: job.update(proc=p),
                             cancelled=lambda: job.get("cancel"))
        if job.get("cancel"):
            raise Cancelled()
        if meta.pop("live", False):  # each recording of a live stream is its own clip, named by its start
            new = safe(f"{name}-{meta['start'][:16].replace('-', '').replace(':', '')}")
            while os.path.exists(os.path.join(CLIPS, new)):
                new += "b"
            os.replace(d, os.path.join(CLIPS, new))
            d = os.path.join(CLIPS, new)
            JOBS[new] = job
            job["renamed"] = new
        st.update(state="done", frac=1.0, t1=time.time())
        old = load_meta(d)
        old.update(meta, center=old.get("center"), truth=old.get("truth"))
        json.dump(old, open(os.path.join(d, "meta.json"), "w"), indent=1)
        job["log"].append("downloaded" + ("" if meta["start"] else "; set the start time") + "\n")
        job["state"] = "done"
    except Exception as e:
        cancelled = job.get("cancel")
        job["log"].append("cancelled\n" if cancelled else f"error: {e}\n")
        st.update(state="cancelled" if cancelled else "error", t1=time.time())
        job["state"] = "cancelled" if cancelled else "error"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _file(self, path):
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        a, b = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            a = int(m[1]) if m[1] else size - int(m[2])
            b = int(m[2]) if m[1] and m[2] else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p == "/":
            return self._file(os.path.join(os.path.dirname(__file__), "app.html"))
        if p == "/api/clips":
            return self._json(clips())
        if p.startswith("/api/job/"):
            j = JOBS.get(safe(p[9:]), {})
            return self._json(dict(state=j.get("state"), kind=j.get("kind"), log="".join(j.get("log", [])),
                                   steps=j.get("steps", []), now=time.time(), renamed=j.get("renamed")))
        if p.startswith("/clips/"):
            f = os.path.realpath(os.path.join(CLIPS, p[7:]))
            if f.startswith(os.path.realpath(CLIPS) + os.sep) and os.path.isfile(f):
                return self._file(f)
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p.startswith("/api/meta/"):
            d = os.path.join(CLIPS, safe(p[10:]))
            meta = load_meta(d)
            meta.update(json.loads(self._body()))
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            return self._json(meta)
        if p.startswith("/api/cancel/"):
            j = JOBS.get(safe(p[12:]), {})
            if j.get("state") == "running":
                j["cancel"] = True
                if j.get("proc") and j["proc"].poll() is None:
                    j["proc"].terminate()
            return self._json(dict(ok=True))
        if p.startswith("/api/delete/"):
            name = safe(p[12:])
            if JOBS.get(name, {}).get("state") == "running":
                return self._json(dict(error="still running"), 409)
            d = os.path.realpath(os.path.join(CLIPS, name))
            if d.startswith(os.path.realpath(CLIPS) + os.sep) and os.path.isdir(d):
                shutil.rmtree(d)
            JOBS.pop(name, None)
            return self._json(dict(ok=True))
        if p.startswith("/api/run/"):
            name = safe(p[9:])
            if JOBS.get(name, {}).get("state") == "running":
                return self._json(dict(error="already running"), 409)
            JOBS[name] = dict(state="running", kind="run", log=[])
            threading.Thread(target=run_job, args=(name, json.loads(self._body() or b"{}")), daemon=True).start()
            return self._json(dict(ok=True))
        if p == "/api/youtube":
            from .youtube import video_id
            req = json.loads(self._body())
            url, offset, length = req["url"].strip(), float(req.get("offset", 0)), float(req.get("length", 900))
            vid = video_id(url)
            if not vid:
                return self._json(dict(error="not a YouTube video link"), 400)
            name = safe(f"yt-{vid}" + (f"-{int(offset)}" if offset else ""))
            if JOBS.get(name, {}).get("state") == "running":
                return self._json(dict(name=name))
            d = os.path.join(CLIPS, name)
            os.makedirs(d, exist_ok=True)
            if not os.path.exists(os.path.join(d, "meta.json")):
                json.dump(dict(title=url, source=url, radius_km=20, elev=0, center=None, truth=None),
                          open(os.path.join(d, "meta.json"), "w"), indent=1)
            JOBS[name] = dict(state="running", kind="download", log=[])
            threading.Thread(target=youtube_job, args=(name, url, offset, length), daemon=True).start()
            return self._json(dict(name=name))
        if p == "/api/upload":
            q = parse_qs(u.query)
            fname = q.get("filename", ["clip.mp4"])[0]
            name = safe(os.path.splitext(fname)[0])
            d = os.path.join(CLIPS, name)
            os.makedirs(d, exist_ok=True)
            raw = os.path.join(d, "upload" + os.path.splitext(fname)[1].lower())
            left = int(self.headers.get("Content-Length", 0))
            with open(raw, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    fh.write(chunk)
                    left -= len(chunk)
            try:
                info = audio.probe(raw)
            except Exception:
                info = dict(start=None, location=None)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", raw, "-vn", "-ac", "1",
                            "-c:a", "aac", "-b:a", "96k", os.path.join(d, "clip.m4a")], check=True)
            os.remove(raw)
            meta = dict(title=fname, source="upload", radius_km=20, elev=0, clock_tol=2,
                        start=info["start"].strftime("%Y-%m-%dT%H:%M:%SZ") if info["start"] else None,
                        center=None, truth=None, note="" if info["start"] else "no time in metadata, set start")
            loc = audio.parse_iso6709(info.get("location"))
            if loc:  # your own test file: use its GPS tag as the known spot
                meta["truth"] = [round(loc[0], 6), round(loc[1], 6)]
                if loc[2] is not None:
                    meta["elev"] = round(loc[2])
                meta["note"] = "known spot and elevation from the file's GPS tag"
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            return self._json(dict(name=name, meta=meta))
        self.send_error(404)


def main(port=8080):
    os.makedirs(CLIPS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"http://127.0.0.1:{port}  (clips in {CLIPS})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
