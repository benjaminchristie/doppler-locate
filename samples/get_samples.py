"""Download test clips whose filming spot is published by the uploader, or make synthetic ones.

usage:
  python samples/get_samples.py                 # all real samples (needs a current yt-dlp)
  python samples/get_samples.py lhr-robocam     # just one
  python samples/get_samples.py --synth         # synthetic clips, no network
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "clips")

# spot = uploader's stated filming location (my lat/lon estimate, about +-300 m); elev in m MSL, rough
SAMPLES = {
    "lhr-robocam": dict(url="https://www.youtube.com/watch?v=j84u714bMgI", spot=[51.478, -0.46], elev=25,
                        note="Flight Focus 365 robocam, 'north runway' only: spot good to 1-2 km"),
    "lax-imperial-hill": dict(url="https://www.youtube.com/watch?v=WEKGSAa3PJU", spot=[33.931, -118.404], elev=40,
                              note="AIRLINE VIDEOS live; commentary + ATC in audio"),
    "syd-sheps-mound": dict(url="https://www.youtube.com/watch?v=DuZbEXyqjiU", spot=[-33.932, 151.176], elev=5,
                            note="SydSquad live; host commentary + ATC"),
    "nrt-sakura-no-yama": dict(url="https://www.youtube.com/watch?v=_Vi4gBDISl4", spot=[35.796, 140.372], elev=40,
                               note="BIG WINGS live; audio unverified"),
    "fra-west": dict(url="https://www.youtube.com/watch?v=0clVk10hSxk", spot=[50.034, 8.523], elev=110,
                     note="Planespotting with Kai; 'West' assumed to be Platform West"),
}


def real(name, offset, length):
    sys.path.insert(0, ROOT)
    from doppler import youtube
    s = SAMPLES[name]
    d = os.path.join(CLIPS, name)
    meta = youtube.fetch(s["url"], d, offset, length)
    meta.update(note=s["note"], center=s["spot"], radius_km=20, elev=s["elev"], truth=s["spot"])
    if not meta["start"]:
        print(f"{name}: no stream start time from YouTube, set it on the page")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"{name}: {d}")


def synth():
    sys.path.insert(0, ROOT)
    from doppler import synth as sy
    for name, dur, n in (("synth-1min", 60, 2), ("synth-15min", 900, 3)):
        d = os.path.join(CLIPS, name)
        t = sy.make(d, seed=6 if dur == 60 else 2, n_pass=n, amp=0.012, duration=dur)
        os.remove(os.path.join(d, "truth.json"))
        from doppler.timeutil import parse_time
        start = parse_time(t["start"]).timestamp() + t["clock_error"]
        meta = dict(title=f"Synthetic, {dur // 60 or 1} min", source="synthetic", note="metadata clock is 3 s off",
                    start=datetime.fromtimestamp(start, timezone.utc).isoformat().replace("+00:00", "Z"),
                    center=[round(t["lat"] + 0.07, 4), round(t["lon"] - 0.09, 4)], radius_km=35, elev=0, truth=[t["lat"], t["lon"]], clock_tol=4)
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        # tracks.csv already covers the clip, so the page won't try to download ADS-B
        json.dump(dict(synthetic=True), open(os.path.join(d, "tracks.json"), "w"))
        print(f"{name}: {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help=f"any of {', '.join(SAMPLES)}")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--offset", type=float, default=3600, help="seconds into the stream")
    ap.add_argument("--length", type=float, default=900)
    a = ap.parse_args()
    if a.synth:
        synth()
    else:
        for n in a.names or SAMPLES:
            try:
                real(n, a.offset, a.length)
            except RuntimeError as e:
                print(f"{n}: {e}")
