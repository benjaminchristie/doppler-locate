# command line: python -m doppler COMMAND; find and serve are the ones the README uses
import argparse
import json
import os
import sys

from . import adsb, audio
from .model import to_enu
from .progress import report


# metres between two points
def _dist(lat, lon, lat0, lon0):
    e, n, _ = to_enu(lat, lon, 0, lat0, lon0)
    return float((e * e + n * n) ** 0.5)


def clip_start(a, path):
    if a.start:
        return adsb.parse_time(a.start).timestamp()
    t = audio.probe(path)["start"]
    if t is None:
        sys.exit(f"no creation time in {path}; pass --start (UTC, e.g. 2025-11-24T14:05:00Z)")
    return t.timestamp()


def cmd_probe(a):
    p = audio.probe(a.file)
    print(f"start (UTC): {p['start']}\nduration:    {p['duration']:.1f} s")
    if p["location"]:
        print(f"location tag: {p['location']}")


def cmd_adsb(a):
    from . import adsbindex
    t0, t1 = adsb.parse_time(a.start), adsb.parse_time(a.end)
    if not a.tar and not a.mlat and adsbindex.has_days(t0.timestamp(), t1.timestamp(), a.store):
        tracks, info = adsbindex.query(t0.timestamp(), t1.timestamp(), a.bbox, a.store)
        adsbindex.write_csv(a.out, tracks, info)
        print(f"{len(tracks)} aircraft from the local index -> {a.out}")
        return
    if not a.tar and not a.mlat and adsb.recent(t0.timestamp()):  # no daily archive yet
        adsb.fetch_recent(t0, t1, a.bbox, a.out, workers=a.workers)
        return
    adsb.fetch(t0, t1, a.bbox, a.out, tar_paths=a.tar, workers=a.workers, mlat=a.mlat)


def cmd_find(a):
    import math
    from datetime import datetime, timezone
    from . import events, find, wide
    if a.center:
        la, lo = a.center
        dl, dn = a.radius / 111.32, a.radius / (111.32 * math.cos(math.radians(la)))
        bbox = [la - dl, la + dl, lo - dn, lo + dn]
    else:
        bbox = a.bbox
    side = max((bbox[1] - bbox[0]) * 111.32, (bbox[3] - bbox[2]) * 111.32 * math.cos(math.radians((bbox[0] + bbox[1]) / 2)))
    if a.jets:  # fan tones reach a few kHz
        a.sr = max(a.sr, 8000)
    x = audio.load_audio(a.file, a.sr)
    dur = len(x) / a.sr
    if a.date:  # time unknown: anywhere in that UTC day
        S, tol = adsb.parse_time(a.date + "T00:00:00Z").timestamp(), None
        window = (S, S + 86399)
    else:
        S, tol = clip_start(a, a.video or a.file), a.clock_tol
        window = (S - tol - 900, S + dur + tol + 900)
    report("search", 0.02, "events")
    ev = events.track_events(x, a.sr)
    report("search", 0.25, f"{len(ev)} events, loading flights")
    src = find.TONAL + (("jet",) if a.jets else ())
    load = wide.Tracks(a.tracks, src) if a.tracks else wide.Index(sources=src)
    kw = dict(temp_c=15.0 if a.temp is None else a.temp, sources=src, elev=a.elev, lapse=a.lapse)
    first_temp = kw["temp_c"]
    # a big area, or a whole day of flights over more than a town, goes tile by tile
    big = not a.tracks and (side > 150 or (tol is None and side > 50))
    far = []
    if big:
        res, _ = wide.search(x, a.sr, S, dur, load, bbox, ev, tol=tol, window=window, log=print, **kw)
    else:
        segs, types = load(None if a.tracks else bbox, *window, True)
        report("search", 0.35, f"{len(segs)} flights")
        res, info = find.search(x, a.sr, S, dur, segs, bbox, ev, tol=tol, log=print, **kw)
    if big and not res:  # nothing anywhere: an empty page for the middle of the area
        mid = wide._grow([(bbox[0] + bbox[1]) / 2] * 2 + [(bbox[2] + bbox[3]) / 2] * 2, 5)
        segs, types = load(mid, window[0], window[0] + dur)
        res, info = find.search(x, a.sr, S, dur, segs, mid, [], tol=tol or 30.0, log=lambda m: None, **kw)
        big = False
    if big:  # the viewer shows the area around the best answer
        r0, far = res[0], res[1:]
        loc = r0["local"]
        segs, types = load(wide._grow(loc["bbox"], 12), loc["start"] - loc["tol"] - 900, loc["start"] + dur + loc["tol"] + 900,
                           True)
    # the weather service knows the air temperature there and then
    if a.temp is None and res and res[0]["score"] > 0:
        from . import weather
        temp = weather.ground_temp(res[0]["lat"], res[0]["lon"], S + res[0]["offset"])
        if temp is not None:
            kw["temp_c"], a.temp = temp, temp
            print(f"air temperature there: {temp:.0f} C")
    # failing that: two or more aircraft at different heights only agree at the right sound speed
    if a.temp is None and res and res[0]["aircraft"] >= 2:
        r0 = dict(res[0], offset=res[0]["offset"] + (S - loc["start"])) if big else res[0]
        temp, sc = find.fit_temp(x, a.sr, loc["start"] if big else S, dur, segs, ev, r0, sources=src, elev=a.elev, lapse=a.lapse)
        if sc[temp] >= 1.02 * sc.get(15.0, 0.0):
            kw["temp_c"] = temp
            print(f"air temperature that fits best: {temp:.0f} C")
    if big:
        top = res[0]
        loc_res, info = find.search(x, a.sr, loc["start"], dur, segs, loc["bbox"], ev, tol=loc["tol"], log=lambda m: None, **kw)
        if loc_res and _dist(loc_res[0]["lat"], loc_res[0]["lon"], top["lat"], top["lon"]) < 1500:
            loc_res[0].update({k: top[k] for k in ("margin", "confident", "tail")})
        for r in loc_res:
            r["offset"] += loc["start"] - S
        res = loc_res + [r for r in far if all(_dist(r["lat"], r["lon"], q["lat"], q["lon"]) > 1000 for q in loc_res)]
    elif kw["temp_c"] != first_temp:
        again, info2 = find.search(x, a.sr, S, dur, segs, bbox, ev, tol=tol, log=lambda m: None, **kw)
        if again and _dist(again[0]["lat"], again[0]["lon"], res[0]["lat"], res[0]["lon"]) < 1500:
            # the temperature was picked for this answer, so it keeps the weaker of its two margins
            again[0]["margin"] = min(again[0]["margin"], res[0]["margin"])
            again[0]["confident"] = again[0]["confident"] and res[0]["confident"]
            res, info = again, info2
    report("search", 0.9, "writing")
    if not res or not res[0]["confident"]:
        why = "no match" if not res or res[0]["score"] <= 0 else "one aircraft" if res[0]["aircraft"] < 2 \
            else "missed passes" if 2 * len(res[0]["missed"]) >= res[0]["aircraft"] else f"margin {res[0]['margin']:.2f}" if res[0]["margin"] < 1.4 \
            else "weak evidence"
        print(f"no confident match ({why})")
    for i, r in enumerate(res[:a.top]):
        d = f"  {_dist(r['lat'], r['lon'], *a.truth):.0f} m from truth" if a.truth else ""
        start = datetime.fromtimestamp(S + r["offset"], timezone.utc)
        print(f"{i + 1:2d}. {r['lat']:.5f}, {r['lon']:.5f}  score {r['score']:.1f}  margin {r['margin']:.2f}  "
              f"start {start:%Y-%m-%d %H:%M:%S}Z  {len(r['events'])} events, {len(r['heard'])} faint, "
              f"{len(r['missed'])} missed{d}")
    if big:  # places elsewhere have no flights loaded here: the viewer lists them
        S2 = loc["start"]
        near = [r for r in res if "match" in r and r.get("local") is None]
        for r in near:
            r["offset"] -= S2 - S
        out = find.summary(x, a.sr, S2, dur, near, info, loc["tol"], bbox, types=types, truth=a.truth)
        out["elsewhere"] = [dict(lat=r["lat"], lon=r["lon"], score=round(r["score"], 2), aircraft=r["aircraft"],
                                 start=datetime.fromtimestamp(S + r["offset"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
                            for r in res if r.get("local") is not None][:12]
    else:
        out = find.summary(x, a.sr, S, dur, res, info, tol, bbox, types=types, truth=a.truth)
    out["temp"] = kw["temp_c"]
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(a.out)
    if a.viz:
        from . import findviz
        print(findviz.build(a.out, a.file, a.viz))
    report("search", 1.0, "done")


def cmd_adsb_index(a):
    from datetime import date
    from . import adsbindex
    adsbindex.build(date.fromisoformat(a.date), a.store, tar_paths=a.tar, workers=a.workers)


def cmd_serve(a):
    from . import serve
    serve.main(a.port)


def cmd_synth(a):
    from . import synth
    t = synth.make(a.out, seed=a.seed, n_pass=a.passes, amp=a.amp, duration=a.duration)
    print(json.dumps(t, indent=1))


def main():
    p = argparse.ArgumentParser(prog="doppler")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("probe", help="start time / location tags from metadata")
    s.add_argument("file")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("adsb", help="pull tracks from adsb.lol archives")
    s.add_argument("--start", required=True, help="ISO time, UTC")
    s.add_argument("--end", required=True)
    s.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LAT0", "LAT1", "LON0", "LON1"))
    s.add_argument("-o", "--out", default="tracks.csv")
    s.add_argument("--tar", nargs="+", help="local tar parts instead of downloading")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--mlat", action="store_true", help="also pull the MLAT-only archive")
    s.add_argument("--store", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "data", "adsb"), help="local index from adsb-index")
    s.set_defaults(fn=cmd_adsb)

    from .adsbindex import DEFAULT_STORE
    s = sub.add_parser("adsb-index", help="stream one day once into a local low-altitude store")
    s.add_argument("--date", required=True, help="UTC day, YYYY-MM-DD")
    s.add_argument("--store", default=DEFAULT_STORE)
    s.add_argument("--tar", nargs="+", help="local tar parts instead of downloading")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_adsb_index)

    s = sub.add_parser("find", help="event-based search: Doppler events, flights that fit, silence, exact curves")
    s.add_argument("file")
    s.add_argument("--tracks", help="ADS-B CSV; default: the local index (adsb-index)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", nargs=4, type=float, metavar=("LAT0", "LAT1", "LON0", "LON1"))
    g.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    s.add_argument("--radius", type=float, default=50.0, help="km, with --center")
    s.add_argument("--video", help="read start time from this file instead")
    s.add_argument("--start", help="ISO start time of the clip, UTC")
    s.add_argument("--date", help="time unknown: search this whole UTC day (YYYY-MM-DD)")
    s.add_argument("--clock-tol", type=float, default=30.0)
    s.add_argument("--elev", type=float, default=None, help="flat ground at this height (m); default terrain tiles")
    s.add_argument("--temp", type=float, default=None, help="air temperature at the ground, C; default: looked up, else fitted")
    s.add_argument("--lapse", type=float, default=6.5, help="K per km of height the air cools by")
    s.add_argument("--jets", action="store_true", help="also consider jets")
    s.add_argument("--sr", type=int, default=4000)
    s.add_argument("--truth", nargs=2, type=float, metavar=("LAT", "LON"))
    s.add_argument("--top", type=int, default=5)
    s.add_argument("-o", "--out", default="find.json")
    s.add_argument("--viz", help="also write the viewer to this directory")
    s.set_defaults(fn=cmd_find)

    s = sub.add_parser("serve", help="local web app over ./clips")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("synth", help="make a synthetic test case")
    s.add_argument("-o", "--out", default="synth")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--passes", type=int, default=3)
    s.add_argument("--amp", type=float, default=0.006, help="tone amplitude at 1 km")
    s.add_argument("--duration", type=float, default=900)
    s.set_defaults(fn=cmd_synth)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
