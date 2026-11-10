# progress lines the web app reads from a step's output: @progress {json}
import json
import time

_last = {}


def report(step, frac, msg="", every=1.0):
    now = time.time()
    if frac < 1 and now - _last.get(step, 0) < every:
        return
    _last[step] = now
    print("@progress " + json.dumps(dict(step=step, frac=round(min(max(frac, 0), 1), 4), msg=msg)), flush=True)
