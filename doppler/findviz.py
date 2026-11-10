# writes the viewer page and audio for a find result
import json
import os
import shutil

import numpy as np


def build(find_json, clip, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = json.load(open(find_json))
    ext = os.path.splitext(clip)[1] or ".m4a"
    shutil.copy(clip, os.path.join(out_dir, "clip" + ext))
    data["audio"] = "clip" + ext
    from .audio import load_audio  # clip loudness, so the page can boost quiet audio
    x = load_audio(clip, 8000)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)
    data["level"] = dict(rms_db=round(20 * np.log10(rms), 1),
                         peak_db=round(20 * np.log10(float(np.percentile(np.abs(x), 99.9)) + 1e-9), 1))
    tpl = open(os.path.join(os.path.dirname(__file__), "findviz.html")).read()
    out = os.path.join(out_dir, "index.html")
    with open(out, "w") as fh:
        fh.write(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    return out
