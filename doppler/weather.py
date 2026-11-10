# air temperature at the ground from open-meteo (no key); None if it can't be had
import json
import urllib.request
from datetime import datetime, timezone

URLS = ("https://archive-api.open-meteo.com/v1/archive", "https://api.open-meteo.com/v1/forecast")


# temperature (C) 2 m above the ground near (lat, lon) at unix time t
# the place is rounded to 0.1 degree: weather doesn't need more, and the service doesn't learn the answer
def ground_temp(lat, lon, t, timeout=15):
    day = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
    q = f"?latitude={lat:.1f}&longitude={lon:.1f}&start_date={day}&end_date={day}&hourly=temperature_2m&timezone=UTC"
    for url in URLS:
        try:
            req = urllib.request.Request(url + q, headers={"User-Agent": "doppler-tracking"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                h = json.load(r)["hourly"]
            hour = datetime.fromtimestamp(t, timezone.utc).hour
            v = h["temperature_2m"][hour]
            if v is not None:
                return float(v)
        except Exception:
            continue
    return None
