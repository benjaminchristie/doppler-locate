import re
from datetime import datetime, timezone


# iso 8601 to aware utc datetime; no zone means utc
# handles Z and +HHMM, which fromisoformat rejects on python 3.9
def parse_time(s):
    s = str(s).strip()
    s = re.sub(r"[Zz]$", "+00:00", s)
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
    m = re.match(r"(.*?\d{2}:\d{2}:\d{2})\.(\d+)(.*)$", s)
    if m:
        s = f"{m.group(1)}.{(m.group(2) + '000000')[:6]}{m.group(3)}"
    d = datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)
