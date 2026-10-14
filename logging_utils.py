import json
import os
from datetime import datetime, timezone

from paths import log_dir, log_path


def write_log(kind, record, test=False):
    os.makedirs(log_dir(), exist_ok=True)
    row = dict(record)
    row["kind"] = kind
    row["test"] = bool(test)
    row["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = log_path(kind, test=test)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return path


def read_log(kind, test=False):
    path = log_path(kind, test=test)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
