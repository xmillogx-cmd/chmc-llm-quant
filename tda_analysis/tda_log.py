"""
tda_log.py — logging of TDA pipeline steps (console + file)
=============================================================

Tee duplicates sys.stdout/sys.stderr to the console AND a log file, so all
step output (prints, warnings, traceback on crash) is saved regardless of
the launch method (batch script or manual). Logs: tda_analysis\\logs\\ — these
are NOT experiment artifacts (there are still exactly 5 in results_v6\\tda_analysis\\).
"""

import sys
import time
from pathlib import Path


class Tee:
    """Stream that writes to several streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def log_file_for(step, model, base_dir=None):
    """Log path: <tda_analysis>/logs/<step>_<model>_<YYYYmmdd_HHMMSS>.log."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / "logs" / f"{step}_{model}_{ts}.log"


def start_logging(log_path):
    """sys.stdout/stderr -> console + file (utf-8). Returns the open file."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    print(f"[log] {log_path}")
    return f
