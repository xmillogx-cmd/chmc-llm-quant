"""
tda_log.py — логирование шагов TDA-пайплайна (консоль + файл)
=============================================================

Tee дублирует sys.stdout/sys.stderr в консоль И в лог-файл, поэтому весь
вывод шага (prints, warnings, traceback при падении) сохраняется независимо
от способа запуска (батник или вручную). Логи: tda_analysis\\logs\\ — это НЕ
артефакты эксперимента (их по-прежнему ровно 5 в results_v6\\tda_analysis\\).
"""

import sys
import time
from pathlib import Path


class Tee:
    """Поток, пишущий одновременно в несколько потоков."""

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
    """Путь лога: <tda_analysis>/logs/<step>_<model>_<YYYYmmdd_HHMMSS>.log."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / "logs" / f"{step}_{model}_{ts}.log"


def start_logging(log_path):
    """sys.stdout/stderr -> консоль + файл (utf-8). Возвращает открытый файл."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    print(f"[log] {log_path}")
    return f
