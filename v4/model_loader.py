"""
model_loader.py — общий модуль загрузки модели с:
  • tqdm прогресс-бар (байты + %)
  • watchdog на скорость (ретраит если < 50 KB/s дольше 30s)
  • автовосстановление при network reset / timeout (до 3 попыток)
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR = Path(__file__).parent.resolve()  # v4/
ROOT_DIR = BASE_DIR.parent                   # cmq_experiment/

DEFAULT_MODEL = "HuggingFaceTB/SmolLM-135M"
LOCAL_DIR = ROOT_DIR / "models" / "smollm-135m"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16


def _get_hf_cache_dir() -> Optional[Path]:
    """Найти модель в HF cache (snapshot directory)."""
    try:
        from huggingface_hub import hf_hub_download
        cached_file = hf_hub_download(
            repo_id=DEFAULT_MODEL,
            filename="config.json",
            local_dir_use_symlinks=False,
        )
        return Path(cached_file).parent
    except Exception as e:
        print(f"[model] HF cache check failed: {e}")
        return None


def _find_model_source() -> Optional[Path]:
    """Найти модель: 1) local dir → 2) HF cache."""
    # 1. Локальная копия
    if _model_exists(LOCAL_DIR):
        print(f"[model] [OK] Local copy: {LOCAL_DIR}")
        return LOCAL_DIR

    # 2. HF cache snapshot
    cache_dir = _get_hf_cache_dir()
    if cache_dir and (cache_dir / "model.safetensors").exists():
        print(f"[model] [OK] HF cache: {cache_dir.name}")
        return cache_dir

    return None


# ──────────────────────────────────────────────────────────────────────
# Speed watchdog — мониторит скорость скачивания в отдельном потоке
# ──────────────────────────────────────────────────────────────────────
class SpeedWatchdog:
    """Если скорость < min_kbps дольше patience сек → поднимает флаг stall."""

    def __init__(self, min_kbps=50.0, patience_s=30):
        self.min_kbps = min_kbps
        self.patience = patience_s
        self._history = []          # (timestamp, bytes_downloaded)
        self._lock = threading.Lock()
        self.stalled = False

    def record(self, downloaded_bytes: int):
        with self._lock:
            self._history.append((time.monotonic(), downloaded_bytes))
            # Храним только последние 60 секунд
            cutoff = time.monotonic() - self.patience - 10
            self._history = [(t, b) for t, b in self._history if t > cutoff]

    def check(self):
        """Проверить stall. Если скорость < threshold → stalled=True."""
        with self._lock:
            if len(self._history) < 2:
                return False
            t0, b0 = self._history[0]
            t1, b1 = self._history[-1]
            dt = t1 - t0
            if dt < 5:
                return False
            speed_kbps = (b1 - b0) / dt / 1024
            if speed_kbps < self.min_kbps:
                self.stalled = True
                return True
        return False


# ──────────────────────────────────────────────────────────────────────
# Progress bar callback для huggingface_hub
# ──────────────────────────────────────────────────────────────────────
class HFProgressBar:
    """Обёртка над tqdm для huggingface_hub download callbacks."""

    def __init__(self, watchdog: SpeedWatchdog):
        self._bars = {}   # {file_id: tqdm instance}
        self._wd = watchdog

    def __call__(self, downloaded: int, total: Optional[int] = None, file_id: str = ""):
        if file_id not in self._bars:
            desc = Path(file_id or "download").name
            self._bars[file_id] = tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"↓ {desc}",
                ncols=120,
            )
        bar = self._bars[file_id]
        bar.update(downloaded - (bar.n if hasattr(bar, 'n') else 0))
        if total and total != bar.total:
            bar.reset(total=total)
            bar.n = downloaded

        self._wd.record(downloaded)

        if total and downloaded >= total:
            bar.close()
            print()  # newline after bar


# ──────────────────────────────────────────────────────────────────────
# Загрузка модели с ретраями
# ──────────────────────────────────────────────────────────────────────
def _model_exists(local_dir: Path) -> bool:
    """Проверка что модель целая (safetensors + config)."""
    has_weights = any(local_dir.glob("*.safetensors")) or any(local_dir.glob("pytorch_model.bin"))
    return has_weights and (local_dir / "config.json").exists()


def load_model(
    model_name: str = DEFAULT_MODEL,
    local_dir: Optional[Path] = None,
    max_retries: int = 3,
) -> AutoModelForCausalLM:
    """
    Загрузить модель с прогресс-баром и автовосстановлением.

    Ищет модель: 1) local dir → 2) HF cache → 3) скачивает при необходимости.
    При network error / stall → автоматический ретраит с exponential backoff.
    """
    # Сначала попробуй найти существующую копию (local или HF cache)
    source_path = _find_model_source()

    if source_path is None:
        print(f"[model] [DL] Downloading: {model_name}")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            wd = SpeedWatchdog(min_kbps=50.0, patience_s=30)
            bar_cb = HFProgressBar(wd)

            # Запускаем watchdog в фоне
            wd_thread = threading.Thread(target=_watchdog_loop, args=(wd,), daemon=True)
            wd_thread.start()

            if source_path:
                # Локальная загрузка из кэша — быстро, без прогресса
                print(f"[model] Loading from {source_path.name}...")
                model = AutoModelForCausalLM.from_pretrained(
                    str(source_path),
                    torch_dtype=DTYPE,
                    device_map=DEVICE,
                )
            else:
                # HF download с прогрессом
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=DTYPE,
                    device_map=DEVICE,
                    resume_download=True,
                )

            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[model] [OK] Loaded {n_params:,} params -> {DEVICE}")
            return model

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Retryable errors
            retryable = wd.stalled or any(kw in msg for kw in [
                "timeout", "connection", "reset", "eof", "interrupted",
                "failed to connect", "ssl", "network",
            ])

            if retryable and attempt < max_retries:
                backoff = min(5 * attempt, 30)
                reason = "SPEED STALL" if wd.stalled else f"{type(e).__name__}"
                print(f"[model] [ERR] {reason}: {e}")
                print(f"[model] [RETRY] {attempt+1}/{max_retries} in {backoff}s...")
                time.sleep(backoff)
                wd.stalled = False
            elif attempt < max_retries:
                backoff = min(3 * attempt, 15)
                print(f"[model] [ERR] {type(e).__name__}: {e}")
                print(f"[model] [RETRY] {attempt+1}/{max_retries} in {backoff}s...")
                time.sleep(backoff)
            else:
                break

    raise RuntimeError(f"Model load failed after {max_retries} attempts") from last_err


def _watchdog_loop(wd: SpeedWatchdog, interval_s=5):
    """Фоновый поток — проверяет stall каждые interval_s секунд."""
    while True:
        time.sleep(interval_s)
        if wd.check():
            print(f"\n[watchdog] [WARN] Download stalled (< {wd.min_kbps} KB/s)")


def load_tokenizer(
    model_name: str = DEFAULT_MODEL,
    local_dir: Optional[Path] = None,
) -> AutoTokenizer:
    """Загрузить токенизатор (быстро, без весов)."""
    source_path = _find_model_source()
    source = str(source_path) if source_path else model_name
    tok = AutoTokenizer.from_pretrained(source, resume_download=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    local_dir: Optional[Path] = None,
    max_retries: int = 3,
):
    """Загрузить модель + токенизатор."""
    tok = load_tokenizer(model_name, local_dir)
    mdl = load_model(model_name, local_dir, max_retries)
    return tok, mdl
