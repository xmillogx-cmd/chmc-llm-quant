"""Direct download of a Hugging Face model file (bypasses stalling snapshot_download).

Streams from the HF resolve URL with resume support (Range header based on
existing partial file) and a progress line every ~5s. Originally written for
TinyLlama's single-file model.safetensors; works for any repo/file pair.

Usage:
    python download_tinyllama_direct.py
    python download_tinyllama_direct.py --repo Qwen/Qwen2.5-0.5B \
        --filename model-00001-of-00002.safetensors --dest-dir models/qwen2.5-0.5b

Defaults: TinyLlama/TinyLlama-1.1B-Chat-v1.0 / model.safetensors into
<repo_root>/models/tinyllama-1.1b/.
"""
import argparse
import sys
import time
from pathlib import Path

import requests

DEFAULT_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_FILENAME = "model.safetensors"
ROOT_DIR = Path(__file__).resolve().parent

CHUNK = 1024 * 1024  # 1 MB
MAX_RETRIES = 8


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stream a single file from an HF repo with resume support.")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--filename", default=DEFAULT_FILENAME,
                    help=f"file to download (default: {DEFAULT_FILENAME})")
    ap.add_argument("--dest-dir", default=None,
                    help="destination directory "
                         "(default: <repo_root>/models/tinyllama-1.1b)")
    return ap.parse_args()


def main():
    args = parse_args()
    repo, filename = args.repo, args.filename
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    if args.dest_dir:
        dest_dir = Path(args.dest_dir)
    else:
        dest_dir = ROOT_DIR / "models" / "tinyllama-1.1b"
    dest = dest_dir / filename
    part = dest.with_suffix(dest.suffix + ".part")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Resolve total size via HEAD
    head = requests.head(url, allow_redirects=True, timeout=30)
    total = int(head.headers.get("Content-Length", 0))
    print(f"Total size: {total/1e9:.3f} GB", flush=True)

    start = part.stat().st_size if part.exists() else 0
    if start >= total:
        print("Already complete, skipping download.", flush=True)
        part.rename(dest)
        return

    print(f"Resuming from {start/1e6:.1f} MB", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        headers = {"Range": f"bytes={start}-"} if start > 0 else {}
        try:
            with requests.get(url, stream=True, headers=headers,
                              allow_redirects=True, timeout=(30, 120)) as r:
                if start > 0 and r.status_code == 200:
                    # Server ignored Range -> restart from 0
                    start = 0
                    part.write_bytes(b"")
                r.raise_for_status()
                mode = "ab" if (start > 0 and r.status_code == 206) else "wb"
                if mode == "wb":
                    start = 0
                t0 = time.time()
                shown = 0
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        start += len(chunk)
                        now = time.time()
                        if now - shown >= 5:
                            shown = now
                            done_pct = 100.0 * start / total
                            print(f"\r{done_pct:5.1f}%  {start/1e6:8.1f}/{total/1e6:.0f} MB",
                                  end="", flush=True)
            print(flush=True)
            break  # completed
        except Exception as e:
            print(f"\n[attempt {attempt}/{MAX_RETRIES}] error: {type(e).__name__}: {e}",
                  flush=True)
            print(f"  partial saved: {start/1e6:.1f} MB — will resume", flush=True)
            time.sleep(3 * attempt)

    if part.stat().st_size >= total:
        part.rename(dest)
        print(f"DONE -> {dest} ({dest.stat().st_size/1e9:.3f} GB)", flush=True)
    else:
        print(f"INCOMPLETE: {part.stat().st_size/1e6:.1f}/{total/1e6:.0f} MB "
              f"(re-run to resume)", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
