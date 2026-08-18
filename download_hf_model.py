r"""Generic HF model downloader (bypasses stalling snapshot_download).

Streams each file from the HF resolve URL with Range-based resume and a
progress line. Fetches the file list from the HF API, then downloads the
files needed to load a CausalLM model (config, tokenizer, weights, index).

Usage:
    python download_hf_model.py <repo> <dest_dir>
    e.g.
    python download_hf_model.py Qwen/Qwen2.5-3B models/qwen2.5-3b
    python download_hf_model.py Qwen/Qwen3-4B   models/qwen3-4b
"""
import sys
import time
from pathlib import Path

import requests

CHUNK = 1024 * 1024          # 1 MB
MAX_RETRIES = 10
API_TIMEOUT = 30

# file extensions / names we want (skip datasets, images, big non-weight blobs)
WANT_EXT = {".json", ".safetensors", ".model", ".txt", ".jinja"}
WANT_NAME = {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "model.safetensors.index.json", "configuration.json",
}


def list_files(repo: str):
    """Return list of (filename, size_bytes) at the repo root via HF API."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    r = requests.get(url, timeout=API_TIMEOUT)
    r.raise_for_status()
    items = r.json()
    out = []
    for it in items:
        if it.get("type") != "file":
            continue
        name = it["path"]
        size = it.get("size", 0)
        if name in WANT_NAME or any(name.endswith(e) for e in WANT_EXT):
            # skip obvious non-model blobs
            if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".pdf")):
                continue
            out.append((name, size))
    return out


def download_one(repo: str, filename: str, dest_dir: Path,
                 expected_size: int = 0) -> bool:
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    # total size via HEAD, fallback to the size from the tree API
    try:
        head = requests.head(url, allow_redirects=True, timeout=API_TIMEOUT)
        total = int(head.headers.get("Content-Length", 0))
    except Exception:
        total = 0
    if total == 0 and expected_size:
        total = expected_size

    if dest.exists():
        if total == 0 or dest.stat().st_size >= total:
            print(f"  [skip] {filename} already present", flush=True)
            return True
        # dest is incomplete (e.g. renamed after HEAD failed) — resume from it
        if part.exists():
            part.unlink()
        dest.rename(part)
        print(f"  [resume] {filename} incomplete on disk, resuming", flush=True)

    if total == 0:
        print(f"  [warn] {filename}: unknown size (HEAD failed), streaming", flush=True)

    start = part.stat().st_size if part.exists() else 0
    if total and start >= total:
        part.rename(dest)
        print(f"  [done] {filename} ({total/1e9:.3f} GB)", flush=True)
        return True

    if start > 0:
        print(f"  [resume] {filename} from {start/1e6:.1f} MB", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        headers = {"Range": f"bytes={start}-"} if start > 0 else {}
        try:
            with requests.get(url, stream=True, headers=headers,
                              allow_redirects=True, timeout=(30, 180)) as r:
                if start > 0 and r.status_code == 200:
                    start = 0
                    part.write_bytes(b"")
                r.raise_for_status()
                mode = "ab" if (start > 0 and r.status_code == 206) else "wb"
                if mode == "wb":
                    start = 0
                shown = 0.0
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        start += len(chunk)
                        now = time.time()
                        if now - shown >= 5:
                            shown = now
                            if total:
                                pct = 100.0 * start / total
                                print(f"\r  {filename}: {pct:5.1f}%  "
                                      f"{start/1e6:8.1f}/{total/1e6:.0f} MB",
                                      end="", flush=True)
                            else:
                                print(f"\r  {filename}: {start/1e6:8.1f} MB",
                                      end="", flush=True)
            print(flush=True)
            break
        except Exception as e:
            print(f"\n  [attempt {attempt}/{MAX_RETRIES}] {filename}: "
                  f"{type(e).__name__}: {e}", flush=True)
            print(f"    partial {start/1e6:.1f} MB — will resume", flush=True)
            time.sleep(3 * attempt)

    if part.exists() and (total == 0 or part.stat().st_size >= total):
        part.rename(dest)
        print(f"  [done] {filename}", flush=True)
        return True

    print(f"  [INCOMPLETE] {filename} "
          f"({(part.stat().st_size if part.exists() else 0)/1e6:.1f} MB) — re-run",
          flush=True)
    return False


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    repo, dest = sys.argv[1], Path(sys.argv[2])
    print(f"Repo: {repo}", flush=True)
    print(f"Dest: {dest}", flush=True)

    print("Fetching file list...", flush=True)
    files = list_files(repo)
    if not files:
        print("  No files found (API error or empty repo?)", flush=True)
        sys.exit(1)

    total_bytes = sum(s for _, s in files)
    print(f"  {len(files)} files, {total_bytes/1e9:.3f} GB total:", flush=True)
    for name, size in sorted(files, key=lambda x: -x[1]):
        print(f"    {size/1e9:8.3f} GB  {name}", flush=True)

    ok = True
    # order: small config/tokenizer first, then weights (biggest last)
    ordered = sorted(files, key=lambda x: (x[1] > 1e8, x[0]))
    for name, size in ordered:
        if not download_one(repo, name, dest, size):
            ok = False

    print(flush=True)
    if ok:
        print(f"ALL DONE -> {dest}", flush=True)
    else:
        print("SOME FILES INCOMPLETE — re-run to resume", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
