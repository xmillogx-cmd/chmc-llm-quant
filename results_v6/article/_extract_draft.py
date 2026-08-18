#!/usr/bin/env python3
"""Извлекает вербальный черновик статьи (user message, 2026-08-18T13:39) из
транскрипта сессии в draft_source.txt для построения HTML."""
import json
import sys

# One-shot helper: pass the session transcript explicitly.
if len(sys.argv) != 2:
    raise SystemExit("Usage: python _extract_draft.py <path-to-session-transcript.jsonl>")
SRC = sys.argv[1]
OUT = __file__.replace("\\", "/").rsplit("/", 1)[0] + "/draft_source.txt"

cands = []  # (line_no, text)
with open(SRC, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue

        def collect(obj):
            if isinstance(obj, str):
                if "Геометрический апокалипсис" in obj and "Пролог" in obj:
                    cands.append((i, obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)
            elif isinstance(obj, list):
                for v in obj:
                    collect(v)

        collect(rec)

if not cands:
    raise SystemExit("draft message not found")
line_no, text = max(cands, key=lambda t: len(t[1]))
# обрезаем инструкцию после P.S. (она не часть статьи)
cut = text.find("Так а тепрь")
if cut > 0:
    text = text[:cut].rstrip()
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(f"[ok] line {line_no} -> {OUT} ({len(text)} chars)")
