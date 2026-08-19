#!/usr/bin/env python3
"""Extracts the verbal article draft (user message, 2026-08-18T13:39) from the
session transcript into draft_source.txt for building the HTML."""
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
                if "\u0413\u0435\u043e\u043c\u0435\u0442\u0440\u0438\u0447\u0435\u0441\u043a\u0438\u0439 \u0430\u043f\u043e\u043a\u0430\u043b\u0438\u043f\u0441\u0438\u0441" in obj and "\u041f\u0440\u043e\u043b\u043e\u0433" in obj:
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
# cut off the instruction after P.S. (it is not part of the article)
cut = text.find("\u0422\u0430\u043a \u0430 \u0442\u0435\u043f\u0440\u044c")
if cut > 0:
    text = text[:cut].rstrip()
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(f"[ok] line {line_no} -> {OUT} ({len(text)} chars)")
