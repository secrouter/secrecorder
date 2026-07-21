#!/usr/bin/env python3
# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Compare two whisper JSON responses — normalized token WER + word-timestamp sanity.

Used for the q4-vs-fp16 accuracy gate (Phase B). WER via difflib opcodes (edit distance over
normalized tokens); no external deps. The timestamp check guards the contract the timeline depends
on: monotonic non-decreasing starts, no negative spans, coverage to ~end of audio.

    ./wer.py REF.json HYP.json [--audio-seconds S]
"""
from __future__ import annotations

import argparse
import difflib
import json
import re


def norm_tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int, float]:
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    errors = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            errors += max(i2 - i1, j2 - j1)  # subs + surplus ins/del for the block
        elif tag == "delete":
            errors += i2 - i1
        elif tag == "insert":
            errors += j2 - j1
    return errors, len(ref), (errors / len(ref) if ref else 0.0)


def ts_sanity(words: list[dict], audio_seconds: float) -> list[str]:
    issues, last_start = [], -1.0
    for i, w in enumerate(words):
        s, e = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        if e < s:
            issues.append(f"word {i} negative span ({s:.2f}>{e:.2f})")
        if s < last_start - 0.05:  # allow tiny diarization/rounding overlap
            issues.append(f"word {i} start regresses ({s:.2f}<{last_start:.2f})")
        last_start = s
    if words and audio_seconds:
        cov = float(words[-1].get("end", 0.0)) / audio_seconds
        if cov < 0.9:
            issues.append(f"last word ends at {cov * 100:.0f}% of audio (dropped tail?)")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("hyp")
    ap.add_argument("--audio-seconds", type=float, default=0.0)
    a = ap.parse_args()
    ref, hyp = json.load(open(a.ref)), json.load(open(a.hyp))
    rt, ht = norm_tokens(ref.get("text", "")), norm_tokens(hyp.get("text", ""))
    errs, n, rate = wer(rt, ht)
    print(f"WER: {rate * 100:.2f}%  ({errs} edits / {n} ref tokens)  |  "
          f"ref {len(rt)} vs hyp {len(ht)} tokens")
    dur = a.audio_seconds or float(hyp.get("duration", 0.0))
    iss = ts_sanity(hyp.get("words", []), dur)
    print("timestamp sanity:", "OK" if not iss else f"{len(iss)} issue(s): " + "; ".join(iss[:5]))


if __name__ == "__main__":
    main()
