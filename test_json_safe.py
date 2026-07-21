#!/usr/bin/env python3
# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the NaN-in-response 500 (run: `python3 test_json_safe.py`).

mlx-whisper emits non-finite floats in segment scores (avg_logprob / compression_ratio /
no_speech_prob) on degenerate audio. Starlette's JSONResponse serializes with allow_nan=False, so
ONE such value 500'd the whole request and lost the episode's transcript. _json_safe scrubs them to
null at the response boundary. This test loads that function out of server.py WITHOUT importing the
heavy ML deps, so it runs anywhere.
"""
import json
import math
import pathlib


def _load_json_safe():
    src = pathlib.Path(__file__).with_name("server.py").read_text()
    start = src.index("def _json_safe")
    end = src.index("def _speaker_embeddings")
    ns = {"math": math}
    exec(compile("from __future__ import annotations\n" + src[start:end], "server.py", "exec"), ns)
    return ns["_json_safe"]


def main() -> None:
    f = _load_json_safe()

    # 1. The exact production failure: a segment score is NaN.
    found = []
    payload = {"words": [{"word": "hi", "start": 0.0, "end": 0.5}],
               "segments": [{"text": "hi", "avg_logprob": float("nan"),
                             "compression_ratio": float("inf"), "no_speech_prob": 0.1}]}
    safe = f(payload, found=found)
    json.dumps(safe, allow_nan=False)  # must NOT raise — that was the 500
    assert safe["segments"][0]["avg_logprob"] is None
    assert safe["segments"][0]["compression_ratio"] is None
    assert safe["segments"][0]["no_speech_prob"] == 0.1     # finite value untouched
    assert safe["words"][0]["start"] == 0.0                 # words the client needs untouched
    assert sorted(found) == ["segments[].avg_logprob", "segments[].compression_ratio"]

    # 2. Non-float types and structure preserved; -inf caught too.
    assert f({"a": "txt", "b": 3, "c": None, "d": [1.0, float("-inf")], "e": True}) == \
        {"a": "txt", "b": 3, "c": None, "d": [1.0, None], "e": True}

    # 3. A clean payload is returned unchanged (no false positives).
    clean = {"language": "en", "words": [{"word": "x", "start": 1.0, "end": 2.0}], "speakers": []}
    found2 = []
    assert f(clean, found=found2) == clean and found2 == []

    # 4. Raw NaN really does break allow_nan=False (guards the premise).
    try:
        json.dumps({"x": float("nan")}, allow_nan=False)
        raise AssertionError("expected ValueError — premise no longer holds")
    except ValueError:
        pass

    print("PASS: NaN/inf scrubbed to null, finite/other values preserved, paths reported")


if __name__ == "__main__":
    main()
