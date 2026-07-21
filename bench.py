#!/usr/bin/env python3
# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the whisper server — POST audio N times (optionally concurrent), report timing + RTF.

Measures client-side wall time and derives realtime-factor from the response's audio ``duration``
(no log scraping). Concurrent mode reports makespan for J simultaneous jobs.

    ./bench.py FILE [--url http://127.0.0.1:9001] [--runs 3] [--jobs 1] [--diarize]
               [--save out.json] [--label fp16]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request
import uuid
import wave


def wav_seconds(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001 — non-WAV / unreadable → fall back to response duration
        return 0.0


def post(url: str, path: str, diarize: bool) -> tuple[float, dict]:
    boundary = uuid.uuid4().hex
    fields = {"model": "whisper-1", "response_format": "verbose_json",
              "timestamp_granularities[]": "word"}
    if diarize:
        fields["diarize"] = "true"
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += f"{v}\r\n".encode()
    with open(path, "rb") as f:
        data = f.read()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/audio/transcriptions",
                                 data=bytes(body), method="POST")
    req.add_header("content-type", f"multipart/form-data; boundary={boundary}")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        payload = json.loads(r.read().decode())
    return time.monotonic() - t0, payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--url", default="http://127.0.0.1:9000")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1, help="concurrent jobs per run (makespan mode)")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--save", help="save the first response JSON here (for wer.py)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    dur = wav_seconds(a.file)
    walls: list[float] = []
    first: dict | None = None
    for i in range(a.runs):
        if a.jobs == 1:
            w, p = post(a.url, a.file, a.diarize)
            walls.append(w)
            first = first or p
        else:
            results: list = [None] * a.jobs
            t0 = time.monotonic()

            def worker(idx: int) -> None:
                results[idx] = post(a.url, a.file, a.diarize)

            ths = [threading.Thread(target=worker, args=(j,)) for j in range(a.jobs)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            walls.append(time.monotonic() - t0)
            first = first or results[0][1]
        print(f"  run {i + 1}/{a.runs}: {walls[-1]:.2f}s")

    if a.save and first is not None:
        json.dump(first, open(a.save, "w"))
    dur = dur or (first.get("duration", 0.0) if first else 0.0)
    n_words = len(first.get("words", [])) if first else 0
    n_spk = len(first.get("speakers", [])) if first else 0
    best, mean = min(walls), statistics.mean(walls)
    tag = f"[{a.label}] " if a.label else ""
    mode = f"{a.jobs} concurrent" if a.jobs > 1 else "serial"
    line = (f"{tag}{os.path.basename(a.file)} ({dur:.0f}s audio, {mode}): "
            f"min {best:.2f}s  mean {mean:.2f}s  | RTF(min) {dur / best:.1f}x | words {n_words}")
    if a.diarize:
        line += f" | speakers {n_spk}"
    print(line)


if __name__ == "__main__":
    main()
