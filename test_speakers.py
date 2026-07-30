# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the speaker library (stdlib-only; runs on py3.9+ without the ML stack).

    python3 test_speakers.py
"""

import sys
import tempfile
import os

from speakers import SpeakerLibrary, SpeakerError, _cosine

ONES = [1.0] * 32
NEAR = ([1.0] * 31) + [0.9]                                # ~1.0 cosine with ONES
HALF = ([1.0] * 16) + ([0.0] * 16)                         # ~0.707 cosine with ONES
ORTHO = [1.0 if i % 2 == 0 else -1.0 for i in range(32)]   # ~0.0 cosine with ONES

_fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        _fails.append(msg)


def raises(fn, msg):
    try:
        fn()
        check(False, msg + " (expected SpeakerError)")
    except SpeakerError:
        check(True, msg)


def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


def main():
    print("cosine:")
    check(approx(_cosine(ONES, ONES), 1.0), "identical vectors -> 1.0")
    check(approx(_cosine(ONES, ORTHO), 0.0), "orthogonal vectors -> 0.0")
    check(approx(_cosine(ONES, HALF), 0.7071), "half-overlap -> ~0.707")
    check(_cosine(ONES, [0.0] * 32) == 0.0, "zero-norm vector -> 0.0 (no divide-by-zero)")
    check(_cosine(ONES, [1.0] * 8) == 0.0, "length mismatch -> 0.0")

    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "nested", "speakers.db")  # also exercises parent-dir creation
        lib = SpeakerLibrary(db)

        print("enroll + read:")
        alice = lib.enroll("Alice", ONES, source="rec1")
        check(alice["id"].startswith("spk_"), "enroll returns spk_ id")
        check(alice["name"] == "Alice" and alice["samples"] == 1, "name + 1 sample")
        check(lib.get(alice["id"])["name"] == "Alice", "get by id")
        check(len(lib.list_speakers()) == 1 and lib.count() == 1, "listed + counted")

        print("validation:")
        raises(lambda: lib.enroll("", ONES), "empty name rejected")
        raises(lambda: lib.enroll("Short", [1.0, 2.0, 3.0]), "too-short vector rejected")
        raises(lambda: lib.enroll("Nan", [float("nan")] * 32), "non-finite vector rejected")
        raises(lambda: lib.enroll("Str", "not-a-vector"), "non-array embedding rejected")
        raises(lambda: lib.add_sample("spk_nope", ONES), "add_sample to unknown speaker rejected")

        print("identify:")
        m = lib.identify(NEAR)
        check(m is not None and m["name"] == "Alice" and m["score"] >= 0.99, "near vector -> Alice")
        check(lib.identify(ORTHO) is None, "orthogonal vector -> no match")
        check(lib.identify(HALF, threshold=0.5) is not None, "0.707 clears threshold 0.5")
        check(lib.identify(HALF, threshold=0.9) is None, "0.707 fails threshold 0.9")
        check(lib.identify([0.0] * 32) is None, "zero vector -> no match")

        print("centroid (multi-sample):")
        lib.add_sample(alice["id"], HALF, source="rec2")
        check(lib.get(alice["id"])["samples"] == 2, "second sample recorded")
        check(lib.identify(ONES) is not None, "ONES still matches the ONES+HALF centroid")

        print("two speakers + identify_many:")
        bob = lib.enroll("Bob", ORTHO)
        many = lib.identify_many({"SPEAKER_00": NEAR, "SPEAKER_01": ORTHO})
        check(many["SPEAKER_00"]["name"] == "Alice", "SPEAKER_00 -> Alice")
        check(many["SPEAKER_01"]["name"] == "Bob", "SPEAKER_01 -> Bob")

        print("delete (cascade):")
        check(lib.delete(alice["id"]) is True, "delete returns True")
        check(lib.get(alice["id"]) is None, "deleted speaker gone")
        check(lib.delete("spk_nope") is False, "delete unknown -> False")
        # Alice's voiceprints must be gone too, else NEAR would still match her centroid.
        check(lib.identify(NEAR) is None, "NEAR no longer matches (voiceprints cascaded)")

        print("persistence (reopen):")
        del lib
        lib2 = SpeakerLibrary(db)
        names = [s["name"] for s in lib2.list_speakers()]
        check(names == ["Bob"], "data survives reopen")
        check(lib2.identify(ORTHO)["name"] == "Bob", "identify works after reopen")

    if _fails:
        print("\n%d FAILED" % len(_fails))
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
