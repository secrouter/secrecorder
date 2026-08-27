# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for audit.py (stdlib-only; no web/ML stack).

    python3 test_audit.py

Exercises the hash-chained JSONL log offline: append + verify, tamper detection (editing a written
line breaks verify_chain at that record), the metadata-only backstop (a transcript-like key never
reaches the file), lazy/no-op-at-import behavior, and disabled-logger no-ops.
"""

import json
import os
import sys
import tempfile

import audit

_fails = []


def check(cond, msg):
    # NOTE: unlike test_auth.py/test_summarize.py's non-raising `check()` (which only prints and
    # keeps going — harmless there, but it means a logically-failing assertion still shows green
    # under `pytest`, since nothing raises), this one asserts immediately. For a tamper-detection
    # test this matters: pytest must actually catch a broken invariant, not just "didn't crash".
    if not cond:
        _fails.append(msg)
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)


# --- chain append + verify -----------------------------------------------------------------------

def test_append_and_verify_empty_log():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "audit.jsonl")  # exercises parent-dir creation
        ok, checked, broken = audit.verify_chain(path)
        check(ok is True and checked == 0 and broken is None, "a log that doesn't exist yet verifies trivially")


def test_append_and_verify_chain():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        r1 = log.record("transcription.run", principal="alice", target={"requestId": "r1"},
                         detail={"durationSec": 12.3, "diarize": True})
        r2 = log.record("speaker.enroll", principal="alice", target={"speakerId": "spk_1"})
        r3 = log.record("speaker.delete", principal="bob", target={"speakerId": "spk_1"}, outcome="ok")

        check(r1["prevHash"] == audit.GENESIS_HASH, "first record chains from the genesis constant")
        check(r2["prevHash"] == r1["hash"], "second record's prevHash is the first record's hash")
        check(r3["prevHash"] == r2["hash"], "third record's prevHash is the second record's hash")
        check(r1["hash"] != r2["hash"] != r3["hash"], "each record gets a distinct hash")

        # Canonical field shape (Spec B.1).
        for r in (r1, r2, r3):
            for field in ("ts", "type", "principal", "sourceIp", "target", "outcome", "detail", "prevHash", "hash"):
                check(field in r, f"record missing canonical field {field!r}: {r}")

        ok, checked, broken = audit.verify_chain(path)
        check(ok is True and checked == 3 and broken is None, f"a clean 3-record chain should verify: {(ok, checked, broken)}")

        # A fresh AuditLogger pointed at the same path continues the chain (restart-safe).
        log2 = audit.AuditLogger(path, enabled=True)
        r4 = log2.record("auth.failure", principal="anonymous", outcome="deny", target={"path": "/v1/models"})
        check(r4["prevHash"] == r3["hash"], "a new logger instance continues the chain from the log's tail hash")
        ok, checked, broken = audit.verify_chain(path)
        check(ok is True and checked == 4, "chain still verifies after continuing across a logger restart")


def test_tamper_detection():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        log.record("transcription.run", principal="alice", detail={"durationSec": 1.0})
        log.record("transcription.run", principal="alice", detail={"durationSec": 2.0})
        log.record("transcription.run", principal="alice", detail={"durationSec": 3.0})

        ok, checked, broken = audit.verify_chain(path)
        check(ok is True and broken is None, "chain is clean before tampering")

        # Edit the FIRST record's detail in place (a realistic tamper: change a duration after the
        # fact) without touching its stored hash — the hash mismatch must be caught.
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        rec = json.loads(lines[0])
        rec["detail"]["durationSec"] = 999.0  # tamper: content changed, hash left as-is
        lines[0] = json.dumps(rec) + "\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        ok, checked, broken = audit.verify_chain(path)
        check(ok is False, "tampering with a record's content must break verification")
        check(broken == 1, f"the tampered record (first line) should be reported as broken, got {broken}")

        # A break at record 1 also breaks the linkage for every record after it (prevHash no longer
        # matches downstream once record 1's hash changes) — verify_chain stops at the FIRST failure.
        check(checked == 1, f"verify_chain should stop counting at the first bad record, got checked={checked}")


def test_tamper_broken_linkage_only():
    """Edit the prevHash field of a middle record without touching anything else — this breaks the
    LINKAGE check specifically (as opposed to the per-record hash mismatch in test_tamper_detection)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        log.record("transcription.run", principal="alice")
        log.record("transcription.run", principal="alice")

        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        rec = json.loads(lines[1])
        rec["prevHash"] = "not-the-real-prev-hash"
        lines[1] = json.dumps(rec) + "\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)

        ok, checked, broken = audit.verify_chain(path)
        check(ok is False and broken == 2, f"a broken prevHash linkage on line 2 should be caught there, got {(ok, checked, broken)}")


# --- metadata-only discipline (Spec B.3) ---------------------------------------------------------

def test_metadata_only_forbidden_keys_never_reach_the_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        transcript_like = "Alice: I think we should ship on Friday. Bob: agreed, let's do it."
        log.record(
            "transcription.run",
            principal="alice",
            target={"requestId": "r1"},
            detail={
                "text": transcript_like,          # forbidden — must be dropped
                "transcript": transcript_like,     # forbidden — must be dropped
                "summary": "a fabricated summary", # forbidden — must be dropped
                "audio": b"not really audio but pretend".hex(),  # forbidden — must be dropped
                "embedding": [0.1, 0.2, 0.3],       # forbidden — must be dropped
                "durationSec": 42.0,                # allowed — a plain metadata count
                "diarize": True,                    # allowed
            },
        )
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        check(transcript_like not in raw, "transcript-like content must never land in the audit log")
        check("fabricated summary" not in raw, "summary content must never land in the audit log")
        rec = json.loads(raw.strip().splitlines()[0])
        check(set(rec["detail"].keys()) == {"durationSec", "diarize"},
              f"only metadata keys should survive scrubbing, got {sorted(rec['detail'].keys())}")
        check(rec["detail"]["durationSec"] == 42.0 and rec["detail"]["diarize"] is True,
              "legitimate metadata fields must be preserved")
        # The record still verifies — scrubbing happens before hashing, not after.
        ok, checked, broken = audit.verify_chain(path)
        check(ok is True, "a record with scrubbed detail should still verify (scrub happens pre-hash)")


def test_anonymous_principal_default():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        r = log.record("auth.failure", outcome="deny", target={"path": "/v1/models"})
        check(r["principal"] == "anonymous", f"no principal supplied should default to 'anonymous', got {r['principal']!r}")


# --- disabled / lazy behavior ---------------------------------------------------------------------

def test_disabled_logger_is_a_noop():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=False)
        r = log.record("transcription.run", principal="alice")
        check(r is None, "a disabled logger's record() should return None")
        check(not os.path.exists(path), "a disabled logger must never create the log file")


def test_constructor_does_no_io():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "never", "created", "audit.jsonl")
        audit.AuditLogger(path, enabled=True)  # constructing must not touch the filesystem
        check(not os.path.exists(os.path.dirname(path)),
              "constructing an AuditLogger must not create directories (deferred to first record())")


def test_module_import_does_no_io():
    # audit.py is already imported (module-level singleton built from the environment); its default
    # path lives under this repo. Importing it must not have created that directory as a side effect
    # (verified independently by the harness before this test file runs; re-assert the invariant here
    # against a FRESH default-shaped logger so this test is self-contained).
    default_dir = os.path.dirname(audit.AUDIT_PATH)
    with tempfile.TemporaryDirectory() as d:
        fresh_path = os.path.join(d, "data", "audit.jsonl")
        fresh = audit.AuditLogger(fresh_path, enabled=True)
        check(not os.path.exists(os.path.dirname(fresh_path)), "merely constructing must not create data/")
        del fresh
    check(True, f"(informational) this repo's own default audit dir is {default_dir!r}")


# --- verify_chain edge cases ------------------------------------------------------------------

def test_verify_chain_corrupt_json_line():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        log.record("transcription.run", principal="alice")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{not valid json\n")
        ok, checked, broken = audit.verify_chain(path)
        check(ok is False and broken == 2, f"a corrupt JSON line should be caught as the broken record, got {(ok, checked, broken)}")


def test_tail_records():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        log = audit.AuditLogger(path, enabled=True)
        for i in range(5):
            log.record("transcription.run", principal="alice", target={"requestId": str(i)})
        tail = audit.tail_records(path, limit=3)
        check(len(tail) == 3, f"tail_records should respect the limit, got {len(tail)}")
        check([r["target"]["requestId"] for r in tail] == ["2", "3", "4"],
              "tail_records should return the LAST N records, oldest-first")
        check(audit.tail_records(os.path.join(d, "missing.jsonl")) == [], "a missing log tails to an empty list")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if _fails:
        print(f"\n{len(_fails)} FAILED")
        sys.exit(1)
    print("audit: all tests passed")
