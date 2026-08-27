# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Tamper-evident audit logging for SecRecorder (CMMC / NIST SP 800-171 AU-3.3.1, 3.3.2, 3.3.8).

SecRecorder is the system of record for CUI voice recordings passing through it, so this trail is
**on by default** — unlike an opt-in agent log, an operator has to explicitly turn it off
(``SECRECORDER_AUDIT_ENABLED=0``) rather than opt in.

Every recorded action — a transcription run, a speaker enrolled/updated/deleted, a summarization
request, an auth failure — becomes one append-only JSONL line, chained with a SHA-256 hash so any
insertion, deletion, or edit downstream is detectable (:func:`verify_chain`). Canonical field names
follow the suite's Spec B.1 (``ts``, ``type``, ``principal``, ``sourceIp``, ``target``, ``outcome``,
``detail``, ``prevHash``, ``hash``) — new to this component, so no legacy names to preserve.

**Metadata only (Spec B.3, absolute).** ``detail`` may hold counts, durations, sizes, flags, model
names, thresholds — never transcript text, a summary, a prompt, or audio/voiceprint bytes.
:func:`_scrub_detail` is a defense-in-depth backstop that drops any key that looks like it might
carry that content, even though call sites are not supposed to pass one.

Modeled on secagent's ``audit.py`` (chained JSONL, sorted-keys-JSON canonicalization, 0700/0600
at-rest hardening, a ``verify_chain`` function) — see that module for the sibling implementation.
Two deliberate differences here, both documented at the point of difference below: the genesis
hash is the literal string ``"GENESIS"`` (Spec B.2's chosen constant) rather than a zero hash, and
directory/file creation is deferred to the first write rather than done in the constructor, so
merely importing this module (e.g. ``python -c "import server"``, or pytest collection) never
touches the filesystem.

Depends only on the stdlib — importable without FastAPI or any ML dependency, matching this repo's
test convention (``test_audit.py`` runs offline).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

# Spec B.2: "genesis constant "GENESIS""" — the literal string, not a hash of anything.
GENESIS_HASH = "GENESIS"

# Keys stripped from `detail` before it is ever canonicalized, hashed, or written — a backstop for
# Spec B.3's metadata-only discipline. Call sites should never pass one of these in the first place;
# this only guards against a future mistake turning into a permanent, tamper-evident CUI leak.
_FORBIDDEN_DETAIL_KEYS = frozenset({
    "text", "transcript", "transcription", "prompt", "completion", "summary",
    "audio", "content", "embedding", "embeddings", "words", "segments",
})

# Default location: a `data/` directory next to this file, mirroring SPEAKER_DB's
# next-to-server.py convention. Overridable in full via SECRECORDER_AUDIT_PATH.
_DEFAULT_AUDIT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "audit.jsonl")

AUDIT_ENABLED = os.environ.get("SECRECORDER_AUDIT_ENABLED", "1").strip().lower() not in (
    "0", "false", "no", "off")
AUDIT_PATH = os.environ.get("SECRECORDER_AUDIT_PATH", "").strip() or _DEFAULT_AUDIT_PATH


def now_iso() -> str:
    """Current UTC time in the ISO-8601 shape this module's records use (Spec B.1 ``ts``).
    Public — the evidence bundle's ``generatedAt`` uses the same clock/format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical(record: dict[str, Any]) -> str:
    """Stable serialization for hashing (key order independent) — sorted-keys-JSON per Spec B.2."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def _hash_record(record_without_hash: dict[str, Any]) -> str:
    """SHA-256 over the canonicalized record (``prevHash`` is one of the canonicalized fields, so
    the chain links through it) — Spec B.2. ``record_without_hash`` must not contain a ``hash`` key."""
    return sha256(_canonical(record_without_hash).encode("utf-8")).hexdigest()


def _harden_path(path: str | Path, mode: int) -> None:
    """Best-effort ``chmod`` for at-rest protection. No-op on platforms without POSIX permissions."""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _scrub_detail(detail: dict[str, Any] | None) -> dict[str, Any]:
    """Drop any key that looks like it could carry transcript/audio/summary content (Spec B.3
    backstop). Prints a warning (never raises) if something was dropped, so a bug is visible rather
    than silently swallowed."""
    if not detail:
        return {}
    if not isinstance(detail, dict):
        print(f"AUDIT-WARN detail must be a dict, got {type(detail).__name__}; dropped", file=sys.stderr)
        return {}
    safe, dropped = {}, []
    for k, v in detail.items():
        if str(k).strip().lower() in _FORBIDDEN_DETAIL_KEYS:
            dropped.append(k)
            continue
        safe[k] = v
    if dropped:
        print(f"AUDIT-WARN dropped forbidden metadata key(s) from audit detail: {dropped}", file=sys.stderr)
    return safe


class AuditLogger:
    """Append-only, hash-chained JSONL audit logger.

    Writes are serialized with a lock; a write failure degrades to stderr and never raises into the
    caller (a hot-path request must never fail *because* logging it failed — Spec B.2). A disabled
    logger's :meth:`record` is a no-op returning ``None``.

    Directory creation, permission hardening, and reading the log's current tail hash are all
    deferred to the first :meth:`record` call (not the constructor) — see the module docstring.
    """

    def __init__(self, path: str | Path | None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._prev_hash = GENESIS_HASH
        self._ready = False

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            _harden_path(self._path.parent, 0o700)  # at-rest hardening (CMMC-2)
            self._prev_hash = _last_hash(self._path) or GENESIS_HASH
        self._ready = True

    def record(
        self,
        type_: str,
        *,
        principal: str | None = None,
        source_ip: str | None = None,
        target: dict[str, Any] | None = None,
        outcome: str = "ok",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append one audit record. Returns the written record, or ``None`` if disabled."""
        if not self.enabled:
            return None
        safe_detail = _scrub_detail(detail)
        with self._lock:
            self._ensure_ready()
            record: dict[str, Any] = {
                "ts": now_iso(),
                "type": type_,
                "principal": principal or "anonymous",
                "sourceIp": source_ip,
                "target": target or {},
                "outcome": outcome,
                "detail": safe_detail,
                "prevHash": self._prev_hash,
            }
            digest = _hash_record(record)
            record["hash"] = digest
            self._prev_hash = digest
            self._write(record)
            return record

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        try:
            if self._path is not None:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                _harden_path(self._path, 0o600)  # owner-only audit log (CMMC-2)
        except OSError as exc:  # never break the request on a logging failure
            print(f"AUDIT-ERROR could not write audit record: {exc}", file=sys.stderr)


def _last_hash(path: Path) -> str | None:
    """Return the hash of the last record in an existing log, to continue the chain across restarts."""
    if not path.exists():
        return None
    last = None
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
    except OSError:
        return None
    if not last:
        return None
    try:
        value = json.loads(last).get("hash")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def verify_chain(path: str | Path) -> tuple[bool, int, int | None]:
    """Validate a JSONL audit log's hash chain.

    Returns ``(ok, checked, broken_at_seq)``: ``checked`` is the number of records read;
    ``broken_at_seq`` is the 1-based ordinal of the first record that fails verification (a
    tampered/edited record, a missing hash, or a broken ``prevHash`` linkage), or ``None`` when
    ``ok``. A log that does not exist yet is treated as trivially valid (nothing to verify) rather
    than a failure — the file is created lazily on the first write.
    """
    p = Path(path)
    if not p.exists():
        return True, 0, None
    prev = GENESIS_HASH
    n = 0
    with open(p, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            n += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                return False, n, n
            stored = record.get("hash")
            if not isinstance(stored, str):
                return False, n, n
            if record.get("prevHash") != prev:
                return False, n, n
            recomputed = _hash_record({k: v for k, v in record.items() if k != "hash"})
            if recomputed != stored:
                return False, n, n
            prev = stored
    return True, n, None


def tail_records(path: str | Path, limit: int = 200) -> list[dict[str, Any]]:
    """The last ``limit`` records (oldest first), for an evidence bundle. Best-effort: a corrupt
    line is skipped rather than raising, since this is a read-only reporting path."""
    p = Path(path)
    if not p.exists():
        return []
    lines: list[str] = []
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(line)
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# Process-wide singleton, built from the environment — the convention the rest of this codebase
# uses (summarize.py's module-level `enabled`, server.py's module-level backend config). Its
# constructor does no I/O (see _ensure_ready above), so importing this module is always safe.
logger = AuditLogger(AUDIT_PATH, enabled=AUDIT_ENABLED)


def record(
    type_: str,
    *,
    principal: str | None = None,
    source_ip: str | None = None,
    target: dict[str, Any] | None = None,
    outcome: str = "ok",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Append one record via the process-wide logger. See :meth:`AuditLogger.record`."""
    return logger.record(type_, principal=principal, source_ip=source_ip, target=target,
                          outcome=outcome, detail=detail)


def status() -> dict[str, Any]:
    """Audit summary for /health."""
    return {"audit_enabled": AUDIT_ENABLED, **({"audit_path": AUDIT_PATH} if AUDIT_ENABLED else {})}
