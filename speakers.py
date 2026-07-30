# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Speaker library — persistent voiceprint enrollment + recognition.

Diarization (``server._diarize``) produces one embedding per speaker (a pyannote
``community-1`` voiceprint) for each recording, but the labels (``SPEAKER_00``…) are
per-recording — the same person is a different label in the next file. This module stores
*named* voiceprints in SQLite and matches a recording's speaker embeddings against them by
cosine similarity, so a person enrolled once is recognized across later recordings.

Design notes:
  * **Stdlib only** (``sqlite3`` + ``math``): importable and unit-testable without torch /
    pyannote / numpy, and with no cold-start cost.
  * Cosine on ``community-1`` voiceprints measures ~0.92-0.98 for the same voice vs <=0.30 for
    different ones, so the default match threshold (``DEFAULT_THRESHOLD``) cleanly separates
    them. Raise it to cut false matches; lower it to tolerate more cross-condition variation.
  * A speaker can hold multiple enrollment samples; recognition compares against their
    **centroid** (mean voiceprint), which is more stable than any single sample.
  * Vectors are validated on the way in (non-empty, all finite): the diarizer already omits
    non-finite voiceprints, and a client-supplied vector must not poison the library.
  * A fresh short-lived connection per call keeps the store safe across the event loop thread
    and ``asyncio.to_thread`` workers without a shared-connection lock.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid

# Cosine similarity threshold above which a voiceprint is treated as the same speaker. The
# server overrides this from SPEAKER_IDENTIFY_THRESHOLD; this constant is the standalone default.
DEFAULT_THRESHOLD = 0.5

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
  id       TEXT PRIMARY KEY,
  name     TEXT NOT NULL,
  created  REAL NOT NULL,
  updated  REAL NOT NULL,
  meta     TEXT
);
CREATE TABLE IF NOT EXISTS voiceprints (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  dim        INTEGER NOT NULL,
  vec        TEXT NOT NULL,
  source     TEXT,
  created    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_vp_speaker ON voiceprints(speaker_id);
"""

# Minimum voiceprint dimensionality — guards against a caller sending a stray scalar/short list.
_MIN_DIM = 16


class SpeakerError(ValueError):
    """Bad input (unknown speaker, malformed vector). The server maps this to a 4xx."""


def _clean_vector(embedding) -> list:
    """Validate + coerce a client/diarizer embedding to a list[float]. Rejects an empty vector,
    a too-short one, or any non-finite value (NaN/±inf) — matching on garbage is worse than
    treating the speaker as unknown."""
    if not isinstance(embedding, (list, tuple)):
        raise SpeakerError("embedding must be an array of numbers")
    try:
        vec = [float(x) for x in embedding]
    except (TypeError, ValueError):
        raise SpeakerError("embedding must contain only numbers")
    if len(vec) < _MIN_DIM:
        raise SpeakerError(f"embedding too short ({len(vec)} < {_MIN_DIM} dims)")
    if not all(math.isfinite(x) for x in vec):
        raise SpeakerError("embedding contains non-finite values")
    return vec


def _cosine(a, b) -> float:
    """Cosine similarity of two equal-length vectors. Returns 0.0 for a zero-norm vector or a
    length mismatch (treated as no match) rather than raising."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _mean(vectors) -> list:
    """Element-wise mean of a non-empty list of equal-length vectors (a speaker's centroid)."""
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    return [s / n for s in acc]


class SpeakerLibrary:
    """SQLite-backed store of named voiceprints, with cosine recognition."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")  # so DELETE speaker cascades to its voiceprints
        return con

    # --- enrollment ---------------------------------------------------------------------------

    def enroll(self, name: str, embedding, source: str = None, meta: dict = None) -> dict:
        """Create a speaker from a first voiceprint. Returns the speaker record."""
        name = (name or "").strip()
        if not name:
            raise SpeakerError("name is required")
        vec = _clean_vector(embedding)
        sid = "spk_" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._connect() as con:
            con.execute(
                "INSERT INTO speakers (id, name, created, updated, meta) VALUES (?,?,?,?,?)",
                (sid, name, now, now, json.dumps(meta) if meta else None),
            )
            con.execute(
                "INSERT INTO voiceprints (speaker_id, dim, vec, source, created) VALUES (?,?,?,?,?)",
                (sid, len(vec), json.dumps(vec), source, now),
            )
        return self.get(sid)

    def add_sample(self, speaker_id: str, embedding, source: str = None) -> dict:
        """Append another voiceprint to an existing speaker (improves recognition robustness)."""
        vec = _clean_vector(embedding)
        now = time.time()
        with self._connect() as con:
            row = con.execute("SELECT id FROM speakers WHERE id = ?", (speaker_id,)).fetchone()
            if row is None:
                raise SpeakerError(f"unknown speaker: {speaker_id}")
            con.execute(
                "INSERT INTO voiceprints (speaker_id, dim, vec, source, created) VALUES (?,?,?,?,?)",
                (speaker_id, len(vec), json.dumps(vec), source, now),
            )
            con.execute("UPDATE speakers SET updated = ? WHERE id = ?", (now, speaker_id))
        return self.get(speaker_id)

    # --- read / delete ------------------------------------------------------------------------

    def list_speakers(self) -> list:
        """All speakers with their sample counts, newest-updated first (no vectors)."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT s.id, s.name, s.created, s.updated, s.meta, "
                "       COUNT(v.id) AS samples "
                "FROM speakers s LEFT JOIN voiceprints v ON v.speaker_id = s.id "
                "GROUP BY s.id ORDER BY s.updated DESC"
            ).fetchall()
        return [self._row_to_speaker(r) for r in rows]

    def get(self, speaker_id: str, with_centroid: bool = False) -> dict:
        """One speaker (or None). ``with_centroid`` adds its mean voiceprint."""
        with self._connect() as con:
            r = con.execute(
                "SELECT s.id, s.name, s.created, s.updated, s.meta, "
                "       COUNT(v.id) AS samples "
                "FROM speakers s LEFT JOIN voiceprints v ON v.speaker_id = s.id "
                "WHERE s.id = ? GROUP BY s.id",
                (speaker_id,),
            ).fetchone()
            if r is None:
                return None
            speaker = self._row_to_speaker(r)
            if with_centroid:
                speaker["centroid"] = self._centroid(con, speaker_id)
        return speaker

    def delete(self, speaker_id: str) -> bool:
        """Remove a speaker and (via cascade) its voiceprints. True if one was removed."""
        with self._connect() as con:
            cur = con.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
        return cur.rowcount > 0

    def count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]

    # --- recognition --------------------------------------------------------------------------

    def identify(self, embedding, threshold: float = None):
        """Best matching speaker for a voiceprint, or None if none clears ``threshold``.
        Returns ``{"speaker_id", "name", "score"}``. Compares against each speaker's centroid."""
        thr = DEFAULT_THRESHOLD if threshold is None else threshold
        try:
            vec = _clean_vector(embedding)
        except SpeakerError:
            return None  # an unusable probe can't match anything
        best = None
        with self._connect() as con:
            for r in con.execute("SELECT id, name FROM speakers").fetchall():
                centroid = self._centroid(con, r["id"])
                if not centroid:
                    continue
                score = _cosine(vec, centroid)
                if best is None or score > best["score"]:
                    best = {"speaker_id": r["id"], "name": r["name"], "score": round(score, 4)}
        if best is not None and best["score"] >= thr:
            return best
        return None

    def identify_many(self, embeddings: dict, threshold: float = None) -> dict:
        """Identify a whole recording's speakers: ``{label: vector}`` → ``{label: match|None}``."""
        return {label: self.identify(vec, threshold) for label, vec in (embeddings or {}).items()}

    # --- internals ----------------------------------------------------------------------------

    def _centroid(self, con: sqlite3.Connection, speaker_id: str):
        rows = con.execute(
            "SELECT vec FROM voiceprints WHERE speaker_id = ?", (speaker_id,)
        ).fetchall()
        vecs = [json.loads(r["vec"]) for r in rows]
        vecs = [v for v in vecs if v]
        if not vecs:
            return None
        dim = len(vecs[0])
        vecs = [v for v in vecs if len(v) == dim]  # ignore any odd-dim sample
        return _mean(vecs)

    @staticmethod
    def _row_to_speaker(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"],
            "name": r["name"],
            "samples": r["samples"],
            "created": r["created"],
            "updated": r["updated"],
            "meta": json.loads(r["meta"]) if r["meta"] else None,
        }
