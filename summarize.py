# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Optional transcript summarization for SecRecorder — via any OpenAI-compatible chat endpoint.

After a transcription (or on demand), SecRecorder can send the transcript to a chat/completions
endpoint and attach a short ``summary``. **Governed by default:** the request carries
``X-Sec-Acting-User`` and is pointed at **SecRouter** so the summarization LLM call is attributed,
budgeted, egress-controlled, and audited as the recording's owner — the same pattern SecChat's
assistant path uses. The endpoint is fully configurable, so it can instead hit **SecLLM** directly
or any arbitrary OpenAI-compatible server.

**Off by default.** Enabled only when ``SECRECORDER_SUMMARIZE_ENABLED`` is truthy AND
``SECRECORDER_SUMMARIZE_ENDPOINT`` (an OpenAI-compatible base URL, e.g. ``…/v1``) is set. Uses the
stdlib (``urllib``) — no new HTTP dependency. A summarization failure never fails the transcription:
the transcript is returned with a ``summary_error`` instead (graceful degradation, like diarization).
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request

ENABLED_FLAG = os.environ.get("SECRECORDER_SUMMARIZE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
# OpenAI-compatible base URL (chat/completions is appended). Default routes through SecRouter for
# governance; secdeploy wires this to the deployment's SecRouter automatically. Empty ⇒ disabled.
ENDPOINT = os.environ.get("SECRECORDER_SUMMARIZE_ENDPOINT", "").strip().rstrip("/")
MODEL = os.environ.get("SECRECORDER_SUMMARIZE_MODEL", "auto").strip()
# Optional bearer for the LLM endpoint (a SecRouter/SecLLM service token). Governance still rides on
# X-Sec-Acting-User; this only authenticates the machine call when the endpoint requires it.
API_KEY = os.environ.get("SECRECORDER_SUMMARIZE_API_KEY", "").strip()
# Fallback attribution when the caller is anonymous (auth off): the service's own identity.
DEFAULT_ACTING_USER = os.environ.get("SECRECORDER_SUMMARIZE_ACTING_USER", "svc-secrecorder").strip()
PROMPT = os.environ.get(
    "SECRECORDER_SUMMARIZE_PROMPT",
    "You are a precise meeting summarizer. Given a transcript, produce a concise summary: a short "
    "overview, the key points as bullets, any decisions, and any action items with owners when "
    "stated. Be faithful to the transcript and never invent details.",
).strip()
TIMEOUT = float(os.environ.get("SECRECORDER_SUMMARIZE_TIMEOUT", "60"))
# Cap the transcript sent to the model (characters) so a multi-hour recording can't blow the
# model's context or the request size. Long transcripts are truncated with a marker.
MAX_CHARS = int(os.environ.get("SECRECORDER_SUMMARIZE_MAX_CHARS", "48000"))

enabled = ENABLED_FLAG and bool(ENDPOINT)


class SummarizeError(Exception):
    """A summarization call failed (endpoint unreachable, non-2xx, or a malformed response)."""


def _summarize_sync(text: str, acting_user: str | None) -> str:
    body = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n\n[transcript truncated]"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": body}],
        "stream": False,
        "temperature": 0.2,
    }).encode("utf-8")
    headers = {
        "content-type": "application/json",
        # Attribute/govern the call as the recording's owner (never SecRecorder's own identity),
        # exactly like SecChat's assistant path — this is what SecRouter reads to apply per-user
        # policy/budget/audit. Falls back to the service identity when the caller is anonymous.
        "X-Sec-Acting-User": acting_user or DEFAULT_ACTING_USER,
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{ENDPOINT}/chat/completions", data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — surface a clean reason, never a raw stack
        raise SummarizeError(str(e)) from e
    try:
        content = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise SummarizeError("malformed chat completion response") from e
    if not isinstance(content, str) or not content.strip():
        raise SummarizeError("empty summary")
    return content.strip()


async def summarize(text: str, acting_user: str | None = None) -> str:
    """Summarize ``text`` via the configured chat endpoint, attributed to ``acting_user``. Runs the
    blocking HTTP call off the event loop. Raises :class:`SummarizeError` on any failure."""
    if not enabled:
        raise SummarizeError("summarization is not configured")
    if not text or not text.strip():
        raise SummarizeError("nothing to summarize")
    return await asyncio.to_thread(_summarize_sync, text, acting_user)


def status() -> dict:
    """Summarization summary for /health."""
    return {"summarize_enabled": enabled, **({"summarize_endpoint": ENDPOINT, "summarize_model": MODEL} if enabled else {})}
