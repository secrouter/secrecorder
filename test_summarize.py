# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for summarize.py (stdlib-only; no ML stack, no live LLM).

    python3 test_summarize.py

The chat/completions HTTP call is monkeypatched, so these run offline and assert the request shape
(governance header, endpoint, truncation) and the response/error handling — never a real model.
"""

import json
import os
import sys

# Configure BEFORE import (module reads env at import time, matching server.py's convention).
os.environ["SECRECORDER_SUMMARIZE_ENABLED"] = "1"
os.environ["SECRECORDER_SUMMARIZE_ENDPOINT"] = "http://secrouter.test/v1"
os.environ["SECRECORDER_SUMMARIZE_MODEL"] = "auto"
os.environ["SECRECORDER_SUMMARIZE_MAX_CHARS"] = "50"

import summarize  # noqa: E402

_fails = []


def check(cond, msg):
    if not cond:
        _fails.append(msg)
        print(f"  FAIL: {msg}")


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_captured = {}


def _install_fake(completion=None, exc=None):
    def fake_urlopen(req, timeout=None):
        _captured.clear()
        _captured["url"] = req.full_url
        _captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        _captured["body"] = json.loads(req.data.decode())
        if exc is not None:
            raise exc
        payload = completion if completion is not None else {"choices": [{"message": {"content": "A short summary."}}]}
        return _FakeResp(json.dumps(payload).encode())

    summarize.urllib.request.urlopen = fake_urlopen


def test_enabled():
    check(summarize.enabled is True, "enabled should be True with flag + endpoint set")


def test_happy_path_and_governance_header():
    _install_fake()
    out = summarize._summarize_sync("Alice: hello. Bob: hi.", "alice")
    check(out == "A short summary.", f"expected summary content, got {out!r}")
    check(_captured["url"] == "http://secrouter.test/v1/chat/completions", f"bad url {_captured['url']!r}")
    # Governance: the call is attributed to the acting user via X-Sec-Acting-User (SecRouter reads it).
    check(_captured["headers"].get("x-sec-acting-user") == "alice", f"missing/wrong acting-user header: {_captured['headers']}")
    check(_captured["body"]["model"] == "auto", "model should pass through")
    roles = [m["role"] for m in _captured["body"]["messages"]]
    check(roles == ["system", "user"], f"expected system+user messages, got {roles}")
    check(_captured["body"]["stream"] is False, "must be non-streaming")


def test_default_acting_user_when_anonymous():
    _install_fake()
    summarize._summarize_sync("text", None)
    check(_captured["headers"].get("x-sec-acting-user") == "svc-secrecorder",
          f"anonymous should fall back to the service identity, got {_captured['headers'].get('x-sec-acting-user')!r}")


def test_truncation():
    _install_fake()
    long = "x" * 200  # MAX_CHARS=50
    summarize._summarize_sync(long, "alice")
    sent = _captured["body"]["messages"][1]["content"]
    check(sent.startswith("x" * 50) and "[transcript truncated]" in sent,
          "over-long transcript should be truncated with a marker")
    check(len(sent) < 200, "truncated content should be shorter than the input")


def test_error_on_transport_failure():
    _install_fake(exc=OSError("connection refused"))
    try:
        summarize._summarize_sync("text", "alice")
        _fails.append("expected SummarizeError on transport failure")
        print("  FAIL: expected SummarizeError on transport failure")
    except summarize.SummarizeError:
        pass


def test_error_on_malformed_response():
    _install_fake(completion={"nope": True})
    try:
        summarize._summarize_sync("text", "alice")
        _fails.append("expected SummarizeError on malformed response")
        print("  FAIL: expected SummarizeError on malformed response")
    except summarize.SummarizeError:
        pass


def test_error_on_empty_summary():
    _install_fake(completion={"choices": [{"message": {"content": "   "}}]})
    try:
        summarize._summarize_sync("text", "alice")
        _fails.append("expected SummarizeError on empty summary")
        print("  FAIL: expected SummarizeError on empty summary")
    except summarize.SummarizeError:
        pass


def test_status():
    st = summarize.status()
    check(st["summarize_enabled"] is True and st["summarize_endpoint"] == "http://secrouter.test/v1",
          f"status should reflect config: {st}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if _fails:
        print(f"\n{len(_fails)} FAILED")
        sys.exit(1)
    print("summarize: all tests passed")
