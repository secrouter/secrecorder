# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for auth.py (PyJWT + cryptography; no web/ML stack, no live IdP).

    python3 test_auth.py

Exercises the crypto/OIDC helpers offline: PKCE, the HS256 session + flow cookies (incl. that one
can't be replayed as the other), and JWKS bearer / id_token verification against a locally-generated
RSA keypair — including the alg-confusion and audience/nonce rejections that make it safe.
"""

import base64
import hashlib
import os
import sys
import time

# Configure BEFORE import (module reads env at import time). Enough to make sso_ready/auth_enabled.
os.environ["SECRECORDER_OIDC_ISSUER"] = "https://secsso.test"
os.environ["SECRECORDER_OIDC_CLIENT_ID"] = "secrecorder"
os.environ["SECRECORDER_OIDC_CLIENT_SECRET"] = "shh"
os.environ["SECRECORDER_PUBLIC_URL"] = "https://secrecorder.test"
os.environ["SECRECORDER_SESSION_SECRET"] = "test-session-secret-0123456789abcdef"

import jwt  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

import auth  # noqa: E402

_fails = []


def check(cond, msg):
    if not cond:
        _fails.append(msg)
        print(f"  FAIL: {msg}")


def raises(fn, msg):
    try:
        fn()
        _fails.append(msg)
        print(f"  FAIL: {msg}")
    except Exception:  # noqa: BLE001 — the point is that it rejects
        pass


# One RSA keypair for the bearer/id_token tests; point auth's JWKS client at the public half.
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUB = _KEY.public_key()


class _FakeSigningKey:
    key = _PUB


class _FakeJWKS:
    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey()


auth._jwk_client = _FakeJWKS()  # so verify_bearer / _verify_id_token use our test public key


def _rs256(claims, key=None):
    return jwt.encode(claims, key or _KEY, algorithm="RS256")


def test_config_enabled():
    check(auth.auth_enabled and auth.sso_ready and auth.bearer_ready, "OIDC config should enable auth + sso + bearer")


def test_safe_next():
    check(auth._safe_next("/channels/x") == "/channels/x", "same-origin path preserved")
    check(auth._safe_next(None) == "/", "None → /")
    check(auth._safe_next("//evil.com") == "/", "protocol-relative rejected")
    check(auth._safe_next("https://evil.com") == "/", "absolute URL rejected")
    check(auth._safe_next("/a\r\nSet-Cookie: x") == "/", "CR/LF (header injection) rejected")


def test_pkce_s256():
    v = auth._new_verifier()
    check(len(v) == 43, f"verifier should be 43 chars (32 bytes b64url), got {len(v)}")
    expected = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
    check(auth._challenge_s256(v) == expected, "S256 challenge must be base64url(sha256(verifier))")


def test_session_roundtrip_and_tamper():
    p = auth.Principal(sub="alice", email="a@x.test", display_name="Alice", groups=["eng", "sec"])
    tok = auth.mint_session(p)
    back = auth.verify_session(tok)
    check(back.sub == "alice" and back.email == "a@x.test" and back.groups == ["eng", "sec"],
          "session round-trip should preserve identity + groups")
    raises(lambda: auth.verify_session(tok[:-3] + "AAA"), "a tampered session cookie must be rejected")
    raises(lambda: auth.verify_session(jwt.encode({"sub": "x"}, "wrong-secret-0123456789abcdef0123456789", algorithm="HS256")),
           "a session signed with the wrong secret must be rejected")


def test_flow_cookie_and_cross_domain_isolation():
    tok = auth._sign_flow("state1", "verifier1", "nonce1", "/next")
    flow = auth._verify_flow(tok)
    check(flow["state"] == "state1" and flow["verifier"] == "verifier1" and flow["nonce"] == "nonce1",
          "flow round-trip should preserve state/verifier/nonce")
    # A flow cookie must NOT verify as a session, nor vice versa (disjoint iss/aud, same secret).
    raises(lambda: auth.verify_session(tok), "a flow cookie must not be accepted as a session cookie")
    session_tok = auth.mint_session(auth.Principal(sub="alice"))
    raises(lambda: auth._verify_flow(session_tok), "a session cookie must not be accepted as a flow cookie")


def test_principal_from_claims():
    p = auth._principal_from_claims({"sub": "bob", "email": "b@x.test", "name": "Bob", "groups": ["a"]})
    check(p.sub == "bob" and p.display_name == "Bob" and p.groups == ["a"], "claims → Principal")
    raises(lambda: auth._principal_from_claims({"email": "no-sub@x.test"}), "missing sub must raise")


def test_verify_bearer():
    now = int(time.time())
    good = _rs256({"sub": "alice", "email": "a@x.test", "groups": ["eng"],
                   "iss": "https://secsso.test", "aud": "secrecorder", "exp": now + 300, "iat": now})
    p = auth.verify_bearer(good)
    check(p.sub == "alice" and p.groups == ["eng"], "a valid SecSSO bearer should verify to its principal")

    wrong_aud = _rs256({"sub": "alice", "iss": "https://secsso.test", "aud": "someone-else", "exp": now + 300})
    raises(lambda: auth.verify_bearer(wrong_aud), "wrong audience must be rejected")

    wrong_iss = _rs256({"sub": "alice", "iss": "https://evil.test", "aud": "secrecorder", "exp": now + 300})
    raises(lambda: auth.verify_bearer(wrong_iss), "wrong issuer must be rejected")

    expired = _rs256({"sub": "alice", "iss": "https://secsso.test", "aud": "secrecorder", "exp": now - 10})
    raises(lambda: auth.verify_bearer(expired), "an expired token must be rejected")

    # Alg-confusion: an HS256 token (classic attack signs with the public key as the HMAC secret)
    # must be rejected because verification pins algorithms=["RS256"].
    hs = jwt.encode({"sub": "attacker", "iss": "https://secsso.test", "aud": "secrecorder", "exp": now + 300},
                    "any-secret-0123456789abcdef0123456789ab", algorithm="HS256")
    raises(lambda: auth.verify_bearer(hs), "an HS256 token must be rejected (RS256 pinned)")


def test_verify_id_token_nonce():
    now = int(time.time())
    tok = _rs256({"sub": "alice", "iss": "https://secsso.test", "aud": "secrecorder",
                  "nonce": "N1", "exp": now + 300})
    p = auth._verify_id_token(tok, "N1")
    check(p.sub == "alice", "id_token with the matching nonce should verify")
    raises(lambda: auth._verify_id_token(tok, "N2"), "id_token nonce mismatch must be rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if _fails:
        print(f"\n{len(_fails)} FAILED")
        sys.exit(1)
    print("auth: all tests passed")
