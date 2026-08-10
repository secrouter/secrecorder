# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Optional SSO auth for SecRecorder — SecSSO (OIDC) integration, off by default.

Two ways in, both validated against the SAME SecSSO issuer, so a locked-down deployment can front
the transcription API and the built-in web UI with the suite's single sign-on:

  * **Bearer JWT** — a programmatic/service client presents ``Authorization: Bearer <token>`` and
    the token is verified against SecSSO's published JWKS (RS256, issuer, audience). This is the
    machine-to-machine path (e.g. SecChat calling SecRecorder on a user's behalf).
  * **Browser login (BFF)** — the built-in web UI logs in through SecSSO with the Authorization
    Code + PKCE flow run entirely server-side; the browser's only credential is an httpOnly
    ``secrecorder_session`` cookie (it never sees an OIDC token). Mirrors SecChat's login BFF.

**Off by default.** With no OIDC env set, ``auth_enabled`` is False and every request passes through
exactly as before — SecRecorder stays an open service. Auth turns on the moment
``SECRECORDER_OIDC_ISSUER`` + ``SECRECORDER_OIDC_AUDIENCE`` are configured (bearer), and the browser
login additionally needs the confidential-client secret + this service's public URL + a session
secret. ``/health`` always stays open (liveness); everything under ``/v1/`` requires a valid
principal once auth is on; the UI at ``/`` bounces to ``/auth/login`` instead of 401ing a browser.

Depends only on **PyJWT[crypto]** + the stdlib (urllib for discovery/token exchange) — no other new
runtime dependency, matching the rest of the suite's supply-chain-conscious posture.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import jwt
# NOTE: FastAPI/Starlette are imported lazily inside install() (the only web-framework-dependent
# part), so this module's crypto/OIDC helpers import with just PyJWT + the stdlib — the unit tests
# exercise them without pulling the web (or ML) stack, matching this repo's test convention.

# ── Config (env) ──────────────────────────────────────────────────────────────────────────────
ISSUER = os.environ.get("SECRECORDER_OIDC_ISSUER", "").strip().rstrip("/")
CLIENT_ID = os.environ.get("SECRECORDER_OIDC_CLIENT_ID", "").strip()
# The token audience to require. Defaults to the client id (Authentik mints id_tokens with
# aud == client_id). Set explicitly only if your access tokens carry a different audience.
AUDIENCE = os.environ.get("SECRECORDER_OIDC_AUDIENCE", "").strip() or CLIENT_ID
CLIENT_SECRET = os.environ.get("SECRECORDER_OIDC_CLIENT_SECRET", "").strip()
PUBLIC_URL = os.environ.get("SECRECORDER_PUBLIC_URL", "").strip().rstrip("/")
SESSION_SECRET = os.environ.get("SECRECORDER_SESSION_SECRET", "").strip()
SESSION_TTL = int(os.environ.get("SECRECORDER_SESSION_TTL", "43200"))  # 12h
# Optional explicit endpoints — skip OIDC discovery (useful in air-gapped setups that don't expose
# /.well-known, or to pin them). When unset they're discovered from the issuer on first use.
JWKS_URL = os.environ.get("SECRECORDER_OIDC_JWKS_URL", "").strip()
AUTHORIZE_URL = os.environ.get("SECRECORDER_OIDC_AUTHORIZE_URL", "").strip()
TOKEN_URL = os.environ.get("SECRECORDER_OIDC_TOKEN_URL", "").strip()

# Bearer validation needs only the issuer + audience. The browser login (BFF) additionally needs
# the confidential-client secret, this service's own public URL (to build the redirect_uri), and a
# session-signing secret. Either capability being present turns enforcement on.
bearer_ready = bool(ISSUER and AUDIENCE)
sso_ready = bool(ISSUER and CLIENT_ID and CLIENT_SECRET and PUBLIC_URL and SESSION_SECRET)
auth_enabled = bearer_ready or sso_ready

SESSION_COOKIE = "secrecorder_session"
FLOW_COOKIE = "secrecorder_oidc_flow"
FLOW_TTL_SECONDS = 600  # 10 min — one IdP round-trip, no longer
SCOPE = "openid profile email groups"
# Own iss/aud for the HS256 cookies, DISJOINT from each other so a captured flow cookie can never
# be replayed as a session cookie (or vice versa) even though both are signed with SESSION_SECRET.
SESSION_ISS = "secrecorder-session"
FLOW_ISS = "secrecorder-oidc-flow"


@dataclass
class Principal:
    """The authenticated caller — the same shape SecChat's Principal carries, reduced to what
    SecRecorder needs (identity + groups + the per-user attribution the summarizer forwards)."""

    sub: str
    email: str | None = None
    display_name: str | None = None
    groups: list[str] = field(default_factory=list)


# ── OIDC discovery (cached per process) ─────────────────────────────────────────────────────────
_discovery: dict | None = None


def _discover() -> dict:
    """Fetch + cache ``<issuer>/.well-known/openid-configuration``. An OIDC discovery document is
    effectively static, so one fetch per process is plenty. Only called for values not pinned via
    the explicit *_URL env vars above."""
    global _discovery
    if _discovery is None:
        with urllib.request.urlopen(f"{ISSUER}/.well-known/openid-configuration", timeout=10) as r:
            _discovery = json.loads(r.read().decode("utf-8"))
    return _discovery


def _jwks_url() -> str:
    return JWKS_URL or _discover()["jwks_uri"]


def _authorize_url() -> str:
    return AUTHORIZE_URL or _discover()["authorization_endpoint"]


def _token_url() -> str:
    return TOKEN_URL or _discover()["token_endpoint"]


# ── Bearer JWT (RS256, JWKS) ────────────────────────────────────────────────────────────────────
_jwk_client: jwt.PyJWKClient | None = None


def _jwks() -> jwt.PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = jwt.PyJWKClient(_jwks_url())  # caches keys; refetches only on a kid miss
    return _jwk_client


def _principal_from_claims(claims: dict) -> Principal:
    sub = claims.get("sub")
    if not sub:
        raise jwt.InvalidTokenError("token missing 'sub'")
    groups = claims.get("groups")
    return Principal(
        sub=str(sub),
        email=claims.get("email") if isinstance(claims.get("email"), str) else None,
        display_name=claims.get("name") if isinstance(claims.get("name"), str) else None,
        groups=[str(g) for g in groups] if isinstance(groups, list) else [],
    )


def verify_bearer(token: str) -> Principal:
    """Verify a SecSSO access/id token against the published JWKS: RS256 signature (never an
    attacker-forgeable alg), issuer, audience, expiry. Raises ``jwt.PyJWTError`` on any failure."""
    key = _jwks().get_signing_key_from_jwt(token).key
    claims = jwt.decode(token, key, algorithms=["RS256"], issuer=ISSUER, audience=AUDIENCE)
    return _principal_from_claims(claims)


# ── Session + flow cookies (HS256, own trust domain) ────────────────────────────────────────────
def mint_session(p: Principal) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": p.sub, "email": p.email, "name": p.display_name, "groups": p.groups,
         "iss": SESSION_ISS, "aud": SESSION_ISS, "iat": now, "exp": now + SESSION_TTL},
        SESSION_SECRET, algorithm="HS256",
    )


def verify_session(token: str) -> Principal:
    claims = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"], issuer=SESSION_ISS, audience=SESSION_ISS)
    return _principal_from_claims(claims)


def _sign_flow(state: str, verifier: str, nonce: str, nxt: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {"state": state, "verifier": verifier, "nonce": nonce, "next": nxt,
         "iss": FLOW_ISS, "aud": FLOW_ISS, "iat": now, "exp": now + FLOW_TTL_SECONDS},
        SESSION_SECRET, algorithm="HS256",
    )


def _verify_flow(token: str) -> dict:
    claims = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"], issuer=FLOW_ISS, audience=FLOW_ISS)
    for k in ("state", "verifier", "nonce", "next"):
        if not isinstance(claims.get(k), str):
            raise jwt.InvalidTokenError("malformed OIDC flow cookie")
    return claims


# ── PKCE + Authorization Code exchange + id_token verification ──────────────────────────────────
def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _new_verifier() -> str:
    return _b64url(secrets.token_bytes(32))  # RFC 7636: 43-char high-entropy verifier


def _challenge_s256(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _build_authorize_url(state: str, nonce: str, challenge: str, redirect_uri: str) -> str:
    q = urllib.parse.urlencode({
        "response_type": "code", "client_id": CLIENT_ID, "redirect_uri": redirect_uri,
        "scope": SCOPE, "state": state, "nonce": nonce,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })
    return f"{_authorize_url()}?{q}"


def _exchange_code(code: str, verifier: str, redirect_uri: str) -> dict:
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
        "redirect_uri": redirect_uri, "code": code, "code_verifier": verifier,
    }).encode("ascii")
    req = urllib.request.Request(
        _token_url(), data=body,
        headers={"content-type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        tok = json.loads(r.read().decode("utf-8"))
    if not isinstance(tok.get("id_token"), str):
        raise jwt.InvalidTokenError("token response missing id_token")
    return tok


def _verify_id_token(id_token: str, nonce: str) -> Principal:
    key = _jwks().get_signing_key_from_jwt(id_token).key
    claims = jwt.decode(id_token, key, algorithms=["RS256"], issuer=ISSUER, audience=CLIENT_ID)
    if claims.get("nonce") != nonce:
        raise jwt.InvalidTokenError("id_token nonce mismatch")
    return _principal_from_claims(claims)


def _safe_next(raw: str | None) -> str:
    """Open-redirect guard: only a same-origin relative path (single leading '/', no '//', no
    control/space chars) is honored; anything else falls back to '/'."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/"
    if any(ord(c) <= 32 or ord(c) == 127 for c in raw):
        return "/"
    return raw


def _secure_cookie() -> bool:
    return PUBLIC_URL.lower().startswith("https")


# ── Principal resolution (bearer OR session) ────────────────────────────────────────────────────
def _resolve_sync(bearer: str | None, session: str | None) -> Principal | None:
    if session:
        try:
            return verify_session(session)
        except jwt.PyJWTError:
            pass
    if bearer and bearer_ready:
        try:
            return verify_bearer(bearer)
        except jwt.PyJWTError:
            pass
    return None


async def _resolve(request) -> Principal | None:
    """Resolve the caller from the session cookie (browser) or the bearer header (service). The JWKS
    verification can do a one-time network fetch, so it runs in a thread to keep the loop free."""
    hdr = request.headers.get("authorization", "")
    bearer = hdr[7:].strip() if hdr[:7].lower() == "bearer " else None
    session = request.cookies.get(SESSION_COOKIE)
    if not bearer and not session:
        return None
    return await asyncio.to_thread(_resolve_sync, bearer, session)


def current_principal(request) -> Principal | None:
    """The principal the auth middleware resolved for this request (None when auth is off or the
    caller is anonymous on an open path). Handlers read this for per-user attribution."""
    return getattr(request.state, "principal", None)


def _is_protected(path: str) -> bool:
    """Everything under /v1/ is the API surface and requires a principal. /health (liveness),
    /auth/* (the login flow itself), and the UI at / are handled separately in the middleware."""
    return path.startswith("/v1/")


def install(app) -> None:
    """Wire the auth routes + enforcement middleware onto the FastAPI app. FastAPI/Starlette are
    imported HERE, not at module top, so the crypto/OIDC helpers above stay importable with only
    PyJWT (the unit tests exercise them without the web stack). A no-op for request handling when
    auth is disabled — the middleware short-circuits — so it's always safe to call."""
    from fastapi import APIRouter, Request, Response
    from fastapi.responses import JSONResponse, RedirectResponse
    from starlette.middleware.base import BaseHTTPMiddleware

    router = APIRouter()

    @router.get("/auth/status")
    def auth_status_route(request: Request) -> dict:
        """Whether SSO is enforced, whether the browser-login flow is available, and — since the
        session cookie is httpOnly (JS can't read it) — WHO the current session belongs to, so the
        UI can show "signed in as …" + a sign-out control. `user` is null when anonymous/auth-off."""
        p = current_principal(request)
        return {"auth_enabled": auth_enabled, "sso": sso_ready, "bearer": bearer_ready,
                "user": {"sub": p.sub, "name": p.display_name or p.email or p.sub} if p else None}

    @router.get("/auth/login")
    async def auth_login(request: Request):
        if not sso_ready:
            return JSONResponse({"error": "sso_not_configured"}, status_code=503)
        try:
            nxt = _safe_next(request.query_params.get("next"))
            state, nonce, verifier = secrets.token_urlsafe(16), secrets.token_urlsafe(16), _new_verifier()
            redirect_uri = f"{PUBLIC_URL}/auth/callback"
            url = await asyncio.to_thread(_build_authorize_url, state, nonce, _challenge_s256(verifier), redirect_uri)
            resp = RedirectResponse(url, status_code=302)
            resp.set_cookie(FLOW_COOKIE, _sign_flow(state, verifier, nonce, nxt), max_age=FLOW_TTL_SECONDS,
                            httponly=True, samesite="lax", secure=_secure_cookie(), path="/")
            return resp
        except Exception:  # noqa: BLE001 — never leak IdP/network internals; generic bounce
            return RedirectResponse("/?auth_error=login_failed", status_code=302)

    @router.get("/auth/callback")
    async def auth_callback(request: Request):
        if not sso_ready:
            return JSONResponse({"error": "sso_not_configured"}, status_code=503)
        try:
            code, state = request.query_params.get("code"), request.query_params.get("state")
            flow_token = request.cookies.get(FLOW_COOKIE)
            if not code or not state or not flow_token:
                raise jwt.InvalidTokenError("callback missing code/state/flow")
            flow = _verify_flow(flow_token)
            if flow["state"] != state:
                raise jwt.InvalidTokenError("state mismatch")
            redirect_uri = f"{PUBLIC_URL}/auth/callback"
            tok = await asyncio.to_thread(_exchange_code, code, flow["verifier"], redirect_uri)
            principal = await asyncio.to_thread(_verify_id_token, tok["id_token"], flow["nonce"])
            resp = RedirectResponse(_safe_next(flow["next"]), status_code=302)
            resp.set_cookie(SESSION_COOKIE, mint_session(principal), max_age=SESSION_TTL,
                            httponly=True, samesite="lax", secure=_secure_cookie(), path="/")
            resp.delete_cookie(FLOW_COOKIE, path="/")
            return resp
        except Exception:  # noqa: BLE001
            return RedirectResponse("/?auth_error=login_failed", status_code=302)

    @router.post("/auth/logout")
    def auth_logout() -> Response:
        resp = Response(status_code=204)
        resp.delete_cookie(SESSION_COOKIE, path="/")
        return resp

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if not auth_enabled:
                request.state.principal = None
                return await call_next(request)
            principal = await _resolve(request)
            request.state.principal = principal
            path = request.url.path
            if principal is None and _is_protected(path):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            # A browser hitting the UI with no session → send it through SSO (only when the browser
            # login is actually available; a bearer-only deployment 401s the API and has no UI login).
            if principal is None and path == "/" and sso_ready:
                return RedirectResponse("/auth/login?next=/", status_code=302)
            return await call_next(request)

    app.include_router(router)
    app.add_middleware(AuthMiddleware)


def status() -> dict:
    """Auth summary for /health."""
    return {"auth_enabled": auth_enabled, "sso_login": sso_ready, "bearer": bearer_ready,
            **({"issuer": ISSUER, "audience": AUDIENCE} if auth_enabled else {})}
