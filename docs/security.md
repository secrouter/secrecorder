# Security

## Auth model

**Off by default.** With no `SECRECORDER_OIDC_*` set, SecRecorder is an open service — every
route (except the deliberately-always-open ones below) answers with no authentication, matching
its pre-existing posture before SSO support was added. This is a real posture, not a placeholder:
any deployment handling CUI must explicitly turn auth on (see [configuration.md](configuration.md)
for the full variable list) — that's a deployer responsibility this component does not (and
cannot safely) infer on its own.

Setting `SECRECORDER_OIDC_ISSUER` + `SECRECORDER_OIDC_CLIENT_ID` requires a valid **SecSSO**
(OIDC) bearer JWT on every `/v1/*` route: a programmatic client sends
`Authorization: Bearer <token>`, verified against the issuer's JWKS (RS256 pinned, plus issuer,
audience, and expiry checks). `/health` and `/auth/*` are never gated — liveness and the login
flow itself have to stay reachable regardless of auth state.

Additionally setting `SECRECORDER_OIDC_CLIENT_SECRET` + `SECRECORDER_PUBLIC_URL` +
`SECRECORDER_SESSION_SECRET` enables a server-side login BFF for the **built-in web UI**: an
unauthenticated browser at `/` is bounced through SecSSO (Authorization Code + PKCE), and the only
credential the browser ever holds afterward is an httpOnly `secrecorder_session` cookie — it never
sees the OIDC token itself. `GET /auth/status` lets the UI show "signed in as …" without the page
being able to read the cookie directly.

There is no separate admin/group concept anywhere in this service — `/v1/audit/verify` and
`/v1/evidence` get exactly the same gate as every other `/v1/*` route (any authenticated
principal, or open when auth is off). Inventing a narrower admin check for just those two routes
would add a new, ungated trust boundary rather than tighten one that already exists.

In the SecRouter suite, SecDeploy wires this automatically (a `secrecorder` OIDC client in
SecSSO + the corresponding env — see [deploy.md](deploy.md)).

## Audit trail: metadata-only discipline

SecRecorder is the system of record for the CUI voice recordings passing through it, so its audit
trail (`audit.py`) is **on by default** — `SECRECORDER_AUDIT_ENABLED=0` is how an operator turns it
*off*, not on. Every transcription run, speaker enroll/update/delete, summarize request, and auth
failure becomes one append-only JSONL record, chained with a SHA-256 hash (`GET /v1/audit/verify`
detects any downstream edit, insertion, or deletion) and hardened at rest (0700 log directory,
0600 log file).

**A record's `detail` is metadata only — counts, durations, sizes, flags, model names, and
thresholds — never transcript text, a generated summary, a prompt, or audio/voiceprint bytes.**
This is an absolute rule, not a best effort: `audit.py`'s `_scrub_detail` is a defense-in-depth
backstop that drops any `detail` key named `text`/`transcript`/`transcription`/`prompt`/
`completion`/`summary`/`audio`/`content`/`embedding`/`embeddings`/`words`/`segments` before a
record is ever canonicalized, hashed, or written, and prints a warning if it had to. Call sites
are not supposed to pass one of these keys in the first place — the scrub only guards against a
future coding mistake becoming a permanent, tamper-evident CUI leak. See
[control-validation.md](control-validation.md) for the control mapping and what SecRecorder's
audit trail explicitly does *not* cover (LLM call governance for summarization is SecRouter's
audit, not this one).

## Voice-biometric considerations

Diarization produces a voiceprint (embedding) per detected speaker; the speaker library
(`speakers.py`) persists *named* voiceprints so the same person is recognized across later
recordings. This is biometric data, handled as follows:

- **Stored locally only.** The library is a local SQLite file (`SPEAKER_DB`, default
  `speakers.db` next to `server.py`), gitignored, with no network sync of its own. Nothing about
  an enrolled voiceprint leaves the box except within a transcription response the caller
  explicitly requested (`speakers[].embedding`) or an explicit speaker-library API call.
- **Names never reach the audit log.** Every speaker-lifecycle audit record (`speaker.enroll`/
  `speaker.update`/`speaker.delete`) carries the opaque `speakerId` as its `target`, never the
  enrolled `name` — matching the metadata-only discipline above. The audit trail can show *that*
  a speaker was enrolled/updated/deleted and *when*, but not *who*.
- **Recognition is opt-in per request** (`identify=true`) and can be disabled host-wide
  (`SPEAKER_LIBRARY=0`), which removes the `/v1/speakers*` endpoints entirely. Diarization itself
  (`WHISPER_ALLOW_DIARIZE=0`) can likewise be disabled where pyannote can't load or isn't wanted
  on a given host.
- **Matching, not identity verification.** Recognition is a cosine-similarity match against
  enrolled centroids (`SPEAKER_IDENTIFY_THRESHOLD`, default `0.5`) — a probabilistic signal for
  labeling a transcript, not an authentication mechanism. It gates nothing security-sensitive by
  itself.

## Summarization governance

Summarization is off by default and, when enabled, is expected to be pointed at **SecRouter**
(`SECRECORDER_SUMMARIZE_ENDPOINT=<secrouter>/v1`) so the LLM call itself is attributed
(`X-Sec-Acting-User`), budgeted, egress-controlled, and audited *at SecRouter* — SecRecorder's own
audit only records that a summarization was *requested* (never the transcript or the summary
text). Pointing the endpoint directly at SecLLM or another server instead removes that governance
layer; that's a deployer choice this component does not warn about beyond what's documented here
and in [control-validation.md](control-validation.md).
