# SecRecorder — Control Validation

Citation style (Spec B.5): bare NIST SP 800-171 r2 IDs in the table below; `FAMILY-ID` in prose
(e.g. `AU-3.3.8`). This is honest scoping, not a claim of full CMMC coverage — see
**Shared Responsibility** at the end for what SecRecorder does *not* own.

## Controls this component implements

| Control | Family | Requirement | Implementation (file:function) | Evidence command |
|---|---|---|---|---|
| 3.3.1 | AU | Create and retain system audit records needed to enable monitoring, analysis, investigation, and reporting of unlawful or unauthorized activity. | `audit.py:AuditLogger.record` — one append-only JSONL record per lifecycle event: `transcription.run`, `speaker.enroll`/`speaker.update`/`speaker.delete`, `summarize.request`, `auth.failure`. Wired at the call sites in `server.py` and `auth.py`'s `AuthMiddleware`. | `curl -s $HOST/v1/evidence \| jq '.auditRecent[-5:]'` |
| 3.3.2 | AU | Ensure that the actions of individual system users can be uniquely traced to those users, to be held accountable for their actions. | Every record's `principal` field carries the OIDC `sub` resolved by `auth.py:current_principal` (or the literal `"anonymous"` when auth is off or the caller isn't authenticated). | `curl -s $HOST/v1/evidence \| jq '.auditRecent[].principal'` |
| 3.3.8 | AU | Protect audit information and audit logging tools from unauthorized access, modification, and deletion. | SHA-256 hash chain over sorted-keys-JSON canonical records (`audit.py:_hash_record`, `audit.py:verify_chain`); genesis constant `"GENESIS"` (Spec B.2); 0700 log directory / 0600 log file permissions applied on first write (`audit.py:_harden_path`). Exposed at `GET /v1/audit/verify`. | `curl -s $HOST/v1/audit/verify` |
| 3.5.2 | IA | Authenticate (or verify) the identities of users, processes, or devices, as a prerequisite to allowing access to organizational systems. | `auth.py` — SecSSO OIDC: bearer JWT verified against JWKS (RS256 pinned, issuer/audience/expiry checked) for service callers, or a server-side Authorization Code + PKCE login (BFF) for the built-in web UI, both against the same issuer. **Off by default** — see Shared Responsibility. | `curl -s $HOST/health \| jq '.auth'` |

## Metadata-only discipline (Spec B.3 — absolute, not a numbered control)

`detail` on every audit record may hold counts, durations, sizes, flags, model names, and
thresholds — **never** transcript text, a generated summary, a prompt, or audio/voiceprint bytes.
`audit.py:_scrub_detail` is a defense-in-depth backstop: it drops any `detail` key named
`text`/`transcript`/`transcription`/`prompt`/`completion`/`summary`/`audio`/`content`/`embedding`/
`embeddings`/`words`/`segments` before the record is ever canonicalized, hashed, or written, and
prints a warning if it had to. Call sites are not supposed to pass one of these in the first place;
this only guards against a future mistake becoming a permanent, tamper-evident CUI leak.
Verified by `test_audit.py:test_metadata_only_forbidden_keys_never_reach_the_file`.

## Note (not a control citation)

**SI — silence gate.** `WHISPER_SILENCE_MAX_DB` (server.py `_peak_volume_db`) rejects
near-silent/degenerate audio before it reaches Whisper, which otherwise hallucinates fabricated
text on silence. A gated request is flagged `detail.silenceGated: true` on its `transcription.run`
audit record. This is included here for completeness; it is not asserted against a specific
NIST SP 800-171 control ID (no overclaiming, per Spec B.5).

## Shared Responsibility

SecRecorder's audit trail covers **its own** actions — the API surface in this repository. The
following are explicitly **not** implemented here:

- **LLM call governance for summarization.** SecRecorder makes exactly one LLM call
  (`summarize.py`), and by default that call is routed through **SecRouter** with
  `X-Sec-Acting-User` set to the recording's owner. Attribution, per-user budget, egress control,
  and audit of *that call* are SecRouter's responsibility and land in **SecRouter's** audit log —
  SecRecorder only records that a summarization was *requested* (`summarize.request`, with
  `detail.governedBy: "secrouter"` and a character count — never the transcript or the summary
  itself). If a deployment points `SECRECORDER_SUMMARIZE_ENDPOINT` at SecLLM or another endpoint
  directly instead, that governance division no longer applies and is the deployer's
  responsibility to reinstate.
- **Authentication is off by default.** SecRecorder ships as an open service (matching its
  pre-existing posture) until `SECRECORDER_OIDC_ISSUER`/`_AUDIENCE` (bearer) or the full BFF set
  (`_CLIENT_ID`/`_CLIENT_SECRET`/`SECRECORDER_PUBLIC_URL`/`_SESSION_SECRET`) are configured. Any
  deployment handling CUI recordings **must** enable OIDC (SecDeploy wires this automatically in
  the SecRouter suite) — until then, IA-3.5.2 is not enforced and every audit record's `principal`
  reads `"anonymous"`.
- **Audit log retention, backup, and SIEM forwarding.** `audit.jsonl` (default
  `data/audit.jsonl` next to `server.py`) is local, append-only storage with no rotation, off-box
  replication, or forwarding built in. Operators should point `SECRECORDER_AUDIT_PATH` at a
  protected volume with its own backup/retention policy and forward it to a SIEM if one is in use —
  the same operator responsibility noted in secagent's audit module.
- **Transport encryption and network access control** (TLS termination, network segmentation) —
  environment-owned, same as the rest of this service.
- **Recording persistence.** SecRecorder does not persist uploaded recordings anywhere — each
  upload is transcribed from a temp file and discarded (see `server.py:_spool_upload`'s `finally`).
  There is consequently no "recording created/deleted/listed" lifecycle to audit; `transcription.run`
  uses a fresh per-request id (`target.requestId`) as the closest available correlation handle. A
  deployment that itself persists recordings (e.g. SecChat Meetings, storing the upload before
  calling this API) owns auditing that persistence lifecycle.
