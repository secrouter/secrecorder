# Configuration

Every environment variable SecRecorder reads, swept from `os.environ` in `server.py`, `auth.py`,
`summarize.py`, and `audit.py` (`speakers.py` reads none — it's configured entirely through
`server.py`'s `SPEAKER_*` vars). `WHISPER_*` / `SPEAKER_*` are typically set in the process
environment (e.g. by `run.sh`); `HF_TOKEN` is read from `.env` next to `server.py` (written by
`install.sh`, chmod 600).

## Whisper / transcription (`server.py`)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_BACKEND` | `auto` | `mlx` \| `faster-whisper` \| `auto` (mlx if importable, else faster-whisper) |
| `WHISPER_MODEL` | per-backend | mlx: `mlx-community/whisper-large-v3-turbo` · faster-whisper: `deepdml/faster-whisper-large-v3-turbo-ct2` |
| `WHISPER_DEVICE` | `auto` | faster-whisper only: `cuda` \| `cpu` \| `auto` |
| `WHISPER_COMPUTE_TYPE` | auto | faster-whisper only: `float16` (cuda) \| `int8` (cpu) \| … |
| `WHISPER_MAX_CONCURRENCY` | `1` | concurrent transcriptions (GPU-serialized via a semaphore) |
| `WHISPER_MAX_UPLOAD_MB` | `0` | upload cap in MiB (`0` = unlimited); over-cap uploads get a 413 |
| `WHISPER_SILENCE_MAX_DB` | `-45` | dead-air guard: a clip whose peak volume falls below this (dBFS) is returned as an empty transcript instead of being sent to Whisper. **Why:** Whisper hallucinates fabricated text when given near-silent/degenerate audio, so this rejects it before ASR rather than after. Set below `0` to enable (any negative dB threshold); a probe that can't measure the level (ffmpeg missing/erroring) fails open and transcribes normally. Flags `detail.silenceGated: true` on the request's audit record. |
| `WHISPER_DIA_CONCURRENCY` | `1` | concurrent diarization jobs (separate semaphore from ASR, so diarizing one episode overlaps the next episode's transcription instead of blocking it) |
| `WHISPER_PREWARM` | `0` (`1` via `run.sh`) | load the model at startup so the first request has no cold-start hit |
| `WHISPER_PREWARM_DIARIZER` | `0` (`1` via `run.sh`) | also load the diarizer at startup |
| `WHISPER_DIARIZE` | `0` | default `diarize` on when a transcription request omits it |
| `WHISPER_ALLOW_DIARIZE` | `1` | hard kill-switch; set `0` where pyannote can't load (e.g. no HF access) |
| `WHISPER_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | diarization pipeline |
| `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) | — | Hugging Face token for the gated diarization model; not needed for transcription |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | internal: lets pyannote's torch ops fall back to CPU when the MPS backend lacks a kernel (Apple only); set via `os.environ.setdefault`, so an operator-set value is respected — rarely needs changing |

## Speaker library (`server.py`)

| Variable | Default | Meaning |
|---|---|---|
| `SPEAKER_LIBRARY` | `1` | enable the speaker library + `/v1/speakers*` endpoints; `0` disables the feature entirely |
| `SPEAKER_DB` | `speakers.db` next to `server.py` | SQLite path for enrolled voiceprints |
| `SPEAKER_IDENTIFY` | `0` | default `identify` on when a transcription request omits it |
| `SPEAKER_IDENTIFY_THRESHOLD` | `0.5` | cosine match cut-off (same voice ≈0.9+, different voice ≈0.3) |

## SSO authentication (`auth.py`)

Off by default — auth only turns on once `SECRECORDER_OIDC_ISSUER` + `SECRECORDER_OIDC_CLIENT_ID`
are both set. See [security.md](security.md) for the full model.

| Variable | Default | Meaning |
|---|---|---|
| `SECRECORDER_OIDC_ISSUER` | — | SecSSO issuer URL — set (with `_CLIENT_ID`) to require a valid bearer token on `/v1/*` |
| `SECRECORDER_OIDC_CLIENT_ID` | — | this service's OIDC client id / default token audience |
| `SECRECORDER_OIDC_AUDIENCE` | `SECRECORDER_OIDC_CLIENT_ID` | token audience, if it differs from the client id |
| `SECRECORDER_OIDC_CLIENT_SECRET` | — | confidential-client secret — enables the browser-login BFF (`/auth/*`) |
| `SECRECORDER_PUBLIC_URL` | — | this service's external URL — builds the OIDC redirect URI and the session cookie's `Secure` flag |
| `SECRECORDER_SESSION_SECRET` | — | signing key for the browser session cookie |
| `SECRECORDER_SESSION_TTL` | `43200` (12h, seconds) | browser session cookie lifetime |
| `SECRECORDER_OIDC_JWKS_URL` | — (OIDC discovery) | override the issuer's discovered JWKS endpoint |
| `SECRECORDER_OIDC_AUTHORIZE_URL` | — (OIDC discovery) | override the issuer's discovered authorize endpoint |
| `SECRECORDER_OIDC_TOKEN_URL` | — (OIDC discovery) | override the issuer's discovered token endpoint |

## Summarization (`summarize.py`)

Off by default — only enabled once `SECRECORDER_SUMMARIZE_ENABLED` is truthy **and**
`SECRECORDER_SUMMARIZE_ENDPOINT` is set.

| Variable | Default | Meaning |
|---|---|---|
| `SECRECORDER_SUMMARIZE_ENABLED` | `0` | enable transcript summarization |
| `SECRECORDER_SUMMARIZE_ENDPOINT` | — | OpenAI-compatible base URL (e.g. SecRouter's `…/v1`); `chat/completions` is appended |
| `SECRECORDER_SUMMARIZE_MODEL` | `auto` | model id for the summary call |
| `SECRECORDER_SUMMARIZE_API_KEY` | — | optional bearer token for the summarization endpoint itself |
| `SECRECORDER_SUMMARIZE_ACTING_USER` | `svc-secrecorder` | fallback `X-Sec-Acting-User` when the caller is anonymous (auth off) |
| `SECRECORDER_SUMMARIZE_CLASSIFICATION` | — | classification level asserted via `x-data-classification` on the summarize call (e.g. `CUI`) — matches a level in SecRouter's `security.classification.levels`; empty omits the header |
| `SECRECORDER_SUMMARIZE_PROMPT` | (built-in meeting-summary prompt) | system prompt for the summary call |
| `SECRECORDER_SUMMARIZE_TIMEOUT` | `60` (seconds) | HTTP timeout for the summarization call |
| `SECRECORDER_SUMMARIZE_MAX_CHARS` | `48000` | transcript characters sent to the model before truncating (with a `[transcript truncated]` marker) |

## Audit trail (`audit.py`)

On by default — see [security.md](security.md) for what's recorded and the metadata-only
discipline.

| Variable | Default | Meaning |
|---|---|---|
| `SECRECORDER_AUDIT_ENABLED` | `1` | tamper-evident audit trail; set `0` to turn it off |
| `SECRECORDER_AUDIT_PATH` | `data/audit.jsonl` next to `audit.py` | path to the hash-chained JSONL audit log |
