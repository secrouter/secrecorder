# Usage

## `POST /v1/audio/transcriptions`

OpenAI-compatible transcription. Multipart form:

| Field | Type | Meaning |
|---|---|---|
| `file` | file, required | audio to transcribe (any format `ffmpeg` can decode) |
| `model` | string | accepted and ignored — this server always uses its one loaded model |
| `response_format` | string | accepted and ignored — always returns `verbose_json` + a top-level `words[]` |
| `timestamp_granularities[]` | string | accepted and ignored — word-level timestamps are always included |
| `prompt` | string, optional | biases/continues decoding (Whisper's `initial_prompt`) |
| `diarize` | `"true"`/`"false"`, optional | label words/segments with a `speaker`; defaults to `WHISPER_DIARIZE` when omitted |
| `identify` | `"true"`/`"false"`, optional | match diarized speakers to the enrolled library by name; implies `diarize=true`; defaults to `SPEAKER_IDENTIFY` when omitted |
| `summarize` | `"true"`/`"false"`, optional | attach a `summary` via the configured summarization endpoint (see [configuration.md](configuration.md)) |

Response (`200`, `verbose_json`-shaped):

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.3,
  "text": "...",
  "words": [{"word": "...", "start": 0.12, "end": 0.34}],
  "segments": [...],
  "speakers": [{"id": "SPEAKER_00", "talk_time": 8.1, "embedding": [...], "name": "...", "speaker_id": "...", "match_score": 0.94}],
  "summary": "..."
}
```

`speakers[]` only appears when `diarize` resolved true; `name`/`speaker_id`/`match_score` only
appear on a speaker matched by `identify`. A diarization failure never fails the request — the
response carries `diarization_error` instead. Requesting `diarize`/`identify` when the
corresponding feature is disabled on this host (`WHISPER_ALLOW_DIARIZE=0` /
`SPEAKER_LIBRARY=0`) returns the plain transcript with `diarization_disabled` /
`identification_disabled: true` rather than an error. A failed summarization degrades to a
`summary_error` field — the transcript itself is never lost.

**Dead-air / silence response.** When `WHISPER_SILENCE_MAX_DB` is enabled (default `-45`, i.e. any
negative threshold) and the clip's peak volume falls below it, the request short-circuits before
reaching Whisper and returns:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 0.0,
  "text": "",
  "words": [],
  "segments": [],
  "speakers": [],
  "silence": true,
  "max_volume_db": -52.3
}
```

This exists because Whisper hallucinates fabricated text on near-silent audio — this response is
the guard, not a bug. The audit record for this request carries `detail.silenceGated: true`.

## Speaker library

Off entirely when `SPEAKER_LIBRARY=0`.

| Route | Meaning |
|---|---|
| `GET /v1/speakers` | list enrolled speakers (id, name, sample count, timestamps) |
| `POST /v1/speakers` | enroll from a voiceprint vector: `{"name": "...", "embedding": [...], "source": "...", "meta": {...}}` — typically a `speakers[].embedding` from a prior transcription |
| `GET /v1/speakers/{id}` | one speaker; `?centroid=1` includes its mean voiceprint |
| `POST /v1/speakers/{id}/samples` | add another voiceprint sample to sharpen recognition: `{"embedding": [...], "source": "..."}` |
| `DELETE /v1/speakers/{id}` | remove a speaker and all its voiceprints |
| `POST /v1/speakers/from-audio` | enroll from an audio sample (multipart `name` + `file`): diarizes the clip and stores the dominant speaker's voiceprint. Requires diarization enabled. Use a clean, single-speaker sample. |

## `POST /v1/summarize`

Standalone summarization for arbitrary text (e.g. a transcript from an earlier call):

```json
{"text": "..."}
```

Returns `{"summary": "..."}`. `503` when summarization isn't configured
(`SECRECORDER_SUMMARIZE_ENABLED`/`_ENDPOINT`); `502` when the LLM call itself fails.

## `GET /v1/audit/verify`

Verifies the audit hash chain (tamper-evidence): `{"ok": true, "checked": N}`, or
`{"ok": false, "checked": N, "brokenAtSeq": K}` naming the first tampered/broken record.

## `GET /v1/evidence`

One-shot CMMC evidence bundle: sanitized config posture (model names, thresholds, paths — never a
secret/token), the audit chain's verification result, the last 200 audit records, and a small
control self-assessment. See [control-validation.md](control-validation.md) for what's implemented
here vs. delegated elsewhere.

## `GET /health`

Liveness + posture: backend/model in use, concurrency settings, diarization/speaker-library/auth/
summarize/audit status. Always open (never gated by auth), so it's safe for a load balancer or
`secproxy` health check.

## `GET /v1/models` / `GET /v1/models/{id}`

OpenAI Models API — this server always serves exactly one model (whichever it loaded at startup).

## Auth routes (only when SSO is configured — see [security.md](security.md))

`GET /auth/login` · `GET /auth/callback` · `POST /auth/logout` · `GET /auth/status`.

## The built-in web UI (`GET /`)

Same-origin single-page UI (no CORS needed), served from `ui.html` next to `server.py`:

- **Record or upload** — capture from the mic or drop an audio file, then transcribe.
- **Label speakers** — toggle diarization on for the request; detected speakers can be named
  inline and saved to the speaker library (`POST /v1/speakers`) directly from the result.
- **Copy / export** — copy the transcript, or export it as Markdown or plain text.
- **Theme toggle** — the ◐ button in the header flips between light/dark, persisted in
  `localStorage` (`secrecorder-theme`); it otherwise follows the browser's
  `prefers-color-scheme`.
- **Sign-in state** — when SSO is configured, `GET /auth/status` drives a "signed in as …" /
  sign-out affordance (the session cookie itself is httpOnly, so the UI reads status via the API
  rather than the cookie).

If `ui.html` is missing next to `server.py`, `/` falls back to a one-line placeholder pointing at
the API — the service still runs.
