# SecRecorder

*aka SpeakerBox* — a self-hosted, OpenAI-compatible Whisper speech-to-text server with optional speaker diarization.
One `server.py` runs on either backend, auto-selected for the host:

- **Apple Silicon** → **MLX** (Metal GPU) via `mlx-whisper`
- **Linux / NVIDIA** → **faster-whisper** (CTranslate2) on CUDA, or CPU anywhere

It speaks the OpenAI audio-transcriptions API, so existing OpenAI clients work by pointing
`base_url` at it. Word-level timestamps are always returned. Optional diarization (pyannote) adds
per-word speaker labels and per-speaker voiceprints; optional recognition names those speakers
across recordings.

## Features

- **Built-in web UI** at `/` — record from the mic or drop an audio file, transcribe with optional
  speaker labels + recognition, enroll speakers into the library, and copy/export the notes
  (Markdown or plain text). Self-contained (system fonts, no external calls), field-console styling.
- **OpenAI-compatible** `POST /v1/audio/transcriptions` (multipart; `verbose_json` with a top-level
  `words[]` array), plus `GET /v1/models` and `GET /health`.
- **Multi-backend**, one codebase (`WHISPER_BACKEND=auto|mlx|faster-whisper`).
- **Speaker diarization** (opt-in): pass `diarize=true` and every word/segment gains a `speaker`,
  plus a `speakers[]` summary. Each speaker also carries a voiceprint `embedding`, so a client that
  chunks long audio can link the same speaker across independently-diarized chunks.
- **Speaker recognition** (opt-in): enroll named voiceprints once, then pass `identify=true` and a
  recording's diarized speakers are matched back to them by cosine similarity — each `speakers[]`
  entry gains `name` + `match_score`. The library is a local SQLite file; manage it via
  `/v1/speakers` or the web UI. No extra model, nothing leaves the box.
- **Optional SSO auth** (off by default): validate SecSSO (OIDC) bearer JWTs on the API, and log the
  built-in UI in through SecSSO — Authorization Code + PKCE, httpOnly session cookie. See below.
- **Optional summarization** (off by default): attach a `summary` to a transcript via any
  OpenAI-compatible chat endpoint — governed through SecRouter by default (per-user attribution +
  audit). Use `summarize=true` on the transcription call, or `POST /v1/summarize` for any text.
- **Built for long recordings:** the blocking transcription runs off the event loop (so `/health`
  stays responsive), GPU work is serialized, uploads stream to disk, and the model is prewarmed at
  startup so a restart has no cold-start hit.
- **Robust output:** non-finite floats from the model (which would otherwise break JSON) are
  scrubbed to `null` rather than failing the request.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) (the installer uses it to manage Python + deps; no system
  Python needed).
- Apple Silicon (for the MLX backend) **or** Linux with an NVIDIA GPU / any CPU (faster-whisper).
- **`ffmpeg`** on `PATH` — decodes audio for the MLX backend and normalises the diarizer's input to
  WAV (all backends). Install it yourself (`brew install ffmpeg` / `apt-get install ffmpeg`), or let
  the installer handle it: `./install.sh --with-ffmpeg`.
- **NVIDIA GPU (faster-whisper backend):** CTranslate2 needs **cuBLAS + cuDNN 9** at runtime and does
  not bundle them. `./run.sh` auto-installs them (the `cuda` extra) when it detects an NVIDIA GPU; to
  pre-install at setup use `./install.sh --with-cuda` (or `uv sync --extra cuda`).
- For diarization only: a Hugging Face token whose account has accepted the
  [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  conditions. Transcription needs no token.

## Quickstart

```bash
git clone <this-repo> secrecorder && cd secrecorder

# (optional) diarization: copy the template, add your HF token
cp secrets.env.example secrets.env
$EDITOR secrets.env

./install.sh secrets.env          # sets up the venv; writes .env (chmod 600) from your secrets
#   ...or ./install.sh --with-ffmpeg secrets.env   to also install ffmpeg if it's missing
./run.sh                          # 127.0.0.1:9000, prewarmed
# HOST=0.0.0.0 ./run.sh           # expose on the LAN/VPN
```

Omit `secrets.env` to install transcription-only:

```bash
./install.sh
```

Open **http://localhost:9000/** for the built-in web UI, or point any OpenAI client at the API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")
client.audio.transcriptions.create(model="whisper-1", file=open("audio.wav", "rb"),
                                    response_format="verbose_json")
```

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_BACKEND` | `auto` | `mlx` \| `faster-whisper` \| `auto` (mlx if importable, else faster-whisper) |
| `WHISPER_MODEL` | per-backend | mlx: `mlx-community/whisper-large-v3-turbo` · faster-whisper: `deepdml/faster-whisper-large-v3-turbo-ct2` |
| `WHISPER_DEVICE` | `auto` | faster-whisper only: `cuda` \| `cpu` \| `auto` |
| `WHISPER_COMPUTE_TYPE` | auto | faster-whisper only: `float16` (cuda) \| `int8` (cpu) \| … |
| `WHISPER_MAX_CONCURRENCY` | `1` | concurrent transcriptions (GPU-serialized) |
| `WHISPER_MAX_UPLOAD_MB` | `0` | upload cap in MiB (0 = unlimited) |
| `WHISPER_PREWARM` | `1` (via run.sh) | load the model at startup |
| `WHISPER_PREWARM_DIARIZER` | `1` (via run.sh) | also load the diarizer at startup |
| `WHISPER_DIARIZE` | `0` | default `diarize` on when the request omits it |
| `WHISPER_ALLOW_DIARIZE` | `1` | hard kill-switch; set `0` where pyannote can't load |
| `WHISPER_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | diarization pipeline |
| `HF_TOKEN` | — | Hugging Face token for the gated diarization model (from `.env`) |
| `SPEAKER_LIBRARY` | `1` | enable the speaker library + `/v1/speakers` endpoints; `0` disables the feature |
| `SPEAKER_DB` | next to `server.py` | SQLite path for enrolled voiceprints (`speakers.db`) |
| `SPEAKER_IDENTIFY` | `0` | default `identify` on when a request omits it |
| `SPEAKER_IDENTIFY_THRESHOLD` | `0.5` | cosine match cut-off (same voice ≈0.9+, different ≈0.3) |
| `SECRECORDER_OIDC_ISSUER` | — | SecSSO issuer — set (with `_CLIENT_ID`) to require SSO on `/v1/*` |
| `SECRECORDER_OIDC_CLIENT_ID` | — | this service's OIDC client id / token audience |
| `SECRECORDER_OIDC_CLIENT_SECRET` | — | confidential-client secret — enables the browser-login BFF |
| `SECRECORDER_PUBLIC_URL` | — | external URL (builds the OIDC redirect + the cookie `Secure` flag) |
| `SECRECORDER_SESSION_SECRET` | — | session-cookie signing key (browser login) |
| `SECRECORDER_SUMMARIZE_ENABLED` | `0` | enable transcript summarization (needs an endpoint) |
| `SECRECORDER_SUMMARIZE_ENDPOINT` | — | OpenAI-compatible base URL (e.g. SecRouter `…/v1`) |
| `SECRECORDER_SUMMARIZE_MODEL` | `auto` | model id for the summary call |
| `SECRECORDER_SUMMARIZE_API_KEY` | — | optional bearer for the summarization endpoint |
| `SECRECORDER_SUMMARIZE_PROMPT` | (built-in) | system prompt for the summary |

`WHISPER_*` are read from the process environment (e.g. `run.sh`); `HF_TOKEN` is read from `.env`
(written by `install.sh`, chmod 600). Per-request, `diarize=true|false` overrides `WHISPER_DIARIZE`.

## Backends

### Apple (MLX / Metal)
`./run.sh`. Defaults to `large-v3-turbo`. Warm calls run many× realtime on Apple Silicon.

### Linux / NVIDIA (faster-whisper)
`WHISPER_BACKEND=faster-whisper WHISPER_DEVICE=cuda ./run.sh`. `install.sh` picks faster-whisper via
the `platform_system == 'Linux'` dependency marker. faster-whisper transcription decodes audio via
bundled PyAV, but **speaker diarization still needs system `ffmpeg`** (the input is normalised to WAV
first — see Requirements). CUDA needs a working driver and a GPU new enough for the current
PyTorch/CTranslate2 build, plus **cuBLAS + cuDNN 9** for CTranslate2 — run `./install.sh --with-cuda`
(or `uv sync --extra cuda`) if they aren't already on the system, or you'll see a CTranslate2 load
error at first transcription.

## Running as a background service

The server runs in the foreground; for persistence use your platform's service manager.
`./install.sh --service` prints a ready-to-use unit for the current platform (a launchd agent on
macOS, a systemd `--user` unit on Linux) with the correct absolute paths filled in.

## Diarization notes

- Speaker labels are per-episode for a single request. Across a **chunked** client, use the
  `speakers[].embedding` voiceprints to link speakers by cosine similarity (same voice typically
  ≥ ~0.9, different voices ≤ ~0.3).
- The speaker **count** is auto-estimated per request; it is not fixed.
- A diarization failure never fails the request — the transcript comes back with a
  `diarization_error` field instead.

## Speaker recognition

Diarization labels speakers *within* one recording (`SPEAKER_00`…); recognition puts **names** on
them **across** recordings. Enrolled voiceprints live in a local SQLite library (`SPEAKER_DB`,
gitignored) — no extra model, and nothing leaves the box.

**Enroll** a speaker (one voiceprint is enough; add more to sharpen it):

- **Web UI** — transcribe with **Label speakers**, then type a name next to a detected speaker and **Save**.
- **From a prior result** — `POST /v1/speakers` with `{"name": "...", "embedding": [...]}`, reusing a
  `speakers[].embedding` returned by a transcription.
- **From an audio sample** — `POST /v1/speakers/from-audio` (multipart `name` + `file`): diarizes the
  clip and stores the dominant speaker's voiceprint. Use a clean, single-speaker sample.

**Recognize** — add `identify=true` to a `/v1/audio/transcriptions` request (it implies
`diarize=true`). Matched `speakers[]` entries gain `name`, `speaker_id`, and `match_score`; unmatched
speakers stay anonymous.

**Manage** — `GET /v1/speakers` (list) · `GET /v1/speakers/{id}` (`?centroid=1` for the mean
voiceprint) · `POST /v1/speakers/{id}/samples` (add a sample) · `DELETE /v1/speakers/{id}`.

Tune `SPEAKER_IDENTIFY_THRESHOLD` (default `0.5`): raise it to cut false matches, lower it to tolerate
more cross-condition variation. `SPEAKER_LIBRARY=0` disables the feature entirely on a locked-down host.

## SSO authentication (optional)

Off by default — with no `SECRECORDER_OIDC_*` set, the API and UI are open, exactly as before. Set
`SECRECORDER_OIDC_ISSUER` + `SECRECORDER_OIDC_CLIENT_ID` to require a valid **SecSSO** (OIDC) token on
every `/v1/*` route: a programmatic client sends `Authorization: Bearer <token>` and it is verified
against the issuer's JWKS (RS256 pinned, plus issuer, audience, expiry). `/health` stays open.

For the **built-in web UI**, additionally set `SECRECORDER_OIDC_CLIENT_SECRET` +
`SECRECORDER_PUBLIC_URL` + `SECRECORDER_SESSION_SECRET` to enable a server-side login BFF: an
unauthenticated browser at `/` is bounced through SecSSO (Authorization Code + PKCE), and the only
credential the browser ever holds is an httpOnly `secrecorder_session` cookie — it never sees an OIDC
token. Routes: `GET /auth/login` · `GET /auth/callback` · `POST /auth/logout` · `GET /auth/status`.

In the SecRouter suite, SecDeploy wires this automatically (a `secrecorder` OIDC client in SecSSO +
the env above). Override the audience with `SECRECORDER_OIDC_AUDIENCE`; skip OIDC discovery with
`SECRECORDER_OIDC_{JWKS,AUTHORIZE,TOKEN}_URL`.

## Summarization (optional)

Off by default. Set `SECRECORDER_SUMMARIZE_ENABLED=1` + `SECRECORDER_SUMMARIZE_ENDPOINT` (an
OpenAI-compatible base URL) to attach a `summary` to transcripts via a chat/completions call. Two
surfaces:

- **inline** — pass `summarize=true` on `POST /v1/audio/transcriptions`; the response gains a
  `summary` (or a `summary_error` if the LLM call fails — the transcript is never lost).
- **standalone** — `POST /v1/summarize` with `{"text": "…"}` summarizes any text (e.g. a prior
  transcript).

**Governed by default:** point the endpoint at **SecRouter** (`…/v1`) and each call carries
`X-Sec-Acting-User` (the authenticated caller), so the summarization LLM call is attributed, budgeted,
egress-controlled, and audited per user — the same governed path SecChat's assistant uses. It can
instead hit SecLLM directly or any endpoint. Tune with
`SECRECORDER_SUMMARIZE_{MODEL,API_KEY,PROMPT,MAX_CHARS,TIMEOUT}`.

## License

Copyright 2026 Austin Probe. Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

Model weights are **not** distributed with this project. Whisper weights (MLX / CTranslate2) and the
pyannote diarization models are downloaded at runtime under their own licenses; the pyannote models
are gated and require accepting their terms and providing `HF_TOKEN`.
