# SecRecorder

*aka SpeakerBox* — a self-hosted, OpenAI-compatible Whisper speech-to-text server with optional
speaker diarization. One `server.py` runs on either backend, auto-selected for the host:
**Apple Silicon** → **MLX** (Metal GPU) via `mlx-whisper`; **Linux / NVIDIA** →
**faster-whisper** (CTranslate2) on CUDA, or CPU anywhere. Part of the
[SecRouter suite](https://github.com/secrouter/secdeploy#the-suite).

It speaks the OpenAI audio-transcriptions API, so existing OpenAI clients work by pointing
`base_url` at it. Word-level timestamps are always returned. Optional diarization (pyannote) adds
per-word speaker labels and per-speaker voiceprints; optional recognition names those speakers
across recordings; optional summarization attaches a governed LLM-generated summary; and every
lifecycle event lands in a tamper-evident, metadata-only audit trail.

## Features

- **Built-in web UI** at `/` — record from the mic or drop an audio file, transcribe with optional
  speaker labels + recognition, enroll speakers into the library, and copy/export the notes.
- **OpenAI-compatible** `POST /v1/audio/transcriptions` (multipart; `verbose_json` with a top-level
  `words[]` array), plus `GET /v1/models` and `GET /health`.
- **Multi-backend**, one codebase (`WHISPER_BACKEND=auto|mlx|faster-whisper`).
- **Speaker diarization + recognition** (both opt-in) — label speakers per recording, then
  recognize enrolled speakers by name across recordings via a local voiceprint library.
- **Dead-air guard** — near-silent audio is returned as an empty transcript instead of letting
  Whisper hallucinate on it.
- **Optional SSO auth** (off by default) and **optional, governed summarization** (off by
  default) — see [docs/security.md](docs/security.md).
- **Tamper-evident audit trail** (on by default) — see [docs/security.md](docs/security.md).
- **Built for long recordings:** transcription runs off the event loop (`/health` stays
  responsive), GPU work is serialized, uploads stream to disk, and the model is prewarmed at
  startup.

## Quickstart

```bash
git clone <this-repo> secrecorder && cd secrecorder

# (optional) diarization: copy the template, add your HF token
cp secrets.env.example secrets.env
$EDITOR secrets.env

./install.sh secrets.env          # sets up the venv; writes .env (chmod 600) from your secrets
./run.sh                          # 127.0.0.1:9000, prewarmed
```

Omit `secrets.env` to install transcription-only. Open **http://localhost:9000/** for the
built-in web UI, or point any OpenAI client at the API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")
client.audio.transcriptions.create(model="whisper-1", file=open("audio.wav", "rb"),
                                    response_format="verbose_json")
```

## Documentation

See [docs/](docs/index.md) for the full reference:

- [docs/configuration.md](docs/configuration.md) — every environment variable
- [docs/usage.md](docs/usage.md) — full API walkthrough + the web UI tour
- [docs/deploy.md](docs/deploy.md) — install prerequisites, running as a service, SecProxy fronting
- [docs/security.md](docs/security.md) — auth model, audit discipline, voice-biometric handling
- [docs/control-validation.md](docs/control-validation.md) — CMMC / NIST SP 800-171 control mapping

## Related suite components

Optional integrations, both off by default: [SecSSO](https://github.com/secrouter/secsso) for
bearer/browser authentication, and [SecRouter](https://github.com/secrouter/secrouter) as the
default governed path for summarization (attribution, budget, egress control, audit — the same
pattern [SecChat](https://github.com/secrouter/secchat)'s assistant uses).

## License

Copyright 2026 Austin Probe. Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

Model weights are **not** distributed with this project. Whisper weights (MLX / CTranslate2) and
the pyannote diarization models are downloaded at runtime under their own licenses; the pyannote
models are gated and require accepting their terms and providing `HF_TOKEN`.
