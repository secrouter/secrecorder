# SecRecorder

*aka SpeakerBox* — a self-hosted, OpenAI-compatible Whisper speech-to-text server with optional speaker diarization.
One `server.py` runs on either backend, auto-selected for the host:

- **Apple Silicon** → **MLX** (Metal GPU) via `mlx-whisper`
- **Linux / NVIDIA** → **faster-whisper** (CTranslate2) on CUDA, or CPU anywhere

It speaks the OpenAI audio-transcriptions API, so existing OpenAI clients work by pointing
`base_url` at it. Word-level timestamps are always returned. Optional diarization (pyannote) adds
per-word speaker labels and per-speaker voiceprints.

## Features

- **OpenAI-compatible** `POST /v1/audio/transcriptions` (multipart; `verbose_json` with a top-level
  `words[]` array), plus `GET /v1/models` and `GET /health`.
- **Multi-backend**, one codebase (`WHISPER_BACKEND=auto|mlx|faster-whisper`).
- **Speaker diarization** (opt-in): pass `diarize=true` and every word/segment gains a `speaker`,
  plus a `speakers[]` summary. Each speaker also carries a voiceprint `embedding`, so a client that
  chunks long audio can link the same speaker across independently-diarized chunks.
- **Built for long recordings:** the blocking transcription runs off the event loop (so `/health`
  stays responsive), GPU work is serialized, uploads stream to disk, and the model is prewarmed at
  startup so a restart has no cold-start hit.
- **Robust output:** non-finite floats from the model (which would otherwise break JSON) are
  scrubbed to `null` rather than failing the request.

## Requirements

- [`uv`](https://docs.astral.sh/uv/) (the installer uses it to manage Python + deps; no system
  Python needed).
- Apple Silicon (for the MLX backend) **or** Linux with an NVIDIA GPU / any CPU (faster-whisper).
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
./run.sh                          # 127.0.0.1:9000, prewarmed
# HOST=0.0.0.0 ./run.sh           # expose on the LAN/VPN
```

Omit `secrets.env` to install transcription-only:

```bash
./install.sh
```

Point any OpenAI client at it:

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
| `WHISPER_MODEL` | per-backend | mlx: `mlx-community/whisper-large-v3-turbo` · faster-whisper: `large-v3` |
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

`WHISPER_*` are read from the process environment (e.g. `run.sh`); `HF_TOKEN` is read from `.env`
(written by `install.sh`, chmod 600). Per-request, `diarize=true|false` overrides `WHISPER_DIARIZE`.

## Backends

### Apple (MLX / Metal)
`./run.sh`. Defaults to `large-v3-turbo`. Warm calls run many× realtime on Apple Silicon.

### Linux / NVIDIA (faster-whisper)
`WHISPER_BACKEND=faster-whisper WHISPER_DEVICE=cuda ./run.sh`. `install.sh` picks faster-whisper via
the `platform_system == 'Linux'` dependency marker. faster-whisper decodes audio via bundled PyAV,
so no system `ffmpeg` is required. CUDA needs a working driver and a GPU new enough for the current
PyTorch/CTranslate2 build.

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

## License

Copyright 2026 Austin Probe. Licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

Model weights are **not** distributed with this project. Whisper weights (MLX / CTranslate2) and the
pyannote diarization models are downloaded at runtime under their own licenses; the pyannote models
are gated and require accepting their terms and providing `HF_TOKEN`.
