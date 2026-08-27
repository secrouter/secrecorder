# Deploy

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — the installer uses it to manage Python + deps; no system
  Python needed.
- Apple Silicon (for the MLX backend) **or** Linux with an NVIDIA GPU / any CPU (faster-whisper).
- **`ffmpeg`** on `PATH` — decodes audio for the MLX backend and normalises the diarizer's input
  to WAV (all backends). Install yourself (`brew install ffmpeg` / `apt-get install ffmpeg`), or
  `./install.sh --with-ffmpeg`.
- **NVIDIA GPU (faster-whisper backend):** CTranslate2 needs cuBLAS + cuDNN 9 at runtime and does
  not bundle them. `./run.sh` auto-installs them (the `cuda` extra) when it detects an NVIDIA GPU;
  to pre-install at setup use `./install.sh --with-cuda` (or `uv sync --extra cuda`).
- For diarization only: a Hugging Face token whose account has accepted the
  [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  conditions. Transcription needs no token.

## Install + run

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

Omit `secrets.env` to install transcription-only (no diarization, no `HF_TOKEN` needed):

```bash
./install.sh
```

`run.sh` sets `WHISPER_PREWARM=1` and `WHISPER_PREWARM_DIARIZER=1` by default, so the model(s)
load at startup rather than on the first request.

### Backends

**Apple (MLX / Metal)** — just `./run.sh`. Defaults to `large-v3-turbo`; warm calls run many×
realtime on Apple Silicon.

**Linux / NVIDIA (faster-whisper)** —
`WHISPER_BACKEND=faster-whisper WHISPER_DEVICE=cuda ./run.sh`. `install.sh` picks faster-whisper
via the `platform_system == 'Linux'` dependency marker. faster-whisper transcription decodes audio
via bundled PyAV, but **speaker diarization still needs system `ffmpeg`** (the input is normalised
to WAV first). CUDA needs a working driver and a GPU new enough for the current PyTorch/
CTranslate2 build, plus cuBLAS + cuDNN 9 for CTranslate2 — run `./install.sh --with-cuda` (or
`uv sync --extra cuda`) if they aren't already on the system, or you'll see a CTranslate2 load
error at first transcription.

## Running as a service

The server runs in the foreground; for persistence use your platform's service manager.
`./install.sh --service` prints a ready-to-use unit for the current platform with the correct
absolute paths filled in:

- **macOS** — a `launchd` agent (`~/Library/LaunchAgents/speakerbox.plist`,
  `launchctl bootstrap gui/$(id -u) …`).
- **Linux** — a `systemd --user` unit (`~/.config/systemd/user/speakerbox.service`,
  `systemctl --user enable --now speakerbox`; `loginctl enable-linger $USER` to survive logout).

In the SecRouter suite's Fedora FIPS target, SecRecorder instead runs as a system service under
its own `secsuite-secrecorder` user (`/opt/secsuite/secrecorder`, `deploy/fedora-fips/systemd/
secrecorder.service` in the suite's `secdeploy` repo), started via `run.sh` and configured from
`/etc/secsuite/secrecorder.env` plus an optional, secdeploy-generated
`/etc/secsuite/secrecorder-addressing.env` (topology-derived OIDC issuer/client/public-URL and the
SecRouter-governed summarize endpoint) — the addressing file is loaded second, so it overrides any
same-named key in the operator's own `.env`.

## Fronting with SecProxy

Standalone, SecRecorder listens directly on `HOST:PORT` (default `127.0.0.1:9000`) with plain
HTTP and no path prefix. In a SecRouter suite deployment where the `edge` tier (**SecProxy**) is
placed, SecRecorder is one of the components SecProxy fronts: it gets a bare `https://<fqdn>`
(443 implied, TLS terminated at the proxy) instead of `http://<fqdn>:9000`, and
`SECRECORDER_PUBLIC_URL` is set to that fronted URL — it's what builds the OIDC redirect
(`<SECRECORDER_PUBLIC_URL>/auth/callback`) and decides the session cookie's `Secure` flag. Without
an edge tier, SecRecorder is reachable directly at its ported URL and none of the above changes;
this is purely how the URL callers use is computed, not a change to the server itself. `/health`
is safe to point a load balancer or SecProxy health check at — it is never gated by auth.
