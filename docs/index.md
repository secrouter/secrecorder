# SecRecorder

*aka SpeakerBox* — a self-hosted, OpenAI-compatible Whisper speech-to-text server with optional
speaker diarization, speaker recognition, transcript summarization, and a tamper-evident audit
trail. One `server.py` runs on either Apple MLX (Metal) or faster-whisper (NVIDIA CUDA / CPU),
auto-selected for the host, and speaks the OpenAI audio-transcriptions API so existing OpenAI
clients work by pointing `base_url` at it.

Part of the [SecRouter suite](https://github.com/secrouter/secdeploy#the-suite).

## Contents

- [configuration.md](configuration.md) — every environment variable, one row each
- [usage.md](usage.md) — API walkthrough + the built-in web UI tour
- [deploy.md](deploy.md) — running it as a service, fronting it with SecProxy, model prerequisites
- [security.md](security.md) — auth model, audit discipline, voice-biometric considerations
- [control-validation.md](control-validation.md) — CMMC / NIST SP 800-171 control mapping

See the repo [README](../README.md) for a quickstart.
