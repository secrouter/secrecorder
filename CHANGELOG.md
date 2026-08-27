# Changelog

All notable changes to SecRecorder (aka SpeakerBox), seeded from `git log --oneline`.

## [Unreleased]

### Security & compliance
- **Tamper-evident audit trail** (`audit.py`) — hash-chained JSONL log of every transcription run,
  speaker enroll/update/delete, summarize request, and auth failure. On by default
  (`SECRECORDER_AUDIT_ENABLED`); `GET /v1/audit/verify` checks the chain, `GET /v1/evidence`
  returns a one-shot CMMC evidence bundle. Metadata only — transcript text, summaries, prompts, and
  audio/voiceprint bytes never reach a record. See [docs/control-validation.md](docs/control-validation.md).
- **Optional SSO auth** (SecSSO/OIDC) — bearer JWT verification on `/v1/*` plus a browser-login BFF
  (Authorization Code + PKCE, httpOnly session cookie) for the built-in web UI. Off by default.
- **Optional, governed summarization** — attach a `summary` to a transcript via any
  OpenAI-compatible chat endpoint, routed through SecRouter by default
  (`X-Sec-Acting-User` attribution) with an egress-classification header
  (`SECRECORDER_SUMMARIZE_CLASSIFICATION`). Off by default.
- **Dead-air guard** (`WHISPER_SILENCE_MAX_DB`) — near-silent/degenerate audio is rejected before
  reaching Whisper (which otherwise hallucinates fabricated text on silence) and returned as an
  empty transcript instead.

### Backend
- CUDA extra (cuBLAS/cuDNN) + `install.sh --with-cuda` for the faster-whisper GPU backend;
  `run.sh` auto-installs the extra when it detects an NVIDIA GPU.
- `large-v3-turbo` as the default model on faster-whisper too (CT2 turbo build), matching MLX.
- `run.sh` no longer forces the MLX model id as the default, which had broken the
  faster-whisper/CUDA path.

## [0.8.2]
- Reset `prevHyp`/`lastTime` on the live-transcription rolling window's forced drop, fixing
  word/phrase duplication in live mode.

## [0.8.1]
- Reduced live-transcription word repetition.

## [0.8.0]
- Live transcription MVP: rolling-window transcription with LocalAgreement stitching, carrying
  committed text forward across windows via `prompt`.

## [0.7.0] and earlier
- Speaker recognition — enroll named voiceprints (`/v1/speakers`, `/v1/speakers/from-audio`) and
  identify speakers across recordings by cosine similarity (`identify=true`).
- Built-in web UI (`GET /`) — record from the mic or upload a file, transcribe with optional
  speaker labels, copy/export the result.
- Initial release: multi-backend (MLX / faster-whisper) OpenAI-compatible Whisper ASR server with
  word-level timestamps and optional speaker diarization (pyannote).
