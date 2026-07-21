#!/usr/bin/env bash
# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
# Start the Whisper server in the foreground.
#   ./run.sh                                  # 127.0.0.1:9000, model large-v3-turbo
#   PORT=9001 ./run.sh                        # override port
#   HOST=0.0.0.0 ./run.sh                     # expose on LAN/VPN (worker on another box)
#   WHISPER_MODEL=mlx-community/whisper-small ./run.sh   # lighter/faster model
set -euo pipefail
cd "$(dirname "$0")"
export WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-large-v3-turbo}"
# Optimized prod defaults (see OPTIMIZATION_PLAN.md): prewarm the model + diarizer at startup so a
# restart has no cold-start hit. fp16 turbo + concurrency=1 stay the defaults (q4 and conc=2 were
# tested and rejected — see bench/phaseB_q4_result.md, bench/phaseC_concurrency_result.md).
export WHISPER_PREWARM="${WHISPER_PREWARM:-1}"
export WHISPER_PREWARM_DIARIZER="${WHISPER_PREWARM_DIARIZER:-1}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-9000}"
echo "SecRecorder -> http://${HOST}:${PORT}   (model: ${WHISPER_MODEL}, prewarm: ${WHISPER_PREWARM})"
exec uv run uvicorn server:app --host "$HOST" --port "$PORT"
