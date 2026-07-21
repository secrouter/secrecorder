#!/usr/bin/env bash
# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
# SecRecorder (aka SpeakerBox) installer.
#
#   ./install.sh [SECRETS_FILE]   set up the venv; if given, write SECRETS_FILE -> .env (chmod 600)
#   ./install.sh --service        print a ready-to-use service unit for this platform, then exit
#
# SECRETS_FILE is a KEY=VALUE env file (see secrets.env.example). Only HF_TOKEN is used today, and
# only for speaker diarization — transcription needs no secrets, so the argument is optional.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

# ---- uv ---------------------------------------------------------------------------------------
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then
  echo "error: 'uv' not found. Install it, then re-run:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# ---- --service: emit a persistence unit and exit ----------------------------------------------
if [ "${1:-}" = "--service" ]; then
  case "$(uname -s)" in
    Darwin)
      cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>speakerbox</string>
  <key>ProgramArguments</key>
  <array><string>$HERE/run.sh</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>HOST</key><string>0.0.0.0</string>
    <key>PATH</key><string>$(dirname "$UV"):/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>WorkingDirectory</key><string>$HERE</string>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HERE/speakerbox.log</string>
  <key>StandardErrorPath</key><string>$HERE/speakerbox.log</string>
</dict></plist>
PLIST
      echo "# Save to ~/Library/LaunchAgents/speakerbox.plist, then:" >&2
      echo "#   launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/speakerbox.plist" >&2
      ;;
    Linux)
      cat <<UNIT
[Unit]
Description=speakerbox (OpenAI-compatible Whisper ASR)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$HERE
Environment=HOST=0.0.0.0
Environment=PATH=$(dirname "$UV"):/usr/local/bin:/usr/bin:/bin
ExecStart=$HERE/run.sh
Restart=always
RestartSec=5
TimeoutStartSec=0

[Install]
WantedBy=default.target
UNIT
      echo "# Save to ~/.config/systemd/user/speakerbox.service, then:" >&2
      echo "#   systemctl --user daemon-reload && systemctl --user enable --now speakerbox" >&2
      echo "#   (survive logout/boot: sudo loginctl enable-linger \$USER)" >&2
      ;;
    *) echo "error: no service template for $(uname -s)" >&2; exit 1 ;;
  esac
  exit 0
fi

# ---- deps -------------------------------------------------------------------------------------
echo "==> installing dependencies with uv (backend auto-selected for this platform)…"
"$UV" sync

# ---- secrets ----------------------------------------------------------------------------------
SECRETS="${1:-}"
if [ -n "$SECRETS" ]; then
  [ -f "$SECRETS" ] || { echo "error: secrets file not found: $SECRETS" >&2; exit 1; }
  umask 077
  cp "$SECRETS" .env
  chmod 600 .env
  if grep -q '^HF_TOKEN=..' .env; then
    echo "==> wrote .env from $SECRETS (chmod 600) — HF_TOKEN present, diarization enabled"
  else
    echo "==> wrote .env from $SECRETS (chmod 600) — note: no HF_TOKEN, so diarization is disabled"
  fi
else
  echo "==> no secrets file given — transcription-only. For diarization:"
  echo "      cp secrets.env.example secrets.env && \$EDITOR secrets.env && ./install.sh secrets.env"
fi

echo
echo "Done. Start it with:"
echo "  ./run.sh                 # 127.0.0.1:9000"
echo "  HOST=0.0.0.0 ./run.sh    # expose on the LAN/VPN"
echo "  ./install.sh --service   # print a launchd/systemd unit for persistence"
