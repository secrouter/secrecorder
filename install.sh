#!/usr/bin/env bash
# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
# SecRecorder (aka SpeakerBox) installer.
#
#   ./install.sh [SECRETS_FILE]   set up the venv; if given, write SECRETS_FILE -> .env (chmod 600)
#   ./install.sh --with-ffmpeg    also install the ffmpeg system binary if it is missing
#   ./install.sh --service        print a ready-to-use service unit for this platform, then exit
#
# Flags combine with SECRETS_FILE, e.g.  ./install.sh --with-ffmpeg secrets.env
# SECRETS_FILE is a KEY=VALUE env file (see secrets.env.example). Only HF_TOKEN is used today, and
# only for speaker diarization — transcription needs no secrets, so the argument is optional.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

# ---- args -------------------------------------------------------------------------------------
SERVICE=0; WITH_FFMPEG=0; SECRETS=""
for arg in "$@"; do
  case "$arg" in
    --service)                      SERVICE=1 ;;
    --with-ffmpeg|--install-ffmpeg) WITH_FFMPEG=1 ;;
    --*) echo "error: unknown option: $arg" >&2; exit 1 ;;
    *)   SECRETS="$arg" ;;
  esac
done

# ---- ffmpeg installer (used below when --with-ffmpeg is set and ffmpeg is missing) ------------
install_ffmpeg() {
  echo "==> installing ffmpeg…"
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null 2>&1 || {
        echo "error: Homebrew not found — install it from https://brew.sh, then: brew install ffmpeg" >&2
        return 1; }
      brew install ffmpeg ;;
    Linux)
      local SUDO=""
      if [ "$(id -u)" -ne 0 ]; then
        command -v sudo >/dev/null 2>&1 && SUDO="sudo" || {
          echo "error: installing ffmpeg needs root — re-run as root or install sudo." >&2; return 1; }
      fi
      if   command -v apt-get >/dev/null 2>&1; then $SUDO apt-get update && $SUDO apt-get install -y ffmpeg
      elif command -v dnf     >/dev/null 2>&1; then $SUDO dnf install -y ffmpeg
      elif command -v yum     >/dev/null 2>&1; then $SUDO yum install -y ffmpeg
      elif command -v pacman  >/dev/null 2>&1; then $SUDO pacman -Sy --noconfirm ffmpeg
      elif command -v zypper  >/dev/null 2>&1; then $SUDO zypper install -y ffmpeg
      elif command -v apk     >/dev/null 2>&1; then $SUDO apk add ffmpeg
      else
        echo "error: no supported package manager (apt/dnf/yum/pacman/zypper/apk) — install ffmpeg manually." >&2
        return 1
      fi ;;
    *) echo "error: automatic ffmpeg install unsupported on $(uname -s) — install it manually." >&2; return 1 ;;
  esac
  hash -r 2>/dev/null || true            # refresh the shell's command lookup
  command -v ffmpeg >/dev/null 2>&1      # succeeds iff ffmpeg is now on PATH
}

# ---- uv ---------------------------------------------------------------------------------------
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then UV="$HOME/.local/bin/uv"; fi
if [ -z "$UV" ]; then
  echo "error: 'uv' not found. Install it, then re-run:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# ---- --service: emit a persistence unit and exit ----------------------------------------------
if [ "$SERVICE" = "1" ]; then
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

# ---- ffmpeg (system binary, not a Python package) ---------------------------------------------
# Audio decode for the MLX backend + the diarizer's WAV normalisation (all backends). Must be on PATH.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg: $(command -v ffmpeg)"
elif [ "$WITH_FFMPEG" = "1" ]; then
  if install_ffmpeg; then
    echo "==> ffmpeg installed: $(command -v ffmpeg)"
  else
    echo "==> ERROR: could not install ffmpeg automatically (see messages above)." >&2
    exit 1
  fi
else
  echo "==> WARNING: 'ffmpeg' not found on PATH — required for the MLX backend and for speaker" >&2
  echo "             diarization/recognition. Re-run with --with-ffmpeg to install it, or:" >&2
  case "$(uname -s)" in
    Darwin) echo "               brew install ffmpeg" >&2 ;;
    Linux)  echo "               sudo apt-get install -y ffmpeg" >&2 ;;
    *)      echo "               install ffmpeg via your package manager." >&2 ;;
  esac
fi

# ---- secrets ----------------------------------------------------------------------------------
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
