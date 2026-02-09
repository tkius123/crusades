#!/bin/bash
# Setup systemd service for Crusades gaming check (Opus classifies new top-5 as gaming or not).
#
# Prerequisite: Set your Cursor API key and repo for the service:
#   sudo mkdir -p /etc/crusades
#   printf 'CURSOR_API_KEY=key-...\nCURSOR_REPO=https://github.com/tkius123/crusades\n' | sudo tee /etc/crusades/gaming-check.env
#   sudo chmod 600 /etc/crusades/gaming-check.env
# Then run this script.
#
# Usage (from repo root):
#   ./richardzhang_work/setup-check-gaming-service.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo "~$CURRENT_USER")

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found. Install it first."
  exit 1
fi

SERVICE_FILE="/tmp/crusades-check-gaming.service"
cp "$SCRIPT_DIR/crusades-check-gaming.service" "$SERVICE_FILE"

sed -i "s|REPLACE_USER|$CURRENT_USER|g" "$SERVICE_FILE"
sed -i "s|REPLACE_WORKDIR|$PROJECT_DIR|g" "$SERVICE_FILE"
sed -i "s|REPLACE_HOME|$USER_HOME|g" "$SERVICE_FILE"

# If env file exists, add it to the unit
if [ -f /etc/crusades/gaming-check.env ]; then
  sed -i '/^ExecStart=/i EnvironmentFile=/etc/crusades/gaming-check.env' "$SERVICE_FILE"
fi

echo "Installing systemd service (requires sudo)..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/crusades-check-gaming.service
sudo systemctl daemon-reload

sudo systemctl enable crusades-check-gaming
sudo systemctl start crusades-check-gaming

echo "Check-gaming service installed and started."
echo "  status:  sudo systemctl status crusades-check-gaming"
echo "  logs:    sudo journalctl -u crusades-check-gaming -f"
echo "  results: $PROJECT_DIR/richardzhang_work/gaming_checks/"
echo "  log:     $PROJECT_DIR/richardzhang_work/check-gaming.log"
echo ""
echo "If the service fails (missing keys), create:"
echo "  sudo mkdir -p /etc/crusades"
echo "  printf 'CURSOR_API_KEY=key-...\nCURSOR_REPO=https://github.com/tkius123/crusades\n' | sudo tee /etc/crusades/gaming-check.env"
echo "  sudo chmod 600 /etc/crusades/gaming-check.env"
echo "  sudo systemctl restart crusades-check-gaming"
