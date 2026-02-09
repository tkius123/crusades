#!/bin/bash
# Setup systemd service for Crusades upstream sync (runs every 60s).
#
# Usage (from repo root):
#   ./richardzhang_work/setup-sync-upstream-service.sh
#
# This installs crusades-sync-upstream.service so it starts on boot and
# keeps running, fetching/merging upstream into main once per minute.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo "~$CURRENT_USER")

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

SERVICE_FILE="/tmp/crusades-sync-upstream.service"
cp "$SCRIPT_DIR/crusades-sync-upstream.service" "$SERVICE_FILE"

sed -i "s|REPLACE_USER|$CURRENT_USER|g" "$SERVICE_FILE"
sed -i "s|REPLACE_WORKDIR|$PROJECT_DIR|g" "$SERVICE_FILE"
sed -i "s|REPLACE_HOME|$USER_HOME|g" "$SERVICE_FILE"

echo "Installing systemd service (requires sudo)..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/crusades-sync-upstream.service
sudo systemctl daemon-reload

sudo systemctl enable crusades-sync-upstream
sudo systemctl start crusades-sync-upstream

echo "Sync-upstream service installed and started."
echo "  status: sudo systemctl status crusades-sync-upstream"
echo "  logs:   sudo journalctl -u crusades-sync-upstream -f"
echo "  merge log: $PROJECT_DIR/richardzhang_work/sync-upstream.log"
