#!/bin/bash
# Setup systemd service for Crusades improve-and-submit.
#
# Usage (from repo root):
#   ./richardzhang_work/setup-improve-service.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo "~$CURRENT_USER")

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found."
  exit 1
fi

SERVICE_FILE="/tmp/crusades-improve.service"
cp "$SCRIPT_DIR/crusades-improve.service" "$SERVICE_FILE"

sed -i "s|REPLACE_USER|$CURRENT_USER|g" "$SERVICE_FILE"
sed -i "s|REPLACE_WORKDIR|$PROJECT_DIR|g" "$SERVICE_FILE"
sed -i "s|REPLACE_HOME|$USER_HOME|g" "$SERVICE_FILE"

if [ -f /etc/crusades/gaming-check.env ]; then
  sed -i '/^ExecStart=/i EnvironmentFile=/etc/crusades/gaming-check.env' "$SERVICE_FILE"
fi

echo "Installing systemd service (requires sudo)..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/crusades-improve.service
sudo systemctl daemon-reload
sudo systemctl enable crusades-improve
sudo systemctl start crusades-improve

echo "Improve service installed and started."
echo "  status:  sudo systemctl status crusades-improve"
echo "  logs:    sudo journalctl -u crusades-improve -f"
echo "  output:  $PROJECT_DIR/richardzhang_work/improved/"
