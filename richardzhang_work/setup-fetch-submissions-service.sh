#!/bin/bash
# Setup systemd service for Crusades top-submissions fetcher.
#
# Usage (from repo root):
#   ./richardzhang_work/setup-fetch-submissions-service.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER=$(whoami)
USER_HOME=$(eval echo "~$CURRENT_USER")

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

SERVICE_FILE="/tmp/crusades-fetch-submissions.service"
cp "$SCRIPT_DIR/crusades-fetch-submissions.service" "$SERVICE_FILE"

sed -i "s|REPLACE_USER|$CURRENT_USER|g" "$SERVICE_FILE"
sed -i "s|REPLACE_WORKDIR|$PROJECT_DIR|g" "$SERVICE_FILE"
sed -i "s|REPLACE_HOME|$USER_HOME|g" "$SERVICE_FILE"

echo "Installing systemd service (requires sudo)..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/crusades-fetch-submissions.service
sudo systemctl daemon-reload

sudo systemctl enable crusades-fetch-submissions
sudo systemctl start crusades-fetch-submissions

echo "Fetch-submissions service installed and started."
echo "  status:  sudo systemctl status crusades-fetch-submissions"
echo "  logs:    sudo journalctl -u crusades-fetch-submissions -f"
echo "  data:    $PROJECT_DIR/richardzhang_work/top_submissions/"
echo "  log:     $PROJECT_DIR/richardzhang_work/fetch-top-submissions.log"
