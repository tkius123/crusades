#!/bin/bash
# Setup script for Crusades API systemd service
# 
# Usage:
#   ./scripts/setup-api-service.sh
#
# This script:
# 1. Creates a systemd service for the Crusades API
# 2. Enables it to start on boot
# 3. Starts the service
#
# Prerequisites:
# - Python virtual environment at .venv/
# - uv installed and dependencies synced (uv sync)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Crusades API Service Setup ===${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CURRENT_USER=$(whoami)

echo "Project directory: $PROJECT_DIR"
echo "Current user: $CURRENT_USER"
echo ""

# Check if venv exists
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo -e "${RED}Error: Virtual environment not found at $PROJECT_DIR/.venv${NC}"
    echo "Please run 'uv sync' first to create the virtual environment."
    exit 1
fi

# Check if crusades-api is available
if [ ! -f "$PROJECT_DIR/.venv/bin/crusades-api" ]; then
    echo -e "${RED}Error: crusades-api not found in venv${NC}"
    echo "Please run 'uv sync' to install dependencies."
    exit 1
fi

echo -e "${GREEN}✓ Virtual environment found${NC}"

# Create service file from template
SERVICE_FILE="/tmp/crusades-api.service"
cp "$SCRIPT_DIR/crusades-api.service" "$SERVICE_FILE"

# Replace placeholders
sed -i "s|REPLACE_USER|$CURRENT_USER|g" "$SERVICE_FILE"
sed -i "s|REPLACE_WORKDIR|$PROJECT_DIR|g" "$SERVICE_FILE"

echo -e "${GREEN}✓ Service file configured${NC}"
echo ""

# Show the configured service file
echo "Service configuration:"
echo "----------------------------------------"
cat "$SERVICE_FILE"
echo "----------------------------------------"
echo ""

# Install the service
echo "Installing systemd service (requires sudo)..."
sudo cp "$SERVICE_FILE" /etc/systemd/system/crusades-api.service
sudo systemctl daemon-reload

echo -e "${GREEN}✓ Service installed${NC}"

# Enable the service
sudo systemctl enable crusades-api
echo -e "${GREEN}✓ Service enabled (will start on boot)${NC}"

# Start the service
echo ""
echo "Starting the service..."
sudo systemctl start crusades-api
sleep 2

# Check status
if sudo systemctl is-active --quiet crusades-api; then
    echo -e "${GREEN}✓ Service is running${NC}"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    echo "Check logs with: sudo journalctl -u crusades-api -n 50"
    exit 1
fi

# Test the API
echo ""
echo "Testing API endpoint..."
sleep 1
HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo "failed")

if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✓ API is responding${NC}"
    echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"
else
    echo -e "${YELLOW}⚠ API not responding yet, may need a moment to start${NC}"
fi

# Get public IP
echo ""
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s icanhazip.com 2>/dev/null || echo "unknown")
echo "=== Setup Complete ==="
echo ""
echo "API URL (local):  http://localhost:8080"
echo "API URL (public): http://$PUBLIC_IP:8080"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status crusades-api   # Check status"
echo "  sudo systemctl restart crusades-api  # Restart"
echo "  sudo systemctl stop crusades-api     # Stop"
echo "  sudo journalctl -u crusades-api -f   # View logs"
echo ""
echo -e "${GREEN}Remember to set CRUSADES_API_URL in Vercel:${NC}"
echo "  CRUSADES_API_URL=http://$PUBLIC_IP:8080"


