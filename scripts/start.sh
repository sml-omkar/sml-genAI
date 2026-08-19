#!/bin/bash
# =============================================================================
# AI-Bot Startup Script for EC2 (t4g.large)
# Run once after setup: chmod +x scripts/start.sh && ./scripts/start.sh
# =============================================================================

set -e

APP_DIR="/home/ubuntu/sml-genAI"
VENV_DIR="$APP_DIR/venv"

echo "=========================================="
echo "  AI-Bot Startup"
echo "=========================================="

# --- 1. Check prerequisites ---
echo ""
echo "[1/6] Checking prerequisites..."

# Python
if ! command -v python3.11 &> /dev/null; then
    echo "  Installing Python 3.11..."
    sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip
fi
echo "  Python: $(python3.11 --version)"

# PostgreSQL
if ! systemctl is-active --quiet postgresql; then
    echo "  Starting PostgreSQL..."
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
fi
echo "  PostgreSQL: running"

# Ollama
if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "  Ollama: installed"

# --- 2. Check/create PostgreSQL database ---
echo ""
echo "[2/6] Setting up database..."

# Create user and database if they don't exist
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='aibot_user'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER aibot_user WITH PASSWORD 'aibot_pass_2024';"

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='aibot_db'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE aibot_db OWNER aibot_user;"

echo "  Database ready."

# --- 3. Setup Python virtual environment ---
echo ""
echo "[3/6] Setting up Python environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$APP_DIR/requirements.txt"
echo "  Virtual environment ready."

# --- 4. Pull Ollama model ---
echo ""
echo "[4/6] Pulling SmolLM 360M model..."

# Start Ollama server in background if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve &
    sleep 3
fi

ollama pull smollm:360m 2>/dev/null || echo "  Model already pulled."
echo "  SmolLM 360M ready."

# --- 5. Initialize database tables ---
echo ""
echo "[5/6] Initializing database..."
cd "$APP_DIR"
python scripts/init_db.py

# --- 6. Start the application ---
echo ""
echo "[6/6] Starting AI-Bot server..."

# Pre-download embedding model (first run only)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')" 2>/dev/null || true

echo ""
echo "=========================================="
echo "  AI-Bot is running!"
echo "  Admin Portal: http://$(hostname -I | awk '{print $1}'):8000/admin"
echo "  API Docs:     http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "  Health:       http://$(hostname -I | awk '{print $1}'):8000/health"
echo ""
echo "  Login: admin@company.com / admin123"
echo "=========================================="

# Start with uvicorn
cd "$APP_DIR"
exec "$VENV_DIR/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000
