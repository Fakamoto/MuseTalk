set -euo pipefail

echo "🔄 Restarting MuseTalk Realtime API server..."

echo "🛑 Killing existing process on port 8000..."
# Try multiple methods to kill the process
pkill -f "fastapi dev.*realtime_api.py.*--port 8000" || true
pkill -f "uv run fastapi dev.*realtime_api.py" || true

# Also try to kill by port number directly
if command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs -r kill -9 || true
elif command -v netstat >/dev/null 2>&1; then
    netstat -tlnp 2>/dev/null | grep :8000 | awk '{print $7}' | cut -d'/' -f1 | xargs -r kill -9 || true
fi

# Wait a moment for the port to be freed
sleep 2

echo "🐍 Activating virtual environment..."
. /workspace/MuseTalk/.venv/bin/activate

echo "🚀 Starting MuseTalk Realtime API server..."
uv run fastapi dev /workspace/MuseTalk/realtime_api.py --port 8000 --host 0.0.0.0
