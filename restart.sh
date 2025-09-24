set -euo pipefail

echo "🔄 Restarting MuseTalk Realtime API server..."

echo "🛑 Killing existing process on port 8000..."
pkill -f 8000 || true

# Wait for the port to be freed
sleep 2

echo "🐍 Activating virtual environment..."
. /workspace/MuseTalk/.venv/bin/activate

echo "🚀 Starting MuseTalk Realtime API server..."
uv run fastapi dev /workspace/MuseTalk/realtime_api.py --port 8000 --host 0.0.0.0
