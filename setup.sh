set -euo pipefail
touch ~/.no_auto_tmux
export UV_HTTP_TIMEOUT=600 UV_HTTP_MAX_RETRIES=8

echo "🔧 Updating package lists and installing system dependencies..."
apt-get update && apt-get install -y git ffmpeg build-essential ninja-build



# mkdir -p /workspace
# cd /workspace
# git clone https://github.com/Fakamoto/MuseTalk
# cd MuseTalk
# /bin/bash ./setup.sh


echo "🐍 Setting up Python environment with uv and venv..."
pip install -U uv
uv venv --python=python3.10 --seed
. .venv/bin/activate


echo "🔥 Installing PyTorch and related packages (CUDA 11.8)..."
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --no-cache-dir -U openmim
uv run mim install mmengine
uv run mim install "mmcv==2.0.1"
uv run mim install "mmdet==3.1.0"
uv run mim install "mmpose==1.1.0"


echo "📦 Installing Python requirements from requirements.txt..."
uv pip install -r requirements.txt

uv pip uninstall opencv-python opencv-contrib-python numpy
uv pip install --no-cache-dir "numpy==1.23.5" "scipy==1.11.4" "pandas==2.2.2"
uv pip install --no-cache-dir --no-deps "opencv-python==4.9.0.80" "opencv-contrib-python==4.9.0.80"


echo "⬇️ Downloading model weights..."
sh ./download_weights.sh || true


# mkdir -p /workspace/MuseTalk/results/v15/avatars
# curl -fL --retry 5 --retry-all-errors -o /tmp/avatar_1_minute_preset.zip "https://audios-avatar.s3.eu-north-1.amazonaws.com/avatar_1_minute_preset.zip"
# unzip -o /tmp/avatar_1_minute_preset.zip -d /workspace/MuseTalk/results/v15/avatars/
# rm /tmp/avatar_1_minute_preset.zip


cp ./restart.sh /restart.sh
cp ./restart.sh /root/restart.sh
chmod +x /restart.sh

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONPATH="/workspace/MuseTalk:${PYTHONPATH:-}"

echo "🚀 Starting MuseTalk Realtime API server in tmux session..."

# Crear sesión tmux para el servidor
tmux new-session -s api
uv run fastapi run fast_api.py --port 8000 --host 0.0.0.0

echo "✅ Sesión tmux 'api' creada con el servidor corriendo."
echo "Para conectarte a la sesión usa: tmux attach -t api"
echo "Para ver sesiones activas: tmux list-sessions"
# uv run fastapi dev realtime_api.py --port 8000 --host 0.0.0.0


