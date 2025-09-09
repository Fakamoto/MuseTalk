set -euo pipefail
touch ~/.no_auto_tmux


echo "🔧 Updating package lists and installing system dependencies..."
apt-get update && apt-get install -y git ffmpeg build-essential ninja-build



# mkdir -p /workspace
# cd /workspace
# git clone https://github.com/Fakamoto/MuseTalk
# cd MuseTalk


echo "🐍 Setting up Python environment with uv and venv..."
pip install -U uv
uv venv --python=python3.10 --seed
. .venv/bin/activate


echo "🔥 Installing PyTorch and related packages (CUDA 11.8)..."
uv pip install --constraint constraints.txt torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --constraint constraints.txt --no-cache-dir -U openmim
uv run mim install mmengine
uv run mim install "mmcv==2.0.1"
uv run mim install "mmdet==3.1.0"
uv run mim install "mmpose==1.1.0"


echo "📦 Installing Python requirements from requirements.txt..."
uv pip install --constraint constraints.txt -r requirements.txt
echo "⬇️ Downloading model weights..."
sh ./download_weights.sh || true


echo "🚀 Starting MuseTalk Realtime API server..."
uv run fastapi dev realtime_api.py --port 8000 --host 0.0.0.0