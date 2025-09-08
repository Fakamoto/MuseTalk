set -euo pipefail

apt-get update && apt-get install -y git ffmpeg build-essential ninja-build

cd /workspace
git clone https://github.com/Fakamoto/MuseTalk
cd MuseTalk

pip install -U uv
uv venv --python=python3.10
. .venv/bin/activate

uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

uv pip install -r requirements.txt
uv pip install -U huggingface_hub

uv pip install -U openmim
uv run mim install mmengine
uv run mim install "mmcv==2.0.1"
uv run mim install "mmdet==3.1.0"
uv run mim install "mmpose==1.1.0"


sh ./download_weights.sh || true

sh inference.sh v1.5 realtime