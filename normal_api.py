#!/usr/bin/env python3
"""
Normal MuseTalk API - Single endpoint that calls scripts.inference via subprocess
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# ====================================================================
# CONFIGURACIÓN
# ====================================================================

VERSION = "v15"
GPU_ID = 0
BATCH_SIZE = 8
RESULT_DIR = "./results"

THIS_DIR = Path(__file__).parent

# ====================================================================
# FASTAPI APP
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Normal MuseTalk API...")
    yield
    print("🛑 Shutting down Normal MuseTalk API...")

app = FastAPI(
    title="Normal MuseTalk API",
    description="Normal API that calls scripts.inference via subprocess",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/generate")
async def generate_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    """
    Generate video using scripts.inference subprocess
    - Receives video and audio files
    - Creates temporary config file
    - Calls scripts.inference via subprocess
    - Returns generated video
    """
    import time
    start_time = time.time()

    try:
        print(f"🎬 [NORMAL-GENERATE] Starting generation")

        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded files with original names
            video_path = temp_path / f"video{Path(video.filename).suffix}"
            audio_path = temp_path / f"audio{Path(audio.filename).suffix}"

            print("💾 Saving uploaded files...")
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

            # Create config file for normal inference (task-based format)
            config_content = f"""task_0:
 video_path: "{video_path}"
 audio_path: "{audio_path}"
"""

            config_path = temp_path / "config.yaml"
            with open(config_path, "w") as f:
                f.write(config_content)

            print(f"📝 Created config: {config_path}")

            # Build CLI command for normal inference
            cmd = [
                "uv", "run", "python", "-m", "scripts.inference",
                "--inference_config", str(config_path),
                "--result_dir", str(temp_path / "results"),
                "--unet_model_path", f"./models/musetalkV15/unet.pth",
                "--unet_config", f"./models/musetalkV15/musetalk.json",
                "--version", VERSION,
                "--gpu_id", str(GPU_ID),
                "--batch_size", str(BATCH_SIZE),
                "--fps", "25"
            ]

            print(f"🚀 Running command: {' '.join(cmd)}")

            # Execute the command
            result = subprocess.run(
                cmd,
                cwd=THIS_DIR,
                capture_output=False,
                text=True,
                timeout=60000  # 10 minutes timeout
            )

            if result.returncode != 0:
                print("❌ Command failed!")
                raise HTTPException(
                    status_code=500,
                    detail=f"Inference failed with return code {result.returncode}"
                )

            print("✅ Inference completed")

            # Find output video - normal inference creates files with different naming
            # Look for the generated video in the results directory
            results_dir = temp_path / "results" / VERSION
            if not results_dir.exists():
                raise HTTPException(status_code=500, detail="Results directory not found")

            # Search for any .mp4 files in the results directory
            mp4_files = list(results_dir.glob("**/*.mp4"))
            if not mp4_files:
                print(f"❌ No output video found in: {results_dir}")
                print("Available files:")
                for file in results_dir.rglob("*"):
                    print(f"  {file}")
                raise HTTPException(status_code=500, detail="No output video found")

            expected_output = mp4_files[0]
            print(f"✅ Found video at: {expected_output}")

            # Calculate duration
            end_time = time.time()
            duration = end_time - start_time
            print(f"Duration: {duration:.2f}s")

            # Return the video
            return FileResponse(
                path=expected_output,
                media_type='video/mp4',
                filename=f"generated_{duration:.2f}s.mp4"
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Generation timed out after 10 minutes")
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration:.2f}s")
        print(f"💥 Error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
