#!/usr/bin/env python3
"""
Simple MuseTalk API - Single endpoint that calls realtime_inference.py via subprocess
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
BATCH_SIZE = 20

THIS_DIR = Path(__file__).parent

# ====================================================================
# FASTAPI APP
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Simple MuseTalk API...")
    yield
    print("🛑 Shutting down Simple MuseTalk API...")

app = FastAPI(
    title="Simple MuseTalk API",
    description="Simple API that calls realtime_inference.py via subprocess",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/generate")
async def generate_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    avatar_id: str = "simple_generated"
):
    """
    Generate video using realtime_inference.py subprocess
    - Receives video and audio files
    - Creates temporary config file
    - Calls realtime_inference.py via subprocess
    - Returns generated video
    """
    import time
    start_time = time.time()

    try:
        print(f"🎬 [SIMPLE-GENERATE] Starting generation for avatar: {avatar_id}")

        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded files
            video_path = temp_path / f"{avatar_id}_video{Path(video.filename).suffix}"
            audio_path = temp_path / f"{avatar_id}_audio{Path(audio.filename).suffix}"

            print("💾 Saving uploaded files...")
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)

            # Create config file
            config_content = f"""{avatar_id}:
 preparation: true
 bbox_shift: 0
 video_path: "{video_path}"
 audio_clips:
   generated: "{audio_path}"
"""

            config_path = temp_path / f"config_{avatar_id}.yaml"
            with open(config_path, "w") as f:
                f.write(config_content)

            print(f"📝 Created config: {config_path}")

            # Build CLI command
            cmd = [
                "python", "scripts/realtime_inference.py",
                "--version", VERSION,
                "--gpu_id", str(GPU_ID),
                "--batch_size", str(BATCH_SIZE),
                "--inference_config", str(config_path)
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

            # Find output video
            expected_output = Path("./results") / VERSION / "avatars" / avatar_id / "vid_output" / "generated.mp4"

            if not expected_output.exists():
                print(f"❌ Output video not found at: {expected_output}")
                print("Checking other possible locations...")

                # Search for any .mp4 files in the avatar directory
                avatar_dir = Path("./results") / VERSION / "avatars" / avatar_id
                if avatar_dir.exists():
                    mp4_files = list(avatar_dir.glob("**/*.mp4"))
                    if mp4_files:
                        expected_output = mp4_files[0]
                        print(f"✅ Found video at: {expected_output}")
                    else:
                        raise HTTPException(status_code=500, detail="No output video found")

            # Calculate duration
            end_time = time.time()
            duration = end_time - start_time
            print(f"Duration: {duration:.2f}s")

            # Return the video
            return FileResponse(
                path=expected_output,
                media_type='video/mp4',
                filename=f"{duration:.2f}s_generated.mp4"
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Generation timed out after 10 minutes")
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration:.2f}s")
        print(f"💥 Error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
