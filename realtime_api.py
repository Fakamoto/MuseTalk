#!/usr/bin/env python3
"""
MuseTalk Realtime API Server
FastAPI server with three endpoints for avatar generation
"""

import os
import subprocess

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

# ====================================================================
# CONFIGURACIÓN GLOBAL
# ====================================================================

VERSION = "v15"
GPU_ID = 0
BBOX_SHIFT = 5
FPS = 25
BATCH_SIZE = 20

# Directorio para archivos temporales
TEMP_DIR = Path("./temp_api_files")
TEMP_DIR.mkdir(exist_ok=True)

# Caché global de avatares preparados
avatar_cache = {}

# ====================================================================
# FASTAPI APP
# ====================================================================

app = FastAPI(
    title="MuseTalk Realtime API",
    description="API for realtime avatar generation with caching",
    version="1.0.0"
)

def get_model_paths():
    """Get model paths based on version"""
    if VERSION == "v1":
        model_dir = "./models/musetalk"
        unet_model_path = f"{model_dir}/pytorch_model.bin"
        unet_config = f"{model_dir}/musetalk.json"
    elif VERSION == "v15":
        model_dir = "./models/musetalkV15"
        unet_model_path = f"{model_dir}/unet.pth"
        unet_config = f"{model_dir}/musetalk.json"
    else:
        raise HTTPException(status_code=400, detail="Invalid version. Use 'v1' or 'v15'")

    return unet_model_path, unet_config

def create_yaml_config(avatar_id: str, video_path: str, audio_clips: dict, preparation: bool = True):
    """Create temporary YAML config file"""
    yaml_content = f"""{avatar_id}:
 preparation: {str(preparation).lower()}
 bbox_shift: {BBOX_SHIFT}
 video_path: "{video_path}"
 audio_clips:
"""

    for audio_name, audio_path in audio_clips.items():
        yaml_content += f"""     {audio_name}: "{audio_path}"
"""

    # Write to temporary config file
    config_path = TEMP_DIR / f"config_{avatar_id}_{hash(str(audio_clips))}.yaml"
    with open(config_path, 'w') as f:
        f.write(yaml_content)

    return config_path

def run_inference(config_path: str):
    """Run realtime inference with given config"""
    unet_model_path, unet_config = get_model_paths()

    cmd_args = [
        "python3", "-m", "scripts.realtime_inference",
        "--inference_config", str(config_path),
        "--gpu_id", str(GPU_ID),
        "--version", VERSION,
        "--bbox_shift", str(BBOX_SHIFT),
        "--fps", str(FPS),
        "--batch_size", str(BATCH_SIZE)
    ]

    try:
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e.stderr}"
        )

def save_uploaded_file(upload_file: UploadFile, filename: str) -> str:
    """Save uploaded file and return path"""
    file_path = TEMP_DIR / filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return str(file_path)

# ====================================================================
# ENDPOINTS
# ====================================================================

@app.post("/generate")
async def generate_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    avatar_id: str = Form("generated_avatar")
):
    """
    Endpoint 1: Generate video from scratch (no caching)
    - Receives video and audio files
    - Processes from scratch (preparation=True)
    - Returns the generated video
    """
    try:
        # Validate file types
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(status_code=400, detail="Video must be MP4, AVI, or MOV")
        if not audio.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            raise HTTPException(status_code=400, detail="Audio must be MP3, WAV, or M4A")

        # Save uploaded files
        video_path = save_uploaded_file(video, f"{avatar_id}_video{video.filename}")
        audio_path = save_uploaded_file(audio, f"{avatar_id}_audio{audio.filename}")

        # Create audio clips dict
        audio_clips = {
            "generated": audio_path
        }

        # Create config file
        config_path = create_yaml_config(avatar_id, video_path, audio_clips, preparation=True)

        # Run inference
        stdout, stderr = run_inference(config_path)

        # Find generated video
        results_dir = Path("./results") / VERSION / "avatars" / avatar_id / "vid_output"
        if results_dir.exists():
            video_files = list(results_dir.glob("*.mp4"))
            if video_files:
                output_video = video_files[0]

                # Clean up temp files
                config_path.unlink(missing_ok=True)

                return FileResponse(
                    path=output_video,
                    media_type='video/mp4',
                    filename=f"{avatar_id}_generated.mp4"
                )

        raise HTTPException(status_code=500, detail="Video generation completed but output file not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/prepare")
async def prepare_avatar(
    video: UploadFile = File(...),
    avatar_id: str = Form("prepared_avatar")
):
    """
    Endpoint 2: Prepare avatar for caching
    - Receives only video file
    - Prepares avatar and saves to cache
    - No audio processing needed
    """
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(status_code=400, detail="Video must be MP4, AVI, or MOV")

        # Save uploaded file
        video_path = save_uploaded_file(video, f"{avatar_id}_prepare{video.filename}")

        # Create dummy audio clips (not used in preparation)
        audio_clips = {
            "dummy": video_path  # Use video path as dummy
        }

        # Create config file with preparation=True
        config_path = create_yaml_config(avatar_id, video_path, audio_clips, preparation=True)

        # Run inference (this will prepare the avatar)
        stdout, stderr = run_inference(config_path)

        # Mark avatar as prepared in cache
        avatar_cache[avatar_id] = {
            "video_path": video_path,
            "prepared": True,
            "bbox_shift": BBOX_SHIFT,
            "version": VERSION
        }

        # Clean up temp config
        config_path.unlink(missing_ok=True)

        return {
            "message": f"Avatar '{avatar_id}' prepared and cached successfully",
            "avatar_id": avatar_id,
            "status": "prepared"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preparation failed: {str(e)}")

@app.post("/generate-fast")
async def generate_fast(
    audio: UploadFile = File(...),
    avatar_id: str = Form(...)
):
    """
    Endpoint 3: Generate video using cached avatar
    - Requires avatar to be prepared first
    - Uses cached avatar for faster generation
    - Only processes audio
    """
    try:
        # Check if avatar is prepared
        if avatar_id not in avatar_cache:
            raise HTTPException(
                status_code=400,
                detail=f"Avatar '{avatar_id}' not found in cache. Use /prepare endpoint first."
            )

        if not avatar_cache[avatar_id]["prepared"]:
            raise HTTPException(
                status_code=400,
                detail=f"Avatar '{avatar_id}' is not prepared. Use /prepare endpoint first."
            )

        # Validate audio file
        if not audio.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            raise HTTPException(status_code=400, detail="Audio must be MP3, WAV, or M4A")

        # Get cached avatar data
        avatar_data = avatar_cache[avatar_id]
        video_path = avatar_data["video_path"]

        # Save uploaded audio
        audio_path = save_uploaded_file(audio, f"{avatar_id}_fast{audio.filename}")

        # Create audio clips dict
        audio_clips = {
            "fast_generated": audio_path
        }

        # Create config file with preparation=False (use cache)
        config_path = create_yaml_config(avatar_id, video_path, audio_clips, preparation=False)

        # Run inference (this will use the cached avatar)
        stdout, stderr = run_inference(config_path)

        # Find generated video
        results_dir = Path("./results") / VERSION / "avatars" / avatar_id / "vid_output"
        if results_dir.exists():
            video_files = list(results_dir.glob("*.mp4"))
            if video_files:
                output_video = video_files[0]

                # Clean up temp files
                config_path.unlink(missing_ok=True)

                return FileResponse(
                    path=output_video,
                    media_type='video/mp4',
                    filename=f"{avatar_id}_fast_generated.mp4"
                )

        raise HTTPException(status_code=500, detail="Fast generation completed but output file not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fast generation failed: {str(e)}")

@app.get("/cache")
async def get_cache_status():
    """Get current cache status"""
    return {
        "cached_avatars": list(avatar_cache.keys()),
        "cache_info": avatar_cache
    }

@app.delete("/cache/{avatar_id}")
async def clear_cache(avatar_id: str):
    """Clear specific avatar from cache"""
    if avatar_id in avatar_cache:
        # Clean up temp files
        avatar_data = avatar_cache[avatar_id]
        video_path = avatar_data.get("video_path")
        if video_path and os.path.exists(video_path):
            os.remove(video_path)

        del avatar_cache[avatar_id]
        return {"message": f"Avatar '{avatar_id}' removed from cache"}
    else:
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found in cache")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MuseTalk Realtime API Server",
        "endpoints": {
            "POST /generate": "Generate video from scratch (video + audio)",
            "POST /prepare": "Prepare avatar for caching (video only)",
            "POST /generate-fast": "Generate video using cached avatar (audio only)",
            "GET /cache": "Get cache status",
            "DELETE /cache/{avatar_id}": "Clear specific avatar from cache"
        },
        "version": VERSION,
        "cached_avatars": list(avatar_cache.keys())
    }
