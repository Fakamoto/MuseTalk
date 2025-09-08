#!/usr/bin/env python3
"""
MuseTalk Realtime API Server
FastAPI server with three endpoints for avatar generation
"""

import os
import subprocess
import time

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
# INICIALIZACIÓN Y VERIFICACIONES
# ====================================================================

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment setup...")

    # Check required directories
    required_dirs = [
        Path("./models"),
        Path("./results"),
        TEMP_DIR
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"⚠️  Directory missing: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")

    # Check model files
    model_files = []
    if VERSION == "v1":
        model_files = [
            Path("./models/musetalk/pytorch_model.bin"),
            Path("./models/musetalk/musetalk.json")
        ]
    elif VERSION == "v15":
        model_files = [
            Path("./models/musetalkV15/unet.pth"),
            Path("./models/musetalkV15/musetalk.json")
        ]

    print("🔍 Checking model files...")
    for model_file in model_files:
        if model_file.exists():
            file_size = model_file.stat().st_size
            print(f"✅ Model file exists: {model_file} ({file_size} bytes)")
        else:
            print(f"❌ Model file missing: {model_file}")

    # Check scripts
    scripts_to_check = [
        Path("./scripts/realtime_inference.py"),
        Path("./scripts/inference.py")
    ]

    print("🔍 Checking script files...")
    for script_file in scripts_to_check:
        if script_file.exists():
            print(f"✅ Script exists: {script_file}")
        else:
            print(f"❌ Script missing: {script_file}")

    print("✅ Environment check completed\n")

# Run environment check
check_environment()

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
    print(f"📝 Creating YAML config for avatar: {avatar_id}")

    yaml_content = f"""{avatar_id}:
 preparation: {str(preparation).lower()}
 bbox_shift: {BBOX_SHIFT}
 video_path: "{video_path}"
 audio_clips:
"""

    for audio_name, audio_path in audio_clips.items():
        yaml_content += f"""     {audio_name}: "{audio_path}"
"""

    print("📄 YAML content to be written:")
    print(yaml_content)

    # Write to temporary config file
    config_path = TEMP_DIR / f"config_{avatar_id}_{hash(str(audio_clips))}.yaml"
    print(f"💾 Writing config to: {config_path}")

    try:
        with open(config_path, 'w') as f:
            f.write(yaml_content)

        # Verify file was written correctly
        if config_path.exists():
            file_size = config_path.stat().st_size
            print(f"✅ Config file written successfully: {file_size} bytes")
        else:
            print("❌ Config file was not created!")

    except Exception as e:
        print(f"❌ Error writing config file: {type(e).__name__}: {str(e)}")
        raise

    return config_path

def run_inference(config_path: str):
    """Run realtime inference with given config"""
    print("🔧 Building inference command...")

    unet_model_path, unet_config = get_model_paths()
    print(f"📦 Model paths: unet={unet_model_path}, config={unet_config}")

    cmd_args = [
        "python3", "-m", "scripts.realtime_inference",
        "--inference_config", str(config_path),
        "--gpu_id", str(GPU_ID),
        "--version", VERSION,
        "--bbox_shift", str(BBOX_SHIFT),
        "--fps", str(FPS),
        "--batch_size", str(BATCH_SIZE)
    ]

    print(f"🚀 Command to execute: {' '.join(cmd_args)}")
    print(f"📍 Working directory: {os.getcwd()}")
    print(f"📄 Config file exists: {config_path.exists()}")
    print(f"📄 Config file size: {config_path.stat().st_size if config_path.exists() else 0} bytes")

    try:
        print("⚡ Executing inference process...")
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            check=True,
            cwd=os.getcwd()
        )

        print("✅ Inference process completed successfully")
        print(f"📝 STDOUT length: {len(result.stdout)} chars")
        print(f"⚠️  STDERR length: {len(result.stderr)} chars")

        if result.stderr:
            print("⚠️  STDERR content:")
            print(result.stderr)

        return result.stdout, result.stderr

    except subprocess.CalledProcessError as e:
        print("❌ Inference process failed!")
        print(f"💥 Exit code: {e.returncode}")
        print(f"💥 STDERR: {e.stderr}")
        print(f"💥 STDOUT: {e.stdout}")

        # Check if config file is readable
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_content = f.read()
                print("📄 Config file content:")
                print(config_content)
            except Exception as config_error:
                print(f"❌ Could not read config file: {config_error}")

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed (exit code {e.returncode}): {e.stderr}"
        )
    except Exception as e:
        print(f"❌ Unexpected error in run_inference: {type(e).__name__}: {str(e)}")
        raise

def save_uploaded_file(upload_file: UploadFile, filename: str) -> str:
    """Save uploaded file and return path"""
    file_path = TEMP_DIR / filename
    print(f"💾 Saving file: {filename}")
    print(f"📍 Target path: {file_path}")

    try:
        # Ensure temp directory exists
        TEMP_DIR.mkdir(exist_ok=True)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        # Verify file was saved correctly
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f"✅ File saved successfully: {file_size} bytes")

            if file_size == 0:
                print("⚠️  Warning: File size is 0 bytes!")

        else:
            print("❌ File was not saved!")

        return str(file_path)

    except Exception as e:
        print(f"❌ Error saving file: {type(e).__name__}: {str(e)}")
        raise

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
    - Returns the generated video with duration as filename
    """
    start_time = time.time()
    print(f"🎬 [GENERATE] Starting video generation for avatar: {avatar_id}")

    try:
        print("📁 Step 1: Validating file types...")
        # Validate file types
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"❌ Invalid video file type: {video.filename}")
            raise HTTPException(status_code=400, detail="Video must be MP4, AVI, or MOV")
        if not audio.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            print(f"❌ Invalid audio file type: {audio.filename}")
            raise HTTPException(status_code=400, detail="Audio must be MP3, WAV, or M4A")
        print("✅ File types validated")

        print("💾 Step 2: Saving uploaded files...")
        # Save uploaded files
        video_path = save_uploaded_file(video, f"{avatar_id}_video{video.filename}")
        audio_path = save_uploaded_file(audio, f"{avatar_id}_audio{audio.filename}")
        print(f"✅ Files saved: video={video_path}, audio={audio_path}")

        print("📝 Step 3: Creating configuration...")
        # Create audio clips dict
        audio_clips = {
            "generated": audio_path
        }

        # Create config file
        config_path = create_yaml_config(avatar_id, video_path, audio_clips, preparation=True)
        print(f"✅ Config created: {config_path}")

        print("🤖 Step 4: Running inference...")
        # Run inference
        stdout, stderr = run_inference(config_path)
        print("✅ Inference completed")

        print("🔍 Step 5: Looking for output video...")
        # Find generated video
        results_dir = Path("./results") / VERSION / "avatars" / avatar_id / "vid_output"
        print(f"🔍 Checking directory: {results_dir}")

        if results_dir.exists():
            print(f"📁 Directory exists, listing contents...")
            all_files = list(results_dir.glob("*"))
            print(f"📁 All files in directory: {[str(f) for f in all_files]}")

            video_files = list(results_dir.glob("*.mp4"))
            print(f"🎥 MP4 files found: {[str(f) for f in video_files]}")

            if video_files:
                output_video = video_files[0]
                print(f"✅ Found output video: {output_video}")

                # Check if file actually exists and has content
                if output_video.exists():
                    file_size = output_video.stat().st_size
                    print(f"📊 Output video size: {file_size} bytes")

                    if file_size == 0:
                        print("❌ Output video file is empty!")
                        raise HTTPException(status_code=500, detail="Generated video file is empty")

                    # Calculate total duration
                    end_time = time.time()
                    duration = end_time - start_time

                    # Print duration to terminal
                    print(f"Duration: {duration:.2f}s")
                    # Clean up temp files
                    print("🧹 Cleaning up temp files...")
                    config_path.unlink(missing_ok=True)
                    print("✅ Cleanup completed")

                    return FileResponse(
                        path=output_video,
                        media_type='video/mp4',
                        filename=f"{duration:.2f}s_generated.mp4"
                    )
                else:
                    print("❌ Output video file does not exist!")
            else:
                print("❌ No MP4 files found in output directory")
                print("📁 Directory contents:", list(results_dir.iterdir()) if results_dir.exists() else "Directory doesn't exist")
        else:
            print(f"❌ Results directory does not exist: {results_dir}")

        print("❌ Video generation completed but output file not found")
        raise HTTPException(status_code=500, detail="Video generation completed but output file not found")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Print error duration if failed
        end_time = time.time()
        duration = end_time - start_time
        print(".2f")
        print(f"💥 Error details: {type(e).__name__}: {str(e)}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
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
    start_time = time.time()

    try:
        print(f"🔧 [PREPARE] Starting avatar preparation for: {avatar_id}")

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

        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time

        # Print duration to terminal
        print(f"Duration: {duration:.2f}s")
        # Mark avatar as prepared in cache
        avatar_cache[avatar_id] = {
            "video_path": video_path,
            "prepared": True,
            "bbox_shift": BBOX_SHIFT,
            "version": VERSION,
            "preparation_time": duration
        }

        # Clean up temp config
        config_path.unlink(missing_ok=True)

        return {
            "message": f"Avatar '{avatar_id}' prepared and cached successfully",
            "avatar_id": avatar_id,
            "status": "prepared",
            "preparation_duration_seconds": duration
        }

    except Exception as e:
        # Print error duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration:.2f}s")
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
    - Returns video with duration as filename
    """
    start_time = time.time()

    try:
        print(f"⚡ [FAST-GENERATE] Starting fast generation for avatar: {avatar_id}")

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

                # Calculate total duration
                end_time = time.time()
                duration = end_time - start_time

                # Print duration to terminal
                print(f"Duration: {duration:.2f}s")
                # Clean up temp files
                config_path.unlink(missing_ok=True)

                return FileResponse(
                    path=output_video,
                    media_type='video/mp4',
                    filename=f"{duration:.2f}s_fast.mp4"
                )

        raise HTTPException(status_code=500, detail="Fast generation completed but output file not found")

    except HTTPException:
        raise
    except Exception as e:
        # Print error duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration:.2f}s")

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
        "message": "MuseTalk Realtime API Server with Enhanced Logging",
        "endpoints": {
            "POST /generate": "Generate video from scratch (video + audio)",
            "POST /prepare": "Prepare avatar for caching (video only)",
            "POST /generate-fast": "Generate video using cached avatar (audio only)",
            "GET /cache": "Get cache status",
            "DELETE /cache/{avatar_id}": "Clear specific avatar from cache"
        },
        "version": VERSION,
        "cached_avatars": list(avatar_cache.keys()),
        "note": "All endpoints now print detailed duration and use duration as filename",
        "features": [
            "Detailed step-by-step logging",
            "Duration tracking in terminal",
            "Duration-based filenames",
            "Enhanced error reporting",
            "Environment verification",
            "File validation and size checking"
        ]
    }

# ====================================================================
# EJECUCIÓN DEL SERVIDOR:
#
# Opción 1 - Con uv (recomendado):
# uv run fastapi run realtime_api.py --port 8000 --host 0.0.0.0
#
# Opción 2 - Con script:
# ./run_enhanced_api.sh
# ====================================================================
