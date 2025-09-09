#!/usr/bin/env python3
"""
MuseTalk Realtime API Server - Folder-based Cache System
FastAPI server with endpoints for avatar generation using persistent folder-based caching

CACHE SYSTEM:
- Caches are stored as folders in ./results/v15/avatars/ with any name
- No longer uses in-memory dictionary, all cache data persists between server restarts
- Cache names can be any string (e.g., avatar1, mycache, test_cache, etc.)
- Cache folders are automatically detected on server startup
- A cache is considered "prepared" when it contains these required files:
  • avatar_info.json
  • coords.pkl
  • full_imgs/ (folder)
  • latents.pt
  • mask/ (folder)
  • mask_coords.pkl
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

def get_available_caches():
    """Obtener lista de caches disponibles desde carpeta ./results/v15/avatars/"""
    cache_dir = Path("./results") / VERSION / "avatars"
    if not cache_dir.exists():
        return []

    # Obtener todas las carpetas (cualquier nombre es válido como cache)
    cache_folders = []
    for folder in cache_dir.iterdir():
        if folder.is_dir():
            cache_folders.append(folder.name)

    return sorted(cache_folders)

def get_cache_info(cache_id: str):
    """Obtener información de un cache específico.

    Un cache se considera preparado cuando contiene todos los archivos requeridos:
    - avatar_info.json
    - coords.pkl
    - latents.pt
    - mask_coords.pkl
    - full_imgs/ (carpeta)
    - mask/ (carpeta)
    """
    cache_dir = Path("./results") / VERSION / "avatars" / cache_id
    if not cache_dir.exists():
        return None

    # Verificar archivos requeridos para un cache preparado
    required_files = [
        "avatar_info.json",
        "coords.pkl",
        "latents.pt",
        "mask_coords.pkl"
    ]

    required_dirs = [
        "full_imgs",
        "mask"
    ]

    missing_files = []
    missing_dirs = []

    # Verificar archivos requeridos
    for file_name in required_files:
        file_path = cache_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    # Verificar carpetas requeridas
    for dir_name in required_dirs:
        dir_path = cache_dir / dir_name
        if not dir_path.exists() or not dir_path.is_dir():
            missing_dirs.append(dir_name)

    # El cache está preparado si todos los archivos y carpetas requeridas existen
    is_prepared = len(missing_files) == 0 and len(missing_dirs) == 0

    return {
        "cache_id": cache_id,
        "prepared": is_prepared,
        "exists": True,
        "required_files": required_files,
        "required_dirs": required_dirs,
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "cache_dir": str(cache_dir)
    }

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

def cleanup_old_configs(avatar_id: str):
    """Clean up old config files for an avatar to prevent conflicts"""
    try:
        if TEMP_DIR.exists():
            for config_file in TEMP_DIR.glob(f"config_{avatar_id}_*.yaml"):
                try:
                    config_file.unlink()
                    print(f"🗑️ Cleaned up old config file: {config_file.name}")
                except Exception as e:
                    print(f"⚠️ Could not delete {config_file.name}: {e}")
    except Exception as e:
        print(f"⚠️ Error during config cleanup: {e}")


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

    # Clean up old config files for this avatar to avoid conflicts
    cleanup_old_configs(avatar_id)

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
    - Prepares avatar and saves to cache folder
    - Cache folder name is specified by avatar_id parameter
    - No audio processing needed
    """
    start_time = time.time()

    try:
        print(f"🔧 [PREPARE] Starting avatar preparation for: {avatar_id}")

        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(status_code=400, detail="Video must be MP4, AVI, or MOV")

        # Verificar si el cache ya existe y está preparado
        available_caches = get_available_caches()
        if avatar_id in available_caches:
            cache_info = get_cache_info(avatar_id)
            if cache_info and cache_info.get("prepared"):
                return {
                    "message": f"Cache '{avatar_id}' already exists and is prepared",
                    "cache_id": avatar_id,
                    "status": "already_prepared",
                    "required_files_present": cache_info.get("required_files", []),
                    "required_dirs_present": cache_info.get("required_dirs", [])
                }

        # Save uploaded file
        video_path = save_uploaded_file(video, f"{avatar_id}_prepare{video.filename}")

        # No audio clips needed during preparation
        audio_clips = {}

        # Create config file with preparation=True
        config_path = create_yaml_config(avatar_id, video_path, audio_clips, preparation=True)

        # Run inference (this will prepare the avatar)
        stdout, stderr = run_inference(config_path)

        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time

        # Print duration to terminal
        print(f"Duration: {duration:.2f}s")

        # Verificar que el cache se haya creado correctamente
        cache_info = get_cache_info(avatar_id)
        if cache_info and cache_info.get("prepared"):
            # Clean up temp config
            config_path.unlink(missing_ok=True)

            return {
                "message": f"Cache '{avatar_id}' prepared successfully",
                "cache_id": avatar_id,
                "status": "prepared",
                "preparation_duration_seconds": duration,
                "required_files_present": cache_info.get("required_files", []),
                "required_dirs_present": cache_info.get("required_dirs", [])
            }
        else:
            # Proporcionar información detallada sobre qué falta
            missing_info = ""
            if cache_info:
                missing_files = cache_info.get("missing_files", [])
                missing_dirs = cache_info.get("missing_dirs", [])
                if missing_files or missing_dirs:
                    missing_info = f" Missing files: {missing_files}, Missing dirs: {missing_dirs}"

            raise HTTPException(
                status_code=500,
                detail=f"Cache preparation completed but required files not found.{missing_info}"
            )

    except HTTPException:
        raise
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
    - Requires avatar to be prepared first (cache folder with required files must exist)
    - Cache must contain: avatar_info.json, coords.pkl, latents.pt, mask_coords.pkl, full_imgs/, mask/
    - Uses cached avatar for faster generation
    - Only processes audio
    - Returns video with duration as filename
    """
    start_time = time.time()

    try:
        print(f"⚡ [FAST-GENERATE] Starting fast generation for cache: {avatar_id}")

        # Check if cache exists and is prepared
        available_caches = get_available_caches()
        if avatar_id not in available_caches:
            raise HTTPException(
                status_code=400,
                detail=f"Cache '{avatar_id}' not found. Use /prepare endpoint first to create it."
            )

        cache_info = get_cache_info(avatar_id)
        if not cache_info or not cache_info.get("prepared"):
            missing_info = ""
            if cache_info:
                missing_files = cache_info.get("missing_files", [])
                missing_dirs = cache_info.get("missing_dirs", [])
                if missing_files or missing_dirs:
                    missing_info = f" Missing files: {missing_files}, Missing dirs: {missing_dirs}"

            raise HTTPException(
                status_code=400,
                detail=f"Cache '{avatar_id}' exists but is not properly prepared.{missing_info} Use /prepare endpoint first."
            )

        # Validate audio file
        if not audio.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
            raise HTTPException(status_code=400, detail="Audio must be MP3, WAV, or M4A")

        # Usar el directorio del cache como referencia para el procesamiento
        cache_dir = Path("./results") / VERSION / "avatars" / avatar_id
        video_path = str(cache_dir)  # El script usará los archivos del cache directamente

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

@app.post("/generate-multi-fast")
async def generate_multi_fast(
    avatar_id: str = Form(...),
    audio_files: list[UploadFile] = File(...)
):
    """
    Endpoint 4: Generate multiple videos using cached avatar (SEQUENTIAL PROCESSING)
    - Requires avatar to be prepared first (cache folder with required files must exist)
    - Cache must contain: avatar_info.json, coords.pkl, latents.pt, mask_coords.pkl, full_imgs/, mask/
    - Processes multiple audio files sequentially
    - Returns a ZIP file containing all generated videos

    Advantages:
    - Allows batch processing of multiple audios
    - Reuses prepared avatar for each audio
    - Returns all results in a single ZIP

    Limitations:
    - Sequential processing (not parallel)
    - Total time = sum of individual processing times
    """
    start_time = time.time()

    try:
        print(f"🎵 [MULTI-FAST-GENERATE] Starting batch generation for cache: {avatar_id} with {len(audio_files)} audio files")

        # Check if cache exists and is prepared
        available_caches = get_available_caches()
        if avatar_id not in available_caches:
            raise HTTPException(
                status_code=400,
                detail=f"Cache '{avatar_id}' not found. Use /prepare endpoint first to create it."
            )

        cache_info = get_cache_info(avatar_id)
        if not cache_info or not cache_info.get("prepared"):
            missing_info = ""
            if cache_info:
                missing_files = cache_info.get("missing_files", [])
                missing_dirs = cache_info.get("missing_dirs", [])
                if missing_files or missing_dirs:
                    missing_info = f" Missing files: {missing_files}, Missing dirs: {missing_dirs}"

            raise HTTPException(
                status_code=400,
                detail=f"Cache '{avatar_id}' exists but is not properly prepared.{missing_info} Use /prepare endpoint first."
            )

        # Validate all audio files
        for i, audio in enumerate(audio_files):
            if not audio.filename.lower().endswith(('.mp3', '.wav', '.m4a')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio file {i+1} ({audio.filename}) must be MP3, WAV, or M4A"
                )

        # Usar el directorio del cache como referencia para el procesamiento
        cache_dir = Path("./results") / VERSION / "avatars" / avatar_id
        video_path = str(cache_dir)  # El script usará los archivos del cache directamente

        # Save all audio files and create clips dict
        audio_clips = {}
        saved_audio_paths = []

        for i, audio in enumerate(audio_files):
            audio_name = f"multi_audio_{i+1}"
            audio_path = save_uploaded_file(audio, f"{avatar_id}_{audio_name}{audio.filename}")
            audio_clips[audio_name] = audio_path
            saved_audio_paths.append(audio_path)

        print(f"💾 Saved {len(audio_files)} audio files for batch processing")

        # Create config file with preparation=False (use cache)
        config_path = create_yaml_config(avatar_id, video_path, audio_clips, preparation=False)

        # Run inference (this will process all audios sequentially)
        stdout, stderr = run_inference(config_path)

        # Find all generated videos
        results_dir = Path("./results") / VERSION / "avatars" / avatar_id / "vid_output"
        generated_videos = []

        if results_dir.exists():
            video_files = list(results_dir.glob("*.mp4"))
            print(f"🎥 Found {len(video_files)} generated video files")

            # Collect all generated videos (they should be named after audio_clips keys)
            for video_file in video_files:
                video_name = video_file.stem
                if any(audio_name in video_name for audio_name in audio_clips.keys()):
                    generated_videos.append(video_file)
                    print(f"✅ Collected video: {video_name}.mp4")

        if not generated_videos:
            raise HTTPException(status_code=500, detail="No videos were generated")

        # Create ZIP file with all videos
        import zipfile
        zip_filename = f"{avatar_id}_batch_{len(generated_videos)}_videos.zip"
        zip_path = TEMP_DIR / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for video_path in generated_videos:
                # Add video to ZIP with a clean name
                video_name = video_path.stem
                zip_file.write(video_path, f"{video_name}.mp4")
                print(f"📦 Added {video_name}.mp4 to ZIP")

        # Calculate total duration
        end_time = time.time()
        duration = end_time - start_time

        # Print duration to terminal
        print(f"Duration: {duration:.2f}s")
        print(f"📊 Processed {len(generated_videos)} videos in batch")

        # Clean up temp files
        config_path.unlink(missing_ok=True)

        return FileResponse(
            path=zip_path,
            media_type='application/zip',
            filename=zip_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        # Print error duration
        end_time = time.time()
        duration = end_time - start_time
        print(f"Duration: {duration:.2f}s")

        raise HTTPException(status_code=500, detail=f"Multi-fast generation failed: {str(e)}")

@app.get("/cache")
async def get_cache_status():
    """Get current cache status from folders"""
    available_caches = get_available_caches()
    cache_info = {}

    for cache_id in available_caches:
        cache_info[cache_id] = get_cache_info(cache_id)

    return {
        "available_caches": available_caches,
        "cache_info": cache_info,
        "total_caches": len(available_caches)
    }

@app.delete("/cache/{avatar_id}")
async def clear_cache(avatar_id: str):
    """Clear specific cache folder"""
    available_caches = get_available_caches()
    if avatar_id not in available_caches:
        raise HTTPException(status_code=404, detail=f"Cache '{avatar_id}' not found")

    try:
        # Eliminar la carpeta completa del cache
        cache_dir = Path("./results") / VERSION / "avatars" / avatar_id
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"🗑️ Cache folder '{avatar_id}' deleted successfully")
            return {"message": f"Cache '{avatar_id}' removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Cache folder '{avatar_id}' not found on disk")

    except Exception as e:
        print(f"❌ Error deleting cache folder: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete cache '{avatar_id}': {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    available_caches = get_available_caches()

    return {
        "message": "MuseTalk Realtime API Server - Folder-based Cache System",
        "endpoints": {
            "POST /generate": "Generate video from scratch (video + audio)",
            "POST /prepare": "Prepare avatar for caching (video only) - creates cache folder with any name",
            "POST /generate-fast": "Generate video using cached avatar (audio only)",
            "POST /generate-multi-fast": "Generate multiple videos using cached avatar (multiple audio files -> ZIP)",
            "GET /cache": "Get cache status from folders",
            "DELETE /cache/{avatar_id}": "Clear specific cache folder"
        },
        "version": VERSION,
        "available_caches": available_caches,
        "cache_directory": f"./results/{VERSION}/avatars/",
        "cache_format": "Any folder name (e.g., avatar1, mycache, test_cache, etc.)",
        "cache_requirements": {
            "required_files": ["avatar_info.json", "coords.pkl", "latents.pt", "mask_coords.pkl"],
            "required_dirs": ["full_imgs", "mask"]
        },
        "note": "Cache system now uses folders instead of memory dict. Cache names can be any string",
        "features": [
            "Folder-based cache system",
            "Persistent cache storage",
            "Detailed step-by-step logging",
            "Duration tracking in terminal",
            "Duration-based filenames",
            "Enhanced error reporting",
            "Environment verification",
            "File validation and size checking",
            "Multi-audio batch processing",
            "ZIP file output for multiple results"
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
