#!/usr/bin/env python3
"""
Fast one-off MuseTalk API - Minimal single endpoint for fastest generation without cache.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse


VERSION = "v15"
BATCH_SIZE = 32


BASE_DIR = Path(__file__).parent.resolve()

app = FastAPI(
    title="Fast One-Off MuseTalk API",
    description="Minimal endpoint for fastest single-shot generation (no cache)",
    version="1.0.0",
)


@app.post("/generate")
async def generate(video: UploadFile = File(...), audio: UploadFile = File(...)):
    """Generate a single video as fast as possible without persistent cache."""
    import time
    start_time = time.time()

    batch_size = BATCH_SIZE

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            video_path = tmp_path / f"video{Path(video.filename).suffix}"
            audio_path = tmp_path / f"audio{Path(audio.filename).suffix}"

            with open(video_path, "wb") as vf:
                import shutil
                shutil.copyfileobj(video.file, vf)
            with open(audio_path, "wb") as af:
                import shutil
                shutil.copyfileobj(audio.file, af)

            config_text = f"""task_0:
 video_path: "{video_path}"
 audio_path: "{audio_path}"
"""
            config_path = tmp_path / "config.yaml"
            with open(config_path, "w") as cf:
                cf.write(config_text)

            results_dir = BASE_DIR / "results" / VERSION
            results_dir.mkdir(parents=True, exist_ok=True)
            unet_path = BASE_DIR / "models" / "musetalkV15" / "unet.pth"
            unet_cfg = BASE_DIR / "models" / "musetalkV15" / "musetalk.json"
            output_name = "timespent_generation.mp4"

            cmd = [
                "uv", "run", str(BASE_DIR / "scripts" / "inference.py"),
                "--inference_config", str(config_path),
                "--result_dir", str(results_dir),
                "--unet_model_path", str(unet_path),
                "--unet_config", str(unet_cfg),
                "--version", VERSION,
                "--use_float16",
                "--batch_size", str(batch_size),
                "--fps", "25",
                "--output_vid_name", output_name,
            ]

            proc = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                capture_output=False,
                text=True,
                timeout=60000,
            )
            if proc.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Inference failed (code {proc.returncode})")

            output_file = results_dir / output_name
            if not output_file.exists():
                raise HTTPException(status_code=500, detail="Output video not found")
            if output_file.stat().st_size == 0:
                raise HTTPException(status_code=500, detail="Output video is empty")

            filename = f"{start_time:.0f}_generation.mp4"

            return FileResponse(
                path=output_file,
                media_type="video/mp4",
                filename=filename
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Generation timed out")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}: {str(e)}")
