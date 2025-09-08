#!/usr/bin/env python3
"""
MuseTalk Realtime Avatar Inference - Ultra Simple
Similar to inference.sh but in Python with easy config
"""

import os
import subprocess
import sys

# ====================================================================
# CONFIGURACIÓN FÁCIL - MODIFICA AQUÍ LO QUE NECESITES
# ====================================================================

# === CONFIGURACIÓN BÁSICA ===
VERSION = "v15"  # "v1" o "v15"
GPU_ID = 0
BBOX_SHIFT = 5
FPS = 25
BATCH_SIZE = 20

# === CONFIGURACIÓN DEL AVATAR ===
AVATAR_ID = "mi_avatar_personalizado"
VIDEO_PATH = "data/video/yongen.mp4"  # TU VIDEO DE 10 MINUTOS
PREPARATION = False  # True = primera vez, False = reutilizar avatar preparado

# === AUDIO CLIPS A PROCESAR ===
AUDIO_CLIPS = {
    "audio_1min": "data/audio/yongen.wav",          # Audio de 1 minuto
    "audio_1_5min": "data/audio/eng.wav",           # Audio de 1.5 minutos
    # Agrega más audios aquí según necesites:
    # "audio_5min": "data/audio/audio_5_minutos.wav",
    # "audio_30seg": "data/audio/audio_30_segundos.wav",
}

# ====================================================================
# FIN DE CONFIGURACIÓN - NO MODIFIQUES NADA ABAJO
# ====================================================================

def create_yaml_config():
    """Create temporary YAML config like realtime.yaml"""
    yaml_content = f"""{AVATAR_ID}:
 preparation: {str(PREPARATION).lower()}
 bbox_shift: {BBOX_SHIFT}
 video_path: "{VIDEO_PATH}"
 audio_clips:
"""

    for audio_name, audio_path in AUDIO_CLIPS.items():
        yaml_content += f"""     {audio_name}: "{audio_path}"
"""

    # Write to temporary config file
    config_path = f"./temp_{AVATAR_ID}_config.yaml"
    with open(config_path, 'w') as f:
        f.write(yaml_content)

    return config_path

def main():
    """Main function - ultra simple like inference.sh"""
    print("=== MuseTalk Realtime Avatar - Ultra Simple ===")
    print(f"Avatar ID: {AVATAR_ID}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Preparation: {PREPARATION}")
    print(f"Audio clips: {len(AUDIO_CLIPS)}")
    print("=" * 50)

    # Define model paths based on version
    if VERSION == "v1":
        model_dir = "./models/musetalk"
        unet_model_path = f"{model_dir}/pytorch_model.bin"
        unet_config = f"{model_dir}/musetalk.json"
    elif VERSION == "v15":
        model_dir = "./models/musetalkV15"
        unet_model_path = f"{model_dir}/unet.pth"
        unet_config = f"{model_dir}/musetalk.json"
    else:
        print("Invalid version. Use 'v1' or 'v15'")
        sys.exit(1)

    # Create temporary config file
    config_path = create_yaml_config()
    print(f"Created config: {config_path}")

    # Build command exactly like inference.sh does
    cmd_args = [
        "python3", "-m", "scripts.realtime_inference",
        "--inference_config", config_path,
        "--gpu_id", str(GPU_ID),
        "--version", VERSION,
        "--bbox_shift", str(BBOX_SHIFT),
        "--fps", str(FPS),
        "--batch_size", str(BATCH_SIZE)
    ]

    print("Running command:")
    print(" ".join(cmd_args))
    print("=" * 50)

    # Execute the command
    try:
        subprocess.run(cmd_args, check=True)
        print("\n✅ All audio clips processed successfully!")

        # Clean up temp config
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"Cleaned up: {config_path}")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
