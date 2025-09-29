#!/usr/bin/env python3
"""
Simple MuseTalk API - Single endpoint that calls realtime_inference.py via subprocess
Enhanced with comprehensive performance monitoring
"""

import os
import subprocess
import tempfile
import shutil
import time
import threading
import json
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# Import inference modules directly (like realtime_api.py)
from scripts.realtime_inference import Avatar, load_all_model, fast_check_ffmpeg
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from omegaconf import OmegaConf
import torch

# Performance monitoring imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - system monitoring disabled")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("⚠️ GPUtil not available - GPU monitoring disabled")

# ====================================================================
# CONFIGURACIÓN
# ====================================================================

VERSION = "v15"
GPU_ID = 0
BATCH_SIZE = 32  # Increased from 20 for better GPU utilization

# Dynamic batch size based on GPU memory
def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory"""
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
            return 64
        elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3080 Ti, etc.
            return 48
        elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
            return 32
        elif gpu_memory_gb >= 8:   # RTX 4070, RTX 3070, etc.
            return 24
        else:  # Lower end GPUs
            return 16
    return 32  # Default for CPU or unknown GPU

THIS_DIR = Path(__file__).parent

# ====================================================================
# PERFORMANCE MONITORING
# ====================================================================

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_api_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimplePerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'steps': {},
            'system_metrics': [],
            'subprocess_metrics': {},
            'file_metrics': {}
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, avatar_id: str):
        """Start comprehensive performance monitoring"""
        self.metrics['start_time'] = time.time()
        self.metrics['avatar_id'] = avatar_id
        self.metrics['mode'] = 'simple_subprocess'
        self.monitoring = True
        
        # Start background monitoring thread
        if PSUTIL_AVAILABLE:
            self.monitor_thread = threading.Thread(target=self._monitor_system_resources)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        
        logger.info(f"🔍 [PERFORMANCE] Started monitoring for {avatar_id} (simple subprocess mode)")
    
    def stop_monitoring(self):
        """Stop monitoring and calculate final metrics"""
        self.monitoring = False
        self.metrics['end_time'] = time.time()
        self.metrics['total_duration'] = self.metrics['end_time'] - self.metrics['start_time']
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        self._log_final_metrics()
    
    def _monitor_system_resources(self):
        """Background thread to monitor system resources"""
        while self.monitoring and PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_gb = memory.used / (1024**3)
                
                # GPU metrics (if available)
                gpu_metrics = self._get_gpu_metrics()
                
                # Store metrics
                timestamp = time.time()
                self.metrics['system_metrics'].append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_used_gb': memory_gb,
                    'memory_percent': memory.percent,
                    'gpu_metrics': gpu_metrics
                })
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                break
    
    def _get_gpu_metrics(self):
        """Get GPU metrics if available"""
        if not GPUTIL_AVAILABLE:
            return None
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'gpu_utilization': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                }
        except:
            pass
        return None
    
    def log_step(self, step_name: str, duration: float, details: dict = None):
        """Log a specific step with timing"""
        step_metrics = {
            'duration': duration,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.metrics['steps'][step_name] = step_metrics
        logger.info(f"⏱️ [STEP] {step_name}: {duration:.3f}s")
    
    def log_gpu_metrics(self, step_name: str):
        """Log GPU metrics for a specific step"""
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            gpu_utilization = self._get_gpu_utilization()
            
            logger.info(f"🎮 [GPU] {step_name}: {gpu_memory_used:.2f}GB used, {gpu_memory_reserved:.2f}GB reserved, {gpu_utilization:.1f}% util")
            
            return {
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_reserved_gb': gpu_memory_reserved,
                'gpu_utilization_percent': gpu_utilization
            }
        return {}
    
    def _get_gpu_utilization(self):
        """Get current GPU utilization percentage"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0
    
    def log_subprocess_metrics(self, subprocess_duration: float, return_code: int, details: dict = None):
        """Log subprocess-specific metrics"""
        self.metrics['subprocess_metrics'] = {
            'duration': subprocess_duration,
            'return_code': return_code,
            'details': details or {}
        }
        logger.info(f"🚀 [SUBPROCESS] Duration: {subprocess_duration:.3f}s, Return code: {return_code}")
    
    def log_file_metrics(self, video_size: int, audio_size: int, output_size: int):
        """Log file size metrics"""
        self.metrics['file_metrics'] = {
            'video_size_bytes': video_size,
            'audio_size_bytes': audio_size,
            'output_size_bytes': output_size,
            'video_size_mb': video_size / (1024 * 1024),
            'audio_size_mb': audio_size / (1024 * 1024),
            'output_size_mb': output_size / (1024 * 1024)
        }
        logger.info(f"📁 [FILES] Video: {video_size / (1024 * 1024):.2f}MB, Audio: {audio_size / (1024 * 1024):.2f}MB, Output: {output_size / (1024 * 1024):.2f}MB")
    
    def _log_final_metrics(self):
        """Log comprehensive final metrics"""
        logger.info("=" * 80)
        logger.info("📊 SIMPLE API PERFORMANCE ANALYSIS REPORT")
        logger.info("=" * 80)
        
        # Basic timing
        logger.info(f"🎯 Avatar ID: {self.metrics['avatar_id']}")
        logger.info(f"🎯 Mode: {self.metrics['mode']}")
        logger.info(f"⏱️ Total Duration: {self.metrics['total_duration']:.3f}s")
        
        # Step breakdown
        logger.info("\n📋 STEP BREAKDOWN:")
        for step_name, metrics in self.metrics['steps'].items():
            duration = metrics['duration']
            percentage = (duration / self.metrics['total_duration']) * 100
            logger.info(f"  {step_name}: {duration:.3f}s ({percentage:.1f}%)")
        
        # Subprocess analysis
        if self.metrics['subprocess_metrics']:
            subprocess_duration = self.metrics['subprocess_metrics']['duration']
            subprocess_percentage = (subprocess_duration / self.metrics['total_duration']) * 100
            logger.info(f"\n🚀 SUBPROCESS ANALYSIS:")
            logger.info(f"  Subprocess Duration: {subprocess_duration:.3f}s ({subprocess_percentage:.1f}%)")
            logger.info(f"  Return Code: {self.metrics['subprocess_metrics']['return_code']}")
        
        # File analysis
        if self.metrics['file_metrics']:
            logger.info(f"\n📁 FILE ANALYSIS:")
            logger.info(f"  Input Video: {self.metrics['file_metrics']['video_size_mb']:.2f} MB")
            logger.info(f"  Input Audio: {self.metrics['file_metrics']['audio_size_mb']:.2f} MB")
            logger.info(f"  Output Video: {self.metrics['file_metrics']['output_size_mb']:.2f} MB")
            
            # Calculate compression ratio
            input_total = self.metrics['file_metrics']['video_size_mb'] + self.metrics['file_metrics']['audio_size_mb']
            output_size = self.metrics['file_metrics']['output_size_mb']
            if input_total > 0:
                compression_ratio = output_size / input_total
                logger.info(f"  Compression Ratio: {compression_ratio:.2f} (output/input)")
        
        # System resource analysis
        if self.metrics['system_metrics']:
            self._analyze_system_resources()
        
        # Save detailed metrics to file
        self._save_metrics_to_file()
        
        logger.info("=" * 80)
    
    def _analyze_system_resources(self):
        """Analyze system resource usage patterns"""
        metrics = self.metrics['system_metrics']
        
        if not metrics:
            return
        
        # CPU analysis
        cpu_values = [m['cpu_percent'] for m in metrics]
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        
        logger.info(f"\n💻 CPU ANALYSIS:")
        logger.info(f"  Average CPU Usage: {avg_cpu:.1f}%")
        logger.info(f"  Peak CPU Usage: {max_cpu:.1f}%")
        logger.info(f"  CPU Cores: {metrics[0]['cpu_count']}")
        
        # Memory analysis
        memory_values = [m['memory_used_gb'] for m in metrics]
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        
        logger.info(f"\n🧠 MEMORY ANALYSIS:")
        logger.info(f"  Average Memory Usage: {avg_memory:.2f} GB")
        logger.info(f"  Peak Memory Usage: {max_memory:.2f} GB")
        
        # GPU analysis
        gpu_metrics = [m['gpu_metrics'] for m in metrics if m['gpu_metrics']]
        if gpu_metrics:
            gpu_util = [g['gpu_utilization'] for g in gpu_metrics]
            gpu_mem = [g['gpu_memory_percent'] for g in gpu_metrics]
            
            avg_gpu_util = sum(gpu_util) / len(gpu_util)
            max_gpu_util = max(gpu_util)
            avg_gpu_mem = sum(gpu_mem) / len(gpu_mem)
            max_gpu_mem = max(gpu_mem)
            
            logger.info(f"\n🎮 GPU ANALYSIS:")
            logger.info(f"  Average GPU Utilization: {avg_gpu_util:.1f}%")
            logger.info(f"  Peak GPU Utilization: {max_gpu_util:.1f}%")
            logger.info(f"  Average GPU Memory: {avg_gpu_mem:.1f}%")
            logger.info(f"  Peak GPU Memory: {max_gpu_mem:.1f}%")
            
            # Performance recommendations
            if avg_gpu_util < 50:
                logger.info(f"  💡 OPTIMIZATION: Low GPU utilization ({avg_gpu_util:.1f}%) - consider increasing batch size")
            if max_gpu_mem < 80:
                logger.info(f"  💡 OPTIMIZATION: GPU memory underutilized ({max_gpu_mem:.1f}%) - consider larger batch size")
            if avg_gpu_util > 90:
                logger.info(f"  ✅ EXCELLENT: High GPU utilization ({avg_gpu_util:.1f}%) - good optimization!")
    
    def _save_metrics_to_file(self):
        """Save detailed metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_api_metrics_{self.metrics['avatar_id']}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info(f"💾 Detailed metrics saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

# Global performance monitor instance
performance_monitor = SimplePerformanceMonitor()

# ====================================================================
# MODEL INITIALIZATION (like realtime_api.py)
# ====================================================================

# Global variables for models (like realtime_api.py)
audio_processor = None
device = None
weight_dtype = None
whisper = None
vae = None
unet = None
pe = None
timesteps = None
fp = None
args = None

def load_models_at_startup():
    """Load all models at startup (copied from realtime_api.py)"""
    global audio_processor, device, weight_dtype, whisper, vae, unet, pe, timesteps, fp, args

    print("🔧 Loading models at startup (this may take a moment)...")

    # Set computing device
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")
    
    # Use dynamic batch size based on GPU memory
    global BATCH_SIZE
    optimal_batch_size = get_optimal_batch_size()
    if optimal_batch_size != BATCH_SIZE:
        print(f"🔄 Adjusting batch size: {BATCH_SIZE} → {optimal_batch_size} (based on GPU memory)")
        BATCH_SIZE = optimal_batch_size
    else:
        print(f"📊 Using batch size: {BATCH_SIZE}")
    
    # GPU Memory Optimizations
    if torch.cuda.is_available():
        print("🚀 Enabling GPU optimizations...")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        torch.cuda.empty_cache()  # Clear GPU cache
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"📊 Available GPU Memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        import sys
        path_separator = ';' if sys.platform == 'win32' else ':'
        ffmpeg_path = "/usr/bin"  # Default ffmpeg path
        os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Get model paths
    if VERSION == "v15":
        model_dir = "./models/musetalkV15"
        unet_model_path = f"{model_dir}/unet.pth"
        unet_config = f"{model_dir}/musetalk.json"
    else:
        model_dir = "./models/musetalk"
        unet_model_path = f"{model_dir}/pytorch_model.bin"
        unet_config = f"{model_dir}/musetalk.json"

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=unet_model_path,
        vae_type="sd-vae",
        unet_config=unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # Initialize audio processor and Whisper model
    whisper_dir = "./models/whisper"
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    weight_dtype = unet.model.dtype
    from transformers import WhisperModel
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser
    if VERSION == "v15":
        fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    else:  # v1
        fp = FaceParsing()

    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.version = VERSION
            self.output_dir = None
            self.extra_margin = 10
            self.parsing_mode = 'jaw'
            self.audio_padding_length_left = 2
            self.audio_padding_length_right = 2
            self.skip_save_images = False
            self.ffmpeg_path = "/usr/bin"
            self.gpu_id = GPU_ID
            self.unet_model_path = unet_model_path
            self.vae_type = "sd-vae"
            self.unet_config = unet_config
            self.whisper_dir = whisper_dir
            self.left_cheek_width = 90
            self.right_cheek_width = 90
            self.inference_config = None
            self.fps = 25
            self.batch_size = BATCH_SIZE

    args = MockArgs()

    print("✅ Models loaded and ready at startup")
    
    # Final GPU memory status
    if torch.cuda.is_available():
        print(f"📊 Final GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB allocated")
        print(f"📊 Final GPU Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB reserved")

def run_inference_direct(config_path: str, output_video_path: Path):
    """Run inference directly (like realtime_api.py) instead of subprocess"""
    print("🔧 Starting direct inference...")

    # Use pre-loaded models from global variables
    global audio_processor, device, weight_dtype, whisper, vae, unet, pe, timesteps, fp, args

    print(f"📍 Output video will be saved to: {output_video_path}")

    # Extract output directory and filename
    output_dir = output_video_path.parent
    output_filename = output_video_path.stem  # without .mp4 extension

    try:
        print("⚡ Executing inference...")

        # Load inference config
        inference_config = OmegaConf.load(config_path)
        print(f"📄 Loaded config: {inference_config}")

        for avatar_id in inference_config:
            data_preparation = inference_config[avatar_id]["preparation"]
            video_path = inference_config[avatar_id]["video_path"]

            if VERSION == "v15":
                bbox_shift = 0
            else:
                bbox_shift = inference_config[avatar_id]["bbox_shift"]

            # Create avatar instance
            avatar = Avatar(
                avatar_id=avatar_id,
                video_path=video_path,
                bbox_shift=bbox_shift,
                batch_size=BATCH_SIZE,
                preparation=data_preparation,
                version=VERSION,
                vae=vae,
                face_parser=fp,
                extra_margin=10,
                parsing_mode='jaw',
                device=device,
                unet=unet,
                pe=pe,
                timesteps=timesteps,
                whisper=whisper,
                audio_processor=audio_processor,
                weight_dtype=weight_dtype,
                audio_padding_length_left=2,
                audio_padding_length_right=2,
                skip_save_images=False)

            # Process audio clips if provided
            audio_clips = inference_config[avatar_id].get("audio_clips", {})
            if audio_clips:  # Only process if there are actual audio clips
                for audio_num, audio_path in audio_clips.items():
                    print(f"🎵 Inferring using: {audio_path}")

                    # Use custom output filename instead of audio_num
                    avatar.inference(audio_path, output_filename, 25, skip_save_images=False)
            else:
                print("No audio clips to process - this appears to be a preparation-only run")

        print("✅ Inference completed successfully")

        # Verify the output file was created
        if not output_video_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Inference completed but output video not found at {output_video_path}"
            )

        file_size = output_video_path.stat().st_size
        if file_size == 0:
            raise HTTPException(status_code=500, detail="Generated video file is empty")

        print(f"✅ Output video verified: {output_video_path} ({file_size} bytes)")
        
        # Memory cleanup after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cache cleared")
        
        return output_video_path

    except Exception as e:
        print(f"❌ Inference failed: {type(e).__name__}: {str(e)}")
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

# ====================================================================
# FASTAPI APP
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Simple MuseTalk API...")
    load_models_at_startup()
    yield
    print("🛑 Shutting down Simple MuseTalk API...")

app = FastAPI(
    title="Simple MuseTalk API",
    description="Simple API with direct inference (no subprocess) and comprehensive performance monitoring",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/generate")
async def generate_video_with_monitoring(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    avatar_id: str = "simple_generated"
):
    """
    Enhanced generate video with comprehensive performance monitoring
    - Receives video and audio files
    - Creates temporary config file
    - Calls realtime_inference.py via subprocess
    - Returns generated video with detailed performance metrics
    """
    # Start performance monitoring
    performance_monitor.start_monitoring(avatar_id)
    
    start_time = time.time()
    logger.info(f"🎬 [SIMPLE-GENERATE] Starting generation for avatar: {avatar_id}")

    try:
        # Step 1: File processing and saving
        step_start = time.time()
        logger.info("📁 Step 1: Processing and saving uploaded files...")
        
        # Create temporary directory for this request
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save uploaded files
            video_path = temp_path / f"{avatar_id}_video{Path(video.filename).suffix}"
            audio_path = temp_path / f"{avatar_id}_audio{Path(audio.filename).suffix}"

            logger.info("💾 Saving uploaded files...")
            with open(video_path, "wb") as f:
                shutil.copyfileobj(video.file, f)
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
            
            # Get file sizes for monitoring
            video_size = video_path.stat().st_size
            audio_size = audio_path.stat().st_size
            
            step_duration = time.time() - step_start
            performance_monitor.log_step("file_processing", step_duration, {
                "video_size_bytes": video_size,
                "audio_size_bytes": audio_size,
                "video_filename": video.filename,
                "audio_filename": audio.filename
            })

            # Step 2: Configuration creation
            step_start = time.time()
            logger.info("📝 Step 2: Creating configuration...")
            
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

            step_duration = time.time() - step_start
            performance_monitor.log_step("config_creation", step_duration, {
                "config_size_bytes": config_path.stat().st_size,
                "preparation_mode": True
            })

            # Step 3: Direct inference execution (like realtime_api.py)
            step_start = time.time()
            logger.info("🚀 Step 3: Executing direct inference...")
            
            # Define output path before inference
            output_video_path = Path("./results") / VERSION / "avatars" / avatar_id / "vid_output" / "generated.mp4"
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run inference directly (like realtime_api.py)
            inference_start = time.time()
            
            # Log GPU metrics before inference
            gpu_metrics_before = performance_monitor.log_gpu_metrics("before_inference")
            
            verified_output_path = run_inference_direct(config_path, output_video_path)
            inference_duration = time.time() - inference_start
            
            # Log GPU metrics after inference
            gpu_metrics_after = performance_monitor.log_gpu_metrics("after_inference")

            # Log successful inference metrics
            performance_monitor.log_subprocess_metrics(inference_duration, 0, {
                "method": "direct_inference",
                "output_path": str(verified_output_path),
                "gpu_metrics_before": gpu_metrics_before,
                "gpu_metrics_after": gpu_metrics_after
            })

            step_duration = time.time() - step_start
            performance_monitor.log_step("direct_inference", step_duration, {
                "inference_duration": inference_duration,
                "output_path": str(verified_output_path)
            })

            logger.info("✅ Direct inference completed")

            # Step 4: Output verification (already have verified_output_path from direct inference)
            step_start = time.time()
            logger.info("🔍 Step 4: Verifying output video...")
            
            # We already have the verified output path from direct inference
            expected_output = verified_output_path
            
            # Get output file size
            output_size = expected_output.stat().st_size
            if output_size == 0:
                raise HTTPException(status_code=500, detail="Generated video file is empty")

            # Log file metrics
            performance_monitor.log_file_metrics(video_size, audio_size, output_size)

            step_duration = time.time() - step_start
            performance_monitor.log_step("output_verification", step_duration, {
                "output_file_size_bytes": output_size,
                "output_path": str(expected_output)
            })

            # Step 5: Response preparation
            step_start = time.time()
            logger.info("📤 Step 5: Preparing response...")
            
            # Calculate total duration
            end_time = time.time()
            total_duration = end_time - start_time
            
            step_duration = time.time() - step_start
            performance_monitor.log_step("response_preparation", step_duration, {
                "total_duration": total_duration
            })

            # Stop monitoring and log final metrics
            performance_monitor.stop_monitoring()

            # Return the video
            return FileResponse(
                path=expected_output,
                media_type='video/mp4',
                filename=f"{total_duration:.2f}s_generated.mp4"
            )

    except subprocess.TimeoutExpired:
        performance_monitor.stop_monitoring()
        logger.error("⏰ Generation timed out after 10 minutes")
        raise HTTPException(status_code=500, detail="Generation timed out after 10 minutes")
    except Exception as e:
        performance_monitor.stop_monitoring()
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"💥 Error after {duration:.2f}s: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
