import os
import sys
import shutil
import requests
import traceback
from pathlib import Path

# Determined if running as a PyInstaller bundle
def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_app_dir()
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model URLs (Example links — these should be verified/updated)
# For YOLO doclayout, we usually have it locally, but we can host it on Dropbox/HF
MODEL_SOURCES = {
    "yolo": {
        "filename": "doclayout_yolo_docstructbench_imgsz1024.pt",
        "url": "https://huggingface.co/antigravity-ai/doclayout-yolo/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt", # Placeholder
        "target": os.path.join(BASE_DIR, "doclayout_yolo_docstructbench_imgsz1024.pt")
    },
    "easyocr_en": {
        "filename": "english_g2.zip",
        "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
        "target_dir": os.path.join(MODELS_DIR, "easyocr/model")
    }
}

def setup_portable_paths():
    """Sets environment variables to force models into the application directory."""
    # Force EasyOCR to use local folder
    os.environ["EASYOCR_MODULE_PATH"] = os.path.join(MODELS_DIR, "easyocr")
    
    # Force MinerU model source (if needed)
    # os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
    
    # Force Torch Cache
    os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, "torch")
    
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "torch"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, "easyocr"), exist_ok=True)

def check_disk_space(required_gb=10):
    """Checks if there is enough free space on the drive where the app is located."""
    total, used, free = shutil.disk_usage(BASE_DIR)
    free_gb = free // (2**30)
    return free_gb >= required_gb, free_gb

def health_check(progress_callback=None):
    """
    Checks if all required models are present.
    Returns: (bool, str) -> (Success, Message)
    """
    try:
        setup_portable_paths()
        errors = []
        
        # 1. Check Disk Space first if download might be needed
        ok, free = check_disk_space(5) # Minimum 5GB for safety
        if not ok:
            return False, f"Not enough disk space! Only {free}GB left. We need at least 5-10GB for AI models."

        # 2. Check YOLO
        yolo_cfg = MODEL_SOURCES["yolo"]
        if not os.path.exists(yolo_cfg["target"]):
            errors.append(f"Missing YOLO weights: {yolo_cfg['filename']}")
            
        # 3. Check Ollama Models
        try:
            import ollama
            try:
                models = [m['name'] for m in ollama.list()['models']]
                required = ["minicpm-v:latest", "llama3.2:1b"]
                for r in required:
                    if r not in models and r.split(':')[0] not in models:
                        errors.append(f"Ollama model missing: {r}")
            except Exception:
                errors.append("Ollama service is not running. Please start Ollama Desktop.")
        except ImportError:
            errors.append("Ollama library not installed.")

        if errors:
            return False, "\n".join(errors)

        return True, "All models active and ready."
        
    except Exception as e:
        return False, f"Health check crashed: {str(e)}\n{traceback.format_exc()}"

def download_model(name, url, dest, progress_callback=None):
    """Downloads a file with progress reporting."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    with open(dest, "wb") as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for data in response.iter_content(chunk_size=4096):
                downloaded += len(data)
                f.write(data)
                if progress_callback:
                    done = int(50 * downloaded / total_size)
                    progress_callback(name, (downloaded / total_size) * 100)

if __name__ == "__main__":
    health_check()
