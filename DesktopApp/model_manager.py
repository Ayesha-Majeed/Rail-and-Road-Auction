import os
import sys
import shutil
import requests
import traceback
from pathlib import Path

# Determined if running as a PyInstaller bundle
def get_app_dir():
    if getattr(sys, 'frozen', False):
        # If frozen, look in the directory of the executable
        # AND check _MEIPASS for bundled files
        exe_dir = os.path.dirname(sys.executable)
        if hasattr(sys, '_MEIPASS'):
            # Bundle root (internal)
            return sys._MEIPASS
        return exe_dir
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_app_dir()
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_SOURCES = {
    "yolo": {
        "filename": "doclayout_yolo_docstructbench_imgsz1024.pt",
        "url": "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt",
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
    Returns: (bool, str, list) -> (Success, Message, MissingFeaturesList)
    """
    try:
        setup_portable_paths()
        errors = []
        missing_models = []
        
        # 1. Check Disk Space
        ok, free = check_disk_space(5) 
        if not ok:
            return False, f"Not enough disk space! Only {free}GB left.", []

        # 2. Check YOLO
        yolo_cfg = MODEL_SOURCES["yolo"]
        if os.path.exists(yolo_cfg["target"]):
            # Safety: If file is too small (< 1MB), it's likely a text error message from the server
            if os.path.getsize(yolo_cfg["target"]) < 1024 * 1024:
                os.remove(yolo_cfg["target"])
                errors.append(f"Corrupt YOLO weights detected (too small). Deleting to retry.")
                missing_models.append("yolo")
        else:
            errors.append(f"Missing YOLO weights: {yolo_cfg['filename']}")
            missing_models.append("yolo")
            
        # 3. Check Ollama Models
        try:
            import ollama
            try:
                # Try to ping service first
                models_info = ollama.list()
                
                # Get models list safely (can be list or dict containing 'models')
                raw_models = models_info.get('models', []) if isinstance(models_info, dict) else getattr(models_info, 'models', [])
                
                models = []
                for m in raw_models:
                    # Handle both dictionary and object formats
                    if isinstance(m, dict):
                        name = m.get('name') or m.get('model')
                    else:
                        name = getattr(m, 'model', getattr(m, 'name', None))
                    
                    if name:
                        models.append(name)
                required = ["minicpm-v:latest", "llama3.2:1b"]
                for r in required:
                    # Check both with and without tag
                    found = False
                    for m in models:
                        if m.split(':')[0] == r.split(':')[0]:
                            found = True
                            break
                    if not found:
                        errors.append(f"Ollama model missing: {r}")
                        missing_models.append(f"ollama:{r}")
            except Exception as e:
                err_msg = str(e).lower()
                if "connection" in err_msg or "refused" in err_msg:
                    errors.append("Ollama service is not running. Please start Ollama Desktop.")
                else:
                    errors.append(f"Ollama error: {str(e)}")
                missing_models.append("ollama_service")
        except ImportError:
            errors.append("Ollama library not installed.")

        if errors:
            return False, "\n".join(errors), missing_models

        return True, "All models active and ready.", []
        
    except Exception as e:
        return False, f"Health check crashed: {str(e)}", []

def ensure_models(progress_callback=None):
    """
    Attempts to download missing YOLO/EasyOCR models.
    Ollama models must be pulled via Ollama service.
    """
    ok, msg, missing = health_check()
    if ok: return True, msg
    
    for item in missing:
        if item == "yolo":
            cfg = MODEL_SOURCES["yolo"]
            if progress_callback: progress_callback("Downloading YOLO weights...", 0)
            download_model("YOLO Weights", cfg["url"], cfg["target"], progress_callback)
        elif item.startswith("ollama:"):
            model_name = item.split(":")[1]
            if progress_callback: progress_callback(f"Pulling {model_name}...", 0)
            try:
                import ollama
                # Ollama library has its own pull but we wrap it
                for chunk in ollama.pull(model_name, stream=True):
                    if 'completed' in chunk and 'total' in chunk:
                        pct = (chunk['completed'] / chunk['total']) * 100
                        if progress_callback: progress_callback(f"Pulling {model_name}", pct)
            except Exception as e:
                print(f"Failed to pull {model_name}: {e}")
                
    # Final check
    ok, msg, _ = health_check()
    return ok, msg

def download_model(name, url, dest, progress_callback=None):
    """Downloads a file with SSL resilience."""
    request_kwargs = {"stream": True, "timeout": 20, "verify": True}
    try:
        import certifi
        request_kwargs["verify"] = certifi.where()
    except ImportError:
        pass

    try:
        try:
            response = requests.get(url, **request_kwargs)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
            # Fallback for corporate/portable environments
            request_kwargs["verify"] = False
            response = requests.get(url, **request_kwargs)

        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        
        with open(dest, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=8192):
                    downloaded += len(data)
                    f.write(data)
                    if progress_callback:
                        progress_callback(name, (downloaded / total_size) * 100)
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")
        raise e

if __name__ == "__main__":
    health_check()
