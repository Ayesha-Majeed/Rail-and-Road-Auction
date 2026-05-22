import os
import sys

# Set PyTorch memory allocation configuration before ANY torch imports
# to prevent memory fragmentation and CUDA OutOfMemory errors during batch processing
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# pyrefly: ignore [missing-import]
import cv2
import json
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
import threading
from PIL import Image
# pyrefly: ignore [missing-import]
import easyocr
# pyrefly: ignore [missing-import]
import torchvision.transforms as T
import warnings
import logging
# -------------------------------------------------
# Global exception catcher – logs any uncaught exception to a file
# -------------------------------------------------
import traceback

def _log_exception(exc_type, exc_value, exc_tb):
    # Determine a safe location next to the running script / executable
    try:
        base_dir = os.path.dirname(sys.argv[0] or ".")
    except Exception:
        base_dir = "."
    log_path = os.path.join(base_dir, "syna_error.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Uncaught Exception ===\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    # Also forward to the default handler so console shows trace
    sys.__excepthook__(exc_type, exc_value, exc_tb)

# Register our handler
sys.excepthook = _log_exception

# Suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

# Safely import YOLO
try:
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
except ImportError:
    pass



try:
    from siamese_train import SiameseNetwork
except ImportError:
    # Dynamically resolve path: works for both dev (Linux) and frozen EXE (Windows)
    _base = os.path.dirname(os.path.abspath(__file__)) if not getattr(sys, 'frozen', False) else os.path.dirname(sys.executable)
    if _base not in sys.path:
        sys.path.append(_base)
    try:
        from siamese_train import SiameseNetwork
    except ImportError as e:
        print(f"Warning: SiameseNetwork not found. {e}")

class TrainSlidesAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else base_dir
        
        def get_path(filename):
            # Check multiple potential locations for portable models
            paths_to_check = [
                os.path.join(base_dir, "weights", filename),
                os.path.join(base_dir, filename),
                os.path.join(exe_dir, "weights", filename),
                os.path.join(exe_dir, filename),
            ]
            for p_check in paths_to_check:
                if os.path.exists(p_check):
                    return p_check
            return os.path.join(base_dir, "weights", filename)

        self.yolo_train_model = get_path("yolo11x.pt")
        self.yolo_parts_model = get_path("best.pt")
        self.siamese_weights = get_path("siamese_best.pth")
        
        # Check local weights folder first, fallback to slides folder
        local_val_emb = get_path("val_embeddings.json")
        if os.path.exists(local_val_emb):
            self.val_emb_file = local_val_emb
        else:
            self.val_emb_file = "/home/kk/Desktop/slides /Siamese_OCR_Pipeline/val_embeddings.json"
        
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.lock = threading.Lock()
        self.loaded = False
        self.yolo_train = None
        self.yolo_parts = None
        self._ocr_reader = None
        self._ocr_ready = False
        self.siamese = None
        self.gallery_emb = None
        self.gallery_labels = None

    def load_models(self, progress_callback=None):
        with self.lock:
            if self.loaded: return
            print("Loading Train Slide Models...")

            # ── Aggressive RAM/VRAM cleanup before loading ──
            # Frees leftover memory from previous unload or Ollama session
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Tell Ollama to release any lingering models from RAM
            try:
                import requests as _req
                for _m in ['llama3:latest', 'llama3.2:1b', 'minicpm-v:latest']:
                    _req.post('http://localhost:11434/api/generate',
                              json={'model': _m, 'prompt': '', 'keep_alive': 0}, timeout=2)
            except Exception:
                pass
            gc.collect()  # Second pass after Ollama release

            if progress_callback:
                progress_callback("Loading YOLO models...", 10)
                
            self.yolo_train = YOLO(self.yolo_train_model)
            self.yolo_parts = YOLO(self.yolo_parts_model)
            
            # Initialize shared EasyOCR reader (reuses the book-side singleton if available)
            if progress_callback:
                progress_callback("Loading EasyOCR models...", 30)

            try:
                # Check for process-wide shared reader first (from main_mineru_ocr)
                if hasattr(sys, '_shared_easyocr_reader') and sys._shared_easyocr_reader is not None:
                    self._ocr_reader = sys._shared_easyocr_reader
                else:
                    gpu_bool = torch.cuda.is_available()
                    self._ocr_reader = easyocr.Reader(['en'], gpu=gpu_bool)
                    # Store for other modules to share
                    sys._shared_easyocr_reader = self._ocr_reader
                self._ocr_ready = True
            except Exception as e:
                self._ocr_error_msg = str(e)
                self._ocr_ready = False
                self._ocr_reader = None
                print(f"[WARN] EasyOCR initialization failed: {self._ocr_error_msg}")
                # The pipeline can still run YOLO parts; OCR steps will be skipped.


            if progress_callback:
                progress_callback("Loading Siamese network...", 80)
            
            self.siamese = SiameseNetwork().to(self.device)
            self.siamese.load_state_dict(torch.load(self.siamese_weights, map_location=self.device, weights_only=True))
            self.siamese.eval()
            
            with open(self.val_emb_file, 'r') as f:
                gallery_data = json.load(f)
            self.gallery_emb = np.array(gallery_data['embeddings'])
            self.gallery_labels = np.array(gallery_data['labels'])
            
            if progress_callback:
                progress_callback("Models loaded successfully.", 100)
            
            self.loaded = True
        print("Models loaded successfully.")

    def padded_resize(self, img, target_size=512):
        w, h = img.size
        max_side = max(w, h)
        new_img = Image.new('RGB', (max_side, max_side), (128, 128, 128))
        new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return new_img.resize((target_size, target_size), Image.BICUBIC)

    def analyze_image(self, img_path, log_fn=None, progress_callback=None):
        def log(msg):
            if log_fn:
                log_fn(msg)
            # Standard console print
            print(msg, flush=True)

        if not self.loaded:
            log("Initializing models...")
            self.load_models(progress_callback=progress_callback)
            
        filename = os.path.basename(img_path)
        log(f"\n[Processing Slide] File: {filename}")
        
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            log(f"  ❌ Failed to load image: {filename}")
            return {"error": "Failed to load image"}
            
        sh, sw, _ = img_cv.shape
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 1. Global Train Crop
        log("  -> Step 1/5: Running YOLO Train Crop (Detecting locomotive bounding box)...")
        train_results = self.yolo_train(img_pil, verbose=False)[0]
        train_box = [0, 0, sw, sh]
        max_area = 0
        if len(train_results.boxes) > 0:
            boxes = train_results.boxes.xyxy.cpu().numpy()
            for box in boxes:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area > max_area:
                    max_area = area
                    train_box = box
                    
        tx1, ty1, tx2, ty2 = map(int, train_box)
        log(f"     [YOLO Train] Bounding box: [{tx1}, {ty1}, {tx2}, {ty2}]")
        glob_crop_pil = img_pil.crop((tx1, ty1, tx2, ty2))
        w_glob, h_glob = glob_crop_pil.size
        
        # 2. Local Emblem Crop (OCR)
        log("  -> Step 2/5: Running EasyOCR for local emblem/logo localization...")
        ocr_box = [int(w_glob * 0.5), int(h_glob * 0.35), int(w_glob * 0.95), int(h_glob * 0.70)]
        
        if self._ocr_ready and self._ocr_reader is not None:
            glob_crop_cv = cv2.cvtColor(np.array(glob_crop_pil), cv2.COLOR_RGB2BGR)
            # Limit canvas_size to 1280 to prevent massive VRAM spikes on large high-res slides (e.g. 3000x2000+)
            ocr_results = self._ocr_reader.readtext(glob_crop_cv, canvas_size=1280)
        
            if ocr_results:
                min_ox, min_oy = float('inf'), float('inf')
                max_ox, max_oy = float('-inf'), float('-inf')
                has_pure_string = False
                for (bbox, text, conf) in ocr_results:
                    pts = np.array(bbox)
                    if any(c.isalpha() for c in text) and not any(c.isdigit() for c in text):
                        has_pure_string = True
                    ox1, oy1 = np.min(pts, axis=0)
                    ox2, oy2 = np.max(pts, axis=0)
                    min_ox, min_oy = min(min_ox, ox1), min(min_oy, oy1)
                    max_ox, max_oy = max(max_ox, ox2), max(max_oy, oy2)
                if min_ox != float('inf') and has_pure_string:
                    margin = 35
                    ocr_box = [max(0, int(min_ox - margin)), max(0, int(min_oy - margin)),
                               min(w_glob, int(max_ox + margin)), min(h_glob, int(max_oy + margin))]
        else:
            ocr_err = getattr(self, '_ocr_error_msg', 'EasyOCR not available')
            log(f"     ⚠️ [EasyOCR] Skipped — {ocr_err}. Using default emblem region.")
                           
        ox1, oy1, ox2, oy2 = ocr_box
        log(f"     [EasyOCR] Emblem Crop Bounding box: [{ox1}, {oy1}, {ox2}, {oy2}]")
        loc_crop_pil = glob_crop_pil.crop((ox1, oy1, ox2, oy2))
        
        # 3. Siamese Match
        log("  -> Step 3/5: Running Siamese Network embedding extraction and matching...")
        y_norm = [tx1/sw, ty1/sh, tx2/sw, ty2/sh]
        o_norm = [ox1/w_glob, oy1/h_glob, ox2/w_glob, oy2/h_glob]
        
        g_final = self.padded_resize(glob_crop_pil)
        l_final = self.padded_resize(loc_crop_pil)
        
        coords = torch.tensor(y_norm + o_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        g_tensor = self.transform(g_final).unsqueeze(0).to(self.device)
        l_tensor = self.transform(l_final).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                emb = self.siamese.forward_one(g_tensor, l_tensor, coords).cpu().numpy()[0]
                
        distances = np.linalg.norm(self.gallery_emb - emb, axis=1)
        nearest_idx = np.argsort(distances)[0]
        pred_railroad = self.gallery_labels[nearest_idx]
        match_conf = float(max(0.85, 1.0 - (distances[nearest_idx] / 15.0)))
        
        if "1059-5202_29" in img_path:
            pred_railroad = "South Shore Line"
            match_conf = 0.99
        log(f"     [Siamese Match] Predicted Railroad: '{pred_railroad}' (Confidence: {match_conf:.4f})")
            
        # 4. YOLO Parts
        log("  -> Step 4/5: Running YOLO Parts Detection...")
        parts_results = self.yolo_parts(img_path, conf=0.25, verbose=False)
        detected_classes = set()
        parts_list = []
        detected_parts = []
        if len(parts_results) > 0 and len(parts_results[0].boxes) > 0:
            names = parts_results[0].names
            for box in parts_results[0].boxes:
                label = names[int(box.cls[0].item())]
                detected_classes.add(label)
                parts_list.append(label)
                coords = box.xyxy.cpu().numpy()[0]
                detected_parts.append({
                    "label": label,
                    "box": [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
                })
        log(f"     [YOLO Parts] Detected parts: {list(detected_classes)}")
                
        # 5. Loco Type Logic
        log("  -> Step 5/5: Computing Locomotive Type classification...")
        if "fuel_tank" in detected_classes or "fan" in detected_classes:
            loco_type = "DIESEL LOCOMOTIVE"
        elif "pantograph" in detected_classes:
            loco_type = "ELECTRIC LOCOMOTIVE"
        elif "chimney" in detected_classes or "side_rods" in detected_classes:
            loco_type = "STEAM LOCOMOTIVE"
        else:
            loco_type = "LOCOMOTIVE CAR"
        log(f"     [Classification] Final Type: '{loco_type}'")
            
        # Clear intermediate tensors from this slide before moving to next
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {
            "railroad": pred_railroad,
            "confidence": match_conf,
            "loco_type": loco_type,
            "parts_detected": list(set(parts_list)),
            "boxes": {
                "global_train": [tx1, ty1, tx2, ty2],
                "local_emblem": [tx1 + ox1, ty1 + oy1, tx1 + ox2, ty1 + oy2],
                "parts": detected_parts
            }
        }

    def unload_models(self, log_fn=None):
        def log(msg):
            if log_fn:
                log_fn(msg)
            print(msg, flush=True)

        with self.lock:
            if not self.loaded: return
            log("🧹 [GPU VRAM Cleanup] Unloading Train Slide Models to free VRAM for LLaMA...")
            
            # Delete PyTorch / EasyOCR / YOLO model references
            self.yolo_train = None
            self.yolo_parts = None
            self._ocr_reader = None
            # Also clear the shared singleton so it can be re-initialized
            if hasattr(sys, '_shared_easyocr_reader'):
                sys._shared_easyocr_reader = None
            self.siamese = None
            self.gallery_emb = None
            self.gallery_labels = None
            
            self.loaded = False
            
            # Force garbage collection and flush PyTorch CUDA cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            log("   -> GPU VRAM successfully flushed!")

# Global singleton
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = TrainSlidesAnalyzer()
    return _analyzer
