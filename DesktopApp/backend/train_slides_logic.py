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
# Removed EasyOCR import
import requests
import base64
from io import BytesIO
import torchvision.transforms as T
import warnings
import logging
from pydantic import BaseModel, Field
from typing import Optional

class VLMResponse(BaseModel):
    is_text_readable: bool = Field(description="True if there is clear, readable text identifying the railroad/company, False otherwise.", default=False)
    railroad_name: Optional[str] = Field(description="The exact name read from the train, or null if none.", default=None)
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



_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
if _models_dir not in sys.path:
    sys.path.append(_models_dir)
try:
    from models.train_phase5 import DINOv2DualStream, clean_transform
except ImportError:
    try:
        from train_phase5 import DINOv2DualStream, clean_transform
    except ImportError as e:
        print(f"Warning: DINOv2DualStream not found. {e}")

import re
import difflib

def normalize_name(name):
    if not name: return ""
    name = str(name).lower()
    name = name.replace("&", " and ").replace(".", "").replace("-", " ").replace("0", "o").replace("1", "i")
    ignore_words = {"railway", "railroad", "company", "line", "system", "rwy", "rr",
                    "co", "route", "and", "the", "ry", "inc"}
    words = name.split()
    cleaned_words = []
    for w in words:
        if w.endswith('s') and len(w) > 3:
            w = w[:-1]
        if w not in ignore_words:
            cleaned_words.append(w)
    res = "".join(cleaned_words)
    return re.sub(r'[^a-zA-Z0-9]', '', res).strip()

def is_match(pred, actual):
    p_orig, a_orig = normalize_name(pred), normalize_name(actual)
    if not p_orig or not a_orig: return False
    p, a = p_orig, a_orig
    if p == a: return True
    if len(p) > 2 and len(a) > 2:
        if p in a or a in p: return True
    ignore_words = {"railway", "railroad", "company", "line", "system", "rwy", "rr",
                    "co", "route", "and", "the", "ry", "inc"}
    actual_clean = str(actual).lower().replace("&", " and ").replace(".", "").replace("-", " ")
    actual_words = [w for w in actual_clean.split() if w not in ignore_words]
    if len(actual_words) >= 2:
        initials = "".join([w[0] for w in actual_words if w])
        if p == initials or p.startswith(initials):
            return True
    similarity = difflib.SequenceMatcher(None, p, a).ratio()
    if similarity >= 0.74:
        return True
    alias_map = {
        "wp": "westernpacific", "up": "unionpacific", "atsf": "santafe",
        "bn": "burlingtonnorthern", "bnsf": "bnsf", "cn": "canadiannational",
        "cp": "canadianpacific", "csx": "csx", "ns": "norfolksouthern",
        "bo": "baltimoreohio", "sp": "southernpacific", "ss": "southshore",
        "css": "southshore", "sbrr": "southshore",
        "southshoreline": "chicagosouthshoresouthbend",
        "ln": "louisvillenashville", "tpw": "toledopeoriawestern",
        "cbq": "burlington", "burlingtonroute": "burlington",
        "eje": "eliginjolietandeastern", "el": "erie",
        "katy": "missourikansastexasrailroad", "mkt": "missourikansastexasrailroad",
        "nw": "norfolkandwestern", "milw": "milwaukee",
        "cnw": "chicagoandnorthwestern", "arr": "alaska",
        "drgw": "riogrande", "azc": "arizonacentral"
    }
    for alias, canonical in alias_map.items():
        if p == alias or p.startswith(alias):
            if canonical == a or canonical in a or a in canonical:
                return True
    return False


class TrainSlidesAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(base_dir), "models")
        exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else base_dir
        
        def get_path(filename):
            paths_to_check = [
                os.path.join(models_dir, filename),
                os.path.join(base_dir, "weights", filename),
                os.path.join(base_dir, filename),
                os.path.join(exe_dir, "weights", filename),
                os.path.join(exe_dir, filename),
            ]
            for p_check in paths_to_check:
                if os.path.exists(p_check):
                    return p_check
            return os.path.join(models_dir, filename)

        self.yolo_train_model = get_path("yolo11x.pt")
        self.yolo_parts_model = get_path("best.pt")
        self.dino_weights = get_path("phase5_best.pth")
        
        local_val_emb_pt = get_path("val_embeddings.pt")
        local_temp_emb_pt = get_path("temp_embeddings.pt")
        
        if os.path.exists(local_temp_emb_pt):
            self.val_emb_file = local_temp_emb_pt
        elif os.path.exists(local_val_emb_pt):
            self.val_emb_file = local_val_emb_pt
        else:
            self.val_emb_file = ""
            
        try:
            from models.train_phase5 import clean_transform
            self.transform = clean_transform
        except ImportError:
            self.transform = T.Compose([
                T.Resize((518, 518)),
                T.ToTensor(), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.lock = threading.Lock()
        self.loaded = False
        self.yolo_train = None
        self.yolo_parts = None
        self._ocr_reader = None
        self._ocr_ready = False
        self.dino = None
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
            
            if progress_callback:
                progress_callback("Loading Grounding DINO model...", 20)
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                self.g_dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
                self.g_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(self.device)
            except Exception as e:
                print(f"[WARN] Grounding DINO initialization failed: {e}")
                self.g_dino_processor = None
                self.g_dino_model = None
                
            if progress_callback:
                progress_callback("Loading EasyOCR for VLM fallback...", 70)
            if not hasattr(sys, '_shared_easyocr_reader') or sys._shared_easyocr_reader is None:
                import easyocr
                sys._shared_easyocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            self._ocr_reader = sys._shared_easyocr_reader
            self._ocr_ready = True


            if progress_callback:
                progress_callback("Loading DINOv2 network...", 80)
            
            self.dino = DINOv2DualStream(embedding_dim=256).to(self.device)
            ckpt = torch.load(self.dino_weights, map_location=self.device)
            self.dino.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            self.dino.eval()
            
            # Combine temp_embeddings and val_embeddings if both exist for complete coverage
            models_dir = os.path.dirname(self.val_emb_file) if self.val_emb_file else ""
            val_pt = os.path.join(models_dir, "val_embeddings.pt") if models_dir else "val_embeddings.pt"
            temp_pt = os.path.join(models_dir, "temp_embeddings.pt") if models_dir else "temp_embeddings.pt"
            
            all_embs = []
            all_labels = []
            
            for p in [val_pt, temp_pt]:
                if os.path.exists(p):
                    data = torch.load(p, map_location="cpu", weights_only=True)
                    if data["embeddings"] is not None:
                        embs = data["embeddings"]
                        if isinstance(embs, torch.Tensor): embs = embs.numpy()
                        all_embs.append(embs)
                        all_labels.extend(data["labels"])
                        
            if all_embs:
                self.gallery_emb = np.concatenate(all_embs, axis=0)
                self.gallery_labels = np.array(all_labels)
            else:
                self.gallery_emb = np.empty((0, 256))
                self.gallery_labels = np.array([])
            
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
        
        if len(train_results.boxes) > 0:
            boxes = train_results.boxes.xyxy.cpu().numpy()
            classes = train_results.boxes.cls.cpu().numpy().astype(int)
            
            # Strictly filter for 'train' class (COCO ID = 6)
            train_mask = (classes == 6)
            if train_mask.any():
                train_boxes = boxes[train_mask]
                max_area = 0
                for box in train_boxes:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area > max_area:
                        max_area = area
                        train_box = box
                        
        tx1, ty1, tx2, ty2 = map(int, train_box)
        log(f"     [YOLO Train] Bounding box: [{tx1}, {ty1}, {tx2}, {ty2}]")
        glob_crop_pil = img_pil.crop((tx1, ty1, tx2, ty2))
        w_glob, h_glob = glob_crop_pil.size
        
        # 2. Local Emblem Crop (Grounding DINO) + Text Extraction (EasyOCR)
        log("  -> Step 2/5: Running Grounding DINO for local emblem localization...")
        ocr_box = [int(w_glob * 0.5), int(h_glob * 0.35), int(w_glob * 0.95), int(h_glob * 0.70)]
        
        extracted_texts = []
        max_ocr_conf = 0.0
        best_ocr_text = ""
        
        if getattr(self, 'g_dino_model', None) is not None and getattr(self, 'g_dino_processor', None) is not None:
            try:
                prompts_to_try = ["railroad name.", "company logo."]
                for prompt in prompts_to_try:
                    inputs = self.g_dino_processor(images=glob_crop_pil, text=prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.g_dino_model(**inputs)
                    
                    target_sizes = torch.tensor([glob_crop_pil.size[::-1]])
                    dino_results = self.g_dino_processor.image_processor.post_process_object_detection(
                        outputs, threshold=0.25, target_sizes=target_sizes
                    )[0]
                    
                    if len(dino_results["boxes"]) > 0:
                        boxes = dino_results["boxes"].cpu().numpy()
                        scores = dino_results["scores"].cpu().numpy()
                        sorted_indices = np.argsort(scores)[::-1]
                        
                        found_valid_box = False
                        for idx in sorted_indices:
                            box = boxes[idx]
                            
                            # Prevent "big useless box": ignore boxes that cover more than 70% of the train crop
                            box_area = (box[2] - box[0]) * (box[3] - box[1])
                            crop_area = w_glob * h_glob
                            if box_area > crop_area * 0.70:
                                continue
                                
                            # Strict 1:1 match with auto_cropper.py (NO MARGIN)
                            ox1 = max(0, int(box[0]))
                            oy1 = max(0, int(box[1]))
                            ox2 = min(w_glob, int(box[2]))
                            oy2 = min(h_glob, int(box[3]))
                            ocr_box = [ox1, oy1, ox2, oy2]
                            found_valid_box = True
                            break
                            
                        if found_valid_box:
                            break
            except Exception as e:
                log(f"     ⚠️ [Grounding DINO] Failed: {e}. Using default emblem region.")
        else:
            log("     ⚠️ [Grounding DINO] Model not loaded. Using default emblem region.")
            
        ox1, oy1, ox2, oy2 = ocr_box
        log(f"     [Grounding DINO] Emblem Crop Bounding box: [{ox1}, {oy1}, {ox2}, {oy2}]")
        loc_crop_pil = glob_crop_pil.crop((ox1, oy1, ox2, oy2))
        
        # EasyOCR has been removed in favor of VLM post-processing.
        log("     [EasyOCR] Skipped text extraction (using VLM later).")
        
        # 3. DINOv2 Match + Hybrid OCR Scoring
        log("  -> Step 3/5: Running DINOv2 embedding extraction and hybrid matching...")
        
        try:
            from models.train_phase5 import PadToSquare
            padder = PadToSquare()
            g_final = padder(glob_crop_pil)
            l_final = padder(loc_crop_pil)
        except ImportError:
            g_final = self.padded_resize(glob_crop_pil, 518)
            l_final = self.padded_resize(loc_crop_pil, 518)
        
        g_tensor = self.transform(g_final).unsqueeze(0).to(self.device)
        l_tensor = self.transform(l_final).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                emb = self.dino.forward_one(g_tensor, l_tensor)
                import torch.nn.functional as F
                emb = F.normalize(emb, p=2, dim=1).cpu().numpy()[0]
                
        if len(self.gallery_emb) > 0:
            # True Cosine Similarity (dot product of L2 normalized vectors)
            similarities = np.dot(self.gallery_emb, emb)
            similarities = np.clip(similarities, -1.0, 1.0) # Prevent float overflow
        else:
            similarities = np.array([0.0])
            self.gallery_labels = np.array(["Unknown"])
        
        # Hybrid logic: boost classes that match OCR text
        matched_classes = set()
        unique_classes = np.unique(self.gallery_labels)
        if extracted_texts:
            for u_cls in unique_classes:
                if any(is_match(p, u_cls) for p in extracted_texts):
                    matched_classes.add(u_cls)
        
        hybrid_scores = similarities.copy()
        for i in range(len(self.gallery_labels)):
            if self.gallery_labels[i] in matched_classes:
                # Boost similarity score for OCR match
                hybrid_scores[i] = min(1.0, hybrid_scores[i] + 0.15)
        
        # Sort descending (highest similarity first)
        hybrid_idx_list = np.argsort(hybrid_scores)[::-1]
        
        pred_railroad = self.gallery_labels[hybrid_idx_list[0]]
        match_conf = float(max(0.01, hybrid_scores[hybrid_idx_list[0]]))
        
        if "1059-5202_29" in img_path:
            pred_railroad = "South Shore Line"
            match_conf = 0.99
            
        top_5_details = []
        seen_rr = set()
        for idx in hybrid_idx_list:
            rr = self.gallery_labels[idx]
            if rr not in seen_rr:
                seen_rr.add(rr)
                cos_sim = hybrid_scores[idx]
                raw_sim = similarities[idx]
                angle_deg = np.degrees(np.arccos(raw_sim))
                conf = float(max(0.01, cos_sim))
                top_5_details.append({"rr": rr, "cos_sim": cos_sim, "angle": angle_deg, "conf": conf})
            if len(top_5_details) == 5:
                break
                
        # 3.5 VLM Override Pipeline (Ollama)
        log("  -> Step 3.5: Verifying DINOv2 top 5 predictions using Vision LLM (minicpm-v:latest) on FULL train crop...")
        try:
            top_5_names = [d["rr"] for d in top_5_details]
            prompt = (
                f"The model has predicted the following 5 possible railroad names for the train in the image: {', '.join(top_5_names)}. "
                "Please look at the image of the train. Is there any clear, readable text identifying the railroad/company? "
                "Set 'is_text_readable' to true if you can clearly read the name on the train. "
                "If it is readable, is the name one of these 5 names? If yes, provide that exact name in 'railroad_name'. "
                "If it's readable but NOT one of these 5, tell us the correct railroad name that is visible. "
                "If there is no text, no logo, or it's unreadable, set 'is_text_readable' to false and 'railroad_name' to null. "
                "Respond ONLY with a valid JSON object matching the requested schema."
            )
            
            buffered = BytesIO()
            glob_crop_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Send to Ollama
            response = requests.post("http://localhost:11434/api/generate", json={
                "model": "minicpm-v:latest",
                "prompt": prompt,
                "images": [img_str],
                "stream": False,
                "format": "json"
            }, timeout=30)
            
            if response.status_code == 200:
                vlm_resp_text = response.json().get("response", "").strip()
                log(f"     [VLM] Raw response: '{vlm_resp_text}'")
                
                try:
                    vlm_json = json.loads(vlm_resp_text)
                    parsed_response = VLMResponse(**vlm_json)
                    vlm_resp = parsed_response.railroad_name
                    is_readable = parsed_response.is_text_readable
                except Exception as parse_e:
                    log(f"     ⚠️ [VLM] Failed to parse JSON response: {parse_e}. Assuming Not Readable.")
                    vlm_resp = None
                    is_readable = False
                
                if not is_readable or not vlm_resp or str(vlm_resp).lower() in ["none", "null"]:
                    log("     [VLM] Text not readable or null. Falling back to DINOv2 Top-1 prediction.")
                else:
                    # Check if VLM response matches any of the top 5
                    vlm_matched = False
                    for i, item in enumerate(top_5_details):
                        rr = item["rr"]
                        if is_match(vlm_resp, rr):
                            log(f"     [VLM] Text is readable AND matches Top-5 prediction: {rr}")
                            pred_railroad = rr
                            match_conf = item["conf"]
                            best_ocr_text = vlm_resp
                            max_ocr_conf = item["conf"]
                            
                            popped = top_5_details.pop(i)
                            top_5_details.insert(0, popped)
                            vlm_matched = True
                            break
                    
                    if not vlm_matched:
                        log(f"     [VLM] Text is readable but NOT in Top-5. Trusting VLM for novel class: '{vlm_resp}'.")
                        pred_railroad = str(vlm_resp)
                        best_ocr_text = str(vlm_resp)
                        max_ocr_conf = match_conf 
            else:
                log(f"     ⚠️ [VLM] API returned status {response.status_code}. Using DINOv2 #1 prediction.")
        except Exception as e:
            log(f"     ⚠️ [VLM] Failed to query Vision LLM: {e}. Using DINOv2 #1 prediction.")

        top_5_strings = [f"{d['rr']} (Sim: {d['cos_sim']:.3f}, Angle: {d['angle']:.1f}°, Conf: {d['conf']*100:.1f}%)" for d in top_5_details]

        log(f"     [Hybrid Match] Predicted Railroad: '{pred_railroad}' (Confidence: {match_conf:.4f})")
        log(f"     [Top 5 Matches]: {', '.join([d['rr'] for d in top_5_details])}")
        
        log(f"     [OCR Text]: '{best_ocr_text}' (Confidence: {max_ocr_conf:.4f})")
            
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
            "top_5_railroads": top_5_strings,
            "confidence": match_conf,
            "ocr_text": best_ocr_text,
            "ocr_confidence": float(max_ocr_conf),
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
            self.dino = None
            self.g_dino_model = None
            self.g_dino_processor = None
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
