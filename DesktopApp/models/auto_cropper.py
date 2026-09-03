import torch
from PIL import Image
import numpy as np
import time
import os

# Global instances for lazy loading
_yolo_model = None
_dino_processor = None
_dino_model = None
_ocr_model = None
_models_loaded = False

def load_models():
    """Lazily loads the auto-cropping models. Warnings print only once."""
    global _yolo_model, _dino_processor, _dino_model, _ocr_model, _models_loaded
    
    if _models_loaded:
        return
    _models_loaded = True
    
    # Load YOLO - auto device (GPU if available)
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolo11x.pt")
    except ImportError:
        print("⚠️ Ultralytics not installed. Global crop (YOLO) will fallback to full image.")
    
    # Load Grounding DINO
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        print("⚠️ Transformers not installed. Local crop (DINO) will fallback to center crop.")
            
    # OCR disabled for now to test YOLO and DINO independently
    _ocr_model = None

def predict_crops(image_path):
    """
    Predicts the global (train) and local (railroad name) bounding boxes.
    Returns:
        {"global_box": [x1, y1, x2, y2], "local_box": [x1, y1, x2, y2]}
    """
    load_models()
    
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Default fallbacks - return None if model fails to find a box
    global_box = None
    local_box = None
    
    yolo_time = 0.0
    dino_time = 0.0
    
    # 1. Global Crop (YOLO) - filter only 'train' class (COCO class ID = 6)
    TRAIN_CLASS_ID = 6
    if _yolo_model is not None:
        try:
            t0 = time.time()
            results = _yolo_model(img, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                classes   = results[0].boxes.cls.cpu().numpy().astype(int)
                confs     = results[0].boxes.conf.cpu().numpy()

                # Strictly only 'train' class boxes
                train_mask = classes == TRAIN_CLASS_ID
                if train_mask.any():
                    train_boxes = boxes_xyxy[train_mask]
                    train_confs = confs[train_mask]
                    # Pick highest confidence train box
                    best = np.argmax(train_confs)
                    global_box = [int(x) for x in train_boxes[best]]
            yolo_time = time.time() - t0
        except Exception as e:
            print("YOLO Prediction Error:", e)
            
    # 2. Local Crop (Grounding DINO inside the YOLO global crop)
    if _dino_model is not None and _dino_processor is not None and global_box is not None:
        try:
            t1 = time.time()
            # Crop the original image to just the train
            train_crop_img = img.crop(global_box)
            
            # Prioritized prompts: Try to find 'railroad name' first. If nothing is found, fallback to 'company logo'.
            prompts_to_try = ["railroad name.", "company logo."]
            best_local_box = None
            
            for prompt in prompts_to_try:
                inputs = _dino_processor(images=train_crop_img, text=prompt, return_tensors="pt").to(_dino_model.device)
                with torch.no_grad():
                    outputs = _dino_model(**inputs)
                
                # Post-process DINO boxes based on the cropped image size
                target_sizes = torch.tensor([train_crop_img.size[::-1]])
                dino_results = _dino_processor.image_processor.post_process_object_detection(
                    outputs, threshold=0.25, target_sizes=target_sizes
                )[0]
                
                if len(dino_results["boxes"]) > 0:
                    boxes = dino_results["boxes"].cpu().numpy()
                    scores = dino_results["scores"].cpu().numpy()
                    sorted_indices = np.argsort(scores)[::-1]
                    
                    for idx in sorted_indices:
                        dino_box = [int(x) for x in boxes[idx]]
                        
                        if dino_box[2] <= dino_box[0] or dino_box[3] <= dino_box[1]:
                            continue
                            
                        # Prevent "big useless box": ignore boxes that cover more than 70% of the train crop
                        box_area = (dino_box[2] - dino_box[0]) * (dino_box[3] - dino_box[1])
                        crop_area = train_crop_img.size[0] * train_crop_img.size[1]
                        if box_area > crop_area * 0.70:
                            continue
                        
                        # Shift the DINO box coordinates back to the original full image scale
                        shifted_box = [
                            dino_box[0] + global_box[0],
                            dino_box[1] + global_box[1],
                            dino_box[2] + global_box[0],
                            dino_box[3] + global_box[1],
                        ]
                        
                        best_local_box = shifted_box
                        break
                
                # If we found a valid box with this prompt, don't try the fallback prompts
                if best_local_box is not None:
                    break
                        
            if best_local_box is not None:
                local_box = best_local_box
            
            dino_time = time.time() - t1
                
        except Exception as e:
            print("DINO Prediction Error:", e)
            
    if global_box is None:
        global_box = [0, 0, w, h]
    if local_box is None:
        if global_box is not None:
            gx1, gy1, gx2, gy2 = global_box
            gw = gx2 - gx1
            gh = gy2 - gy1
            local_box = [
                gx1 + int(gw * 0.25),
                gy1 + int(gh * 0.25),
                gx1 + int(gw * 0.75),
                gy1 + int(gh * 0.75)
            ]
        else:
            local_box = [int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]
            
    print(f"[{os.path.basename(image_path)}] YOLO: {yolo_time:.2f}s | DINO: {dino_time:.2f}s")
            
    return {
        "global_box": global_box,
        "local_box": local_box
    }
