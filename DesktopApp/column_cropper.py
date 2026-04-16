# """
# Column Cropper
# ==============
# Steps:
# 1. DocLayout-YOLO  → bboxes
# 2. Gap Detection   → columns (cluster-based)
# 3. Crop each column from original image
# 4. Downscale if longest side > MAX_SIZE
# 5. Save crops to output folder

# Requirements:
#     pip install doclayout-yolo opencv-python python-dotenv

# Usage:
#     python column_cropper.py
# """

# import os
# import sys
# import cv2
# import numpy as np
# from pathlib import Path

# # ─── LOAD .env ────────────────────────────────
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     print("❌ python-dotenv not found. Run: pip install python-dotenv")
#     sys.exit(1)

# # ─── OCR import (lazy — avoid circular import) ─
# def _get_ocr():
#     from ollama_column_ocr import process_page
#     return process_page

# # ─── CONFIG ───────────────────────────────────
# IMAGE_FOLDER  = os.getenv("IMAGE_FOLDER",  "./failed_cases")
# OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./ocr_results")
# CROPS_FOLDER  = os.path.join(OUTPUT_FOLDER, "column_crops")
# YOLO_WEIGHTS  = os.getenv("YOLO_WEIGHTS",  "doclayout_yolo_docstructbench_imgsz1024.pt")
# MAX_SIZE      = int(os.getenv("MAX_SIZE",  "2048"))   # longest side limit


# # ══════════════════════════════════════════════
# #  STEP 1 — YOLO Detection
# # ══════════════════════════════════════════════

# def run_yolo(image_path: str) -> list:
#     try:
#         from doclayout_yolo import YOLOv10
#     except ImportError:
#         print("❌ doclayout-yolo not found. Run: pip install doclayout-yolo")
#         sys.exit(1)

#     model = YOLOv10(YOLO_WEIGHTS)
#     results = model.predict(
#         source=image_path,
#         imgsz=1024,
#         conf=0.25,
#         iou=0.45,
#         verbose=False
#     )

#     boxes = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             boxes.append({
#                 "x1": x1, "y1": y1,
#                 "x2": x2, "y2": y2,
#             })
#     return boxes


# # ══════════════════════════════════════════════
# #  STEP 2 — Gap Detection (cluster-based)
# # ══════════════════════════════════════════════

# def find_all_gaps(boxes: list) -> list:
#     """
#     Cluster-based gap detection.
#     Boxes ka x-range overlap check karo — jahan nahi overlap wahan gap.
#     """
#     if len(boxes) < 2:
#         return []

#     sorted_boxes = sorted(boxes, key=lambda b: (b["x1"] + b["x2"]) / 2)

#     clusters = []
#     current_cluster = [sorted_boxes[0]]
#     current_max_x2  = sorted_boxes[0]["x2"]

#     for b in sorted_boxes[1:]:
#         if b["x1"] <= current_max_x2:
#             current_cluster.append(b)
#             current_max_x2 = max(current_max_x2, b["x2"])
#         else:
#             clusters.append(current_cluster)
#             current_cluster = [b]
#             current_max_x2  = b["x2"]
#     clusters.append(current_cluster)

#     if len(clusters) < 2:
#         return []

#     gaps = []
#     for i in range(len(clusters) - 1):
#         gap_start = max(b["x2"] for b in clusters[i])
#         gap_end   = min(b["x1"] for b in clusters[i + 1])
#         if gap_start < gap_end:
#             gaps.append({"start": gap_start, "end": gap_end})

#     return gaps


# # ══════════════════════════════════════════════
# #  STEP 3 — Column Boundaries
# # ══════════════════════════════════════════════

# def get_column_boundaries(boxes: list, gaps: list, img_width: int, img_height: int) -> list:
#     """
#     Gaps se column boundaries nikalo.
#     Har column ki x1, x2 define karo.
#     y1=0, y2=img_height (full height crop).

#     Returns: [{col_name, x1, y1, x2, y2}, ...]
#     """
#     if not gaps:
#         # Single column — poori image
#         return [{
#             "col_name": "C1",
#             "x1": 0, "y1": 0,
#             "x2": img_width, "y2": img_height
#         }]

#     boundaries = []

#     # C1: image start → first gap start
#     boundaries.append({
#         "col_name": "C1",
#         "x1": 0,
#         "y1": 0,
#         "x2": int(gaps[0]["start"]),
#         "y2": img_height
#     })

#     # C2, C3... : gap end → next gap start
#     for i in range(len(gaps) - 1):
#         boundaries.append({
#             "col_name": f"C{i+2}",
#             "x1": int(gaps[i]["end"]),
#             "y1": 0,
#             "x2": int(gaps[i+1]["start"]),
#             "y2": img_height
#         })

#     # Last column: last gap end → image end
#     boundaries.append({
#         "col_name": f"C{len(gaps)+1}",
#         "x1": int(gaps[-1]["end"]),
#         "y1": 0,
#         "x2": img_width,
#         "y2": img_height
#     })

#     return boundaries


# # ══════════════════════════════════════════════
# #  STEP 4 — Crop + Downscale
# # ══════════════════════════════════════════════

# def crop_and_resize(img: np.ndarray, col: dict) -> np.ndarray:
#     """
#     Column crop karo aur agar longest side > MAX_SIZE toh downscale.
#     LANCZOS4 use karo — text sharpest rehta hai.
#     """
#     x1 = max(0, col["x1"])
#     y1 = max(0, col["y1"])
#     x2 = min(img.shape[1], col["x2"])
#     y2 = min(img.shape[0], col["y2"])

#     crop = img[y1:y2, x1:x2]

#     h, w = crop.shape[:2]
#     longest = max(h, w)

#     if longest > MAX_SIZE:
#         scale = MAX_SIZE / longest
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         crop = cv2.resize(crop, (new_w, new_h),
#                           interpolation=cv2.INTER_LANCZOS4)
#         print(f"        → Resized: {w}x{h} → {new_w}x{new_h}")
#     else:
#         print(f"        → Size OK: {w}x{h} (no resize needed)")

#     return crop


# # ══════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════

# def process_image(image_path: str) -> list:
#     """
#     Ek image ke saare column crops nikalo.
#     Returns: [{col_name, crop_path, width, height}, ...]
#     """
#     print(f"\n{'='*55}")
#     print(f"  {Path(image_path).name}")
#     print(f"{'='*55}")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"  ❌ Could not load image")
#         return []

#     H, W = img.shape[:2]
#     print(f"  📐 Original size: {W}x{H}px")

#     # Step 1 — YOLO
#     print("  [1/3] Running DocLayout-YOLO...")
#     boxes = run_yolo(image_path)
#     print(f"        → {len(boxes)} boxes detected")
#     if not boxes:
#         print("  ⚠️  No boxes — skipping")
#         return []

#     # Step 2 — Gaps
#     print("  [2/3] Finding column gaps...")
#     gaps = find_all_gaps(boxes)
#     if gaps:
#         for i, g in enumerate(gaps):
#             print(f"        → GAP{i+1}: x={g['start']:.0f} → {g['end']:.0f}px")
#         print(f"        → {len(gaps)+1} columns detected")
#     else:
#         print("        → No gaps (single column)")

#     # Step 3 — Crop + Save
#     print("  [3/3] Cropping columns...")
#     columns = get_column_boundaries(boxes, gaps, W, H)

#     os.makedirs(CROPS_FOLDER, exist_ok=True)
#     stem = Path(image_path).stem

#     saved_crops = []
#     for col in columns:
#         crop = crop_and_resize(img, col)
#         ch, cw = crop.shape[:2]

#         crop_filename = f"{stem}_{col['col_name']}.jpg"
#         crop_path     = os.path.join(CROPS_FOLDER, crop_filename)

#         cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         print(f"        → Saved {col['col_name']}: {cw}x{ch}px  →  {crop_filename}")

#         saved_crops.append({
#             "col_name":  col["col_name"],
#             "crop_path": crop_path,
#             "width":     cw,
#             "height":    ch
#         })

#     return saved_crops


# def process_image_and_ocr(image_path: str) -> dict:
#     """
#     Ek image crop karo — phir turant OCR karo.
#     Returns: page OCR result dict
#     """
#     crops = process_image(image_path)
#     if not crops:
#         return {}

#     crop_paths = [c["crop_path"] for c in crops]
#     page_name  = Path(image_path).stem

#     print(f"  🤖 Starting OCR for {len(crop_paths)} column(s)...")
#     process_page = _get_ocr()
#     return process_page(page_name, crop_paths)


# def main():
#     print("\n" + "="*55)
#     print("  COLUMN CROPPER")
#     print("="*55)
#     print(f"  IMAGE_FOLDER : {IMAGE_FOLDER}")
#     print(f"  CROPS_FOLDER : {CROPS_FOLDER}")
#     print(f"  MAX_SIZE     : {MAX_SIZE}px (longest side)\n")

#     supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
#     folder    = Path(IMAGE_FOLDER)

#     if not folder.exists():
#         print(f"❌ Folder not found: {IMAGE_FOLDER}")
#         sys.exit(1)

#     images = sorted(set(
#         p for ext in supported
#         for p in list(folder.glob(f"*{ext}")) + list(folder.glob(f"*{ext.upper()}"))
#     ))

#     if not images:
#         print(f"❌ No images found in {IMAGE_FOLDER}")
#         sys.exit(1)

#     print(f"📁 Found {len(images)} image(s)\n")

#     from ollama_column_ocr import check_ollama, save_results
#     import time

#     # Ollama check pehle
#     if not check_ollama():
#         sys.exit(1)

#     all_pages   = []
#     total_start = time.time()

#     for img_path in images:
#         page_result = process_image_and_ocr(str(img_path))
#         if page_result:
#             all_pages.append(page_result)

#     # Save all results
#     if all_pages:
#         text_folder, json_path = save_results(all_pages)
#         total_time  = round(time.time() - total_start, 2)
#         total_words = sum(p.get("total_words", 0) for p in all_pages)

#         print(f"\n{'='*55}")
#         print(f"  ✅ ALL DONE!")
#         print(f"  Pages     : {len(all_pages)}")
#         print(f"  Words     : {total_words}")
#         print(f"  Time      : {total_time}s")
#         print(f"  Texts     : {text_folder}")
#         print(f"  Report    : {json_path}")
#         print(f"{'='*55}\n")


# if __name__ == "__main__":
#     main()




## YOLO CPU 


# """
# Column Cropper
# ==============
# Steps:
# 1. DocLayout-YOLO  → bboxes
# 2. Gap Detection   → columns (cluster-based)
# 3. Crop each column from original image
# 4. Downscale if longest side > MAX_SIZE
# 5. Save crops to output folder

# Requirements:
#     pip install doclayout-yolo opencv-python python-dotenv

# Usage:
#     python column_cropper.py
# """

# import os
# import sys
# import cv2
# import numpy as np
# from pathlib import Path

# # ─── LOAD .env ────────────────────────────────
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     print("❌ python-dotenv not found. Run: pip install python-dotenv")
#     sys.exit(1)

# # ─── OCR import (lazy — avoid circular import) ─
# def _get_ocr():
#     from ollama_column_ocr import process_page
#     return process_page

# # ─── CONFIG ───────────────────────────────────
# IMAGE_FOLDER  = os.getenv("IMAGE_FOLDER",  "./failed_cases")
# OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./ocr_results")
# CROPS_FOLDER  = os.path.join(OUTPUT_FOLDER, "column_crops")
# YOLO_WEIGHTS  = os.getenv("YOLO_WEIGHTS",  "doclayout_yolo_docstructbench_imgsz1024.pt")
# MAX_SIZE      = int(os.getenv("MAX_SIZE",  "2048"))   # longest side limit


# # ══════════════════════════════════════════════
# #  STEP 1 — YOLO Detection
# # ══════════════════════════════════════════════

# def run_yolo(image_path: str) -> list:
#     try:
#         from doclayout_yolo import YOLOv10
#     except ImportError:
#         print("❌ doclayout-yolo not found. Run: pip install doclayout-yolo")
#         sys.exit(1)

#     # CPU pe chalao — Ollama ne GPU memory full kar rakhi hai
#     model = YOLOv10(YOLO_WEIGHTS)
#     model.to("cpu")
#     results = model.predict(
#         source=image_path,
#         imgsz=1024,
#         conf=0.25,
#         iou=0.45,
#         verbose=False,
#         device="cpu"
#     )

#     boxes = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             boxes.append({
#                 "x1": x1, "y1": y1,
#                 "x2": x2, "y2": y2,
#             })
#     return boxes


# # ══════════════════════════════════════════════
# #  STEP 2 — Gap Detection (cluster-based)
# # ══════════════════════════════════════════════

# def find_all_gaps(boxes: list) -> list:
#     """
#     Cluster-based gap detection.
#     Boxes ka x-range overlap check karo — jahan nahi overlap wahan gap.
#     """
#     if len(boxes) < 2:
#         return []

#     sorted_boxes = sorted(boxes, key=lambda b: (b["x1"] + b["x2"]) / 2)

#     clusters = []
#     current_cluster = [sorted_boxes[0]]
#     current_max_x2  = sorted_boxes[0]["x2"]

#     for b in sorted_boxes[1:]:
#         if b["x1"] <= current_max_x2:
#             current_cluster.append(b)
#             current_max_x2 = max(current_max_x2, b["x2"])
#         else:
#             clusters.append(current_cluster)
#             current_cluster = [b]
#             current_max_x2  = b["x2"]
#     clusters.append(current_cluster)

#     if len(clusters) < 2:
#         return []

#     gaps = []
#     for i in range(len(clusters) - 1):
#         gap_start = max(b["x2"] for b in clusters[i])
#         gap_end   = min(b["x1"] for b in clusters[i + 1])
#         if gap_start < gap_end:
#             gaps.append({"start": gap_start, "end": gap_end})

#     return gaps


# # ══════════════════════════════════════════════
# #  STEP 3 — Column Boundaries
# # ══════════════════════════════════════════════

# def get_column_boundaries(boxes: list, gaps: list, img_width: int, img_height: int) -> list:
#     """
#     Gaps se column boundaries nikalo.
#     Har column ki x1, x2 define karo.
#     y1=0, y2=img_height (full height crop).

#     Returns: [{col_name, x1, y1, x2, y2}, ...]
#     """
#     if not gaps:
#         # Single column — poori image
#         return [{
#             "col_name": "C1",
#             "x1": 0, "y1": 0,
#             "x2": img_width, "y2": img_height
#         }]

#     boundaries = []

#     # C1: image start → first gap start
#     boundaries.append({
#         "col_name": "C1",
#         "x1": 0,
#         "y1": 0,
#         "x2": int(gaps[0]["start"]),
#         "y2": img_height
#     })

#     # C2, C3... : gap end → next gap start
#     for i in range(len(gaps) - 1):
#         boundaries.append({
#             "col_name": f"C{i+2}",
#             "x1": int(gaps[i]["end"]),
#             "y1": 0,
#             "x2": int(gaps[i+1]["start"]),
#             "y2": img_height
#         })

#     # Last column: last gap end → image end
#     boundaries.append({
#         "col_name": f"C{len(gaps)+1}",
#         "x1": int(gaps[-1]["end"]),
#         "y1": 0,
#         "x2": img_width,
#         "y2": img_height
#     })

#     return boundaries


# # ══════════════════════════════════════════════
# #  STEP 4 — Crop + Downscale
# # ══════════════════════════════════════════════

# def crop_and_resize(img: np.ndarray, col: dict) -> np.ndarray:
#     """
#     Column crop karo aur agar longest side > MAX_SIZE toh downscale.
#     LANCZOS4 use karo — text sharpest rehta hai.
#     """
#     x1 = max(0, col["x1"])
#     y1 = max(0, col["y1"])
#     x2 = min(img.shape[1], col["x2"])
#     y2 = min(img.shape[0], col["y2"])

#     crop = img[y1:y2, x1:x2]

#     h, w = crop.shape[:2]
#     longest = max(h, w)

#     if longest > MAX_SIZE:
#         scale = MAX_SIZE / longest
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         crop = cv2.resize(crop, (new_w, new_h),
#                           interpolation=cv2.INTER_LANCZOS4)
#         print(f"        → Resized: {w}x{h} → {new_w}x{new_h}")
#     else:
#         print(f"        → Size OK: {w}x{h} (no resize needed)")

#     return crop


# # ══════════════════════════════════════════════
# #  MAIN
# # ══════════════════════════════════════════════

# def process_image(image_path: str) -> list:
#     """
#     Ek image ke saare column crops nikalo.
#     Returns: [{col_name, crop_path, width, height}, ...]
#     """
#     print(f"\n{'='*55}")
#     print(f"  {Path(image_path).name}")
#     print(f"{'='*55}")

#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"  ❌ Could not load image")
#         return []

#     H, W = img.shape[:2]
#     print(f"  📐 Original size: {W}x{H}px")

#     # Step 1 — YOLO
#     print("  [1/3] Running DocLayout-YOLO...")
#     boxes = run_yolo(image_path)
#     print(f"        → {len(boxes)} boxes detected")
#     if not boxes:
#         print("  ⚠️  No boxes — skipping")
#         return []

#     # Step 2 — Gaps
#     print("  [2/3] Finding column gaps...")
#     gaps = find_all_gaps(boxes)
#     if gaps:
#         for i, g in enumerate(gaps):
#             print(f"        → GAP{i+1}: x={g['start']:.0f} → {g['end']:.0f}px")
#         print(f"        → {len(gaps)+1} columns detected")
#     else:
#         print("        → No gaps (single column)")

#     # Step 3 — Crop + Save
#     print("  [3/3] Cropping columns...")
#     columns = get_column_boundaries(boxes, gaps, W, H)

#     os.makedirs(CROPS_FOLDER, exist_ok=True)
#     stem = Path(image_path).stem

#     saved_crops = []
#     for col in columns:
#         crop = crop_and_resize(img, col)
#         ch, cw = crop.shape[:2]

#         crop_filename = f"{stem}_{col['col_name']}.jpg"
#         crop_path     = os.path.join(CROPS_FOLDER, crop_filename)

#         cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
#         print(f"        → Saved {col['col_name']}: {cw}x{ch}px  →  {crop_filename}")

#         saved_crops.append({
#             "col_name":  col["col_name"],
#             "crop_path": crop_path,
#             "width":     cw,
#             "height":    ch
#         })

#     return saved_crops


# def process_image_and_ocr(image_path: str) -> dict:
#     """
#     Ek image crop karo — phir turant OCR karo.
#     Returns: page OCR result dict
#     """
#     crops = process_image(image_path)
#     if not crops:
#         return {}

#     crop_paths = [c["crop_path"] for c in crops]
#     page_name  = Path(image_path).stem

#     print(f"  🤖 Starting OCR for {len(crop_paths)} column(s)...")
#     process_page = _get_ocr()
#     return process_page(page_name, crop_paths)


# def main():
#     print("\n" + "="*55)
#     print("  COLUMN CROPPER")
#     print("="*55)
#     print(f"  IMAGE_FOLDER : {IMAGE_FOLDER}")
#     print(f"  CROPS_FOLDER : {CROPS_FOLDER}")
#     print(f"  MAX_SIZE     : {MAX_SIZE}px (longest side)\n")

#     supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
#     folder    = Path(IMAGE_FOLDER)

#     if not folder.exists():
#         print(f"❌ Folder not found: {IMAGE_FOLDER}")
#         sys.exit(1)

#     images = sorted(set(
#         p for ext in supported
#         for p in list(folder.glob(f"*{ext}")) + list(folder.glob(f"*{ext.upper()}"))
#     ))

#     if not images:
#         print(f"❌ No images found in {IMAGE_FOLDER}")
#         sys.exit(1)

#     print(f"📁 Found {len(images)} image(s)\n")

#     from ollama_column_ocr import check_ollama, save_results
#     import time

#     # Ollama check pehle
#     if not check_ollama():
#         sys.exit(1)

#     all_pages   = []
#     total_start = time.time()

#     for img_path in images:
#         page_result = process_image_and_ocr(str(img_path))
#         if page_result:
#             all_pages.append(page_result)

#     # Save all results
#     if all_pages:
#         text_folder, json_path = save_results(all_pages)
#         total_time  = round(time.time() - total_start, 2)
#         total_words = sum(p.get("total_words", 0) for p in all_pages)

#         print(f"\n{'='*55}")
#         print(f"  ✅ ALL DONE!")
#         print(f"  Pages     : {len(all_pages)}")
#         print(f"  Words     : {total_words}")
#         print(f"  Time      : {total_time}s")
#         print(f"  Texts     : {text_folder}")
#         print(f"  Report    : {json_path}")
#         print(f"{'='*55}\n")


# if __name__ == "__main__":
#     main()



## Column Croppings without ollama 



"""
Column Cropper (Crops Only)
===========================
Steps:
1. DocLayout-YOLO  → bboxes
2. Gap Detection   → columns
3. Crop + save to output folder
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("❌ python-dotenv not found. Run: pip install python-dotenv")
    sys.exit(1)

# ─── CONFIG ───────────────────────────────────
import model_manager
model_manager.setup_portable_paths()

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER  = os.getenv("IMAGE_FOLDER",  os.path.join(BASE_DIR, "failed_cases"))
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", os.path.join(BASE_DIR, "ocr_results"))
CROPS_FOLDER  = os.path.join(OUTPUT_FOLDER, "column_crops")
YOLO_WEIGHTS  = os.getenv("YOLO_WEIGHTS",  model_manager.MODEL_SOURCES["yolo"]["target"])
MAX_SIZE      = int(os.getenv("MAX_SIZE",  "2048"))


def run_yolo(image_path: str) -> list:
    try:
        from doclayout_yolo import YOLOv10
    except ImportError:
        print("❌ doclayout-yolo not found.")
        sys.exit(1)

    model = YOLOv10(YOLO_WEIGHTS)
    model.to("cpu")
    results = model.predict(
        source=image_path, imgsz=1024,
        conf=0.25, iou=0.45, verbose=False, device="cpu"
    )
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def find_all_gaps(boxes: list) -> list:
    if len(boxes) < 2:
        return []

    sorted_boxes    = sorted(boxes, key=lambda b: (b["x1"] + b["x2"]) / 2)
    clusters        = []
    current_cluster = [sorted_boxes[0]]
    current_max_x2  = sorted_boxes[0]["x2"]

    for b in sorted_boxes[1:]:
        if b["x1"] <= current_max_x2:
            current_cluster.append(b)
            current_max_x2 = max(current_max_x2, b["x2"])
        else:
            clusters.append(current_cluster)
            current_cluster = [b]
            current_max_x2  = b["x2"]
    clusters.append(current_cluster)

    if len(clusters) < 2:
        return []

    gaps = []
    for i in range(len(clusters) - 1):
        gap_start = max(b["x2"] for b in clusters[i])
        gap_end   = min(b["x1"] for b in clusters[i + 1])
        if gap_start < gap_end:
            gaps.append({"start": gap_start, "end": gap_end})
    return gaps


def get_column_boundaries(boxes, gaps, img_width, img_height):
    if not gaps:
        return [{"col_name": "C1", "x1": 0, "y1": 0, "x2": img_width, "y2": img_height}]

    boundaries = [{"col_name": "C1", "x1": 0, "y1": 0,
                   "x2": int(gaps[0]["start"]), "y2": img_height}]

    for i in range(len(gaps) - 1):
        boundaries.append({
            "col_name": f"C{i+2}",
            "x1": int(gaps[i]["end"]), "y1": 0,
            "x2": int(gaps[i+1]["start"]), "y2": img_height
        })

    boundaries.append({
        "col_name": f"C{len(gaps)+1}",
        "x1": int(gaps[-1]["end"]), "y1": 0,
        "x2": img_width, "y2": img_height
    })
    return boundaries


def crop_and_resize(img, col):
    x1 = max(0, col["x1"]);  y1 = max(0, col["y1"])
    x2 = min(img.shape[1], col["x2"]);  y2 = min(img.shape[0], col["y2"])
    crop = img[y1:y2, x1:x2]

    h, w = crop.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        crop  = cv2.resize(crop, (int(w*scale), int(h*scale)),
                           interpolation=cv2.INTER_LANCZOS4)
    return crop


def process_image(image_path: str) -> list:
    print(f"\n{'='*50}\n  {Path(image_path).name}\n{'='*50}")

    img = cv2.imread(image_path)
    if img is None:
        print("  ❌ Could not load image"); return []

    H, W = img.shape[:2]
    print(f"  📐 Size: {W}x{H}px")

    print("  [1/3] YOLO detection...")
    boxes = run_yolo(image_path)
    print(f"        → {len(boxes)} boxes")
    if not boxes:
        print("  ⚠️  No boxes — skipping"); return []

    print("  [2/3] Gap detection...")
    gaps = find_all_gaps(boxes)
    print(f"        → {len(gaps)+1} column(s)" if gaps else "        → Single column")

    print("  [3/3] Cropping...")
    columns = get_column_boundaries(boxes, gaps, W, H)
    os.makedirs(CROPS_FOLDER, exist_ok=True)
    stem = Path(image_path).stem

    saved = []
    for col in columns:
        crop     = crop_and_resize(img, col)
        ch, cw   = crop.shape[:2]
        filename = f"{stem}_{col['col_name']}.jpg"
        out_path = os.path.join(CROPS_FOLDER, filename)
        cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"        → {col['col_name']}: {cw}x{ch}px  → {filename}")
        saved.append({"col_name": col["col_name"], "crop_path": out_path,
                      "width": cw, "height": ch})
    return saved


def main():
    print(f"\n{'='*50}\n  COLUMN CROPPER\n{'='*50}")
    print(f"  INPUT  : {IMAGE_FOLDER}")
    print(f"  OUTPUT : {CROPS_FOLDER}\n")

    folder = Path(IMAGE_FOLDER)
    if not folder.exists():
        print(f"❌ Folder not found: {IMAGE_FOLDER}"); sys.exit(1)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = sorted(set(
        p for ext in supported
        for p in list(folder.glob(f"*{ext}")) + list(folder.glob(f"*{ext.upper()}"))
    ))

    if not images:
        print(f"❌ No images in {IMAGE_FOLDER}"); sys.exit(1)

    print(f"📁 Found {len(images)} image(s)\n")

    total_crops = 0
    for img_path in images:
        crops = process_image(str(img_path))
        total_crops += len(crops)

    print(f"\n{'='*50}")
    print(f"  ✅ Done! {total_crops} crops saved → {CROPS_FOLDER}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()