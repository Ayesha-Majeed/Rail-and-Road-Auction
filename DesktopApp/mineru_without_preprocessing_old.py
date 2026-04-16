"""
Book Text Extractor with MinerU (mineru v2.7+)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Images only (JPEG/PNG) — no PDF needed.

Features:
  ✅ Image input (JPEG/PNG)
  ✅ Single-page processing (page splitting removed)
  ✅ NO redundant preprocessing — MinerU handles it internally
  ✅ GPU accelerated (auto-detected)
"""

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import cv2
import numpy as np
from loguru import logger

# ── Force CUDA device 0 BEFORE any other imports ──
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Global Torch Monkey-Patch to avoid "Invalid device id" crash ──
try:
    import torch
    
    _orig_get_props = torch.cuda.get_device_properties
    
    class MockDeviceProps:
        def __init__(self):
            self.name = "GeForce RTX 8GB (Mocked)"
            self.major = 8
            self.minor = 9
            self.total_memory = 8 * (1024 ** 3)
            self.multi_processor_count = 12
            
    def patched_get_device_properties(device):
        try:
            return _orig_get_props(device)
        except Exception:
            # Universal fallback for any MinerU or Torch submodule failing device index checks
            return MockDeviceProps()
            
    torch.cuda.get_device_properties = patched_get_device_properties
    
    # Also patch MinerU's internal VRAM utility just in case
    import mineru.utils.model_utils as mineru_utils
    mineru_utils.get_vram = lambda dev: 8 
except Exception: pass

def ensure_mineru_config():
    config_path = os.path.expanduser("~/mineru.json")
    config = {
        "latex-delimiter-config": {"inline": {"left": "$", "right": "$"}, "display": {"left": "$$", "right": "$$"}}, 
        "llm-aided-config": {"enable": False},
        "layout_reader": {"enable": False} # Disable crashing module (User Option 3)
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                existing = json.load(f)
            existing["layout_reader"] = {"enable": False}
            config = existing
        except: pass

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("✅ ~/mineru.json enforced (layout_reader disabled)")

def load_image(image_path: str):
    """Just read the image — no preprocessing needed, MinerU does it."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return img

def detect_layout(img):
    """Simplified layout detection: treats every image as a single PAGE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    logger.info("  � Layout: SINGLE PAGE (page splitting disabled)")
    return [("PAGE", (0, 0, w, h))]

def save_split_preview(img, regions, image_path: str, output_dir: str = None):
    """Draws a box around the PAGE area being processed. Saves to output_dir to avoid polluting input."""
    vis = img.copy()
    for label, (x1, y1, x2, y2) in regions:
        cv2.rectangle(vis, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (0, 200, 0), 6)
    basename = os.path.splitext(os.path.basename(image_path))[0] + "_layout.png"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, basename)
    else:
        out_path = os.path.splitext(image_path)[0] + "_layout.png"
    cv2.imwrite(out_path, vis)
    logger.info(f"  🖼️  Layout preview → {out_path}")

def save_region_crop(img_color, region: tuple, tmp_dir: str) -> str:
    x1, y1, x2, y2 = region
    crop = img_color[y1:y2, x1:x2]
    tmp_png = os.path.join(tmp_dir, "region_crop.png")
    cv2.imwrite(tmp_png, crop)
    return tmp_png

def load_mineru_models(backend: str = "pipeline"):
    actual_backend = "vlm-auto-engine" if backend == "vlm" else backend
    method = "auto" if backend == "vlm" else "ocr"
    logger.info(f"🔧 Warming up MinerU models ({actual_backend})...")
    try:
        from mineru.cli.common import do_parse, images_bytes_to_pdf_bytes
        ensure_mineru_config()
        pdf_bytes = images_bytes_to_pdf_bytes(cv2.imencode(".png", np.full((100, 100, 3), 255, dtype=np.uint8))[1].tobytes())
        with tempfile.TemporaryDirectory() as tmp:
            do_parse(output_dir=tmp, pdf_file_names=["warmup"], pdf_bytes_list=[pdf_bytes], p_lang_list=["en"], parse_method=method, backend=actual_backend)
        logger.info("✅ Models warmed up!")
    except Exception as e: logger.warning(f"⚠️ Warmup skipped: {e}")

def run_mineru(image_path: str, output_dir: str, backend: str = "pipeline") -> dict:
    from mineru.cli.common import do_parse, images_bytes_to_pdf_bytes
    actual_backend = "vlm-auto-engine" if backend == "vlm" else backend
    method = "auto" if backend == "vlm" else "ocr"
    logger.info(f"  🚀 Running MinerU ({actual_backend}/{method})...")
    os.makedirs(output_dir, exist_ok=True)
    with open(image_path, "rb") as f: pdf_bytes = images_bytes_to_pdf_bytes(f.read())
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    local_dir = os.path.join(output_dir, base_name)
    import torch
    if torch.cuda.is_available(): 
        torch.cuda.synchronize() # Wait for previous ops (User Option 2)
        torch.cuda.empty_cache()
    do_parse(output_dir=local_dir, pdf_file_names=[base_name], pdf_bytes_list=[pdf_bytes], p_lang_list=["en"], parse_method=method, backend=actual_backend)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    md_files = glob.glob(os.path.join(output_dir, "**", "*.md"), recursive=True)
    markdown_text = ""
    if md_files:
        best_md = max(md_files, key=os.path.getsize)
        with open(best_md, "r", encoding="utf-8") as f: markdown_text = f.read()
    return {"markdown": markdown_text}

def markdown_to_plain(md: str) -> str:
    text = md
    text = re.sub(r"!\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^\|.*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def extract_text(image_path, save_preview=True, backend="pipeline", model_source="huggingface", output_dir=None):
    if not os.path.exists(image_path): return "❌ File not found"
    logger.info(f"\n🔄 Processing: {os.path.basename(image_path)}")
    if model_source == "modelscope": os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
    else: os.environ.pop("MINERU_MODEL_SOURCE", None)
    
    ensure_mineru_config()
    try:
        # Load raw image (no preprocessing — MinerU handles it internally)
        img_color = load_image(image_path)
        if img_color is None: return "❌ Could not read image"

        # Detect regions (always single PAGE now)
        regions = detect_layout(img_color)

        # Output dir setup
        if output_dir is None: output_dir = os.path.splitext(image_path)[0] + "_mineru_output"
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save layout preview to OUTPUT dir (not input dir)
        if save_preview:
            save_split_preview(img_color, regions, image_path, output_dir)

        final_output = ""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for label, region in regions:
                logger.info(f"\n  🔍 Region: {label}")

                x1, y1, x2, y2 = region
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    logger.warning("     ⚠️  Too small, skipping")
                    continue

                # Crop raw image and pass directly to MinerU
                tmp_img    = save_region_crop(img_color, region, tmp_dir)
                region_out = os.path.join(output_dir, label)
                result     = run_mineru(tmp_img, region_out, backend)
                md_text    = result["markdown"]

                if not md_text.strip():
                    logger.warning(f"     ⚠️  No text in {label}")
                    continue

                plain_text = markdown_to_plain(md_text)

                final_output += plain_text + "\n"
                logger.info(f"  ✅ {label}: {len(plain_text)} chars")
                with open(os.path.join(output_dir, f"{label}.md"), "w", encoding="utf-8") as f: f.write(md_text)

        return final_output.strip() if final_output else "No text extracted."
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="pipeline", choices=["pipeline", "vlm"])
    parser.add_argument("--source", type=str, default="huggingface")
    args = parser.parse_args()
    images = [args.path] if (args.path and os.path.isfile(args.path)) else sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.jpg")))
    if not images: return logger.error("❌ No images found!")
    try:
        load_mineru_models(backend=args.backend)
        for img_path in images:
            out = extract_text(img_path, backend=args.backend, model_source=args.source)
            print(f"\n📄 RESULTS: {os.path.basename(img_path)}\n" + "="*50 + "\n" + out)
    finally:
        try:
            import torch
            if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
        except: pass
        # Force kill vllm hanging processes
        try:
            import signal, psutil
            current = psutil.Process()
            for child in current.children(recursive=True):
                try: child.send_signal(signal.SIGTERM)
                except: pass
        except: pass
        os._exit(0)

if __name__ == "__main__":
    main()