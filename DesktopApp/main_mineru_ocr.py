"""
MinerU OCR — Per-Book Automation Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Groups images by Book ID, processes each book separately.

Filename convention:
  642_01.jpg  → Book 642, front_cover
  642_02.jpg  → Book 642, back_cover
  642_03.jpg  → Book 642, interior
  1189_01.jpg → Book 1189, front_cover
  ...

Per-book output:
  mineru_results/
  ├── 642/
  │   ├── ocr_output.txt
  │   ├── ocr_output.json
  │   ├── color_palette.png
  │   └── book_metadata.json   ← title + colors + description
  ├── 1189/
  │   └── ...

Usage:
  python main_mineru_ocr.py                      # full pipeline
  python main_mineru_ocr.py --source modelscope  # if HuggingFace blocked
  python main_mineru_ocr.py --no-preview         # skip layout previews
  python main_mineru_ocr.py --no-ai              # skip title/color/description
"""

import os
import sys
import shutil
import subprocess
import requests
import re
import json
import time
import easyocr
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from PIL import Image, ImageDraw
import ollama
import cv2
import numpy as np
from dotenv import load_dotenv

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

# ─── Portable Model Management ─────────────────────
import model_manager
model_manager.setup_portable_paths()
PORTABLE_BASE = model_manager.BASE_DIR

load_dotenv(os.path.join(PORTABLE_BASE, ".env"))

sys.path.insert(0, PORTABLE_BASE)
import mineru_without_preprocessing_old as mineru
import column_cropper as cropper

# ════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════
RAW_IMAGE_FOLDER = os.getenv("IMAGE_FOLDER",  os.path.join(PORTABLE_BASE, "test_dataset"))
CROPS_FOLDER     = os.getenv("CROPS_FOLDER",  os.path.join(PORTABLE_BASE, "doclayout_column_cropings/column_crops"))
OUTPUT_FOLDER    = os.getenv("OUTPUT_FOLDER", os.path.join(PORTABLE_BASE, "mineru_results"))
VISION_MODEL     = os.getenv("VISION_MODEL",  "minicpm-v")
TEXT_MODEL       = os.getenv("TEXT_MODEL",    "llama3.2:1b") # 1B for absolute 8GB stability
SEPARATOR        = "─" * 60

# Set PyTorch memory allocation configuration for flexibility
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER,  exist_ok=True)


# ─── Lazy EasyOCR Engine ───────────────────────────────────────────
_easy_reader = None

def get_easyocr_reader():
    """Lazy singleton: loads EasyOCR with GPU support."""
    global _easy_reader
    if _easy_reader is None:
        log_progress("Initializing EasyOCR Reader...")
        import torch
        gpu_bool = torch.cuda.is_available()
        _easy_reader = easyocr.Reader(['en'], gpu=gpu_bool)
    return _easy_reader

def unload_easyocr():
    """Unloads EasyOCR to free VRAM for Ollama Vision."""
    global _easy_reader
    if _easy_reader is not None:
        log_progress("Unloading EasyOCR to free VRAM...")
        del _easy_reader
        _easy_reader = None
        cleanup_gpu()

# ════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════

# ─── Global Log Redirector ───
LOG_CALLBACK = None

def log(msg: str = ""):
    if LOG_CALLBACK:
        try: LOG_CALLBACK(msg)
        except: pass
    print(msg, flush=True)

def log_step(title: str):
    log()
    log("═" * 60)
    log(f"  {title}")
    log("═" * 60)

def log_part(title: str):
    log()
    log(f"  ── {title} {'─' * max(1, 54 - len(title))}")

def log_progress(msg: str):
    # For UI, we don't want the trailing '...', we want a clean message
    if LOG_CALLBACK:
        try: LOG_CALLBACK(f"  ⏳ {msg}")
        except: pass
    print(f"  ⏳ {msg} ...", end=" ", flush=True)

def log_done(extra: str = ""):
    if LOG_CALLBACK:
        try: LOG_CALLBACK(f"    ✓ {extra}")
        except: pass
    print(f"✓  {extra}", flush=True)

def log_fail(reason: str = ""):
    if LOG_CALLBACK:
        try: LOG_CALLBACK(f"    ✗ {reason}")
        except: pass
    print(f"✗  {reason}", flush=True)

def log_info(key: str, value: str):
    msg = f"  {key:<20}: {value}"
    if LOG_CALLBACK:
        try: LOG_CALLBACK(msg)
        except: pass
    print(msg, flush=True)

def log_warn(msg: str):
    log(f"  ⚠️  {msg}")

def log_error(msg: str):
    log(f"  ❌ {msg}")

def log_book_banner(book_id: str, current: int, total: int):
    log_step(f"({current}/{total}) Processing Book: {book_id}")

# ════════════════════════════════════════════════════
# PAGE TYPE + BOOK ID DETECTION
# ════════════════════════════════════════════════════

def detect_type(page_id: str, total_pages: int = 0) -> str:
    """
    User Rules:
    - 1 page book   : _01 is front_cover (VLM)
    - 2 page book   : _01 is front_cover (VLM), _02 is interior (Pipeline)
    - 3+ page book  : _01 is front_cover (VLM), _02 is back_cover (VLM), others are interior (Pipeline)
    
    Robust check: splits on underscore and checks numeric value.
    """
    parts = page_id.rsplit("_", 1)
    if len(parts) < 2 or not parts[1].isdigit():
        return "interior"
    
    val = int(parts[1])
    
    if val == 1:
        return "front_cover"
    
    if val == 2:
        if total_pages == 2:
            return "interior"
        return "back_cover"
    
    return "interior"

def get_book_id(filename: str) -> str:
    """
    '642_01.jpg' → '642'
    '1189_03.jpg' → '1189'
    Splits on LAST underscore-number group.
    """
    stem  = Path(filename).stem          # e.g. 642_01
    parts = stem.rsplit("_", 1)          # ['642', '01']
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]                  # '642'
    return stem

def group_images_by_book(raw_folder: str) -> dict:
    folder    = Path(raw_folder)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if not folder.exists():
        log_error(f"IMAGE_FOLDER not found: {raw_folder}")
        sys.exit(1)

    books = defaultdict(list)
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in supported:
            continue
        if "_layout" in f.stem:
            continue
        book_id = get_book_id(f.name)
        books[book_id].append(str(f))

    for book_id in books:
        books[book_id] = sorted(books[book_id])

    return dict(books)


# ════════════════════════════════════════════════════
# STEP 1 — CROP ONE BOOK
# ════════════════════════════════════════════════════

def crop_book(book_images: list, book_crops_folder: str) -> dict:
    cropper.CROPS_FOLDER = book_crops_folder
    os.makedirs(book_crops_folder, exist_ok=True)

    pages = defaultdict(list)
    total_pages = len(book_images)
    
    for img_path in book_images:
        stem     = Path(img_path).stem
        page_id  = stem
        img_type = detect_type(page_id, total_pages)

        log(f"  [{img_type}]  {Path(img_path).name}")
        
        # USER REQUEST: Skip YOLO for covers (just use full image)
        if img_type in ("front_cover", "back_cover"):
            log_done("skipping YOLO for cover — using full image")
            dest_path = os.path.join(book_crops_folder, f"{page_id}_C1.jpg") # C1 is standard
            shutil.copy2(img_path, dest_path)
            pages[page_id].append(("C1", dest_path))
            continue

        log_progress("    YOLO cropping")
        try:
            crops = cropper.process_image(img_path)
            if crops:
                log_done(f"{len(crops)} crop(s)")
                for c in crops:
                    pages[page_id].append((c["col_name"], c["crop_path"]))
                    log_info(f"    {c['col_name']}", f"{c['width']}x{c['height']}px → {Path(c['crop_path']).name}")
            else:
                # FALLBACK: If YOLO finds nothing, don't fail! Use the full image as C1
                log(f"  ⚠️  No columns detected by YOLO. Falling back to full image.")
                dest_path = os.path.join(book_crops_folder, f"{page_id}_C1.jpg")
                shutil.copy2(img_path, dest_path)
                pages[page_id].append(("C1", dest_path))
                log_done("full image used as fallback")
        except Exception as e:
            log_fail(f"YOLO engine error: {e}. Falling back to full image.")
            dest_path = os.path.join(book_crops_folder, f"{page_id}_C1.jpg")
            shutil.copy2(img_path, dest_path)
            pages[page_id].append(("C1", dest_path))

    for page_id in pages:
        pages[page_id] = sorted(pages[page_id], key=lambda x: x[0])

    return dict(pages)


# ════════════════════════════════════════════════════
# STEP 2 — OCR ONE BOOK
# ════════════════════════════════════════════════════

def ocr_book(pages: dict, book_output_folder: str, args, total_pages: int = 0) -> dict:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    front_cover_img      = ""
    back_cover_img       = ""
    interior_texts       = []
    all_results          = []
    total_start          = time.time()

    total_pages = total_pages or len(pages)
    for i, (page_id, col_files) in enumerate(sorted(pages.items()), 1):
        # Trigger cleanup at start of each page
        cleanup_gpu()
        
        img_type       = detect_type(page_id, total_pages)
        # USER REQUEST: Remove VLM backend as it's too heavy. Use pipeline for all.
        backend_to_use = "pipeline"

        log()
        log(f"  [{i}/{len(pages)}]  {page_id}  [{img_type}]  "
            f"{len(col_files)} col(s)  [backend: {backend_to_use}]")

        t_start   = time.time()
        col_texts = []

        for col_name, crop_path in col_files:
            log_progress(f"  {col_name} → EasyOCR")
            
            try:
                # USER REQUEST: Use EasyOCR instead of MinerU for interior text
                reader = get_easyocr_reader()
                result = reader.readtext(crop_path, detail=0)
                text = "\n".join(result)

                if img_type in ("front_cover", "back_cover") and (not text or text.strip() == ""):
                    log_progress("    Using Ollama Vision OCR for cover text")
                    text = extract_title_from_cover_image(
                        crop_path, 
                        custom_prompt="OCR task: Read every word in this image exactly as printed. Output ONLY the raw text."
                    )
                
                # Cleanup after each column
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                if not args.no_ai:
                    if img_type == "front_cover" and (col_name == "C1" or not front_cover_img):
                        front_cover_img = crop_path
                        log_info("    ↳ Front cover captured", Path(crop_path).name)

                    elif img_type == "back_cover" and (col_name == "C1" or not back_cover_img):
                        back_cover_img = crop_path
                        log_info("    ↳ Back cover captured", Path(crop_path).name)

                if text and not text.startswith("❌"):
                    clean = re.sub(r'^={50,}\n.*?PAGE.*?\n={50,}\n', '', text, flags=re.MULTILINE)
                    clean = re.sub(r'^={50,}\n.*?\n={50,}\n',        '', clean, flags=re.MULTILINE)
                    clean = clean.strip()
                    col_texts.append(clean)
                    log_done(f"{len(clean.split())} words")

                    if clean:
                        log()
                        log(f"    📜 EXTRACTED TEXT ({img_type} — {col_name}):")
                        log("    " + "─" * 40)
                        for line in clean.splitlines():
                            if line.strip():
                                log(f"      {line.strip()}")
                        log("    " + "─" * 40)
                        log()

                else:
                    log_fail(text if text and text.startswith("❌") else "no text")
                    col_texts.append("")

            except Exception as e:
                log_fail(str(e))
                col_texts.append("")

        combined = "\n".join(t for t in col_texts if t)
        elapsed  = round(time.time() - t_start, 2)
        status   = "success" if combined else "error"

        log(f"  └─ {len(combined.split())} words  [{elapsed}s]  [{status}]")

        if total_pages == 1 and img_type == "front_cover" and combined:
            interior_texts.append(combined)
        elif img_type == "interior" and combined:
            interior_texts.append(combined)

        all_results.append({
            "image":          f"{page_id}.jpg",
            "type":           img_type,
            "status":         status,
            "extracted_text": combined,
            "words":          len(combined.split()),
            "time_seconds":   elapsed,
            "backend_used":   backend_to_use
        })

    total_time = round(time.time() - total_start, 2)
    save_ocr_outputs(all_results, book_output_folder)
    
    log_info("Pages processed", f"{len(all_results)}")
    log_info("OCR time",        f"{total_time}s")

    return {
        "results":               all_results,
        "front_cover_img":       front_cover_img,
        "back_cover_img":        back_cover_img,
        "interior_texts":        interior_texts
    }


def save_ocr_outputs(all_results: list, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = os.path.join(output_folder, "ocr_output.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(SEPARATOR + "\n")
            f.write(f"IMAGE : {r['image']}\n")
            f.write(f"TYPE  : {r['type']}\n")
            f.write(f"TEXT  :\n")
            clean = r["extracted_text"]
            clean = re.sub(r'^={50,}\n.*?PAGE.*?PAGE.*?\n={50,}\n', '', clean, flags=re.MULTILINE)
            clean = re.sub(r'^={50,}\n.*?\n={50,}\n',               '', clean, flags=re.MULTILINE)
            f.write(clean.strip())
            f.write("\n\n")

    log_info("  TXT saved", txt_path)


# ════════════════════════════════════════════════════
# AI EXTRACTION HELPER
# ════════════════════════════════════════════════════

def extract_title_from_cover_image(image_path: str, custom_prompt: str = "") -> str:
    log_progress(f"Sending cover to {VISION_MODEL}")
    
    prompt = custom_prompt or """Look at this book cover carefully. What is the PRIMARY TITLE of this book?
Rules:
1. Extract only the MAIN TITLE and its SUBTITLE.
2. Ignore decorative text, secondary headings, lists of cities, or marketing slogans.
3. Be concise and professional.
4. Return ONLY the title text. No explanation. No "The title is...". Just the title itself."""

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}]
        )
        title = response["message"]["content"].strip()
        # Clean up AI phrases
        title = re.sub(r'^(The title is|Title:|Book title:|The book title is|This is)[:\-]?\s*', '', title, flags=re.IGNORECASE).strip()
        # Consolidate multiline titles into one line
        title = " ".join([line.strip() for line in title.split('\n') if line.strip()])
        title = title.strip('"\'').strip()
        
        log_done(f'"{title}"')
        return title
    except Exception as e:
        log_fail(str(e))
        return ""


def extract_author_from_cover(front: str, back: str = "") -> str:
    """
    Stage 1+2: Try front cover, then back cover via Vision LLM.
    Returns author string or "" if not found.
    """
    images = [p for p in [front, back] if p and os.path.exists(p)]
    if not images:
        return ""

    prompt = """Look at this book cover image carefully.
Your task: Find the PRIMARY AUTHOR NAME — the person who wrote the book.

CRITICAL PRIORITY:
- Look for a name preceded by "by", "Written by", or "Author:". These are HIGHEST priority.
- If there are multiple names (e.g. at top and bottom), the one with "By" IS THE AUTHOR.
- Example: "PHILIP R. HASTINGS" (top) and "By Douglas M. Nelson" (bottom) -> Return "Douglas M. Nelson".

WHAT TO AVOID (DO NOT RETURN THESE):
- Names preceded by "with" (e.g., "with James P. Shuman") — these are often subjects or collaborators, not the primary author.
- Names preceded by "featuring", "introduction by", "foreword by" (unless no other name exists).
- Organization/Society names.
- Book titles or subtitles.

RULES:
1. Return ONLY the human author's name (e.g., "Jeremy F. Plant").
2. ABSOLUTELY NO SENTENCES. NO PREFIXES. NO "By".
3. Capture the FULL name exactly as printed (e.g., "SAMUEL O. DUNN", not "SAMUEL DUNN").
4. If multiple authors exist, return both (e.g., "Smith and Jones").
5. Return "NOT_FOUND" if no human name is visible.
6. STRICT RULE: Reject railroad companies, transit systems, electric lines, MUSEUMS, SOCIETIES, or JOURNALS.
7. ABSOLUTELY REJECT names from advertisements (e.g., "Harborside Warehouse", "Hotel", "Restaurant").
8. NO CONVERSATION. One line only. Max 6 words. """


    labels = ["front cover", "back cover"]
    for i, img in enumerate(images):
        log_progress(f"Checking {labels[i]} for author")
        try:
            resp = ollama.chat(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [img]}]
            )
            res = resp["message"]["content"].strip().strip('"').strip()
            # Clean common AI prefixes
            res = re.sub(
                r'^(Author:|By:|Written by|Authored by|The author is|Human author is|The human author name is|Human author[:\-]?|The human author[:\-]?|The name of the human author is)[:\-]?\s*',
                '', res, flags=re.IGNORECASE
            ).strip()
            # Consolidate multiline authors into one line
            res = " ".join([line.strip() for line in res.split('\n') if line.strip()]).strip()

            validated = _sanity_check_author(res)
            if validated:
                log_done(f'"{validated}"')
                return validated
        except Exception as e:
            log_fail(str(e))

    log_fail("not found on covers")
    return ""


def extract_author_from_text(text: str, book_title: str = "") -> str:
    """
    Stage 3: Search interior OCR text via Text LLM for author name.
    """
    if not text.strip():
        return ""
    log_progress("Checking interior text for author")

    prompt = f"""Find the HUMAN AUTHOR NAME in the following book text.

- Patterns like "By John Smith", "Author: Jane Doe", "Written by Jeremy Wilson"
- Copyright notices like "© 2001 John Smith" (the person's name)
- Title-page lines listing the author's name clearly.

STRICT PRIORITY: Always prefer the name with "By", "Written by", or "Author:" over a standalone name.

STRICT EXCLUSIONS — do NOT return these:
- Names preceded by "with" (e.g., "with James P. Shuman") — these are often subjects
- Organization names, societies, associations, or institutions
- Publisher or imprint names
- The book title or subtitle
- Editor names (ABSOLUTELY NO EDITORS. "Edited by John Smith" means the Author is NOT_FOUND)
- Any name followed by Inc., Ltd., Co., Press, Society, Association, Institute, University
- Names in a "Contents" or "Table of Contents" list (these are section authors, not the book author)

RULES:
1. Do NOT return the "EDITOR". If only an editor is listed, return: NOT_FOUND
2. If the text is a Table of Contents listing many different authors for different articles, do NOT pick one. Return: NOT_FOUND
3. Return ONLY the human author/editor's name, exactly as written.
4. ABSOLUTELY NO SENTENCES. NO PREFIXES. NO "By".
5. Capture the FULL name exactly as printed (e.g., "SAMUEL O. DUNN", not "Uel O. Dunn").
6. Do NOT invent. If no name exists, return exactly: NOT_FOUND
7. STRICT RULE: Reject railroad companies, transport systems, MUSEUMS, or JOURNALS (e.g., "Pacific Electric", "Railroad History").
8. If multiple section authors exist but no overall book author/editor, return: NOT_FOUND
9. NO CONVERSATION. One line only. Max 6 words.

Book title (for context only): {book_title}
Text to search:
{text[:3000]}"""

    try:
        resp = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
        res = resp["message"]["content"].strip().strip('"').strip()

        res = re.sub(
            r'^(Author:|By:|Written by|Authored by|The author is|Human author is|The human author name is|Human author[:\-]?|The human author[:\-]?|The name of the human author is)[:\-]?\s*',
            '', res, flags=re.IGNORECASE
        ).strip()
        # Consolidate multiline authors into one line
        res = " ".join([line.strip() for line in res.split('\n') if line.strip()]).strip()

        validated = _sanity_check_author(res)
        if validated:
            log_done(f'"{validated}"')
            return validated
    except Exception as e:
        log_fail(str(e))

    log_fail("not found in text")
    return ""


def _sanity_check_author(raw: str) -> str:
    """Reject obviously wrong author responses (orgs, bad AI replies, etc.)."""
    if not raw or not raw.strip():
        return ""
    s = raw.strip().strip('"').strip("'").strip()
    up = s.upper()

    bad_exact = {
        "NOT_FOUND", "NOT FOUND", "YES", "NO", "TRUE", "FALSE", "NONE",
        "N/A", "NA", "UNKNOWN", "NOT APPLICABLE", "AUTHOR", "AUTHOR NAME",
        "NO AUTHOR", "NO AUTHOR FOUND", "AUTHOR NOT FOUND"
    }
    if up in bad_exact:
        return ""

    # Reject if too long (real person names rarely exceed 6 words)
    if len(s.split()) > 8:
        return ""

    # Reject organization-indicator keywords anywhere in the string
    org_markers = [
        " INC", " INC.", " LLC", " LTD", " LTD.", " CO.",
        "SOCIETY", "ASSOCIATION", "INSTITUTE", "NATIONAL ",
        "UNIVERSITY", " PRESS", "HISTORICAL", "FOUNDATION",
        "COMMITTEE", "DEPARTMENT", "BUREAU", "COUNCIL",
        "GEOGRAPHIC", "EDITORIAL", "STAFF WRITE",
        "RAILROAD", "RAILWAY", "ELECTRIC", "PACIFIC", "TRANSPORT",
        "SYSTEM", "LINES", "ROUTE", "COMPANY", "CORP.", "CORPORATION",
        "MUSEUM", "ARCHIVE", "LIBRARY", "SOCIETY", "ASSOCIATION", "INSTITUTE",
        "COMMISSION", "AUTHORITY", "DEPT", "DEPARTMENT", "COLLECTION",
        "PUBLICATION", "JOURNAL", "BULLETIN", "MAGAZINE", "NEWSLETTER", "REVIEW",
        "HISTORICAL", "VOLUME", "NUMBER", "EDITOR", "EDITED BY",
        "WAREHOUSE", "SERVICES", "STATION", "RAILYARD", "HOTEL", "RESTAURANT"
    ]
    for marker in org_markers:
        if marker in up:
            return "NOT_FOUND"

    # Reject if it starts with conversational patterns
    conversational = [
        "the book", "this book", "i can", "i found", "based on",
        "there is", "it appears", "it seems", "the title", "the cover",
        "the author", "no author", "the human", "the name", "written by",
        "this is", "here is", "i believe", "is the author", "author: ",
        "by: "
    ]
    for c in conversational:
        if up.startswith(c.upper()):
            return "NOT_FOUND"

    return s




def generate_description(interior_text: str, title: str = "") -> str:
    if not interior_text.strip():
        return ""
    log_progress(f"Generating description via {TEXT_MODEL}")
    
    prompt = f"""Write a strictly 1-2 line direct description of this book based on the provided text.
Title: {title}
Text: {interior_text[:4000]}
Rules: 
1. Provide ONLY the description text (maximum 2 sentences). 
2. ABSOLUTELY NO introductory phrases (e.g., "This book is...", "Here is a description...", "The text describes...").
3. Start IMMEDIATELY with the subject matter.
4. DO NOT include headers like "Summary:" or "Description:".
5. Be professional and concise. One paragraph only."""
    
    try:
        response = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
        desc = response["message"]["content"].strip()
        # Clean up any potential AI intro meta-talk if it still slips through
        desc = re.sub(r'^(Here is|This is|A summary of|Description:).*?[:\-]\s*', '', desc, flags=re.IGNORECASE).strip()
        log_done("done")
        return desc
    except Exception as e:
        log_fail(str(e))
        return ""

def generate_description_from_images(image_paths: list, title: str = "") -> str:
    """Fallback: Generates description by sending images directly to Vision LLM."""
    valid_paths = [p for p in image_paths if p and os.path.exists(p)]
    if not valid_paths:
        return ""

    log_progress(f"Generating description from images via {VISION_MODEL}")
    
    prompt = f"""Read the text on these book pages and write a strictly 1-2 line direct description of the book's content.
Title: {title}
Rules:
1. Use ONLY the text visible on the pages.
2. Provide ONLY the description text (maximum 2 sentences).
3. ABSOLUTELY NO introductory phrases (e.g., "This book is...", "Here is a description...").
4. Start IMMEDIATELY with the subject matter.
5. DO NOT include headers like "Summary:" or "Description:". """

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": prompt, "images": valid_paths[:3]}]
        )
        desc = response["message"]["content"].strip()
        desc = re.sub(r'^(Here is|This is|A summary of|Description:).*?[:\-]\s*', '', desc, flags=re.IGNORECASE).strip()
        log_done("done")
        return desc
    except Exception as e:
        log_fail(str(e))
        return ""

def _sanity_check_edition(raw: str) -> str:
    """Lightweight sanity check — reject obviously wrong responses without regex strictness."""
    if not raw or not raw.strip():
        return ""
    
    # Handle escaped variants like NOT\_FOUND
    s = raw.replace('\\', '').strip().strip('"').strip("'").strip()
    up = s.upper()
    
    # Reject known bad responses
    bad_exact = {"NOT_FOUND", "YES", "NO", "TRUE", "FALSE", "NONE", "EDITION", 
                 "N/A", "NA", "UNKNOWN", "NOT FOUND", "NOT APPLICABLE", "NULL"}
    if up in bad_exact:
        return ""
    
    # NEW: REJECT Common book subjects/keywords that AI often hallucinations as editions
    subject_blacklist = [
        "DIESEL", "STEAM", "TRAIN", "RAILROAD", "RAILWAY", "PACIFIC", "UNION",
        "LOCOMOTIVE", "ENGINE", "PENNSYLVANIA", "ELECTRIC", "TRANSPORT",
        "SYSTEM", "LINE", "ROAD", "STEEL", "COAL", "BLACK DIAMOND", "BLACK GOLD",
        "VOLUME", "CHAPTER", "INDEX", "ROSTER", "PHOTO", "COLLECTION"
    ]
    for subj in subject_blacklist:
        if subj in up:
            # If it has "EDITION" or "PRINTING" as well, it MIGHT be okay (e.g. "Limited Railroad Edition")
            # otherwise reject.
            if not any(k in up for k in ["EDITION", "PRINTING", "REVISED"]):
                return ""

    # NEW STRICT RULE: Must contain at least some letters. 
    # Editions like "385' .5" (hallucinated call numbers) have no letters.
    # We require at least 2 letters to be a valid edition phrase (e.g. "1st", "Rev").
    letters = re.findall(r'[A-Z]', up)
    if len(letters) < 1: 
        return ""

    # GLOBAL KEYWORD ENFORCEMENT: 
    # Generic ordinals like "FIRST" or "1ST" are forbidden unless they 
    # have context like "EDITION", "PRINTING", etc.
    edition_keywords = [
        "EDITION", "PRINTING", "REVISED", "ANNIVERSARY", 
        "IMPRESSION", "SPECIAL", "COLLECTOR", "UPDATE", "REPRINT"
    ]
    if not any(k in up for k in edition_keywords):
        return ""

    if re.search(r'^[A-Z]{1,3}\d+', up) or re.search(r"\d+[\.']\d+", up):
        return ""
    
    # Reject strings containing technical cataloging terms
    catalog_terms = ["CATALOGING", "CATALOG", "LIBRARY OF CONGRESS", "DEWEY", "LOC ", "L.C. ", "CIP "]
    for t in catalog_terms:
        if t in up:
            return ""

    # Reject seasons, months, and standalone years (these are publication dates for journals)
    seasons = {"SPRING", "SUMMER", "AUTUMN", "WINTER", "FALL"}
    months = {"JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"}
    
    # Block month ranges (e.g. JULY-DEC, JAN-JUN)
    if "-" in up or "—" in up:
        parts = re.split(r'[-\u2014]', up)
        if any(p.strip() in months for p in parts):
            return ""

    # Reject patterns like "July 1989" (these are publication dates for journals)
    # But ONLY if the string is primarily just that date.
    # If it contains "EDITION" or "PRINTING", we keep the date (User request).
    if re.search(r'(?i)(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}', up):
        if not any(k in up for k in ["EDITION", "PRINTING", "ANNIVERSARY", "REVISED"]):
            return ""

    # Reject prices (e.g. $54.95, 20.00 GBP)
    if re.search(r'\$\s*\d+[\.,]\d{2}', s) or re.search(r'\b\d+[\.,]\d{2}\s*(USD|GBP|EUR|CAD|AUD)\b', up):
        return ""
    
    # Reject ISBNs
    if "ISBN" in up:
        return ""

    if up in seasons or up in months:
        return ""
    
    # Reject strings that are primarily a year (e.g., "1987", "1941")
    if re.search(r'\b(18|19|20)\d{2}\b', s) and len(s.split()) <= 2:
        # Check if "EDITION" is in the string (e.g. "2024 Edition" is okay, but "1941" is not)
        if "EDITION" not in up:
            return ""
    
    # Reject if it contains journal numbering patterns (Volume, Issue, Year, No.)
    # unless it explicitly says "EDITION", "PRINTING", or "ANNIVERSARY"
    if re.search(r'\b(VOL\.?|VOLUME|ISSUE|NO\.?|YEAR)\b', up):
        if not any(k in up for k in ["EDITION", "PRINTING", "ANNIVERSARY", "REVISED"]):
            return ""
    
    # Reject responses that are too long (editions are short phrases, max ~6 words)
    if len(s.split()) > 8:
        return ""
    
    # Reject if it starts with conversational patterns
    conversational = ["the book", "this book", "i can", "i found", "based on", "there is", "it appears", "it seems", "the edition is", "it is listed as"]
    for c in conversational:
        if up.startswith(c.upper()):
            # Try to strip it instead of just rejecting? No, user wants strict 1-2 words.
            # But let's try to extract the relevant part if it's there.
            s = re.sub(r'^(?i)(the book was|this book was|the edition is|it is listed as|the book|this book|i found|i can|based on|there is|it appears|it seems)[:\-]?\s*', '', s)
            up = s.upper()

    # Re-check length after stripping
    if len(s.split()) > 4: # Strict limit for editions
        return ""

    return s

def find_edition_via_regex(text: str) -> str:
    """Quickly search for edition keywords in OCR text."""
    if not text: return ""
    # Broaden regex to capture the following date or dash (e.g. "6th Printing — June 1993")
    patterns = [
        r'(?i)\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:printing|edition)[^\n,.]+[\n,.]?',
        r'(?i)\b(?:1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|\d+th)\s+(?:printing|edition)[^\n,.]+[\n,.]?',
        r'(?i)\banniversary\s+edition[^\n,.]*',
        r'(?i)\bupdated\s+edition[^\n,.]*',
        r'(?i)\brevised\s+edition[^\n,.]*',
        r'(?i)\brevised\s+and\s+updated[^\n,.]*',
        r'(?i)\brevised\b[^\n,.]*',
    ]
    found = []
    for p in patterns:
        matches = re.findall(p, text)
        for m in matches:
            if isinstance(m, tuple):
                phrase = f"{m[0]} {m[1]}"
            else:
                phrase = m
            # Convert "1st" to 1 for sorting
            num = 0
            num_match = re.search(r'\d+', phrase)
            if num_match:
                num = int(num_match.group(0))
            else:
                words = {"first":1, "second":2, "third":3, "fourth":4, "fifth":5, "sixth":6, "seventh":7, "eighth":8, "ninth":9, "tenth":10}
                for w, n in words.items():
                    if w in phrase.lower():
                        num = n
                        break
            found.append((num, phrase.strip()))
    
    if found:
        # Sort by number to get the highest (most recent) printing
        found.sort(key=lambda x: x[0], reverse=True)
        return found[0][1]
    return ""


def extract_edition_from_cover(front: str, back: str = "") -> str:
    """
    Find edition/printing info via Vision LLM.
    'front' can be a path string or a list of path strings.
    """
    if isinstance(front, list):
        images = [p for p in front if p and os.path.exists(p)]
    else:
        images = [p for p in [front, back] if p and os.path.exists(p)]
    
    if not images: return ""
    
    prompt = f"""Find the edition or printing information on this book page.
RULES:
1. Return ONLY the exact edition/printing phrase (e.g. "First Edition", "3rd Printing").
2. ABSOLUTELY NO technical codes, Dewey Decimal numbers, or Library of Congress metadata (e.g. IGNORE "385'.5", "TF725").
3. ABSOLUTELY NO sentences, introductory phrases, or explanations. 
4. MUST BE CONCISE: 1-3 words maximum.
5. If there are multiple printings listed (e.g. 1st, 2nd, 3rd), return ONLY the LATEST (highest) one.
6. INCLUDE THE DATE if it is listed with the printing (e.g. "6th Printing, June 1993").
7. If no edition info is found, return ONLY: NOT_FOUND
8. IGNORE barcodes and Library of Congress classification numbers."""
    
    for img in images:
        log_progress(f"Checking {Path(img).name} for edition")
        try:
            resp = ollama.chat(model=VISION_MODEL, messages=[{"role": "user", "content": prompt, "images": [img]}])
            res = resp["message"]["content"].strip().strip('"').strip()
            
            # If AI returned multiple lines/items, pick the most relevant one
            if "," in res or "\n" in res:
                parts = [p.strip() for p in re.split(r',|\n', res)]
                keywords = ["EDITION", "PRINTING", "REVISED", "ANNIVERSARY", "UPDATED", "EXPANDED", 
                            "DELUXE", "LIMITED", "COLLECTOR", "SPECIAL", "ABRIDGED", "UNABRIDGED",
                            "1ST", "2ND", "3RD", "FIRST", "SECOND", "THIRD"]
                for p in parts:
                    if any(k in p.upper() for k in keywords):
                        res = p
                        break
            
            validated = _sanity_check_edition(res)
            if validated:
                log_done(validated)
                return validated
        except: pass
    log_fail("not found")
    return ""

def extract_edition_from_text(text: str, book_title: str = "") -> str:
    if not text.strip(): return ""
    log_progress("Checking interior for edition")
    
    prompt = f"""Find edition or printing info.
    
    SYSTEM ROLE: You are a SINCERE and STOIC extraction robot. 
    CRITICAL RULE: YOU ARE FORBIDDEN FROM GUESSING OR INVENTING YEARS.
    CRITICAL RULE: IF NO KEYWORD (Edition, Printing, Revised, Impression, Copyright) IS FOUND, RETURN ONLY: NOT_FOUND

    INSTRUCTIONS:
    1. Extract ONLY the exact phrase (e.g. "Third Edition", "First Printing"). 
    2. RETURN 'NOT_FOUND' if the information is missing, ambiguous, or only a publication date.
    3. ABSOLUTELY NO sentences, conversational text, or descriptions.
    4. NO dates except those explicitly attached to a printing number.
    5. Max 1-5 words.
    
    Book title: {book_title}
    Text:
    {text[:3000]}"""
    
    try:
        resp = ollama.chat(model=TEXT_MODEL, messages=[{"role": "user", "content": prompt}])
        res = resp["message"]["content"].strip().strip('"').strip()
        
        validated = _sanity_check_edition(res)
        if validated:
            log_done(validated)
            return validated
    except: pass
    log_fail("not found")
    return ""




def save_book_metadata(book_id, title, description, output_folder, edition="", author="", isbn=""):
    """Saves book metadata as JSON for DB sync."""
    def _clean_val(v, default="Not Found"):
        if not v: return default
        # Remove backslashes (common in AI escapes)
        clean = v.replace('\\', '').strip()
        if not clean or clean.upper() in ["NOT_FOUND", "NOT FOUND"]:
            return default
        return clean

    metadata = {
        "book_id":        book_id,
        "title":          _clean_val(title),
        "author":         _clean_val(author),
        "edition":        _clean_val(edition),
        "isbn":           _clean_val(isbn),
        "description":    _clean_val(description),
        "sync_date":      datetime.now().isoformat(),
        "status":         "pending"
    }
    with open(os.path.join(output_folder, "book_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    log(f"\n  📚 Book {book_id} Metadata:")
    log_info("  Title",   metadata.get("title", "Not Found"))
    log_info("  Author",  metadata.get("author", "Not Found"))
    log_info("  Edition", metadata.get("edition", "Not Found"))
    log_info("  ISBN",    metadata.get("isbn", "Not Found"))


def stop_ollama():
    """Aggressively stops Ollama and unloads all models to free VRAM (Cross-platform)."""
    import requests as _req
    import time
    import platform
    
    # 1. API Unload (Standard way to free VRAM)
    try:
        ps = _req.get("http://localhost:11434/api/ps", timeout=2)
        if ps.status_code == 200:
            models = ps.json().get("models", [])
            for m in models:
                try:
                    # Setting keep_alive=0 tells Ollama to unload the model immediately
                    _req.post("http://localhost:11434/api/generate", 
                             json={"model": m.get("name"), "keep_alive": 0}, timeout=1)
                except: continue
            time.sleep(0.5)
    except: pass

    # 2. System-Level Kill (Fallback for extreme memory pressure)
    try:
        if platform.system() == "Windows":
            # Using taskkill /F to force stop blocking processes
            os.system("taskkill /F /IM ollama_llama_server.exe /T 2>nul >nul")
            os.system("taskkill /F /IM ollama.exe /T 2>nul >nul")
        else:
            os.system("pkill -9 -f 'ollama_llama_server' 2>/dev/null || true")
            os.system("pkill -9 -f 'ollama' 2>/dev/null || true")
    except: pass
    
    time.sleep(1.0)
    cleanup_gpu()

def cleanup_gpu():
    """Forces garbage collection and flushes CUDA cache."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def start_ollama():
    """Starts the Ollama server in the background (if not already running)."""
    # Check if Ollama is already listening
    try:
        import requests
        res = requests.get("http://localhost:11434/", timeout=0.5)
        if res.status_code == 200:
            return # Already running
    except: pass

    # Start as a background process
    os.system("ollama serve &>/dev/null &")
    time.sleep(3)

def cleanup_intermediate_files(book_crops_dir: str):
    """Deletes the per-book crops folder and all its contents (mineru outputs, etc.)"""
    # Check if cleanup is enabled in .env
    if os.getenv("CLEANUP_ENABLED", "true").lower() != "true":
        log_info("Cleanup", "Skipped (Disabled in .env)")
        return
        
    try:
        if os.path.exists(book_crops_dir):
            time.sleep(0.5)  # Small buffer for slow I/O or file locks
            shutil.rmtree(book_crops_dir)
            log_done(f"Cleanup complete: {book_crops_dir}")
        else:
            log_done(f"Nothing to clean: {book_crops_dir}")
    except Exception as e:
        log_warn(f"Cleanup failed for {book_crops_dir}: {e}")

def cleanup_gpu():
    import signal, psutil
    try:
        current = psutil.Process()
        for child in current.children(recursive=True):
            try: child.send_signal(signal.SIGTERM)
            except: pass
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--source", default="huggingface")
        parser.add_argument("--no-preview", action="store_true")
        parser.add_argument("--no-ai", action="store_true")
        args = parser.parse_args()

        log_step("STARTING PIPELINE")
        books = group_images_by_book(RAW_IMAGE_FOLDER)
        if not books: return
        
        all_books_summary = []
        book_list = sorted(books.items())

        for num, (bid, imgs) in enumerate(book_list, 1):
            log_book_banner(bid, num, len(book_list))
            out = os.path.join(OUTPUT_FOLDER, bid)
            os.makedirs(out, exist_ok=True)

            pages = crop_book(imgs, os.path.join(CROPS_FOLDER, bid))
            ocr_data = ocr_book(pages, out, args, total_pages=len(imgs))

            if not args.no_ai:
                start_ollama()
                title = extract_title_from_cover_image(ocr_data["front_cover_img"])
                edition = extract_edition_from_cover(ocr_data["front_cover_img"], ocr_data["back_cover_img"])
                if not edition: edition = extract_edition_from_text("\n".join(ocr_data["interior_texts"]), title)
                
                interior_text = "\n".join(ocr_data["interior_texts"]).strip()
                if interior_text:
                    desc = generate_description(interior_text, title)
                elif len(imgs) <= 2:
                    # Fallback for 1-2 page books with no extracted text
                    fallback_images = [ocr_data.get("front_cover_img"), ocr_data.get("back_cover_img")]
                    desc = generate_description_from_images(fallback_images, title)
                else:
                    desc = ""
                
                save_book_metadata(bid, title, desc, out, edition)
                
                # --- Disk Cleanup (Final step for this book) ---
                log_progress(f"Cleaning up intermediate files for {bid}")
                cleanup_intermediate_files(os.path.join(CROPS_FOLDER, bid))

                if num < len(book_list): stop_ollama()
                
                all_books_summary.append({ "book_id": bid, "title": title, "description": desc })

    except Exception as e:
        log_error(f"Global Error: {e}")
        import traceback
        log(traceback.format_exc())
    finally:
        cleanup_gpu()
        log_step("PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
