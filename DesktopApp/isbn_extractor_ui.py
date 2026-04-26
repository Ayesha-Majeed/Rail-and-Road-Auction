import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import ollama
import requests
import easyocr
import torch
from PIL import Image

# ─── Load Environment ────────────────────────────────────────────────────────
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
# Prefer external .env if it exists next to the EXE, else use internal
exe_env = os.path.join(os.path.dirname(sys.executable), ".env")
internal_env = os.path.join(BASE_DIR, ".env")
env_path = exe_env if os.path.exists(exe_env) else internal_env
load_dotenv(env_path)

# ─── Configuration ────────────────────────────────────────────────────────────
VISION_MODEL   = os.getenv("VISION_MODEL", "minicpm-v:latest")
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "AIzaSyCbMpGA8FnOH2idwhsQ_UtDQ7BeqMEtB-0")

def check_env_loaded(log_fn=print):
    """Log the environment status for debugging."""
    if not GOOGLE_BOOKS_API_KEY:
        log_fn(f"  ⚠️ Google Books API Key missing. (Searched: {env_path})")
    else:
        log_fn(f"  ✅ Environment loaded from: {BASE_DIR}")

# ─── Shared EasyOCR Engine (Singleton across modules) ───────────────────────
_ocr_reader = None

def _get_ocr_reader(log_fn=print):
    """Lazy singleton: loads EasyOCR with GPU support, shared via sys attribute."""
    global _ocr_reader
    
    # 1. Check module-local singleton
    if _ocr_reader is not None:
        return _ocr_reader
    
    # 2. Check process-wide singleton (to avoid conflict with main_mineru_ocr)
    if hasattr(sys, '_shared_easyocr_reader') and sys._shared_easyocr_reader is not None:
        _ocr_reader = sys._shared_easyocr_reader
        return _ocr_reader

    try:
        log_fn("    ⏳ Initializing ISBN EasyOCR Reader...")
        import torch
        # Force CPU for ISBN pass on Windows IF it's a portable build, 
        # to prevent CUDA context hangs during rapid sync.
        gpu_bool = torch.cuda.is_available()
        
        # LOGIC: If on Windows, we check if GPU is already busy
        if sys.platform == "win32" and gpu_bool:
            # Safe check: if VRAM is extremely low, fallback to CPU
            try:
                free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_vram < 500 * 1024 * 1024: # Less than 500MB
                    log_fn("    ⚠️ Low VRAM detected. Using CPU for ISBN pass for stability.")
                    gpu_bool = False
            except: pass

        _ocr_reader = easyocr.Reader(['en'], gpu=gpu_bool)
        # Store for other modules to find
        sys._shared_easyocr_reader = _ocr_reader
        log_fn(f"    ✅ EasyOCR Ready (GPU={gpu_bool})")
    except Exception as e:
        log_fn(f"    ❌ EasyOCR init failed: {e}")
        _ocr_reader = None
    return _ocr_reader

def unload_isbn_reader():
    """Unloads the shared reader from memory."""
    global _ocr_reader
    if hasattr(sys, '_shared_easyocr_reader'):
        del sys._shared_easyocr_reader
        sys._shared_easyocr_reader = None
    
    if _ocr_reader is not None:
        print("🧹 Unloading Shared EasyOCR to free VRAM...")
        del _ocr_reader
        _ocr_reader = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

# ─── ISBN VALIDATION LOGIC ──────────────────────────────────────────────────
def validate_isbn10(isbn: str) -> bool:
    if len(isbn) != 10: return False
    total = 0
    for i, ch in enumerate(isbn):
        val = 10 if ch == "X" else (int(ch) if ch.isdigit() else -1)
        if val == -1: return False
        total += val * (10 - i)
    return total % 11 == 0

def validate_isbn13(isbn: str) -> bool:
    if len(isbn) != 13: return False
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(isbn))
    return total % 10 == 0

def normalize_isbn(raw: str) -> str:
    """Strip everything except digits and X."""
    return "".join(c for c in raw.upper() if c.isdigit() or c == 'X')

def isbn10_to_isbn13(isbn10: str) -> str:
    """Convert ISBN-10 to ISBN-13."""
    body = "978" + isbn10[:9]
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(body))
    check = (10 - (total % 10)) % 10
    return body + str(check)

# ─── CORE EXTRACTION ─────────────────────────────────────────────────────────

def strict_extraction_regex(raw_text: str) -> list[str]:
    """
    Tiered extraction to handle perfect and messy OCR output.
    1. Tier 1: Official Regex (Strict hyphens/spaces/underscores)
    2. Surgical Fix: Allow dots ONLY if preceded by a strong ISBN label.
    """
    labeled_isbns = []
    bare_isbns = []
    likely_isbns = [] # Labeled but failed checksum
    
    # ── TIER 1: MATCHING ──
    # Label pattern allows standard ISBN prefix plus full phrases.
    label_pattern = r'(ISBN(?:-1[03])?|International Standard Book Number|Standard Book Number)'
    # Digit pattern: allows digits, hyphens, spaces, underscores AND common misreads like |, I, l.
    # WE ALSO ALLOW DOTS '.' conditionally inside the logic below.
    digit_pattern = r'((?:97[89][-\s_.]*|[|Il][-\s_.]*97[89][-\s_.]*)?(?:[0-9|Il][-\s_.|Il]*){8,9}[0-9X|Il])\b'
    
    full_pattern = rf'({label_pattern}[:\s._-]*)?{digit_pattern}'
    
    for match_obj in re.finditer(full_pattern, raw_text, re.IGNORECASE):
        label_part = match_obj.group(1) or ""
        m = match_obj.group(3) # Group 3 is the digit part (Group 2 was label)
        
        has_strong_label = any(kw in label_part.upper() for kw in ["ISBN", "STANDARD BOOK NUMBER"])
        
        # SURGICAL DOT PROTECTION:
        # If there is no strong label, we DO NOT ALLOW dots in the ISBN string.
        # This prevents picking up prices like $3.18.
        if not has_strong_label and "." in m:
            continue

        # USER REQUEST: Hardened Filter (Robustly skip Phone/Fax/Box numbers)
        prefix_start = max(0, match_obj.start() - 40)
        prefix_text = raw_text[prefix_start:match_obj.start()].upper()
        clean_prefix = re.sub(r'[\s:._\-]+', '', prefix_text)
        
        exclusion_tags = ["FAX", "TEL", "PHONE", "TELEPHONE", "TOLL", "POBOX", "BOX", "OFFICE", "PAGER", "TF23", "TF."]
        if any(tag in clean_prefix for tag in exclusion_tags):
            continue
            
        # Normalize common misreads: | -> 1, I -> 1, l -> 1
        m_norm = m.replace("|", "1").replace("I", "1").replace("l", "1").replace(".", "-") # Convert dot to hyphen for processing
        clean = normalize_isbn(m_norm)
        if len(clean) == 9: clean = "0" + clean
        
        target_list = labeled_isbns if has_strong_label else bare_isbns
        
        if len(clean) == 10:
            if validate_isbn10(clean):
                if clean not in (labeled_isbns + bare_isbns):
                    target_list.append(clean)
            elif has_strong_label:
                if clean not in likely_isbns:
                    likely_isbns.append(clean)
        elif len(clean) == 13:
            if validate_isbn13(clean):
                if clean not in (labeled_isbns + bare_isbns):
                    target_list.append(clean)
            elif has_strong_label:
                if clean not in likely_isbns:
                    likely_isbns.append(clean)

    if labeled_isbns:
        return labeled_isbns
    if likely_isbns:
        return likely_isbns
    return bare_isbns

    # ── TIER 2: LABEL-ANCHORED FUZZY EXTRACTION ──
    # If Tier 1 fails, find the label and deep-clean the following text.
    # Handles dots (0.916) and other mess naturally.
    labels = ["ISBN", "ISBN-10", "ISBN-13", "STANDARD BOOK NUMBER", "INTERNATIONAL STANDARD BOOK NUMBER"]
    for label in labels:
        for match in re.finditer(re.escape(label), raw_text, re.IGNORECASE):
            start = match.end()
            segment = raw_text[start : start + 35]
            
            # Normalize common misreads: | -> 1, I -> 1, l -> 1
            segment_norm = segment.replace("|", "1").replace("I", "1").replace("l", "1")
            clean = normalize_isbn(segment_norm)
            
            for length in [13, 10, 9]:
                candidate = clean[:length]
                if len(candidate) == 9: candidate = "0" + candidate
                
                if len(candidate) == 10:
                    if validate_isbn10(candidate) and candidate not in valid_isbns:
                        valid_isbns.append(candidate)
                        break
                elif len(candidate) == 13:
                    if validate_isbn13(candidate) and candidate not in valid_isbns:
                        valid_isbns.append(candidate)
                        break
                        
    return valid_isbns

def extract_isbn_from_image(image_path: str, log_fn=print, is_cover=False):
    """
    3-Tier Extraction Hierarchy:
    1. Literal OCR + Strict Regex
    2. Label-Anchored Fuzzy Extraction
    3. AI Smart Parsing (Zero-hallucination fallback)
    
    Returns: tuple(list[str], str) — (list of valid ISBNs, raw OCR text)
    """
    full_ocr_text = ""
    
    log_fn(f"    ⏳ Scanning {Path(image_path).name}...")
    reader = _get_ocr_reader(log_fn)
    if reader:
        try:
            results = reader.readtext(image_path)
            raw_lines = [res[1] for res in results]
            full_ocr_text = " ".join(raw_lines)
            
            # Log raw text for debugging ISBN extraction issues
            if full_ocr_text.strip():
                log_fn(f"    📜 [DEBUG] Full OCR text for ISBN search:")
                log_fn(f"    {'─'*40}")
                # Show up to 2000 chars for deep debugging
                preview_text = full_ocr_text[:2000] + ("..." if len(full_ocr_text) > 2000 else "")
                for line in preview_text.splitlines():
                    if line.strip():
                        log_fn(f"      {line.strip()}")
                log_fn(f"    {'─'*40}")

            isbns = strict_extraction_regex(full_ocr_text)
            if isbns:
                return isbns, full_ocr_text
        except Exception as e:
            log_fn(f"  ⚠️ OCR failed on {Path(image_path).name}: {e}")

    # Tier 3: AI Smart Parsing
    if full_ocr_text.strip():
        log_fn("    ⏳ ISBN Tier 3: AI Smart Parsing...")
        text_prompt = (
            "SYSTEM: You are a precision extraction machine. Find the ISBN-10 or ISBN-13.\n"
            "OCR text often misreads '|', 'I', or 'l' as the number '1'.\n"
            "OCR text can be very messy. Look for any labels like 'ISBN' or 'International Standard Book Number'.\n\n"
            "RULES:\n"
            "1. Output ONLY the digits and X. No intro, no explanation, no chatter.\n"
            "2. DO NOT GUESS. If you cannot find the actual numbers in the text, you MUST output 'NO ISBN FOUND'.\n"
            "3. Use the literal text provided below. Do not use external knowledge.\n\n"
            f"OCR TEXT:\n{full_ocr_text}"
        )
        try:
            response = ollama.chat(
                model="llama3.2:1b" if "llama3.2:1b" in str(ollama.list()) else VISION_MODEL,
                messages=[{"role": "user", "content": text_prompt}]
            )
            raw_res = response["message"]["content"].strip()
            isbns_ai = strict_extraction_regex(raw_res)
            if isbns_ai:
                return isbns_ai, full_ocr_text
        except Exception as e:
            log_fn(f"  ⚠️ AI parsing failed: {e}")

    return [], full_ocr_text

# ─── METADATA ENRICHMENT ─────────────────────────────────────────────────────

def fetch_metadata_google(isbn: str, log_fn=print) -> dict | None:
    if not GOOGLE_BOOKS_API_KEY:
        log_fn("  ⚠️ Google Books API Key missing.")
        return None

    # Hardened headers to bypass common bot filters
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key={GOOGLE_BOOKS_API_KEY}"
    # Log obfuscated URL for debugging
    safe_key = f"{GOOGLE_BOOKS_API_KEY[:4]}...{GOOGLE_BOOKS_API_KEY[-4:]}" if len(GOOGLE_BOOKS_API_KEY)>8 else "..."
    log_fn(f"  🌍 Calling Google API: q=isbn:{isbn} (key={safe_key})")
    
    # SENIOR-GRADE: SSL Resilience for Windows Portable App
    request_kwargs = {"headers": headers, "timeout": 15, "verify": True}
    try:
        import certifi
        request_kwargs["verify"] = certifi.where()
    except ImportError:
        pass

    for attempt in range(1, 4):  # Up to 3 attempts
        try:
            try:
                resp = requests.get(url, **request_kwargs)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as ssl_err:
                log_fn(f"  ⚠️ SSL/Connection error on attempt {attempt}: {ssl_err}")
                log_fn("  🔧 Attempting SSL fallback (verify=False)...")
                # Final fallback for corporate/portable environments with missing certs
                request_kwargs["verify"] = False
                resp = requests.get(url, **request_kwargs)

            data = resp.json()
            
            if resp.status_code == 200:
                if "items" in data:
                    v = data["items"][0]["volumeInfo"]
                    # Extract other volumes sharing the same ISBN
                    other_vols = [item["volumeInfo"].get("title") for item in data["items"][1:] if "volumeInfo" in item]
                    
                    log_fn(f"  📡 API Success | Items: {data.get('totalItems', 0)}")
                    return {
                        "title":          v.get("title", "N/A"),
                        "subtitle":       v.get("subtitle", "N/A"),
                        "authors":        ", ".join(v.get("authors", ["N/A"])),
                        "publisher":      v.get("publisher", "N/A"),
                        "published_date": v.get("publishedDate", "N/A"),
                        "description":    v.get("description", "N/A"),
                        "isbn_10":        next((i["identifier"] for i in v.get("industryIdentifiers", []) if i["type"] == "ISBN_10"), "N/A"),
                        "isbn_13":        next((i["identifier"] for i in v.get("industryIdentifiers", []) if i["type"] == "ISBN_13"), isbn),
                        "categories":     ", ".join(v.get("categories", ["N/A"])),
                        "page_count":     v.get("pageCount", "N/A"),
                        "thumbnail":      v.get("imageLinks", {}).get("thumbnail", "N/A"),
                        "edition":        "N/A",
                        "other_volumes":  other_vols
                    }
                else:
                    log_fn(f"  📡 API OK | No items found for: {isbn}")
                    return None
            
            elif resp.status_code == 503:
                log_fn(f"  ⏳ API 503 (Unavailable) - Attempt {attempt}/3. Waiting...")
                if attempt < 3:
                    time.sleep(2 * attempt)
                    continue
            
            # Handle other errors
            if "error" in data:
                log_fn(f"  ❌ API Error ({resp.status_code}): {data['error'].get('message')}")
            else:
                log_fn(f"  📡 API Status: {resp.status_code}")
            return None

        except Exception as e:
            log_fn(f"  ⚠️ Meta API error (Attempt {attempt}): {e}")
            if attempt < 3:
                time.sleep(1)
                continue
            return None
    
    return None

def fetch_metadata_local_ai(image_path: str, accumulated_text: str = "", log_fn=print) -> dict:
    """Zero-logic fallback using Vision model for basic info + collected OCR text."""
    # Use text context if available (limit to 3000 chars for prompt safety)
    text_context = f"\nEXTRACTED TEXT CONTEXT:\n{accumulated_text[:3000]}\n" if accumulated_text else ""
    
    prompt = (
        "Extract these details from this book image (Front Cover):\n"
        "1. Title\n2. Author(s)\n3. Edition (ONLY if explicitly mentioned, otherwise 'N/A')\n"
        "4. Description (1-2 sentences summarizing the book based on title and provided text context below)\n"
        f"{text_context}\n"
        "RULES: If any detail is missing or ambiguous, return 'N/A'. DO NOT GUESS DATES OR YEARS.\n"
        "Output ONLY a JSON block with keys: title, authors, edition, description."
    )
    try:
        resp = ollama.chat(model=VISION_MODEL, messages=[{"role":"user", "content":prompt, "images":[image_path]}])
        text = resp["message"]["content"]
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        log_fn(f"  ⚠️ Local AI fallback failed: {e}")
    return {"title":"N/A", "authors":"N/A", "edition":"N/A", "description":"N/A"}

# ─── CORE PIPELINE ───────────────────────────────────────────────────────────

def process_book(book_id: str, files: list[str], log_fn=print) -> dict:
    """
    Main entry point for processing a single book.
    Returns a result dictionary.
    """
    log_fn("-" * 60)
    log_fn(f"📚 PROCESSING BOOK: {book_id}")
    
    # SENIOR-GRADE: Log where we are loading keys from
    check_env_loaded(log_fn)
    
    log_fn(f"📄 Found {len(files)} files.")

    # Sort files by page number suffix (e.g., _01.jpg, -04.png)
    file_map = {}
    for f in files:
        m = re.search(r'[\-_](\d+)\.\w+$', f)
        if m:
            file_map[int(m.group(1))] = f

    official_isbn = None
    source_page = None
    collected_ocr_texts = []  # Collect all OCR text for later reuse (e.g., edition detection)

    # Step 1: Scan Page 4
    if 4 in file_map:
        log_fn(f"  🔍 Checking Page 4: {Path(file_map[4]).name}")
        res4, ocr4 = extract_isbn_from_image(file_map[4], log_fn)
        if ocr4: collected_ocr_texts.append(ocr4)
        if res4:
            log_fn(f"  ✅ Found on Page 4: {res4[0]}")
            # Cross-verify with Page 2
            if 2 in file_map:
                log_fn(f"  🛡️ Verifying with Page 2...")
                res2, ocr2 = extract_isbn_from_image(file_map[2], log_fn, is_cover=True)
                if ocr2: collected_ocr_texts.append(ocr2)
                if res2 and res2[0] == res4[0]:
                    log_fn(f"  ⭐ Match confirmed: {res2[0]}")
                    official_isbn = res2[0]
                    source_page = 2
                else:
                    log_fn(f"  ⚠️ No match or P2 empty. Preferring P4: {res4[0]}")
                    official_isbn = res4[0]
                    source_page = 4
            else:
                official_isbn = res4[0]
                source_page = 4

    # Step 2: Fallback sequence 3 -> 2 -> 1
    if not official_isbn:
        log_fn("  🔍 Triggering fallback sequence (3, 2, 1)...")
        for pn in [3, 2, 1]:
            if pn in file_map:
                log_fn(f"  🔍 Checking Page {pn}...")
                res, ocr_txt = extract_isbn_from_image(file_map[pn], log_fn, is_cover=(pn==2))
                if ocr_txt: collected_ocr_texts.append(ocr_txt)
                if res:
                    log_fn(f"  ✅ Found on Page {pn}: {res[0]}")
                    official_isbn = res[0]
                    source_page = pn
                    break

    # Final Stage: Metadata
    result = {
        "book_id": book_id, 
        "isbn": official_isbn or "N/A", 
        "source_page": source_page,
        "metadata": None,
        "ocr_texts": collected_ocr_texts  # Reusable for edition detection
    }
    
    if official_isbn:
        meta = fetch_metadata_google(official_isbn, log_fn)
        if not meta:
            log_fn("  🌍 No Google metadata. Using Local AI Fallback...")
            # USER REQUEST: Use all collected text for description
            all_text = "\n".join(collected_ocr_texts)
            # ALWAYS use Front Cover (Page 1) for visual extraction if available
            fallback_img = file_map.get(1, file_map.get(source_page, files[0]))
            meta = fetch_metadata_local_ai(fallback_img, all_text, log_fn)
        
        result["metadata"] = meta
        log_fn(f"  🎉 SUCCESS: {meta.get('title', 'Unknown Title')}")
    else:
        log_fn("  ❌ NO ISBN FOUND. Attempting visual extraction from front cover...")
        all_text = "\n".join(collected_ocr_texts)
        fallback_img = file_map.get(1, files[0])
        meta = fetch_metadata_local_ai(fallback_img, all_text, log_fn)
        result["metadata"] = meta
        if meta and meta.get("title") != "N/A":
            log_fn(f"  🎉 Visual Success: {meta.get('title')}")
        else:
            log_fn("  ❌ Visual extraction failed.")

    return result

# ─── MAIN CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Professional ISBN Extraction Engine")
    parser.add_argument("files", nargs="+", help="Image files for a single book")
    parser.add_argument("--id", help="Optional Manual Book ID")
    parser.add_argument("--out", help="Output JSON file", default="isbn_results.json")
    
    args = parser.parse_args()
    
    # Auto-detect ID if not provided
    book_id = args.id
    if not book_id:
        stem = Path(args.files[0]).stem
        parts = re.split(r'[_\-]', stem)
        book_id = "-".join(parts[:-1]) if len(parts) >= 2 else stem

    result = process_book(book_id, args.files)
    
    # Save to JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Done. Results saved to {args.out}")

if __name__ == "__main__":
    main()