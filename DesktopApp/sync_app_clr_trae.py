import customtkinter as ctk
from tkinter import filedialog, messagebox, PhotoImage, Canvas
import threading
import json
import os
import sys
import re
import time
import queue
import subprocess
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# ─── OCR Pipeline (same folder) ──────────────────────────────────────────────
import model_manager

try:
    import main_mineru_ocr as ocr_pipeline
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  main_mineru_ocr not found — OCR pipeline disabled")

try:
    import isbn_extractor_ui as isbn_logic
    ISBN_LOGIC_AVAILABLE = True
except ImportError:
    ISBN_LOGIC_AVAILABLE = False
    print("⚠️  isbn_extractor_ui not found — ISBN first-pass disabled")

# ─── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("light")

# ─── Colour palette ───────────────────────────────────────────────────────────
C = {
    "bg":       "#FFFFFF",
    "white":    "#FFFFFF",
    "card":     "#F5F2EC",      # cream tint
    "olive":    "#8C7B5D",      # primary branding
    "olive_h":  "#AA9874",      # hover
    "olive_dk": "#6B5C41",      # dark hover
    "red":      "#BC4B34",
    "red_h":    "#9B3A28",
    "border":   "#E4E7EC",
    "text":     "#1D2939",
    "muted":    "#667085",
    "light":    "#98A2B3",
    "hdr":      "#344054",
    "s_g_bg":   "#ECFDF3", "s_g_fg": "#027A48",   # Uploading / Complete
    "s_o_bg":   "#FDF1E6", "s_o_fg": "#BC4B34",   # Queued
    "s_b_bg":   "#D0E8FD", "s_b_fg": "#1A4A8A",   # Info
    "s_p_bg":   "#E8D0FD", "s_p_fg": "#4A1A8A",   # Processing
    "s_skip_bg":"#F3F4F6", "s_skip_fg":"#6B7280",  # Skipped
    "s_fail_bg":"#FEF3F2", "s_fail_fg":"#B42318",  # Failed
}

CONFIG_FILE = os.path.join(model_manager.BASE_DIR, "sync_config.json")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

STATUS_STYLES = {
    "Uploading":  (C["s_g_bg"],    "#039855"),
    "Complete":   (C["s_g_bg"],    "#039855"),
    "Queued":     ("#F1DDCA",      "#BC4B34"),
    "Processing": (C["s_p_bg"],    C["s_p_fg"]),
    "Skipped":    (C["s_skip_bg"], C["s_skip_fg"]),
    "Failed":     (C["s_fail_bg"], C["s_fail_fg"]),
}


# ─── Book Grouper ─────────────────────────────────────────────────────────────
class BookGrouper:
    # Supports underscores, hyphens, and spaces: BookID_001.jpg, BookID-001.jpg, BookID 001.jpg
    PATTERN = re.compile(r"^(.+?)[_\-\s](\d+)\.(jpg|jpeg|png|tiff|bmp|webp)$", re.IGNORECASE)

    def group(self, folder: str, recursive: bool = False) -> dict:
        books = defaultdict(list)
        if recursive:
            for root, _, files in os.walk(folder):
                for fn in files:
                    if fn.startswith("."): continue
                    m = self.PATTERN.match(fn)
                    if m:
                        book_id  = m.group(1).strip()
                        if not book_id: continue
                        page_num = int(m.group(2))
                        books[book_id].append((page_num, os.path.join(root, fn)))
        else:
            for fn in os.listdir(folder):
                if fn.startswith("."): continue
                fp = os.path.join(folder, fn)
                if not os.path.isfile(fp):
                    continue
                m = self.PATTERN.match(fn)
                if m:
                    book_id  = m.group(1).strip()
                    if not book_id: continue
                    page_num = int(m.group(2))
                    books[book_id].append((page_num, fp))
        for book_id in books:
            books[book_id].sort(key=lambda x: x[0])
        return dict(books)

    def build_document(self, book_id: str, pages: list) -> dict:
        total = len(pages)
        doc = {
            "book_id":        book_id,
            "total_pages":    total,
            "front_cover":    None,
            "back_cover":     None,
            "interior_pages": [],
            "status":         "complete",
            "synced_at":      datetime.now().isoformat(),
        }
        page_nums      = [p[0] for p in pages]
        has_duplicates = len(page_nums) != len(set(page_nums))
        missing        = self._check_missing(page_nums)

        if has_duplicates:
            doc["status"] = "warning_duplicates"
        if missing:
            doc["status"] = "warning_missing_pages"
            doc["missing_pages"] = missing

        sorted_nums = sorted(page_nums)
        for page_num, filepath in pages:
            fname = os.path.basename(filepath)
            entry = {"page_id": f"{book_id}_{str(page_num).zfill(3)}",
                     "file_name": fname, "file_path": filepath}
            if total == 2:
                if page_num == min(page_nums):
                    entry["note"] = "Combined Front & Back Cover"
                    doc["front_cover"] = entry
                else:
                    entry.update({"page_number": page_num, "type": "interior"})
                    doc["interior_pages"].append(entry)
            else:
                if page_num == sorted_nums[0]:
                    doc["front_cover"] = entry
                elif page_num == sorted_nums[1]:
                    doc["back_cover"] = entry
                else:
                    entry.update({"page_number": page_num, "type": "interior"})
                    doc["interior_pages"].append(entry)
        return doc

    def _check_missing(self, page_nums):
        if not page_nums:
            return []
        full = set(range(min(page_nums), max(page_nums) + 1))
        return sorted(full - set(page_nums))


# ─── Crypto Utilities ────────────────────────────────────────────────────────
class CryptoUtils:
    """
    Handles AES-256-GCM decryption for tokens encrypted by Node.js.
    Format expected: iv.at.ct (dot-separated Base64 segments)
    """
    @staticmethod
    def decrypt_token(encrypted_str: str, hex_key: str):
        if not encrypted_str or not hex_key:
            return None
        try:
            import base64
            import hashlib
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            # 1. Prepare Key (Verified strategy: SHA-256 of the secret string)
            key = hashlib.sha256(hex_key.encode('utf-8')).digest()
            
            # 2. Parse Segments (Node.js style: iv.tag.cipher)
            parts = encrypted_str.split('.')
            if len(parts) != 3:
                return None
            
            iv_b64, tag_b64, cipher_b64 = parts

            def b64_decode(s):
                # Add padding if needed
                s += '=' * (-len(s) % 4)
                # Try URL-safe first (standard for modern Node.js)
                try:
                    return base64.urlsafe_b64decode(s)
                except:
                    return base64.b64decode(s)

            iv = b64_decode(iv_b64)
            tag = b64_decode(tag_b64)
            ciphertext = b64_decode(cipher_b64)

            # 3. Decrypt (AESGCM expects cipher + tag combined)
            aesgcm = AESGCM(key)
            decrypted = aesgcm.decrypt(iv, ciphertext + tag, None)
            return decrypted.decode('utf-8')
        except Exception as e:
            # Silent fail for lookup iteration
            return None


# ─── MongoDB Connector ────────────────────────────────────────────────────────
class DBConnector:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False

    def connect(self):
        max_retries = 3
        last_error = ""
        
        for attempt in range(1, max_retries + 1):
            try:
                from pymongo import MongoClient
                # Increase timeout slightly and try connecting
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=15000)
                self.client.admin.command("ping")
                self.db = self.client[self.db_name]
                self.connected = True
                return True, f"Connected to MongoDB — DB: '{self.db_name}'"
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    time.sleep(1.5) # Wait before retry
                continue
        
        self.connected = False
        return False, last_error

    def book_exists(self, collection, book_id):
        try:
            return self.db[collection].find_one({"book_id": book_id}) is not None
        except:
            return False

    def find_user_by_token(self, token: str):
        """
        Iterates through users and decrypts their stored tokens to find a match.
        The secret key is provided by the collaborator (Wasi Shah).
        """
        SECRET_KEY = os.environ.get("CONNECTION_SECRET_KEY", "78752a9db25d08be9e4702510374164335e63863aae30e8e212ac79a8884c354") # Fallback for local dev
        try:
            # 1. Get all users who have an encrypted token
            users = self.db["users"].find({"desktopConnectionTokenEnc": {"$exists": True}})
            for user in users:
                enc_token = user.get("desktopConnectionTokenEnc")
                if not enc_token: continue
                
                # 2. Attempt Decryption using verified strategy
                decrypted = CryptoUtils.decrypt_token(enc_token, SECRET_KEY)
                
                # 3. Compare with input
                if decrypted == token:
                    return user
            return None
        except Exception as e:
            print(f"❌ User lookup failed: {e}")
            return None

    def insert_book(self, collection, doc):
        result = self.db[collection].insert_one(doc)
        return str(result.inserted_id)

    def ping(self):
        try:
            if not self.client:
                return False
            self.client.admin.command("ping")
            self.connected = True
            return True
        except Exception:
            self.connected = False
            return False

    def disconnect(self):
        try:
            if self.client:
                self.client.close()
        except:
            pass
        self.connected = False


# ─── Main Application ─────────────────────────────────────────────────────────
class SyncApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rail & Road — Book Sync")

        # Window sizing — 90% of screen, centered
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h   = int(sw * 0.90), int(sh * 0.90)
        x, y   = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(800, 560)

        # ─── Model Health Check ────────────────────
        self.after(100, self._check_models_on_start)

        # Store screen info for responsive scaling
        self._win_w = w
        self._screen_w = sw

        # Compute font scale based on window width (reference = 1280px)
        # Uses actual tkinter pixel measurement — works correctly on Linux HiDPI too
        self._compute_fonts(w)
        # State
        self.db_connector  = None
        self.sync_running  = False
        self.log_queue     = queue.Queue()
        self.total_ok      = 0
        self.total_skip    = 0
        self.total_fail    = 0
        self.activity_rows = {}   # book_id → row widgets dict
        self._detail_windows = {} # book_id → open Toplevel window
        self._pending_ids  = set() # IDs currently being added to UI
        self.row_order     = []   # insertion order for rows
        self._folder_selected_session = False
        self.current_user = None

        self._load_config()
        self._build_ui()
        self._poll_log()
        # Print detected scale so user can verify
        self.after(500, self._debug_scale)
        # Bind resize to update padding AND fonts
        self.bind("<Configure>", self._on_resize)

    def _get_os_scale(self):
        """
        Detect the OS-level display scale factor on Linux/Ubuntu.
        On HiDPI / Wayland-scaled desktops the GTK scale env var or
        winfo_fpixels tell us the real multiplier (1x, 1.5x, 2x, etc.).
        Returns a float >= 1.0.
        """
        import platform, os as _os
        if platform.system() != "Linux":
            return 1.0
        # 1. GDK_SCALE env var — set by GNOME/Wayland for integer scaling
        gdk = _os.environ.get("GDK_SCALE", "").strip()
        try:
            v = float(gdk)
            if v >= 1.0:
                return v
        except (ValueError, TypeError):
            pass
        # 2. QT_SCALE_FACTOR
        qt = _os.environ.get("QT_SCALE_FACTOR", "").strip()
        try:
            v = float(qt)
            if v >= 1.0:
                return v
        except (ValueError, TypeError):
            pass
        # 3. winfo_fpixels: pixels per inch — 96dpi = 1x, 192dpi = 2x
        try:
            dpi = self.winfo_fpixels("1i")
            if dpi > 0:
                return max(1.0, round(dpi / 96.0 * 4) / 4)  # round to nearest 0.25
        except Exception:
            pass
        return 1.0

    def _debug_scale(self):
        """Print detected OS scale and font sizes to terminal for debugging."""
        os_scale = self._get_os_scale()
        win_w    = self.winfo_width()
        print(f"[FontDebug] win_w={win_w}  os_scale={os_scale:.2f}  "
              f"_scale={self._scale:.2f}  "
              f"heading={self.F['heading']}  label={self.F['label']}  "
              f"muted={self.F['muted']}")

    def _compute_fonts(self, win_w):
        """
        Calculate font sizes proportional to window width + OS display scale.
        On Ubuntu HiDPI (GDK_SCALE=2) the window's logical pixel count is
        already halved by the compositor, so we must multiply fonts back up.
        """
        os_scale = self._get_os_scale()
        # Scale relative to 1280px design width, then apply OS HiDPI factor
        s = (win_w / 1280.0) * os_scale
        s = max(0.85, min(s, 2.0))
        self._scale = s

        def fs(b): return max(11, int(round(b * s)))
        self._fs = fs
        self.F = {
            "logo":    fs(22),
            "section": fs(20),
            "heading": fs(16),
            "label":   fs(13),
            "input":   fs(13),
            "btn":     fs(14),
            "table_h": fs(13),
            "table_b": fs(12),
            "badge":   fs(11),
            "muted":   fs(11),
        }
        self.SB = max(180, int(200 * s))

    def _on_resize(self, event=None):
        """Update padding AND font sizes whenever window is resized."""
        if event and event.widget is not self:
            return
        new_w = self.winfo_width()
        if new_w < 50:
            return
        # Recompute fonts for new width
        self._compute_fonts(new_w)
        self._win_w = new_w
        # Update padding on main sections
        pad = 450 if new_w > 3000 else max(24, int(new_w * 0.06))
        for widget_name in ("_dashboard_label", "_cards_frame", "_act_frame_outer"):
            widget = getattr(self, widget_name, None)
            if widget:
                try:
                    widget.grid_configure(padx=pad)
                except Exception:
                    pass
        # Refresh all registered widget fonts
        self._refresh_fonts()

    def _refresh_fonts(self):
        """Update fonts on all registered widgets."""
        font_map = getattr(self, "_font_registry", {})
        for key, (widget, base_size, family, weight) in list(font_map.items()):
            try:
                new_size = max(9, int(round(base_size * self._scale)))
                widget.configure(font=ctk.CTkFont(family=family, size=new_size, weight=weight))
            except Exception:
                font_map.pop(key, None)

    def _reg(self, widget, base_size, family="Inter", weight="normal"):
        """Register a widget for live font updates on resize."""
        if not hasattr(self, "_font_registry"):
            self._font_registry = {}
        self._font_registry[id(widget)] = (widget, base_size, family, weight)
        return widget

    def _px(self, base):
        """Scale a pixel dimension by the current display scale factor."""
        return max(1, int(round(base * self._scale)))

    # ── Config ────────────────────────────────────────────────────────────────
    def _load_config(self):
        defaults = {
            "folder_path": "",
            "books_path":  "",
            "slides_path": "",
            "mongo_uri":   "",
            "db_name":     "Test",
            "collection":  "Book Data",
            "watch_mode":  False,
            "interval":    30,
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    defaults.update(json.load(f))
            except:
                pass
        
        # Override with Environment Variables for Security (Docker/Env compatible)
        env_uri = os.environ.get("MONGO_URI", "").strip()
        if env_uri:
            defaults["mongo_uri"] = env_uri
            
        env_db = os.environ.get("DB_NAME", "").strip()
        if env_db:
            defaults["db_name"] = env_db
            
        env_coll = os.environ.get("COLLECTION", "").strip()
        if env_coll:
            defaults["collection"] = env_coll

        # Default Sync Interval from environment
        env_interval = os.environ.get("SYNC_INTERVAL", "").strip()
        if env_interval.isdigit():
            defaults["interval"] = int(env_interval)
            
        self.config = defaults

    def _save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)

    # ─── Build UI ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.configure(fg_color=C["bg"])
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()

    # ── Header ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, corner_radius=self._px(0),
                           fg_color=C["white"],
                           border_width=1, border_color=C["border"])
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)  # center stretches

        # ── Right: connection badge + settings ────────────────────────────────
        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.grid(row=0, column=2, sticky="e", padx=24, pady=16)

        self._reg(ctk.CTkLabel(right, text="Connection Status :",
                     font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="normal"),
                     text_color="#000000"), 13, "Inter", "normal").pack(side="left", padx=(0, 8))

        self.conn_badge = ctk.CTkFrame(right, corner_radius=self._px(15),
                                        fg_color=C["s_skip_bg"])  # neutral until verified
        self.conn_badge.pack(side="left", padx=(0, 16), ipady=4)

        icon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")
        g  = os.path.join(icon_dir, "connection_icon.png")
        gy = os.path.join(icon_dir, "connection_icon_grey.png")
        r  = os.path.join(icon_dir, "connection_icon_red.png")
        def _mk_icon(p):
            if os.path.exists(p):
                try:
                    from PIL import Image
                    return ctk.CTkImage(light_image=Image.open(p), size=(20, 20))
                except Exception:
                    try:
                        return PhotoImage(file=p)
                    except Exception:
                        return None
            return None
        self._conn_icon_green = _mk_icon(g)
        self._conn_icon_grey  = _mk_icon(gy) or None
        self._conn_icon_red   = _mk_icon(r)  or None

        self.conn_icon = ctk.CTkLabel(self.conn_badge,
                                      text="" if self._conn_icon_grey else "!",
                                      image=self._conn_icon_grey,
                                      font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                                      text_color=C["s_skip_fg"])
        self._reg(self.conn_icon, 13, "Inter", "bold")
        self.conn_icon.pack(side="left", padx=(10, 6))
        self.conn_text = ctk.CTkLabel(self.conn_badge, text="Token Required",
                                      font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                                      text_color=C["s_skip_fg"])
        self._reg(self.conn_text, 13, "Inter", "bold")
        self.conn_text.pack(side="left", padx=(0, 10))
        self._set_conn_visual("neutral")

        self._reg(ctk.CTkButton(right, text="⚙", width=self._px(36), height=self._px(36),
                      corner_radius=self._px(18),
                      fg_color="transparent",
                      border_width=1, border_color=C["border"],
                      text_color=C["muted"],
                      hover_color="#F5F5F5",
                      font=ctk.CTkFont(size=self.F["heading"]),
                      command=self._open_settings), 16, "Inter", "normal").pack(side="left", padx=(0, 8))

        self._reg(ctk.CTkButton(right, text="↻", width=self._px(36), height=self._px(36),
                      corner_radius=self._px(18),
                      fg_color="transparent",
                      border_width=1, border_color=C["border"],
                      text_color=C["muted"],
                      hover_color="#F5F5F5",
                      font=ctk.CTkFont(size=self.F["heading"]),
                      command=self._check_models_manual), 16, "Inter", "normal").pack(side="left")

    def _set_conn_visual(self, state):
        try:
            self.conn_icon.configure(image=None, text="")
        except Exception:
            pass
        if state == "neutral":
            self.conn_badge.configure(fg_color=C["s_skip_bg"]) 
            self.conn_text.configure(text="Token Required", text_color=C["s_skip_fg"]) 
            if getattr(self, "_conn_icon_grey", None):
                self.conn_icon.configure(image=self._conn_icon_grey, text="")
            else:
                self.conn_icon.configure(image=None, text="!", text_color=C["s_skip_fg"]) 
        elif state == "connecting":
            self.conn_badge.configure(fg_color="#FEF9C3") 
            self.conn_text.configure(text="Connecting...", text_color="#92400E") 
            if getattr(self, "_conn_icon_grey", None):
                self.conn_icon.configure(image=self._conn_icon_grey, text="")
            else:
                self.conn_icon.configure(image=None, text="!", text_color="#92400E") 
        elif state == "active":
            self.conn_badge.configure(fg_color=C["s_g_bg"]) 
            self.conn_text.configure(text="Active", text_color=C["s_g_fg"]) 
            if getattr(self, "_conn_icon_green", None):
                self.conn_icon.configure(image=self._conn_icon_green, text="")
            else:
                self.conn_icon.configure(image=None, text="✔", text_color=C["s_g_fg"]) 
        elif state == "invalid":
            self.conn_badge.configure(fg_color=C["s_fail_bg"]) 
            self.conn_text.configure(text="Invalid Token", text_color=C["s_fail_fg"]) 
            if getattr(self, "_conn_icon_red", None):
                self.conn_icon.configure(image=self._conn_icon_red, text="")
            else:
                self.conn_icon.configure(image=None, text="!", text_color=C["s_fail_fg"]) 
        elif state == "failed":
            self.conn_badge.configure(fg_color=C["s_fail_bg"]) 
            self.conn_text.configure(text="Failed", text_color=C["s_fail_fg"]) 
            if getattr(self, "_conn_icon_red", None):
                self.conn_icon.configure(image=self._conn_icon_red, text="")
            else:
                self.conn_icon.configure(image=None, text="!", text_color=C["s_fail_fg"]) 
        elif state == "offline":
            self.conn_badge.configure(fg_color=C["s_fail_bg"]) 
            self.conn_text.configure(text="Offline", text_color=C["s_fail_fg"]) 
            if getattr(self, "_conn_icon_red", None):
                self.conn_icon.configure(image=self._conn_icon_red, text="")
            else:
                self.conn_icon.configure(image=None, text="!", text_color=C["s_fail_fg"]) 

    def _check_models_on_start(self):
        ok, msg, missing = model_manager.health_check()
        if not ok:
            if any(m == "yolo" or m.startswith("ollama:") for m in missing):
                ans = messagebox.askyesno("AI Models Missing", 
                                        f"Some AI models are missing:\n\n{msg}\n\n"
                                        "Would you like to download/pull them now?")
                if ans:
                    self._show_model_downloader()
            else:
                messagebox.showwarning("AI Service Error", msg)
        else:
            print(f"✅ {msg}")

    def _check_models_manual(self):
        self._set_conn_visual("connecting")
        ok, msg, missing = model_manager.health_check()
        if ok:
            messagebox.showinfo("Health Check", "All systems active!")
            self._set_conn_visual("active" if self.db_connector and self.db_connector.connected else "neutral")
        else:
            ans = messagebox.askyesno("Health Check Result", f"Issues found:\n\n{msg}\n\nTry to fix/download?")
            if ans:
                self._show_model_downloader()
            else:
                self._set_conn_visual("failed")
    
    def _show_model_downloader(self):
        win = ctk.CTkToplevel(self)
        win.title("Download Status")
        w_px, h_px = self._px(550), self._px(220)
        win.geometry(f"{w_px}x{h_px}")
        win.attributes("-topmost", True)
        win.resizable(False, False)
        win.configure(fg_color=C["bg"])
        
        # Center on parent
        self.update_idletasks()
        px, py = self.winfo_x(), self.winfo_y()
        pw, ph = self.winfo_width(), self.winfo_height()
        win.geometry(f"+{px + (pw - w_px)//2}+{py + (ph - h_px)//2}")

        lbl = ctk.CTkLabel(win, text="Preparing models...", 
                           font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                           text_color=C["text"])
        lbl.pack(pady=(self._px(40), self._px(20)))
        
        prog = ctk.CTkProgressBar(win, width=self._px(360), height=self._px(12),
                                   progress_color=C["olive"], fg_color=C["border"])
        prog.pack(pady=self._px(10))
        prog.set(0)
        
        status = ctk.CTkLabel(win, text="Starting download...", 
                              font=ctk.CTkFont(family="Inter", size=self.F["label"]),
                              text_color=C["muted"])
        status.pack(pady=self._px(5))

        def _update_ui(msg, pct):
            try:
                lbl.configure(text=msg)
                prog.set(pct / 100)
                status.configure(text=f"Progress: {pct:.1f}%")
                win.update_idletasks()
            except Exception:
                pass

        def _run():
            try:
                ok, final_msg = model_manager.ensure_models(progress_callback=_update_ui)
                # Close the progress window FIRST
                try: win.destroy()
                except Exception: pass
                
                if ok:
                    messagebox.showinfo("Success", "YOLO weights and AI models successfully downloaded!")
                else:
                    messagebox.showerror("Error", f"Failed to setup models:\n{final_msg}")
            except Exception as e:
                try: win.destroy()
                except: pass
                print(f"Downloader error: {e}")

        threading.Thread(target=_run, daemon=True).start()

    # ── Body (scrollable) ─────────────────────────────────────────────────────
    def _build_body(self):
        self.scroll = ctk.CTkScrollableFrame(self, fg_color=C["bg"], corner_radius=0)
        self.scroll.grid(row=1, column=0, sticky="nsew")
        self.scroll.grid_columnconfigure(0, weight=1)

        # Responsive side padding
        init_pad = 450 if self._win_w > 3000 else max(24, int(self._win_w * 0.06))

        # Dashboard title
        lbl = ctk.CTkLabel(self.scroll, text="Dashboard",
                     font=ctk.CTkFont(family="Inter", size=self.F["section"], weight="bold"),
                     text_color=C["text"])
        self._dashboard_label = self._reg(lbl, 20, "Inter", "bold")
        self._dashboard_label.grid(
            row=0, column=0, sticky="w", padx=init_pad, pady=(28, 18))

        # ── 3 Cards — token card (left, wider) + upload pair (right) ────────
        cards = ctk.CTkFrame(self.scroll, fg_color="transparent")
        cards.grid(row=1, column=0, sticky="ew", padx=init_pad, pady=(0, 24))
        self._cards_frame = cards
        # Use uniform group — all 4 logical "units" same size
        # token card = 2 units wide, each upload card = 1 unit
        # so token card = 50%, each upload = 25%
        cards.grid_columnconfigure(0, weight=40, uniform="col")  # token — 40%
        cards.grid_columnconfigure(1, weight=60, uniform="col")  # upload pair — 60%
        cards.grid_rowconfigure(0, weight=1)

        self._build_token_card(cards)

        # Upload cards sub-frame — each upload gets equal half of col=1
        upload_pair = ctk.CTkFrame(cards, fg_color="transparent")
        upload_pair.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=4)
        upload_pair.grid_columnconfigure(0, weight=1, uniform="up")
        upload_pair.grid_columnconfigure(1, weight=1, uniform="up")
        upload_pair.grid_rowconfigure(0, weight=1)

        self._build_upload_card(upload_pair, col=0,
                                title="Upload Slides",
                                desc="Automatically analyze your scanned slides with AI to extract colors, logos, text, and catalog-ready details.",
                                bg="#8C7B5D", title_color="#FFFFFF", desc_color="#E5E5E5",
                                icon_bg="#AA9874",
                                cmd=lambda: self._browse_folder("slides"),
                                active=False, # Deactivated as requested
                                width=None, height=None, border_color="#E4E7EC",
                                title_size=17, desc_size=14, wrap=218, icon_pady=(0, self._px(48)),
                                icon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "upload_icon.png"))
        self._build_upload_card(upload_pair, col=1,
                                title="Upload Books",
                                desc="Process book cover images to identify titles, authors, and generate complete auction-ready descriptions.",
                                bg="#F5F2EC", title_color="#090909", desc_color="#808080",
                                icon_bg="#AA9874",
                                cmd=lambda: self._browse_folder("books"),
                                width=None, height=None, border_color="#E4E7EC",
                                title_size=17, desc_size=14, wrap=218, icon_pady=(0, self._px(48)),
                                icon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "upload_icon.png"))

        # ── Recent Activity ───────────────────────────────────────────────────
        self._build_activity_section()

        # NOTE: Hidden log box removed as it was causing UI glitches and hang-ups
        # specifically on Linux/Ubuntu when updated rapidly at height=0.
        self.log_box = None

    # ── Token Card ─────────────────────────────────────────────────────────────
    def _build_token_card(self, parent):
        card = ctk.CTkFrame(parent, corner_radius=self._px(10),
                            fg_color=C["white"],
                            border_width=2, border_color=C["border"])
        card.grid(row=0, column=0, padx=(0, 8), pady=4, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=0, column=0, padx=self._px(19), pady=self._px(27), sticky="ew")
        inner.grid_columnconfigure(0, weight=1)

        self._reg(ctk.CTkLabel(inner, text="Enter Secure Connection Token",
                     font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                     text_color="#293751",
                     anchor="w"), 16, "Inter", "bold").grid(row=0, column=0, sticky="ew", pady=(0, 8))

        token_desc = self._reg(ctk.CTkLabel(inner,
                     text="Paste the token provided from the web app to securely\nlink this desktop client to your cloud OCR system.",
                     font=ctk.CTkFont(family="Outfit", size=self._fs(14)),
                     text_color="#000000",
                     justify="left", anchor="w"), 14, "Outfit", "normal")
        token_desc.grid(row=1, column=0, sticky="ew", pady=(0, 16))

        def _token_wrap(e=None):
            try:
                token_desc.configure(wraplength=max(220, inner.winfo_width() - 24))
            except Exception:
                pass

        inner.bind("<Configure>", _token_wrap, add="+")
        _token_wrap()

        self._reg(ctk.CTkLabel(inner, text="Token",
                     font=ctk.CTkFont(family="Inter", size=self.F["label"]),
                     text_color="#000000",
                     anchor="w"), 13, "Inter", "normal").grid(row=2, column=0, sticky="ew", pady=(0, 4))

        self.token_entry = ctk.CTkEntry(
            inner,
            placeholder_text="Paste your secure connection key",
            height=self._px(50), corner_radius=self._px(10),
            border_width=1, border_color=C["border"],
            fg_color=C["card"],
            font=ctk.CTkFont(family="Inter", size=self.F["input"]))
        self._reg(self.token_entry, 13, "Inter", "normal")
        self.token_entry.grid(row=3, column=0, sticky="ew", pady=(12, 12))
        try:
            self.token_entry.bind("<Key>", lambda e: self.token_entry.configure(border_color=C["border"]))
            # USER REQUEST: Select All (Ctrl+A)
            def _select_all(e):
                self.token_entry.select_range(0, 'end')
                self.token_entry.icursor('end')
                return "break" # Prevent default handling
            self.token_entry.bind("<Control-a>", _select_all)
            self.token_entry.bind("<Control-A>", _select_all)
            # USER REQUEST: Enter to Verify
            self.token_entry.bind("<Return>", lambda e: self._test_conn())
        except Exception:
            pass

        self.btn_token = ctk.CTkButton(
            inner, text="Continue",
            height=self._px(50), corner_radius=self._px(10),
            font=ctk.CTkFont(family="Raleway", size=self._fs(16), weight="bold"),
            fg_color="#8C7B5D", hover_color="#7A6B50",
            text_color="white",
            command=self._test_conn)
        self._reg(self.btn_token, 16, "Raleway", "bold")
        self.btn_token.grid(row=4, column=0, sticky="ew")

    # ── Upload Card ────────────────────────────────────────────────────────────
    def _build_upload_card(self, parent, col, title, desc,
                           bg, title_color, desc_color, icon_bg, cmd,
                           width=None, height=None, border_color=None,
                           title_size=None, desc_size=None, wrap=None, icon_pady=None,
                           icon_path=None, active=True, show_manual=False):
        # Outer card — use grid row=0 in parent
        card = ctk.CTkFrame(parent, corner_radius=self._px(16), fg_color=bg,
                            border_width=1, border_color=border_color or C["border"])
        card.grid(row=0, column=col, padx=8, pady=4, sticky="nsew")
        card.grid_rowconfigure(0, weight=1)
        card.grid_columnconfigure(0, weight=1)
        
        # If not active, change appearance and disable command
        if not active:
            card.configure(fg_color="#F5F5F5", border_color="#E0E0E0")
            title_color = "#A0A0A0"
            desc_color = "#C0C0C0"
            icon_bg = "#E0E0E0"
            cmd = lambda: None # Disable command

        def _cb(_e=None):
            try:
                cmd()
            except Exception:
                pass

        # Content frame inside card
        cf = ctk.CTkFrame(card, fg_color="transparent")
        cf.grid(row=0, column=0, sticky="nsew", padx=self._px(24), pady=self._px(24))
        cf.grid_columnconfigure(0, weight=1)

        # Row 0: icon
        icon_box = ctk.CTkFrame(cf, width=self._px(48), height=self._px(48),
                                corner_radius=self._px(12), fg_color=icon_bg)
        icon_box.grid(row=0, column=0, sticky="w", pady=icon_pady or (0, 16))
        icon_box.grid_propagate(False)
        _icon_px = self._px(24)
        if icon_path and os.path.exists(icon_path):
            if not hasattr(self, "_img_cache"): self._img_cache = {}
            key = (icon_path, _icon_px, _icon_px)
            if key not in self._img_cache:
                try:
                    from PIL import Image
                    self._img_cache[key] = ctk.CTkImage(
                        light_image=Image.open(icon_path),
                        size=(_icon_px, _icon_px))
                except Exception:
                    try:
                        self._img_cache[key] = PhotoImage(file=icon_path)
                    except Exception:
                        self._img_cache[key] = None
            img = self._img_cache.get(key)
            if img is not None:
                ctk.CTkLabel(icon_box, text="", image=img,
                             fg_color="transparent").place(relx=0.5, rely=0.5, anchor="center")
            else:
                ctk.CTkLabel(icon_box, text="⬆",
                             font=ctk.CTkFont(size=_icon_px),
                             fg_color="transparent",
                             text_color="white").place(relx=0.5, rely=0.5, anchor="center")
        else:
            ctk.CTkLabel(icon_box, text="⬆",
                         font=ctk.CTkFont(size=_icon_px),
                         fg_color="transparent",
                         text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        # Row 1: title
        _title_base = title_size or 16
        _title_scaled = self._fs(_title_base)
        _title_lbl = self._reg(ctk.CTkLabel(cf, text=title,
                     font=ctk.CTkFont(family="Inter", size=_title_scaled, weight="bold"),
                     text_color=title_color,
                     anchor="w", justify="left"), _title_base, "Inter", "bold")
        _title_lbl.grid(row=1, column=0, sticky="ew", pady=(0, self._px(8)))

        _desc_base = desc_size or 11
        _desc_scaled = self._fs(_desc_base)
        _desc_lbl = self._reg(ctk.CTkLabel(cf, text=desc,
                     font=ctk.CTkFont(family="Outfit", size=_desc_scaled),
                     text_color=desc_color,
                     wraplength=180, justify="left",
                     anchor="w"), _desc_base, "Outfit", "normal")
        _desc_lbl.grid(row=2, column=0, sticky="ew")

        if show_manual:
            m_btn = ctk.CTkButton(cf, text="Manual Image Selection",
                                  height=self._px(32), corner_radius=self._px(8),
                                  font=ctk.CTkFont(family="Inter", size=self._fs(12), weight="bold"),
                                  fg_color="transparent", border_width=1, border_color=C["border"],
                                  text_color=C["muted"], hover_color="#F0F0F0",
                                  command=self._browse_manual)
            m_btn.grid(row=3, column=0, sticky="w", pady=(12, 0))

        def _update_wrap(e=None):
            try:
                w = max(140, cf.winfo_width() - 48)
                _title_lbl.configure(wraplength=w)
                _desc_lbl.configure(wraplength=w)
            except Exception:
                pass
        cf.bind("<Configure>", _update_wrap, add="+")
        _update_wrap()

        def _bind_all(w):
            try:
                w.configure(cursor="hand2" if active else "arrow") # Cursor change based on active
                w.bind("<Button-1>", _cb if active else lambda e: None) # Bind only if active
            except Exception:
                pass
            for ch in getattr(w, "winfo_children", lambda: [])():
                _bind_all(ch)
        _bind_all(card)

    # ── Recent Activity ────────────────────────────────────────────────────────
    def _build_activity_section(self):
        init_pad = 450 if self._win_w > 3000 else max(176, int(self._win_w * 0.13))
        self.act_frame = ctk.CTkFrame(self.scroll,
                                      fg_color=C["white"],
                                      border_width=1, border_color=C["border"],
                                      corner_radius=self._px(12))
        self.act_frame.grid(row=2, column=0, sticky="ew", padx=init_pad, pady=(0, 32))
        self._act_frame_outer = self.act_frame
        self.act_frame.grid_columnconfigure(0, weight=1)

        # Section title
        self._reg(ctk.CTkLabel(self.act_frame, text="Recent Activity",
                     font=ctk.CTkFont(family="Outfit", size=self.F["heading"], weight="bold"),
                     text_color=C["text"]), 16, "Outfit", "bold").pack(anchor="w", padx=20, pady=(16, 0))
        ctk.CTkFrame(self.act_frame, height=1, fg_color=C["border"]).pack(fill="x", padx=20, pady=(8, 0))

        # Table header
        th = ctk.CTkFrame(self.act_frame, fg_color="transparent", height=self._px(35))
        th.pack(fill="x", padx=20, pady=(8, 0))
        th.grid_columnconfigure(0, weight=4, uniform="th")
        th.grid_columnconfigure(1, weight=3, uniform="th")
        th.grid_columnconfigure(2, weight=2, uniform="th")
        th.grid_columnconfigure(3, weight=2, uniform="th")
        self._table_header = th

        for i, col_name in enumerate(["File Name", "Status", "Type", "Upload Time"]):
            self._reg(ctk.CTkLabel(th, text=col_name,
                         font=ctk.CTkFont(family="Outfit", size=self.F["table_h"], weight="bold"),
                         text_color="#000000", anchor="w"), 13, "Outfit", "bold").grid(
                row=0, column=i, sticky="ew", padx=(4, 0), pady=8)

        # Header divider
        ctk.CTkFrame(self.act_frame, height=2, fg_color="#D0D5DD").pack(fill="x", padx=20)

        # Rows container — CTkScrollableFrame same as original
        self.rows_container = ctk.CTkScrollableFrame(self.act_frame, fg_color="#FFFFFF", height=self._px(380),
                                                    scrollbar_button_color=C["olive"], 
                                                    scrollbar_button_hover_color=C["olive_h"])
        self.rows_container.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        self.rows_container.grid_columnconfigure(0, weight=4, uniform="rc")
        self.rows_container.grid_columnconfigure(1, weight=3, uniform="rc")
        self.rows_container.grid_columnconfigure(2, weight=2, uniform="rc")
        self.rows_container.grid_columnconfigure(3, weight=2, uniform="rc")

        # Demo rows removed; rows will appear during sync

    def _add_activity_row(self, book_id, status, type_, timestamp):
        """Add a new row to the activity table — optimized for performance."""
        bg, fg = STATUS_STYLES.get(status, (C["s_o_bg"], C["s_o_fg"]))

        # Row frame
        _pad = self._px(22)
        row = ctk.CTkFrame(self.rows_container, fg_color="#FFFFFF")
        row.pack(fill="x")
        row.grid_columnconfigure(0, weight=4, uniform="rc")
        row.grid_columnconfigure(1, weight=3, uniform="rc")
        row.grid_columnconfigure(2, weight=2, uniform="rc")
        row.grid_columnconfigure(3, weight=2, uniform="rc")

        # File name
        name_lbl = ctk.CTkLabel(row, text=book_id,
                                font=ctk.CTkFont(family="Outfit",
                                                 size=self.F["table_b"], weight="bold"),
                                text_color=C["text"], anchor="w")
        name_lbl.grid(row=0, column=0, sticky="ew", padx=(4, 0), pady=_pad)

        # Status badge
        badge = ctk.CTkLabel(row, text=status,
                             width=self._px(80), height=self._px(23), corner_radius=self._px(12),
                             fg_color=bg, text_color=fg,
                             font=ctk.CTkFont(family="Outfit",
                                              size=self.F["badge"], weight="normal"),
                             padx=12, anchor="center") # Centered for better look
        badge.grid(row=0, column=1, sticky="w", pady=_pad)

        # Type
        type_lbl = ctk.CTkLabel(row, text=type_,
                                font=ctk.CTkFont(family="Outfit", size=self.F["table_b"]),
                                text_color=C["muted"], anchor="w")
        type_lbl.grid(row=0, column=2, sticky="ew", pady=_pad)

        # Time
        time_lbl = ctk.CTkLabel(row, text=timestamp,
                                font=ctk.CTkFont(family="Outfit", size=self.F["table_b"]),
                                text_color=C["muted"], anchor="w")
        time_lbl.grid(row=0, column=3, sticky="ew", pady=_pad)

        # Divider
        ctk.CTkFrame(self.rows_container, height=2, fg_color="#D0D5DD").pack(fill="x")

        # Recursive binding for scroll events
        def _bind_mouse_wheel(widget):
            # Universal binding for both Linux and Windows/macOS
            if sys.platform.startswith("linux"):
                widget.bind("<Button-4>", lambda e: self.rows_container._parent_canvas.yview_scroll(-1, "units"), add="+")
                widget.bind("<Button-5>", lambda e: self.rows_container._parent_canvas.yview_scroll(1, "units"), add="+")
            else:
                widget.bind("<MouseWheel>", lambda e: self.rows_container._parent_canvas.yview_scroll(int(-1*(e.delta/120)), "units"), add="+")
            
            for child in widget.winfo_children():
                _bind_mouse_wheel(child)
        _bind_mouse_wheel(row)

        def _open_detail(_e=None):
            # Check current status from badge
            current_status = badge.cget("text")
            if current_status not in ["Complete", "Skipped"]:
                self._log(f"ℹ️ Detail view unavailable: {book_id} is {current_status}")
                return

            try:
                coll = self.config.get("collection", "Book Data")
                # Use a small wait/after to ensure UI responsiveness if needed,
                # but for simplicity we keep it here for now as it's a single read.
                doc = self.db_connector.db[coll].find_one({"book_id": book_id}) if (self.db_connector and self.db_connector.connected) else None
            except Exception:
                doc = None

            # DEBOUNCE: Check if a detail window for this book is already open
            existing_win = self._detail_windows.get(book_id)
            if existing_win and existing_win.winfo_exists():
                existing_win.focus_set()
                existing_win.lift()
                # On some Linux WMs, lift() might not be enough; deiconify helps
                if existing_win.state() == "iconic":
                    existing_win.deiconify()
                return

            win = ctk.CTkToplevel(self)
            self._detail_windows[book_id] = win
            
            def _on_close():
                if book_id in self._detail_windows:
                    del self._detail_windows[book_id]
                win.destroy()
            
            win.protocol("WM_DELETE_WINDOW", _on_close)
            win.title(f"Details — {book_id}")
            win.withdraw()  # Hide immediately to prevent flickering during setup

            # Proportional sizing — 70% of screen
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w, h   = int(sw * 0.70), int(sh * 0.70)
            win.geometry(f"{w}x{h}")
            win.minsize(w, h)
            win.resizable(True, True)
            win.initial_w = w
            win.initial_h = h
            # Transient windows often lack maximize buttons on Linux; disable it for detail view
            # win.transient(self) 
            
            win.grid_columnconfigure(0, weight=1)
            win.grid_rowconfigure(1, weight=1)

            topbar = ctk.CTkFrame(win, fg_color=C["white"], corner_radius=0)
            topbar.grid(row=0, column=0, sticky="ew")
            fit_var = ctk.BooleanVar(value=True)
            thumbs_var = ctk.BooleanVar(value=True)
            info_var = ctk.BooleanVar(value=True)
            
            # Larger controls for better accessibility
            # "Fit Image" button removed as requested
            ctk.CTkCheckBox(topbar, text="Show Thumbnails", variable=thumbs_var,
                            font=ctk.CTkFont(family="Inter", size=self.F["heading"])).pack(side="left", padx=16, pady=12)
            # Pack controls
            # for wdg in topbar.winfo_children(): # This loop is now redundant as the checkbox is packed directly
            #     try:
            #         wdg.pack(side="left", padx=16, pady=12)
            #     except Exception:
            #         pass
            
            def _toggle_info():
                if info_var.get():
                    info.grid_remove()
                    toggle_btn.configure(text="Show Info")
                    info_var.set(False)
                else:
                    info.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
                    toggle_btn.configure(text="Hide Info")
                    info_var.set(True)

                win.zoom_factor = 1.0
                _sync_detail_split()
                if hasattr(win, "_render_main_cmd"):
                    win._render_main_cmd()

            def _on_mouse_wheel(event):
                px, py = win.winfo_pointerx(), win.winfo_pointery()

                def _inside(w):
                    if not w or not w.winfo_exists() or not w.winfo_ismapped():
                        return False
                    x1, y1 = w.winfo_rootx(), w.winfo_rooty()
                    x2, y2 = x1 + w.winfo_width(), y1 + w.winfo_height()
                    return x1 <= px <= x2 and y1 <= py <= y2

                if _inside(preview):
                    if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
                        win.zoom_factor = min(5.0, win.zoom_factor * 1.1)
                    elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
                        win.zoom_factor = max(1.0, win.zoom_factor / 1.1)
                    if win.zoom_factor <= 1.01:
                        win.zoom_factor = 1.0
                        win.pan_x = 0.5
                        win.pan_y = 0.5
                    if hasattr(win, "_render_main_cmd"):
                        win._render_main_cmd()
                    return "break"

                sc = getattr(win, "_info_scroll_canvas", None)
                if _inside(info) and sc is not None:
                    if event.num == 4:
                        step = -4
                    elif event.num == 5:
                        step = 4
                    else:
                        delta = getattr(event, "delta", 0)
                        blocks = max(1, int(abs(delta) / 120))
                        step = -4 * blocks if delta > 0 else 4 * blocks
                    sc.yview_scroll(step, "units")
                    return "break"

            win.zoom_factor = 1.0
            win.pan_x = 0.5  # Normalized center (0.0 = left edge, 1.0 = right edge)
            win.pan_y = 0.5  # Normalized center (0.0 = top edge, 1.0 = bottom edge)
            win._drag_start = None  # Track drag start position

            def _bind_zoom(w):
                w.bind("<MouseWheel>", _on_mouse_wheel, add="+")
                w.bind("<Button-4>", _on_mouse_wheel, add="+")
                w.bind("<Button-5>", _on_mouse_wheel, add="+")

            win.bind("<MouseWheel>", _on_mouse_wheel, add="+")
            win.bind("<Button-4>", _on_mouse_wheel, add="+")
            win.bind("<Button-5>", _on_mouse_wheel, add="+")

            toggle_btn = ctk.CTkButton(topbar, text="Hide Info", height=40, corner_radius=10,
                                       font=ctk.CTkFont(family="Inter", size=self.F["btn"], weight="bold"),
                                       fg_color=C["olive"], hover_color=C["olive_h"], text_color="white",
                                       command=_toggle_info)
            toggle_btn.pack(side="left", padx=16)
            ctk.CTkButton(topbar, text="Close", height=42, corner_radius=12,
                          font=ctk.CTkFont(family="Inter", size=self.F["btn"], weight="bold"),
                          fg_color=C["olive"], hover_color=C["olive_h"], text_color="white",
                          command=lambda: _close_win()).pack(side="right", padx=20)

            body = ctk.CTkFrame(win, fg_color=C["white"], corner_radius=0)
            body.grid(row=1, column=0, sticky="nsew")
            body.grid_columnconfigure(0, weight=55)
            body.grid_columnconfigure(1, weight=45)
            body.grid_rowconfigure(0, weight=1)
            body.grid_rowconfigure(1, weight=0)
            body.grid_propagate(False)

            preview = ctk.CTkFrame(body, fg_color=C["white"], corner_radius=10)
            preview.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
            preview.grid_propagate(False)

            # --- Layout Stabilization Loader ---
            loader_container = ctk.CTkFrame(preview, fg_color="transparent")
            loader_container.place(relx=0.5, rely=0.5, anchor="center")

            spinner_canvas = Canvas(loader_container, width=64, height=64, bd=0, highlightthickness=0, bg=C["white"])
            spinner_canvas.pack(pady=(0, 10))
            loader_arc = spinner_canvas.create_arc(8, 8, 56, 56, start=0, extent=300, style="arc", outline=C["olive"], width=6)

            ctk.CTkLabel(loader_container, text="STABILIZING LAYOUT...", 
                         font=ctk.CTkFont(family="Inter", size=16, weight="bold"),
                         text_color=C["muted"]).pack(pady=(0, 2))

            spinner_state = {"angle": 0, "job": None}

            def _close_win():
                """Smooth closing: cancel timers, hide instantly, then destroy."""
                job = spinner_state.get("job")
                if job:
                    try: win.after_cancel(job)
                    except: pass
                win.withdraw()
                win.destroy()

            win.protocol("WM_DELETE_WINDOW", _close_win)

            def _spin_loader():
                if not win.winfo_exists() or not loader_container.winfo_exists():
                    return
                spinner_state["angle"] = (spinner_state["angle"] + 14) % 360
                spinner_canvas.itemconfigure(loader_arc, start=spinner_state["angle"])
                spinner_state["job"] = win.after(28, _spin_loader)

            def _stop_loader():
                job = spinner_state.get("job")
                if job:
                    try:
                        win.after_cancel(job)
                    except Exception:
                        pass
                spinner_state["job"] = None
                try:
                    loader_container.destroy()
                except Exception:
                    pass

            _spin_loader()
            info = ctk.CTkFrame(body, fg_color=C["white"], corner_radius=10,
                                 border_width=1, border_color=C["border"])
            info.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
            info.grid_propagate(False)
            status_lbl = ctk.CTkLabel(win, text="", font=ctk.CTkFont(family="Outfit", size=self.F["muted"]),
                                  text_color=C["muted"], anchor="w", justify="left") 
            status_lbl.grid(row=2, column=0, sticky="ew", padx=12, pady=(0,8))
            win._info_scroll_canvas = None

            def _update_status_wrap(_e=None):
                try:
                    status_lbl.configure(wraplength=max(240, win.winfo_width() - 32))
                except Exception:
                    pass

            def _sync_detail_split(_e=None):
                try:
                    bw = max(360, body.winfo_width() - 24)
                except Exception:
                    return
                if info_var.get():
                    left = int(bw * 0.55)
                    left = max(220, min(left, bw - 220))
                    right = max(220, bw - left)
                    body.grid_columnconfigure(0, weight=0, minsize=left)
                    body.grid_columnconfigure(1, weight=0, minsize=right)
                else:
                    body.grid_columnconfigure(0, weight=1, minsize=max(220, bw))
                    body.grid_columnconfigure(1, weight=0, minsize=0)

            win.bind("<Configure>", _update_status_wrap, add="+")
            body.bind("<Configure>", _sync_detail_split, add="+")

            thumbs_wrap = ctk.CTkFrame(body, fg_color="transparent")
            thumbs_wrap.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0,12))
            _sync_detail_split()
            
            # Show window and process images after a short delay to keep UI snappy
            def _deferred_init():
                if not win.winfo_exists(): return
                
                # Center properly
                win.update_idletasks()
                # Use current app pos + size to center relative to main window
                ax, ay = self.winfo_x(), self.winfo_y()
                aw, ah = self.winfo_width(), self.winfo_height()
                # Use initial values if winfo returns stale 1x1 coords
                ww = getattr(win, "initial_w", win.winfo_width())
                wh = getattr(win, "initial_h", win.winfo_height())
                nx = max(0, ax + (aw - ww) // 2)
                ny = max(0, ay + (ah - wh) // 2)
                
                # Combine size and position to enforce centering immediately
                win.geometry(f"{ww}x{wh}+{nx}+{ny}")
                win.deiconify()
                # Force a full layout pass AFTER deiconify to fix "jumping" glitch
                win.update()
                
                # Critical: Grab set can cause hangs on some Linux WMs if called too early
                try: win.grab_set()
                except: pass

                def _wait_layout_ready(attempt=0):
                    if not win.winfo_exists():
                        return
                    win.update_idletasks()
                    if preview.winfo_width() >= 240 and preview.winfo_height() >= 240:
                        _stop_loader()
                        _init_images()
                        return
                    if attempt < 30:
                        win.after(40, lambda: _wait_layout_ready(attempt + 1))
                    else:
                        _stop_loader()
                        _init_images()

                win.after(40, _wait_layout_ready)

            def _init_images():
                try:
                    from PIL import Image, ExifTags, ImageDraw
                    PIL_OK = True
                except Exception:
                    PIL_OK = False

                items = []
                if doc:
                    if doc.get("front_cover"): items.append((doc["front_cover"], "front_cover")) 
                    if doc.get("back_cover"): items.append((doc["back_cover"], "back_cover")) 
                    for it in doc.get("interior_pages", []): items.append((it, it.get("type","interior")))

                search_roots = []
                for key in ("folder_path", "books_path", "slides_path"):
                    root = (self.config.get(key, "") or "").strip()
                    if root and os.path.isdir(root) and root not in search_roots:
                        search_roots.append(root)

                unresolved = []
                paths = []
                for ent, _ in items:
                    raw = (ent.get("file_path") or ent.get("file_name") or "").strip()
                    if raw and os.path.exists(raw):
                        paths.append(raw)
                    else:
                        unresolved.append((ent, raw))

                if unresolved and search_roots:
                    idx_exact = {}
                    idx_noext = {}
                    for root in search_roots:
                        for r, _, files in os.walk(root):
                            for fn in files:
                                full = os.path.join(r, fn)
                                lk = fn.lower()
                                if lk not in idx_exact:
                                    idx_exact[lk] = full
                                stem = os.path.splitext(lk)[0]
                                if stem not in idx_noext:
                                    idx_noext[stem] = full

                    for ent, raw in unresolved:
                        candidates = []
                        f_name = (ent.get("file_name") or "").strip()
                        p_id = (ent.get("page_id") or "").strip()
                        raw_base = os.path.basename((raw or "").strip())
                        if f_name:
                            candidates.append(f_name)
                        if raw_base:
                            candidates.append(raw_base)
                        if p_id:
                            candidates.append(p_id)

                        found = None
                        for cand in candidates:
                            lk = cand.lower()
                            found = idx_exact.get(lk) or idx_noext.get(os.path.splitext(lk)[0])
                            if found and os.path.exists(found):
                                break

                        if not found and p_id:
                            pid = p_id.lower()
                            for fn_l, full in idx_exact.items():
                                stem = os.path.splitext(fn_l)[0]
                                if stem == pid or stem.startswith(pid + "_"):
                                    found = full
                                    break

                        if not found and book_id:
                            bid = str(book_id).lower()
                            for fn_l, full in idx_exact.items():
                                stem = os.path.splitext(fn_l)[0]
                                if stem == bid or stem.startswith(bid + "_"):
                                    found = full
                                    break

                        if found and os.path.exists(found):
                            paths.append(found)

                if paths:
                    uniq = []
                    seen = set()
                    for p in paths:
                        np = os.path.normcase(os.path.normpath(p))
                        if np in seen:
                            continue
                        seen.add(np)
                        uniq.append(p)
                    paths = uniq

                def _meta_for_path(p):
                    np = os.path.normcase(os.path.normpath(p))
                    bp = os.path.basename(p).lower()
                    for ent, typ in items:
                        fp = (ent.get("file_path") or ent.get("file_name") or "").strip()
                        if not fp:
                            continue
                        nfp = os.path.normcase(os.path.normpath(fp))
                        if nfp == np or os.path.basename(fp).lower() == bp:
                            return {
                                "book_id": ent.get("page_id", "").split("_")[0] if ent.get("page_id") else (doc.get("book_id", book_id) if doc else book_id),
                                "page_id": ent.get("page_id") or os.path.basename(fp or p),
                                "type": typ,
                                "file_name": ent.get("file_name") or os.path.basename(fp or p)
                            }
                    return {"book_id": book_id, "page_id": os.path.basename(p), "type": "unknown", "file_name": os.path.basename(p)}

                idx_var = ctk.IntVar(value=0)
                main_lbl = ctk.CTkLabel(preview, text="", anchor="center")
                main_lbl.pack(expand=True, fill="both", padx=0, pady=0)
                _bind_zoom(main_lbl)
                _bind_zoom(preview)

                # --- Mouse drag panning for zoomed image ---
                win._pan_pending = None  # Throttle timer ID

                def _on_drag_start(event):
                    if win.zoom_factor > 1.0:
                        win._drag_start = (event.x, event.y)
                        main_lbl.configure(cursor="fleur")

                def _on_drag_motion(event):
                    if win._drag_start and win.zoom_factor > 1.0:
                        dx = event.x - win._drag_start[0]
                        dy = event.y - win._drag_start[1]
                        win._drag_start = (event.x, event.y)
                        # Convert pixel drag to normalized pan offset
                        sensitivity = 0.002 / win.zoom_factor
                        win.pan_x = max(0.0, min(1.0, win.pan_x - dx * sensitivity * 2))
                        win.pan_y = max(0.0, min(1.0, win.pan_y - dy * sensitivity * 2))
                        # Throttle: schedule render only if not already pending (~30fps)
                        if win._pan_pending is None:
                            win._pan_pending = win.after(33, _flush_pan)

                def _flush_pan():
                    win._pan_pending = None
                    if win.winfo_exists() and hasattr(win, "_render_pan_cmd"):
                        win._render_pan_cmd()

                def _on_drag_end(event):
                    win._drag_start = None
                    if win.zoom_factor > 1.0:
                        main_lbl.configure(cursor="fleur")
                    else:
                        main_lbl.configure(cursor="")

                main_lbl.bind("<ButtonPress-1>", _on_drag_start)
                main_lbl.bind("<B1-Motion>", _on_drag_motion)
                main_lbl.bind("<ButtonRelease-1>", _on_drag_end)

                def _hex_to_rgb(h):
                    h = h.lstrip('#')
                    return tuple(int(h[i:i+2], 16) for i in (0,2,4))
                olive_rgb = _hex_to_rgb(C["olive"])
                olive_h_rgb = _hex_to_rgb(C["olive_h"])
                border_rgb = _hex_to_rgb(C["border"])
                
                def _mk_arrow(side="left", hover=False, dim=48):
                    if not PIL_OK: return None
                    img = Image.new("RGBA", (dim, dim), (0,0,0,0))
                    drw = ImageDraw.Draw(img)
                    if hover:
                        bg = (*border_rgb, 220)
                        drw.rounded_rectangle([0,0,dim,dim], radius=dim//2, fill=bg)
                    ax = dim//2
                    pad = dim//4
                    if side == "left":
                        pts = [(ax+pad//2, pad), (ax-pad//2, dim//2), (ax+pad//2, dim-pad)]
                    else:
                        pts = [(ax-pad//2, pad), (ax+pad//2, dim//2), (ax-pad//2, dim-pad)]
                    drw.polygon(pts, fill=(olive_h_rgb if hover else olive_rgb))
                    return ctk.CTkImage(light_image=img, size=(dim, dim))

                left_img = _mk_arrow("left", hover=False)
                left_img_h = _mk_arrow("left", hover=True)
                right_img = _mk_arrow("right", hover=False)
                right_img_h = _mk_arrow("right", hover=True)
                
                prev_btn = ctk.CTkLabel(preview, text="", image=left_img)
                next_btn = ctk.CTkLabel(preview, text="", image=right_img)
                prev_btn.place(relx=0.03, rely=0.5, anchor="w")
                next_btn.place(relx=0.97, rely=0.5, anchor="e")
                prev_btn.configure(cursor="hand2")
                next_btn.configure(cursor="hand2")
                prev_btn.bind("<Enter>", lambda e: prev_btn.configure(image=left_img_h))
                prev_btn.bind("<Leave>", lambda e: prev_btn.configure(image=left_img))
                next_btn.bind("<Enter>", lambda e: next_btn.configure(image=right_img_h))
                next_btn.bind("<Leave>", lambda e: next_btn.configure(image=right_img))

                _exif_cache = {}
                def _exif_info(p):
                    if p in _exif_cache:
                        return _exif_cache[p]
                    info_lines = []
                    try:
                        sz = os.path.getsize(p)
                        mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
                        info_lines.append(("Path", os.path.basename(p)))
                        info_lines.append(("Size", f"{sz//1024} KB"))
                        info_lines.append(("Modified", mt))
                        if PIL_OK:
                            im = Image.open(p)
                            w, h = im.size
                            info_lines.append(("Resolution", f"{w}x{h}"))
                            exif = getattr(im, "_getexif", lambda: None)() or {}
                            tagmap = getattr(ExifTags, "TAGS", {})
                            for k in (271, 272, 306, 305, 282, 283, 37386, 37385, 33437, 34850, 37383, 37379):
                                if k in exif:
                                    nm = tagmap.get(k, str(k))
                                    info_lines.append((nm, str(exif[k])))
                    except Exception:
                        pass
                    _exif_cache[p] = info_lines
                    return info_lines

                # --- AI Results / Info Panel ---
                has_ai = doc and doc.get("ocr_completed")

                if has_ai:
                    # ── AI Results View ──────────────────────────────
                    # Scrollable to handle long content
                    info_wrap = ctk.CTkFrame(info, fg_color="transparent")
                    info_wrap.pack(fill="both", expand=True, padx=4, pady=4)

                    info_scroll = ctk.CTkScrollableFrame(
                        info_wrap,
                        fg_color="transparent",
                        scrollbar_fg_color="#E5E7EB",
                        scrollbar_button_color=C["olive_dk"],
                        scrollbar_button_hover_color=C["olive"]
                    )
                    info_scroll.pack(side="left", fill="both", expand=True)
                    win._info_scroll_canvas = info_scroll._parent_canvas

                    # -- Modern Card-Based UI --
                    
                    # Section 1: Generated Title (Blue Card)
                    title_card = ctk.CTkFrame(info_scroll, fg_color="#FFFFFF", border_width=1, border_color="#E5E7EB", corner_radius=12)
                    title_card.pack(fill="x", padx=12, pady=(12, 6))

                    # Title Header
                    title_hdr = ctk.CTkFrame(title_card, fg_color="#EFF6FF", height=48, corner_radius=0)
                    title_hdr.pack(fill="x")
                    title_hdr.pack_propagate(False)

                    # Title Badge + Label
                    # --- Dynamic Wrapping Helper ---
                    def _on_label_resize(event, lbl, padding=48):
                        if event and lbl.winfo_exists():
                            lbl.configure(wraplength=max(120, event.width - padding))


                    title_badge_wrap = ctk.CTkFrame(title_hdr, fg_color="transparent")
                    title_badge_wrap.pack(side="left", padx=24)

                    ctk.CTkLabel(title_badge_wrap, text="Ai", width=32, height=32,
                                 corner_radius=6, fg_color="#DBEAFE", text_color="#2563EB",
                                 font=ctk.CTkFont(family="Inter", size=12, weight="bold")
                                 ).pack(side="left")

                    ctk.CTkLabel(title_badge_wrap, text="Generated Title",
                                 font=ctk.CTkFont(family="Inter", size=24, weight="bold"),
                                 text_color="#1E3A8A").pack(side="left", padx=(10, 0))

                    # --- Interactive Title Edit (Premium Popup) ---
                    def _on_edit_title(_e=None):
                        edit_win = ctk.CTkToplevel(win)
                        edit_win.title("Correct Title")
                        edit_win.transient(win)
                        
                        # Balanced Responsive Sizing: Clamp the dimensions
                        edit_win.update_idletasks()
                        pw, ph = win.winfo_width(), win.winfo_height()
                        # Default to reasonable sizes if parent not fully mapped
                        if pw < 400: pw = 1400 
                        if ph < 400: ph = 900
                        
                        nw = int(pw * 0.50) # 50% of parent width
                        nh = int(ph * 0.45) # 45% of parent height
                        # Clamp for elegance: Stay between 600-750 wide and 420-470 high
                        nw = min(750, max(600, nw))
                        nh = min(470, max(420, nh))
                        
                        edit_win.minsize(550, 400)
                        
                        # Center relative to detail view
                        wx, wy = win.winfo_x(), win.winfo_y()
                        nx, ny = wx + (pw - nw)//2, wy + (ph - nh)//2
                        edit_win.geometry(f"{nw}x{nh}+{nx}+{ny}")
                        
                        edit_win.wait_visibility()
                        edit_win.grab_set()

                        ctk.CTkLabel(edit_win, text="✏️ Edit Book Title", 
                                     font=ctk.CTkFont(family="Outfit", size=26, weight="bold"),
                                     text_color="#1E293B").pack(pady=(30, 10))

                        entry_frame = ctk.CTkFrame(edit_win, fg_color="#F8FAFC", corner_radius=12, border_width=1, border_color="#E2E8F0")
                        entry_frame.pack(fill="x", padx=60, pady=10)
                        
                        entry = ctk.CTkEntry(entry_frame, height=60, border_width=0, fg_color="transparent",
                                            font=ctk.CTkFont(family="Inter", size=20, weight="bold"),
                                            placeholder_text="Type corrected title here...")
                        entry.pack(fill="x", padx=20)
                        entry.insert(0, title_lbl.cget("text"))

                        # Subtitle edit row
                        ctk.CTkLabel(edit_win, text="Subtitle", 
                                     font=ctk.CTkFont(family="Inter", size=16),
                                     text_color="#64748B").pack(pady=(10, 5))
                        
                        sub_entry_frame = ctk.CTkFrame(edit_win, fg_color="#F8FAFC", corner_radius=12, border_width=1, border_color="#E2E8F0")
                        sub_entry_frame.pack(fill="x", padx=60, pady=0)
                        
                        sub_entry = ctk.CTkEntry(sub_entry_frame, height=50, border_width=0, fg_color="transparent",
                                                font=ctk.CTkFont(family="Inter", size=18),
                                                placeholder_text="Type corrected subtitle here...")
                        sub_entry.pack(fill="x", padx=20)
                        sub_val = subtitle_lbl.cget("text") if subtitle_lbl else ""
                        sub_entry.insert(0, sub_val)
                        
                        entry.focus()

                        def _save():
                            val = entry.get().strip()
                            sub_val = sub_entry.get().strip()
                            if val:
                                title_lbl.configure(text=val)
                                if subtitle_lbl:
                                    subtitle_lbl.configure(text=sub_val)
                                
                                if self.db_connector and self.db_connector.connected:
                                    try:
                                        update_fields = {"title": val, "subtitle": sub_val}
                                        self.db_connector.db[coll].update_one({"book_id": book_id}, {"$set": update_fields})
                                        self._log(f"✅ Title/Subtitle updated for {book_id}")
                                    except Exception as err: self._log(f"❌ DB Update Error: {err}")
                                edit_win.destroy()

                        btn_row = ctk.CTkFrame(edit_win, fg_color="transparent")
                        btn_row.pack(pady=30)
                        
                        ctk.CTkButton(btn_row, text="Cancel", width=160, height=50, corner_radius=10,
                                     fg_color="#F1F5F9", text_color="#64748B", hover_color="#E2E8F0",
                                     command=edit_win.destroy).pack(side="left", padx=10)
                                     
                        ctk.CTkButton(btn_row, text="Save Changes", width=220, height=50, corner_radius=10,
                                     fg_color="#0F172A", text_color="white", hover_color="#1E293B",
                                     font=ctk.CTkFont(family="Inter", size=18, weight="bold"),
                                     command=_save).pack(side="left", padx=10)
                        
                        # Bind Enter key
                        edit_win.bind("<Return>", lambda e: _save())

                    # Removed _on_edit_desc handler as requested.

                    # Edit Button (Title)
                    title_edit = ctk.CTkLabel(title_hdr, text="✎ Edit",
                                           font=ctk.CTkFont(family="Inter", size=20, weight="normal"),
                                           text_color="#2563EB", cursor="hand2")
                    title_edit.pack(side="right", padx=24)
                    title_edit.bind("<Button-1>", _on_edit_title)

                    # Title Body
                    title_body = ctk.CTkFrame(title_card, fg_color="transparent")
                    title_body.pack(fill="x", padx=24, pady=24)

                    ai_title = doc.get("title", "") or "(Not Found)"
                    title_lbl = ctk.CTkLabel(title_body, text=ai_title,
                                 font=ctk.CTkFont(family="Outfit", size=32, weight="bold"),
                                 text_color="#0F172A", anchor="w",
                                 justify="left", wraplength=420
                                 )
                    title_lbl.pack(anchor="w", fill="x")

                    ai_subtitle = doc.get("subtitle", "")
                    subtitle_lbl = None
                    if ai_subtitle and ai_subtitle != "N/A":
                        subtitle_lbl = ctk.CTkLabel(title_body, text=ai_subtitle,
                                     font=ctk.CTkFont(family="Inter", size=24, weight="normal"),
                                     text_color="#475569", anchor="w",
                                     justify="left", wraplength=420
                                     )
                        subtitle_lbl.pack(anchor="w", fill="x", pady=(8, 0))

                    def _on_title_resize(e, l=title_lbl, sl=subtitle_lbl):
                        w = e.width - 64
                        l.configure(wraplength=w)
                        if sl: sl.configure(wraplength=w)

                    title_card.bind("<Configure>", _on_title_resize, add="+")


                    # Section 2: Generated Description (Purple Card)
                    desc_card = ctk.CTkFrame(info_scroll, fg_color="#FFFFFF", border_width=1, border_color="#E5E7EB", corner_radius=12)
                    desc_card.pack(fill="x", padx=12, pady=6)

                    # Description Header
                    desc_hdr = ctk.CTkFrame(desc_card, fg_color="#FAF5FF", height=48, corner_radius=0)
                    desc_hdr.pack(fill="x")
                    desc_hdr.pack_propagate(False)

                    # Description Badge + Label
                    desc_badge_wrap = ctk.CTkFrame(desc_hdr, fg_color="transparent")
                    desc_badge_wrap.pack(side="left", padx=24)

                    ctk.CTkLabel(desc_badge_wrap, text="Ai", width=32, height=32,
                                 corner_radius=6, fg_color="#F3E8FF", text_color="#9C25EB",
                                 font=ctk.CTkFont(family="Inter", size=12, weight="bold")
                                 ).pack(side="left")

                    ctk.CTkLabel(desc_badge_wrap, text="Generated Description",
                                 font=ctk.CTkFont(family="Inter", size=24, weight="bold"),
                                 text_color="#581C87").pack(side="left", padx=(10, 0))

                    # Removed Edit Button from Description as requested.

                    # Description Body
                    description_val = doc.get("description", "") or "(Not Found)"
                    desc_lbl = ctk.CTkLabel(desc_card, text=description_val,
                                 font=ctk.CTkFont(family="Inter", size=22),
                                 text_color="#000000", anchor="w",
                                 justify="left", wraplength=420
                                 )
                    desc_lbl.pack(anchor="w", padx=24, pady=24, fill="x")
                    desc_card.bind("<Configure>", lambda e, l=desc_lbl: _on_label_resize(e, l, 64), add="+")

                    def _refresh_ai_wrap(_e=None):
                        w = max(220, info.winfo_width() - 72)
                        t_size = max(18, min(42, int(w / 12))) # Allow larger title
                        s_size = max(16, min(30, int(t_size * 0.85))) # Allow larger subtitle (up to 30px)
                        d_size = max(15, min(24, int(w / 22))) # Slightly larger description
                        
                        title_lbl.configure(wraplength=w, font=ctk.CTkFont(family="Outfit", size=t_size, weight="bold"))
                        if subtitle_lbl:
                            subtitle_lbl.configure(wraplength=w, font=ctk.CTkFont(family="Inter", size=s_size))
                        desc_lbl.configure(wraplength=w, font=ctk.CTkFont(family="Inter", size=d_size))

                    info.bind("<Configure>", _refresh_ai_wrap, add="+")
                    _refresh_ai_wrap()


                    # Section 3: Detected Traits (Gray/White Card)
                    traits_card = ctk.CTkFrame(info_scroll, fg_color="#FFFFFF", border_width=1, border_color="#E5E7EB", corner_radius=12)
                    traits_card.pack(fill="x", padx=12, pady=6)

                    # Traits Header
                    traits_hdr = ctk.CTkFrame(traits_card, fg_color="transparent", height=48, corner_radius=0)
                    traits_hdr.pack(fill="x")
                    traits_hdr.pack_propagate(False)

                    ctk.CTkLabel(traits_hdr, text="🏷️ Detected Traits",
                                 font=ctk.CTkFont(family="Inter", size=24, weight="bold"),
                                 text_color="#111827").pack(side="left", padx=24)

                    # Traits Body
                    traits_body = ctk.CTkFrame(traits_card, fg_color="transparent")
                    traits_body.pack(fill="x", padx=24, pady=(0, 24))

                    # 2. Author Subsection
                    ctk.CTkLabel(traits_body, text="Author",
                                 font=ctk.CTkFont(family="Inter", size=20, weight="normal"),
                                 text_color="#6B7280").pack(anchor="w", pady=(0, 10))

                    author_val = doc.get("author", "") or "(Not Found)"
                    author_pill = ctk.CTkFrame(traits_body, fg_color="#EFF6FF", border_width=1, border_color="#BFDBFE", corner_radius=10)
                    author_pill.pack(anchor="w")
                    author_lbl = ctk.CTkLabel(author_pill, text=author_val,
                                 font=ctk.CTkFont(family="Inter", size=20, weight="bold"),
                                 text_color="#1D4ED8", padx=20, pady=12,
                                 justify="left", anchor="w")
                    author_lbl.pack(fill="x")

                    # 3. Edition Subsection
                    ctk.CTkLabel(traits_body, text="Edition",
                                 font=ctk.CTkFont(family="Inter", size=20, weight="normal"),
                                 text_color="#6B7280").pack(anchor="w", pady=(16, 10))
                    
                    edition_val = doc.get("edition", "") or "(Not Found)"
                    edition_pill = ctk.CTkFrame(traits_body, fg_color="#FFFBEB", border_width=1, border_color="#FEF3C7", corner_radius=10)
                    edition_pill.pack(anchor="w")
                    edition_lbl = ctk.CTkLabel(edition_pill, text=edition_val,
                                 font=ctk.CTkFont(family="Inter", size=20, weight="bold"),
                                 text_color="#B45309", padx=20, pady=12,
                                 justify="left", anchor="w")
                    edition_lbl.pack(fill="x")

                    # 4. ISBN Subsection
                    ctk.CTkLabel(traits_body, text="ISBN",
                                 font=ctk.CTkFont(family="Inter", size=20, weight="normal"),
                                 text_color="#6B7280").pack(anchor="w", pady=(16, 10))
                    
                    isbn_val = doc.get("isbn", "") or "(Not Found)"
                    isbn_pill = ctk.CTkFrame(traits_body, fg_color="#F3F4F6", border_width=1, border_color="#D1D5DB", corner_radius=10)
                    isbn_pill.pack(anchor="w")
                    isbn_lbl = ctk.CTkLabel(isbn_pill, text=isbn_val,
                                 font=ctk.CTkFont(family="Inter", size=20, weight="bold"),
                                 text_color="#374151", padx=20, pady=12,
                                 justify="left", anchor="w")
                    isbn_lbl.pack(fill="x")

                    def _update_trait_wrap(e=None):
                        wv = max(160, traits_card.winfo_width() - 100)
                        try:
                            author_lbl.configure(wraplength=wv)
                            edition_lbl.configure(wraplength=wv)
                            isbn_lbl.configure(wraplength=wv)
                        except Exception:
                            pass

                    traits_card.bind("<Configure>", _update_trait_wrap, add="+")
                    _update_trait_wrap()

                    ctk.CTkLabel(info_scroll, text=f"Book ID: {book_id}",
                                 font=ctk.CTkFont(family="Inter", size=18),
                                 text_color="#9CA3AF").pack(anchor="w", padx=24, pady=12)

                else:
                    info_wrap = ctk.CTkFrame(info, fg_color="transparent")
                    info_wrap.pack(fill="both", expand=True, padx=4, pady=4)

                    info_scroll = ctk.CTkScrollableFrame(
                        info_wrap,
                        fg_color="transparent",
                        scrollbar_fg_color="#E5E7EB",
                        scrollbar_button_color=C["olive_dk"],
                        scrollbar_button_hover_color=C["olive"]
                    )
                    info_scroll.pack(side="left", fill="both", expand=True)

                    info_vsb = ctk.CTkScrollbar(
                        info_wrap,
                        orientation="vertical",
                        width=16,
                        fg_color="#F3F4F6",
                        button_color=C["olive_dk"],
                        button_hover_color=C["olive"]
                    )
                    info_vsb.pack(side="right", fill="y", padx=(6, 0))
                    info_scroll._parent_canvas.configure(yscrollcommand=info_vsb.set)
                    info_vsb.configure(command=info_scroll._parent_canvas.yview)
                    win._info_scroll_canvas = info_scroll._parent_canvas

                    _info_title_lbl = ctk.CTkLabel(info_scroll, text="Image Info",
                                     font=ctk.CTkFont(family="Outfit", size=self.F["heading"], weight="bold"),
                                     text_color=C["text"])
                    _info_title_lbl.pack(anchor="w", padx=12, pady=(12,6))

                    _info_meta_labels = []
                    for _ in range(3):
                        lbl = ctk.CTkLabel(info_scroll, text="",
                                           font=ctk.CTkFont(family="Inter", size=self.F["label"]),
                                           text_color=C["hdr"], anchor="w", justify="left")
                        lbl.pack(anchor="w", padx=12, fill="x")
                        _info_meta_labels.append(lbl)

                    _info_exif_labels = []
                    for _ in range(12):
                        lbl = ctk.CTkLabel(info_scroll, text="",
                                           font=ctk.CTkFont(family="Inter", size=self.F["label"]),
                                           text_color=C["muted"], anchor="w", justify="left")
                        lbl.pack(anchor="w", padx=12, fill="x")
                        _info_exif_labels.append(lbl)

                def _render_info(p):
                    if has_ai:
                        return

                    def _clip(v, n=64):
                        s = str(v or "")
                        return s if len(s) <= n else (s[:n - 1] + "…")

                    m = _meta_for_path(p)
                    meta_items = [("Book ID", _clip(m["book_id"], 40)), ("Page ID", _clip(m["page_id"], 40)), ("Type", _clip(m["type"], 30))]
                    for i, (k, v) in enumerate(meta_items):
                        _info_meta_labels[i].configure(text=f"{k}: {v}")

                    exif_items = _exif_info(p)
                    for i, lbl in enumerate(_info_exif_labels):
                        if i < len(exif_items):
                            k, v = exif_items[i]
                            lbl.configure(text=f"{_clip(k, 24)}: {_clip(v, 72)}")
                        else:
                            lbl.configure(text="")

                # --- PIL image cache for instant navigation ---
                _pil_cache = {}

                def _render_main():
                    if not win.winfo_exists(): return
                    i = max(0, min(idx_var.get(), len(paths) - 1))
                    if not paths: return
                    p = paths[i]
                    bn = os.path.basename(p)
                    status_lbl.configure(text=bn if len(bn) <= 90 else (bn[:89] + "…"))
                    if PIL_OK:
                        try:
                            if p not in _pil_cache:
                                _pil_cache[p] = Image.open(p)
                            im = _pil_cache[p]
                            # Use actual frame dimensions for perfect fit
                            # If winfo_width is not yet mapped (1), fallback to a proportional estimate
                            cur_w = preview.winfo_width()
                            cur_h = preview.winfo_height()
                            if cur_w < 120 or cur_h < 120:
                                win.after(40, _render_main)
                                return
                            pw = max(320, cur_w - 20)
                            ph = max(360, cur_h - 20)

                            w, h = im.size
                            
                            # Apply zoom factor
                            zf = getattr(win, "zoom_factor", 1.0)
                            
                            if zf > 1.0:
                                # ZOOM MODE: crop a portion of the original image
                                # using pan offsets for navigation
                                crop_w = int(w / zf)
                                crop_h = int(h / zf)
                                # Use pan offsets (0.0-1.0) to position the crop window
                                cx = int(win.pan_x * w)
                                cy = int(win.pan_y * h)
                                x1 = max(0, cx - crop_w // 2)
                                y1 = max(0, cy - crop_h // 2)
                                # Clamp to image bounds
                                if x1 + crop_w > w: x1 = w - crop_w
                                if y1 + crop_h > h: y1 = h - crop_h
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(w, x1 + crop_w)
                                y2 = min(h, y1 + crop_h)
                                cropped = im.crop((x1, y1, x2, y2))
                                # Fit the cropped portion into the panel
                                cw, ch = cropped.size
                                r = min(pw / float(cw), ph / float(ch))
                                sz = (max(1, int(cw * r)), max(1, int(ch * r)))
                                img = ctk.CTkImage(light_image=cropped, size=sz)
                                main_lbl.configure(cursor="fleur")  # Drag cursor when zoomed
                            else:
                                # NORMAL MODE: fit whole image to panel
                                if fit_var.get():
                                    r = min(pw/float(w), ph/float(h))
                                else:
                                    r = min(1.0, min(pw/float(w), ph/float(h)))
                                sz = (max(1, int(w*r)), max(1, int(h*r)))
                                img = ctk.CTkImage(light_image=im, size=sz)
                                main_lbl.configure(cursor="")  # Normal cursor
                            
                            main_lbl.configure(image=img)
                            main_lbl.image = img
                            main_lbl.configure(text="")
                        except Exception:
                            main_lbl.configure(text="Preview unavailable")
                    else:
                        main_lbl.configure(text="Install Pillow for previews (pip install pillow)")
                    _render_info(p)

                # Store refresh command for toggle access
                win._render_main_cmd = _render_main

                # Lightweight pan-only render (skips _render_info to prevent flicker)
                def _render_pan_only():
                    if not win.winfo_exists(): return
                    i = max(0, min(idx_var.get(), len(paths) - 1))
                    if not paths: return
                    p = paths[i]
                    zf = getattr(win, "zoom_factor", 1.0)
                    if zf <= 1.0 or not PIL_OK: return
                    try:
                        if p not in _pil_cache:
                            _pil_cache[p] = Image.open(p)
                        im = _pil_cache[p]
                        pw = max(320, preview.winfo_width() - 20)
                        ph = max(360, preview.winfo_height() - 20)
                        w, h = im.size
                        crop_w = int(w / zf)
                        crop_h = int(h / zf)
                        cx = int(win.pan_x * w)
                        cy = int(win.pan_y * h)
                        x1 = max(0, cx - crop_w // 2)
                        y1 = max(0, cy - crop_h // 2)
                        if x1 + crop_w > w: x1 = w - crop_w
                        if y1 + crop_h > h: y1 = h - crop_h
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(w, x1 + crop_w)
                        y2 = min(h, y1 + crop_h)
                        cropped = im.crop((x1, y1, x2, y2))
                        cw, ch = cropped.size
                        r = min(pw / float(cw), ph / float(ch))
                        sz = (max(1, int(cw * r)), max(1, int(ch * r)))
                        img = ctk.CTkImage(light_image=cropped, size=sz)
                        main_lbl.configure(image=img)
                        main_lbl.image = img
                    except Exception:
                        pass
                win._render_pan_cmd = _render_pan_only

                # Persistent widget cache for thumbnails to prevent flickering
                win.thumb_widgets = []

                def _render_thumbs():
                    if not win.winfo_exists(): return
                    if not thumbs_var.get():
                        for ch in thumbs_wrap.winfo_children(): ch.destroy()
                        win.thumb_widgets = []
                        return

                    new_idx = idx_var.get()
                    
                    # 1. INITIAL BUILD: Create widgets only if needed
                    if not win.thumb_widgets:
                        for ch in thumbs_wrap.winfo_children(): ch.destroy()
                        try:
                            from PIL import Image as PILImage
                        except Exception:
                            PILImage = None
                        
                        thumbs_container = ctk.CTkFrame(thumbs_wrap, fg_color="transparent")
                        thumbs_container.pack(fill="x", expand=True)
                        
                        for i, p in enumerate(paths[:60]):
                            is_active = (i == new_idx)
                            cell = ctk.CTkFrame(thumbs_container, fg_color=C["card"] if is_active else "transparent", 
                                               corner_radius=8, border_width=2 if is_active else 0, 
                                               border_color=C["olive"])
                            cell.grid(row=0, column=i, padx=4, pady=4)
                            
                            if PILImage:
                                try:
                                    im = PILImage.open(p)
                                    tsize = int(100 * self._scale)
                                    timg = ctk.CTkImage(light_image=im, size=(tsize, int(tsize * 1.33)))
                                    img_lbl = ctk.CTkLabel(cell, text="", image=timg)
                                    img_lbl.pack(padx=2, pady=2)
                                except: pass
                            
                            fname = os.path.basename(p)
                            if len(fname) > 12: fname = fname[:9] + "..."
                            lab = ctk.CTkLabel(cell, text=fname,
                                               font=ctk.CTkFont(family="Inter", size=self.F["muted"]),
                                               text_color=C["olive"] if is_active else C["muted"]) 
                            lab.pack(padx=4, pady=(0,2))
                            
                            win.thumb_widgets.append({"cell": cell, "label": lab})
                            def _mk_cb(idx=i):
                                return lambda _e=None: (
                                    setattr(win, "zoom_factor", 1.0),
                                    setattr(win, "pan_x", 0.5),
                                    setattr(win, "pan_y", 0.5),
                                    idx_var.set(idx), 
                                    _render_main(), 
                                    _render_thumbs()
                                )
                            cell.bind("<Button-1>", _mk_cb())
                            for ch in cell.winfo_children(): ch.bind("<Button-1>", _mk_cb())
                    
                    # 2. UPDATE ONLY: Just refresh the highlights
                    else:
                        for i, widgets in enumerate(win.thumb_widgets):
                            is_active = (i == new_idx)
                            widgets["cell"].configure(
                                fg_color=C["card"] if is_active else "transparent",
                                border_width=2 if is_active else 0,
                                border_color=C["olive"]
                            )
                            widgets["label"].configure(
                                text_color=C["olive"] if is_active else C["muted"]
                            )

                def _prev(_e=None):
                    idx_var.set(max(0, idx_var.get() - 1))
                    win.zoom_factor = 1.0; win.pan_x = 0.5; win.pan_y = 0.5
                    _render_main()
                def _next(_e=None):
                    idx_var.set(min(len(paths) - 1, idx_var.get() + 1))
                    win.zoom_factor = 1.0; win.pan_x = 0.5; win.pan_y = 0.5
                    _render_main()

                
                prev_btn.bind("<Button-1>", _prev)
                next_btn.bind("<Button-1>", _next)
                try:
                    win.bind("<Left>", _prev)
                    win.bind("<Right>", _next)
                except Exception:
                    pass

                if paths:
                    idx_var.set(0)
                    _render_main()
                    _render_thumbs()
                    def _refresh(*_):
                        _render_main()
                        _render_thumbs()
                    fit_var.trace_add("write", _refresh)
                    thumbs_var.trace_add("write", _refresh)
                else:
                    ctk.CTkLabel(preview, text="No images", font=ctk.CTkFont(size=self.F["heading"]))\
                        .pack(expand=True)
                    status_lbl.configure(text="No local image files found for this book")

            # Start the deferred initialization
            win.after(100, _deferred_init)


        # Robust binding: bind to ALL elements in the row so clicking anywhere works
        for w in [row, name_lbl, badge, type_lbl, time_lbl]:
            w.configure(cursor="hand2")
            w.bind("<Button-1>", _open_detail)

        widgets = {
            "row":      row,
            "name_lbl": name_lbl,
            "badge":    badge,
            "type_lbl": type_lbl,
            "time_lbl": time_lbl,
        }
        self.activity_rows[book_id] = widgets
        self.row_order.append(book_id)
        return widgets

    def update_activity_row(self, book_id, status, type_, timestamp):
        """Update existing row or create new one. Called from main thread via after()."""
        if not book_id: return
        
        bg, fg = STATUS_STYLES.get(status, (C["s_o_bg"], C["s_o_fg"]))

        if book_id in self.activity_rows:
            w = self.activity_rows[book_id]
            w["badge"].configure(text=status, fg_color=bg, text_color=fg)
            w["time_lbl"].configure(text=timestamp)
            w["type_lbl"].configure(text=type_)
        elif book_id not in self._pending_ids:
            # Mark as pending to prevent duplicate rows from rapid updates
            self._pending_ids.add(book_id)
            self._add_activity_row(book_id, status, type_, timestamp)
            # Remove from pending after creation (it's now in activity_rows)
            self._pending_ids.discard(book_id)

    def _clear_activity(self):
        """Clear all rows from the activity table."""
        # 1. Destroy all widgets in the scrollable container
        for ch in self.rows_container.winfo_children():
            try:
                ch.destroy()
            except Exception:
                pass
        
        # 2. Reset data structures
        self.activity_rows = {}
        self.row_order = []
        self._pending_ids = set()
        
        # 3. Reset session counters
        self.total_ok = 0
        self.total_skip = 0
        self.total_fail = 0

    # ── Settings Window ────────────────────────────────────────────────────────
    def _open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("Settings")
        win.transient(self)

        # On Linux CTkToplevel needs to be fully mapped before grab_set & geometry
        win.withdraw()  # hide until ready

        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(win, fg_color=C["bg"])
        scroll.grid(row=0, column=0, sticky="nsew")
        scroll.grid_columnconfigure(0, weight=1)

        # Section: Automation
        ctk.CTkLabel(scroll, text="⚙️  Automation Settings",
                     font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                     text_color=C["text"]).grid(
            row=0, column=0, sticky="w", padx=24, pady=(24, 12))

        auto_card = self._settings_card(scroll, 1)

        ctk.CTkLabel(auto_card, text="Background Sync",
                     font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                     text_color=C["text"]).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))

        self.watch_var = ctk.BooleanVar(value=self.config.get("watch_mode", False))
        ctk.CTkSwitch(auto_card, text="Enable automatic folder watching",
                      variable=self.watch_var,
                      font=ctk.CTkFont(family="Inter", size=self.F["label"]),
                      progress_color=C["olive"],
                      button_color=C["white"]).grid(
            row=1, column=0, sticky="w", padx=20, pady=(0, 16))

        ctk.CTkLabel(auto_card, text="Refresh Interval (seconds)",
                     font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                     text_color=C["hdr"]).grid(row=2, column=0, sticky="w", padx=20, pady=(0, 4))

        self.interval_var = ctk.StringVar(value=str(self.config.get("interval", 30)))
        ctk.CTkEntry(auto_card, textvariable=self.interval_var,
                     width=self._px(140), height=self._px(44),
                     border_width=1, border_color=C["border"],
                     fg_color=C["card"],
                     font=ctk.CTkFont(family="Inter", size=self.F["input"])).grid(
            row=3, column=0, sticky="w", padx=20, pady=(0, 16))

        ctk.CTkButton(auto_card, text="💾  Save Settings",
                      height=self._px(44), corner_radius=self._px(10),
                      font=ctk.CTkFont(family="Inter", size=self.F["btn"], weight="bold"),
                      fg_color=C["olive"], hover_color=C["olive_dk"],
                      text_color=C["white"],
                      command=self._save_settings).grid(
            row=4, column=0, sticky="w", padx=20, pady=(0, 20))

        # Section: Database
        ctk.CTkLabel(scroll, text="🗄️  Database Connection",
                     font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                     text_color=C["text"]).grid(
            row=2, column=0, sticky="w", padx=24, pady=(20, 12))

        db_card = self._settings_card(scroll, 3)
        db_card.grid_columnconfigure(1, weight=1)

        fields = [
            ("MongoDB URI",       "mongo_uri",  "mongodb+srv://user:pass@cluster..."),
            ("Database Name",     "db_name",    "e.g. Test"),
            ("Collection Name",   "collection", "e.g. Book Data"),
        ]
        self._db_vars = {}
        for row_i, (lbl, key, ph) in enumerate(fields):
            ctk.CTkLabel(db_card, text=lbl,
                         font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                         text_color=C["hdr"]).grid(
                row=row_i, column=0, padx=(20, 12), pady=10, sticky="w")
            var = ctk.StringVar(value=self.config.get(key, ""))
            self._db_vars[key] = var
            e = ctk.CTkEntry(db_card, textvariable=var,
                             placeholder_text=ph, height=self._px(44),
                             border_width=1, border_color=C["border"],
                             fg_color=C["card"], text_color=C["text"],
                             font=ctk.CTkFont(family="Inter", size=self.F["input"]),
                             state="disabled")
            e.grid(row=row_i, column=1, padx=(0, 20), pady=10, sticky="ew")

        ctk.CTkButton(db_card, text="🔒  DB Config (Locked)",
                      height=self._px(44), corner_radius=self._px(10),
                      font=ctk.CTkFont(family="Inter", size=self.F["btn"], weight="bold"),
                      fg_color=C["olive"], hover_color=C["olive_dk"],
                      text_color=C["white"],
                      state="disabled").grid(
            row=len(fields), column=0, columnspan=2,
            sticky="ew", padx=20, pady=(8, 20))

        # Responsive size
        sw = max(520, int(self.winfo_width() * 0.40))
        sh = max(560, int(self.winfo_height() * 0.70))
        win.geometry(f"{sw}x{sh}")

        # Center and show after all content is built
        win.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - win.winfo_width())  // 2
        y = self.winfo_y() + (self.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")
        win.deiconify()   # show now
        win.grab_set()    # grab after visible — avoids Linux blank window bug

    def _settings_card(self, parent, row):
        c = ctk.CTkFrame(parent, corner_radius=self._px(12),
                         fg_color=C["white"],
                         border_width=1, border_color=C["border"])
        c.grid(row=row, column=0, padx=24, pady=(0, 8), sticky="ew")
        c.grid_columnconfigure(0, weight=1)
        return c

    # ── Actions ────────────────────────────────────────────────────────────────
    def _browse_folder(self, mode="books"):
        if not self.current_user:
            messagebox.showwarning("Authorization Required", "Please enter a valid token to authorize before uploading.")
            return
        # Senior Approach: Default to /host_data if it exists (for Docker volume support)
        init_dir = "/host_data" if os.path.exists("/host_data") else "/"
        path = filedialog.askdirectory(title=f"Select {mode.title()} Folder", initialdir=init_dir)
        if path:
            g = BookGrouper()
            grouped = g.group(path)
            if not grouped:
                messagebox.showerror("Invalid Folder",
                                     "No book images found in this folder.\n"
                                     "Files must be named like BookID_001.jpg (e.g., 7151_001.jpg).\n"
                                     "Please select a valid folder.")
                return
            
            # ── CLEAR PREVIOUS ACTIVITY ──
            self._clear_activity()
            
            self._folder_selected_session = True
            if mode == "slides":
                self.config["slides_path"] = path
            else:
                self.config["books_path"] = path
            self.config["folder_path"] = path
            self._save_config()
            self._log(f"📁 {mode.title()} folder set: {path}")
            if self.db_connector and self.db_connector.ping():
                if not self.sync_running:
                    self._start_sync()
            else:
                self.conn_badge.configure(fg_color=C["s_fail_bg"])
                self.conn_text.configure(text="Offline", text_color=C["s_fail_fg"])
                if getattr(self, "_conn_icon_red", None):
                    self.conn_icon.configure(image=self._conn_icon_red)
                messagebox.showinfo("Folder Selected",
                                    f"{mode.title()} folder set. Connect to start syncing.\n{path}")

    def _test_conn(self):
        token = self.token_entry.get().strip()
        if not token:
            messagebox.showerror("Missing Token",
                                 "Please paste your connection token.")
            return

        # Configuration Priority: OS Env (including .env) > JSON Config > Hardcoded Fallback
        uri = (os.environ.get("MONGO_URI") or 
               self.config.get("mongo_uri") or 
               "mongodb+srv://mlbenchpvtltd:HDqDr62jK1vK50x9@cluster0.pdhd1qx.mongodb.net/Test").strip()
        
        # Save used URI to config if it's not already there
        if uri and not self.config.get("mongo_uri"):
            self.config["mongo_uri"] = uri
            self._save_config()
        
        if not uri:
            messagebox.showerror("Missing Database URI",
                                 "Database connection is not configured.")
            return

        self._set_conn_visual("connecting")
        self.update()

        dbname = self.config.get("db_name", "Test")
        conn   = DBConnector(uri, dbname)
        ok, msg = conn.connect()

        if ok:
            user = conn.find_user_by_token(token)
            if not user:
                self._set_conn_visual("invalid")
                try:
                    self.token_entry.configure(border_color=C["s_fail_fg"]) 
                    self.token_entry.focus_set()
                except Exception:
                    pass
                messagebox.showerror("Invalid Token", "Token not found in users collection.")
                return
            self.db_connector = conn
            self.current_user = {"id": str(user.get("_id")), "username": user.get("username", "")}
            # Mask and lock token UI
            try:
                self.token_entry.configure(show="•", state="disabled")
            except Exception:
                pass
            try:
                self.btn_token.configure(text="Verified", state="disabled")
            except Exception:
                pass
            self._set_conn_visual("active")
            self._log(f"✅ Connected → {dbname} as {self.current_user.get('username','user')}")
            # Auto-start sync only if user selected a folder in this session
            if self._folder_selected_session and not self.sync_running:
                self._start_sync()
        else:
            self._set_conn_visual("failed")
            messagebox.showerror("Connection Failed", msg)

    def _save_settings(self):
        self.config["watch_mode"] = self.watch_var.get()
        try:
            self.config["interval"] = int(self.interval_var.get())
        except ValueError:
            self.config["interval"] = 30
        self._save_config()
        self._log("⚙️ Settings saved.")
        messagebox.showinfo("Saved", "Settings saved!")

    def _save_db_settings(self):
        for key, var in self._db_vars.items():
            self.config[key] = var.get().strip()
        self._save_config()
        self._log("💾 DB config saved.")
        messagebox.showinfo("Saved", "Database config saved!")
        
    # ── Sync Worker ────────────────────────────────────────────────────────────
    def _start_sync(self):
        folder = self.config.get("folder_path", "").strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Folder",
                                   "Please select a source folder first(click Upload Books or Upload Slides).")
            return
        if not self.db_connector or not self.db_connector.connected:
            messagebox.showwarning("Not Connected", "Please connect first.")
            return
        if not self.current_user:
            messagebox.showwarning("Not Authorized", "Please enter a valid token to authorize before syncing.")
            return

        # USER REQUEST: Explicit check for Models / OCR
        if not OCR_AVAILABLE:
            messagebox.showwarning("Models Missing", 
                "OCR Pipeline (main_mineru_ocr.py) or dependencies are missing.\n"
                "Syncing is disabled until models are installed.")
            return
            
        # Check if Ollama is running (simple check)
        try:
            import ollama
            # Just test if we can ping the server
            ollama.list() 
        except Exception:
            self._log("⚠️ Warning: Ollama server not responding. AI extraction might fail.")
            if not messagebox.askyesno("Ollama Missing", 
                "Ollama is either not installed or not running.\n"
                "AI extraction (Title, Color, Description) will fail.\n\n"
                "Continue anyway?"):
                return

        self.total_ok = self.total_skip = self.total_fail = 0
        self.sync_running = True
        self._log("🚀 Sync started!")
        threading.Thread(target=self._worker, daemon=True).start()

    # ── OCR Pipeline Helper ─────────────────────────────────────────────────
    def _process_book_ocr(self, book_id, book_pages, ts):
        """
        Run the full OCR pipeline for one book:
          YOLO crop → MinerU OCR → Ollama AI (title, colors, description)
        Returns dict with {title, description, colors} or None on failure.
        """
        if not OCR_AVAILABLE:
            return None

        BASE = os.path.dirname(os.path.abspath(__file__))
        crops_base  = os.getenv("CROPS_FOLDER",
                                os.path.join(BASE, "..", "doclayout_column_cropings", "column_crops"))
        output_base = os.getenv("OUTPUT_FOLDER",
                                os.path.join(BASE, "..", "mineru_results"))

        book_crops_folder  = os.path.join(crops_base,  book_id)
        book_output_folder = os.path.join(output_base, book_id)
        os.makedirs(book_output_folder, exist_ok=True)

        # Extract sorted image paths from (page_num, filepath) tuples
        image_paths = [fp for _, fp in sorted(book_pages, key=lambda x: x[0])]

        # ── GPU Cleanup: Flush VRAM from previous book ────
        try:
            self._log(f"  🧹 Flushing GPU memory before {book_id}…")
            ocr_pipeline.stop_ollama()
            ocr_pipeline.cleanup_gpu()
        except Exception as e:
            self._log(f"  ⚠️ GPU flush warning: {e}")

        # ── Phase 0: ISBN First-Pass (New) ─────────────────
        isbn_meta = None
        official_isbn = "N/A"
        isbn_source_page = 3 # Default to copyright page
        
        if ISBN_LOGIC_AVAILABLE:
            self.after(0, lambda b=book_id, t=ts:
                       self.update_activity_row(b, "Processing", "Searching ISBN…", t))
            self.after(0, self.update)
            
            self._log(f"  🔍 Checking for ISBN logic for {book_id}…")
            try:
                # Use normalized log function for isbn_logic
                res = isbn_logic.process_book(book_id, image_paths, log_fn=self._log)
                isbn_logic.unload_isbn_reader() # FREE GPU IMMEDIATELY
                
                if res:
                    official_isbn = res.get("isbn", "N/A")
                    isbn_source_page = res.get("source_page") or 3  # Fallback to copyright page
                    isbn_ocr_texts = res.get("ocr_texts", [])  # Captured OCR text from ISBN pages
                    if official_isbn != "N/A":
                        isbn_meta = res.get("metadata")
                        self._log(f"  ✅ ISBN Found: {official_isbn} (Page {isbn_source_page})")
            except Exception as e:
                self._log(f"  ⚠️ ISBN search error: {e}")

        # Initialize metadata from ISBN pass if available
        title_str   = (isbn_meta.get("title") if isbn_meta else "") or ""
        subtitle    = (isbn_meta.get("subtitle") if isbn_meta else "") or ""
        author      = (isbn_meta.get("authors") if isbn_meta else "") or ""
        edition     = (isbn_meta.get("edition") if isbn_meta else "") or ""
        description = (isbn_meta.get("description") if isbn_meta else "") or ""
        colors      = []
        interior_text_for_edition = ""
        isbn_ocr_texts = locals().get('isbn_ocr_texts', [])

        # Track if metadata needs AI generation:
        # Only generate if API explicitly returned N/A or no ISBN was found
        api_title_missing       = (not isbn_meta) or (isbn_meta and isbn_meta.get("title") in ("N/A", "", None))
        api_edition_missing     = (not isbn_meta) or (isbn_meta and isbn_meta.get("edition") in ("N/A", "", None))
        api_description_missing = (not isbn_meta) or (isbn_meta and isbn_meta.get("description") in ("N/A", "", None))

        # Convert "N/A" to empty string for targeted fallback
        if title_str == "N/A": title_str = ""
        if author == "N/A": author = ""
        if edition == "N/A": edition = ""
        if description == "N/A": description = ""

        # Optimization: If all fields are already found via ISBN, skip heavy OCR
        all_meta_found = all([title_str, author, edition, description])
        interior_text = ""
        ocr_data = None
        
        # ── API Metadata Verification Checkpoint (User Request) ──
        self._log(f"\n  📡 API METADATA CHECKPOINT:")
        self._log(f"    • Title:       {title_str or 'N/A'}")
        self._log(f"    • Subtitle:    {subtitle or 'N/A'}")
        self._log(f"    • Author:      {author or 'N/A'}")
        self._log(f"    • Edition:     {edition or 'N/A'}")
        self._log(f"    • Description: {description or 'N/A'}")
        self._log("")

        # Decision: Which pages need processing
        # MEGA OPTIMIZATION: If we have Description via ISBN, we don't need YOLO/OCR at all!
        # We only need the interior OCR/YOLO for generating the description.
        needs_interior_ocr = (not description)

        if needs_interior_ocr:
            # If we only need description, focus on interior pages ONLY (Skip 1 and 2)
            if not description and all([title_str, author, edition]):
                self._log(f"  🎯 Only Description missing. Processing interior pages only (Skipping covers).")
                # User rule: Skip 1 and 2 for description OCR
                images_to_process = [fp for pn, fp in sorted(book_pages, key=lambda x: x[0]) if pn not in (1, 2)]
                # If no interior (short book), process whatever is left BUT still skip 1/2 if possible
                if not images_to_process:
                    images_to_process = image_paths 
            else:
                images_to_process = image_paths

            # ── Phase 1: YOLO Crop ────────────────────────
            self.after(0, lambda b=book_id, t=ts:
                       self.update_activity_row(b, "Processing", "Cropping…", t))
            self.after(0, self.update)
            time.sleep(0.2)
            
            self._log(f"  ✂️  Cropping {book_id} ({len(images_to_process)} pages)…")
            try:
                pages = ocr_pipeline.crop_book(images_to_process, book_crops_folder)
            except Exception as e:
                self._log(f"  ❌ Crop failed: {e}")
                return None
            
            if not pages:
                self._log(f"  ⚠️ No crops for {book_id}")
                return None
            
            self._log(f"  ✅ {len(pages)} page(s) cropped")

            # ── Phase 2: MinerU OCR ───────────────────────
            # Only run MinerU if description or interior text is still needed
            interior_text = ""
            if not description:
                self.after(0, lambda b=book_id, t=ts:
                           self.update_activity_row(b, "Processing", "OCR…", t))
                self.after(0, self.update)
                time.sleep(0.2)
                
                self._log(f"  📝 Running OCR on {book_id}…")
                from types import SimpleNamespace
                ocr_source = os.getenv("OCR_SOURCE", "huggingface") 
                no_ai_mode = os.getenv("NO_AI", "false").lower() == "true"
                ocr_args = SimpleNamespace(no_preview=True, no_ai=no_ai_mode, source=ocr_source)

                try:
                    ocr_pipeline.stop_ollama()
                    ocr_data = ocr_pipeline.ocr_book(pages, book_output_folder, ocr_args, total_pages=len(images_to_process))
                    
                    # USER REQUEST: Unload EasyOCR immediately to free VRAM for Phase 3 (Ollama)
                    ocr_pipeline.unload_easyocr()
                    
                    if ocr_data:
                        interior_text = "\n".join(ocr_data.get("interior_texts", []))
                except Exception as e:
                    self._log(f"  ❌ OCR failed: {e}")
            else:
                self._log(f"  ⏭️ Skipping MinerU OCR (Description found via ISBN)")

        # ── Phase 3: Targeted AI Pipeline (Fill Gaps) ───
        self.after(0, lambda b=book_id, t=ts:
                   self.update_activity_row(b, "Processing", "AI…", t))
        self.after(0, self.update)
        time.sleep(0.2)

        # Start Ollama for AI models
        try:
            self._log(f"  🤖 Starting Ollama for AI pipeline…")
            ocr_pipeline.start_ollama()
        except Exception as e:
            self._log(f"  ❌ Failed to start Ollama: {e}")

        # 1. Title (only if API returned N/A or no ISBN was found)
        if not title_str and api_title_missing:
            try:
                front_cover = next((fp for pn, fp in book_pages if pn == 1), image_paths[0])
                title_str = ocr_pipeline.extract_title_from_cover_image(front_cover)
            except: pass
        elif title_str:
            self._log(f"  ✅ Using API title")

        # 2. Author (if missing)
        if not author and author != "N/A":
            try:
                front_cover = next((fp for pn, fp in book_pages if pn == 1), image_paths[0])
                back_cover  = next((fp for pn, fp in book_pages if pn == 2), None)
                author = ocr_pipeline.extract_author_from_cover(front_cover, back_cover)
            except: pass

        # 3. Edition (only if API returned N/A or no ISBN was found)
        if not edition and api_edition_missing:
            try:
                # USER REQUEST: Prioritized search. isbn_source_page first, then others. Stop if found.
                priority_pages = []
                if isbn_source_page: priority_pages.append(isbn_source_page)
                
                # Pages 2, 4 (copyright), 1 (front), last (back) are most likely
                others = [2, 4, 1, len(book_pages)]
                for p in others:
                    if p not in priority_pages: priority_pages.append(p)

                # Map page numbers to file paths
                target_images = []
                for pn in priority_pages:
                    img = next((fp for p_num, fp in book_pages if p_num == pn), None)
                    if img and img not in target_images:
                        target_images.append(img)
                
                if target_images:
                    self._log(f"  🧠 Vision LLM checking {len(target_images)} priority pages for edition…")
                    # extract_edition_from_cover iterates through images and returns as soon as validated
                    edition = ocr_pipeline.extract_edition_from_cover(target_images)
                
                if not edition:
                    # FALLBACK 1: Use the already-extracted OCR text from MinerU
                    if interior_text.strip():
                        self._log("  🔍 Vision failed → trying text-based edition search (interior text)…")
                        edition = ocr_pipeline.extract_edition_from_text(interior_text, title_str)
                    
                    # FALLBACK 2: Use the ISBN OCR text (copyright page text captured during ISBN phase)
                    if not edition and isbn_ocr_texts:
                        combined_isbn_text = "\n".join(isbn_ocr_texts)
                        self._log("  🔍 Trying text-based edition search (ISBN page text)…")
                        edition = ocr_pipeline.extract_edition_from_text(combined_isbn_text, title_str)
                    
                    if not edition:
                        self._log("  ⚠️ No edition info found on priority pages.")
            except Exception as e:
                self._log(f"  ⚠️ Targeted edition search failed: {e}")
        elif edition:
            self._log(f"  ✅ Using API edition (skipping AI generation)")

        # 4. Description (only if API returned N/A or no ISBN was found)
        if not description and api_description_missing:
            try:
                if interior_text.strip():
                    description = ocr_pipeline.generate_description(interior_text, title_str)
                else:
                    description = ocr_pipeline.generate_description_from_images(image_paths, title_str)
            except: pass
        elif description:
            self._log(f"  ✅ Using API description (skipping AI generation)")
        try:
            ocr_pipeline.save_book_metadata(
                book_id, title_str, description,
                book_output_folder, edition=edition, author=author, isbn=official_isbn)
        except Exception as e:
            self._log(f"  ⚠️ Metadata saving error: {e}")

        # ── Final Cleanup ──
        try:
            self._log(f"  🧹 Cleaning up intermediate files for {book_id}…")
            # Only cleanup if we actually created crops
            if os.path.exists(book_crops_folder):
                ocr_pipeline.cleanup_intermediate_files(book_crops_folder)
        except Exception as e:
            self._log(f"  ⚠️ Cleanup failed: {e}")

        # CRITICAL HARDENING: Detect if extraction failed completely
        # If we have no title and no ISBN, this data is useless. Return None to prevent syncing.
        if not title_str and (not official_isbn or official_isbn == "N/A"):
            self._log(f"  ❌ AI EXTRACTION FAILED: No title or ISBN found. Skipping sync.")
            return None

        return {
            "title":          title_str,
            "subtitle":       subtitle,
            "author":         author,
            "edition":        edition,
            "isbn":           official_isbn,
            "description":    description,
            "status":         "completed",
            "modified":       ts,
            "output_folder":  book_output_folder
        }

    def _worker(self):
        watch    = self.config.get("watch_mode", False)
        interval = self.config.get("interval", 30)
        synced   = set()
        grouper  = BookGrouper()

        try:
            # Start with a clean GPU slate
            if OCR_AVAILABLE:
                ocr_pipeline.cleanup_gpu()

            while self.sync_running:
                if not self.current_user:
                    self._log("❌ Not authorized: token required.")
                    break
                if not (self.db_connector and self.db_connector.ping()):
                    self.after(0, lambda: self._set_conn_visual("offline"))
                    self._log("❌ Connection lost. Stopping sync.")
                    break
                folder = self.config.get("folder_path", "").strip()
                if not folder:
                    break

                books   = grouper.group(folder)
                new_ids = [bid for bid in books if bid not in synced]

                if new_ids:
                    coll = self.config.get("collection", "Book Data")
                    self._log(f"📚 {len(new_ids)} new book(s) found…")

                    sorted_ids = sorted(new_ids)
                    for book_id in sorted_ids:
                        if not self.sync_running:
                            break

                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Show this book as Queued first
                        self.after(0, lambda b=book_id, t=ts:
                                   self.update_activity_row(b, "Queued", "Book", t))
                        self.after(0, self.update)
                        time.sleep(0.15)

                        # Now mark as Processing
                        self.after(0, lambda b=book_id, t=ts:
                                   self.update_activity_row(b, "Processing", "Book", t))
                        self.after(0, self.update)
                        time.sleep(0.3)  # Delay for Ubuntu to process the update

                        self._log(f"  📖 {book_id}…")

                        # Check DB
                        try:
                            if self.db_connector.book_exists(coll, book_id):
                                self._log(f"  ⏭️  Already exists, skipping.")
                                self.total_skip += 1
                                synced.add(book_id)
                                self.after(0, lambda b=book_id, t=ts:
                                           self.update_activity_row(b, "Skipped", "Book", t))
                                self.after(0, self.update)
                                time.sleep(0.15)
                                continue
                        except Exception as e:
                            self._log(f"  ⚠️ DB check error: {e}")

                        # Build base document
                        doc = grouper.build_document(book_id, books[book_id])
                        if self.current_user:
                            doc["user_id"] = self.current_user.get("id")

                        # ── OCR Pipeline ──────────────────────────────────
                        ai_result = None
                        if OCR_AVAILABLE:
                            try:
                                ai_result = self._process_book_ocr(
                                    book_id, books[book_id], ts)
                                if ai_result:
                                    doc["title"]          = ai_result.get("title", "")
                                    doc["subtitle"]       = ai_result.get("subtitle", "")
                                    doc["author"]         = ai_result.get("author", "Not Found")
                                    doc["edition"]        = ai_result.get("edition", "Not Specified")
                                    doc["isbn"]           = ai_result.get("isbn", "N/A")
                                    doc["description"]    = ai_result.get("description", "")
                                    doc["ocr_completed"]  = True
                                    self._log(f"  🎉 AI done: {book_id}")
                                else:
                                    doc["ocr_completed"] = False
                                    self._log(f"  ⚠️ OCR returned no results for {book_id}")
                            except Exception as e:
                                doc["ocr_completed"] = False
                                self._log(f"  ⚠️ OCR pipeline error: {e}")
                        else:
                            self._log(f"  ❌ ERROR: OCR pipeline not available (models missing). Skipping book.")
                            # Mark as failed in UI
                            self.after(0, lambda b=book_id, t=ts:
                                       self.update_activity_row(b, "Failed", "No Models", t))
                            self.after(0, self.update)
                            continue

                        # ── Insert to DB ──────────────────────────────────
                        if ai_result:
                            try:
                                self.db_connector.insert_book(coll, doc)
                                self.total_ok += 1
                                synced.add(book_id)
                                self._log(f"  ✅ Synced: {book_id}")
                                
                                # --- Final Results Cleanup (Delete local mineru_results after success) ---
                                try:
                                    if ai_result.get("output_folder"):
                                        ocr_pipeline.cleanup_intermediate_files(ai_result["output_folder"])
                                        self._log(f"  🧹 Cleaned local results: {book_id}")
                                except Exception as e:
                                    self._log(f"  ⚠️ Result cleanup failed: {e}")
                                
                                self.after(0, lambda b=book_id, t=ts:
                                           self.update_activity_row(b, "Complete", "Book", t))
                            except Exception as e:
                                self._log(f"  ❌ DB Error: {e}")
                                self.total_fail += 1
                                self.after(0, lambda b=book_id, t=ts:
                                           self.update_activity_row(b, "Failed", "DB Error", t))
                                continue
                        else:
                            # If we reached here without ai_result, skip sync
                            self._log(f"  ⚠️ Skipping sync for {book_id} (No metadata extracted)")
                            self.total_fail += 1
                            self.after(0, lambda b=book_id, t=ts:
                                       self.update_activity_row(b, "Failed", "No Metadata", t))
                            continue

                        self.after(0, self.update)
                        time.sleep(0.15)

                    self._log(f"✅ Pass done — OK:{self.total_ok} Skip:{self.total_skip} Fail:{self.total_fail}")
                else:
                    self._log("✅ All up to date.")

                if not watch:
                    break
                time.sleep(interval)

        finally:
            self.sync_running = False
            if OCR_AVAILABLE:
                ocr_pipeline.cleanup_gpu()
            self.after(0, lambda: self.conn_badge.configure(fg_color="#F3F4F6"))
            self.after(0, lambda: self.conn_text.configure(text="Idle", text_color="#6B7280"))
            self.after(0, lambda: self.conn_icon.configure(image=self._conn_icon_grey or self._conn_icon_green))
            self._log("⏹️ Sync finished.")

    # ── Logging ────────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}]  {msg}\n")
        # Mirror to terminal for debugging and user visibility
        print(f"[{ts}]  {msg}", flush=True)

    def _poll_log(self):
        batch = ""
        count = 0
        while not self.log_queue.empty() and count < 100:
            batch += self.log_queue.get_nowait()
            count += 1
        
        if batch:
            # We skip internal log box insertion to prevent UI lag on Ubuntu
            # The log_queue is still useful for debugging if needed later.
            pass
        self.after(200, self._poll_log)


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SyncApp()
    app.mainloop()
