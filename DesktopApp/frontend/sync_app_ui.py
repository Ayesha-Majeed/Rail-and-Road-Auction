import customtkinter as ctk
import os
import sys
import platform
import ssl
import traceback
import io
import multiprocessing
import json
import re
import time
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tkinter import filedialog, PhotoImage, Canvas
from dotenv import load_dotenv

from backend.crypto_utils import CryptoUtils
from backend.db_connector import DBConnector
from backend.book_grouper import BookGrouper
import backend.model_manager as model_manager
from CTkMessagebox import CTkMessagebox

# --- Monkey-patch CustomTkinter ScrollableFrame to fix Python 3.12 string widget bug ---
from customtkinter.windows.widgets.ctk_scrollable_frame import CTkScrollableFrame

original_check = CTkScrollableFrame._check_if_valid_scroll

def _patched_check(self, widget):
    if isinstance(widget, str):
        try:
            widget = self.nametowidget(widget)
        except Exception:
            return False
    return original_check(self, widget)

CTkScrollableFrame._check_if_valid_scroll = _patched_check
# ---------------------------------------------------------------------------------------

class ModernMessageBox:
    def __init__(self):
        self.app = None

    def _create_dialog(self, title, message, mtype, parent=None):
        target_app = parent if parent else self.app
        if not target_app:
            print(f"[{title}] {message}")
            return True if mtype == "question" else None

        # Use global app theme (C) which is defined later in the file
        try:
            bg_color = C.get("bg", "#FFFFFF")
            text_color = C.get("text", "#1D2939")
            olive_btn = C.get("olive", "#8C7B5D")
            olive_hover = C.get("olive_h", "#AA9874")
            red_btn = C.get("red", "#BC4B34")
            red_hover = C.get("red_h", "#9B3A28")
        except NameError:
            bg_color = "#FFFFFF"
            text_color = "#1D2939"
            olive_btn = "#8C7B5D"
            olive_hover = "#AA9874"
            red_btn = "#BC4B34"
            red_hover = "#9B3A28"

        top = ctk.CTkToplevel(target_app, fg_color=bg_color)
        top.title(title)
        
        px = getattr(self.app, "_px", lambda x: x)
        fs = getattr(self.app, "F", {}).get("label", 14)
        
        # Medium size window
        w, h = px(350), px(120)
        
        # Center on app
        if hasattr(target_app, "winfo_x"):
            x = target_app.winfo_x() + (target_app.winfo_width() // 2) - (w // 2)
            y = target_app.winfo_y() + (target_app.winfo_height() // 2) - (h // 2)
            top.geometry(f"{w}x{h}+{x}+{y}")
        else:
            top.geometry(f"{w}x{h}")
            
        top.resizable(False, False)
        top.transient(target_app)
        top.attributes("-topmost", True)
        
        # Determine button color based on message type
        btn_color = red_btn if mtype == "error" else olive_btn
        btn_hover = red_hover if mtype == "error" else olive_hover
        
        frame = ctk.CTkFrame(top, fg_color=bg_color, corner_radius=0)
        frame.pack(fill="both", expand=True, padx=px(20), pady=px(20))
        
        lbl = ctk.CTkLabel(frame, text=message, font=ctk.CTkFont("Inter", size=fs), text_color=text_color, wraplength=w - px(40))
        lbl.pack(expand=True, fill="both", pady=(0, px(20)))
        
        result = [False]
        def _close(res):
            result[0] = res
            top.destroy()
            
        btn_frame = ctk.CTkFrame(frame, fg_color=bg_color, corner_radius=0)
        btn_frame.pack(fill="x")
        
        btn_font = ctk.CTkFont("Inter", size=fs, weight="bold")
        
        if mtype == "question":
            btn_frame.grid_columnconfigure((0, 1), weight=1)
            btn_no = ctk.CTkButton(btn_frame, text="No", font=btn_font, fg_color="#E4E7EC", hover_color="#D0D5DD", text_color=text_color, command=lambda: _close(False), width=px(80), height=px(36), corner_radius=px(6))
            btn_no.grid(row=0, column=0, padx=px(10), sticky="e")
            
            btn_yes = ctk.CTkButton(btn_frame, text="Yes", font=btn_font, fg_color=btn_color, hover_color=btn_hover, text_color="white", command=lambda: _close(True), width=px(80), height=px(36), corner_radius=px(6))
            btn_yes.grid(row=0, column=1, padx=px(10), sticky="w")
        else:
            btn_ok = ctk.CTkButton(btn_frame, text="OK", font=btn_font, fg_color=btn_color, hover_color=btn_hover, text_color="white", command=lambda: _close(True), width=px(100), height=px(36), corner_radius=px(6))
            btn_ok.pack(anchor="center")
            
        top.update_idletasks()
        try:
            top.wait_visibility()
            top.grab_set()
        except Exception:
            pass
        top.update()
        top.wait_window()
        return result[0]

    def showinfo(self, title, message, parent=None):
        self._create_dialog(title, message, "info", parent=parent)

    def showerror(self, title, message, parent=None):
        self._create_dialog(title, message, "error", parent=parent)

    def showwarning(self, title, message, parent=None):
        self._create_dialog(title, message, "warning", parent=parent)

    def askyesno(self, title, message, parent=None):
        return self._create_dialog(title, message, "question", parent=parent)

messagebox = ModernMessageBox()
# OCR Pipeline
OCR_IMPORT_ERROR = None
try:
    import backend.main_mineru_ocr as ocr_pipeline
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    OCR_IMPORT_ERROR = str(e)
    print(f"⚠️  main_mineru_ocr not found or dependency error: {e}")
except Exception as e:
    OCR_AVAILABLE = False
    OCR_IMPORT_ERROR = str(e)
    print(f"⚠️  Unexpected error loading OCR pipeline: {e}")

try:
    import frontend.isbn_extractor_ui as isbn_logic
    ISBN_LOGIC_AVAILABLE = True
except ImportError:
    ISBN_LOGIC_AVAILABLE = False
    print("⚠️  isbn_extractor_ui not found — ISBN first-pass disabled")

def get_app_dir():
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        if hasattr(sys, '_MEIPASS'):
            return sys._MEIPASS
        return exe_dir
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_app_dir()
exe_env = os.path.join(os.path.dirname(sys.executable), ".env")
internal_env = os.path.join(BASE_DIR, ".env")
env_path = exe_env if os.path.exists(exe_env) else internal_env
load_dotenv(env_path)

INTERNAL_BASE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(INTERNAL_BASE, "icons")
if not os.path.exists(ASSETS_DIR):
    ASSETS_DIR = os.path.join(BASE_DIR, "frontend", "icons")

ctk.set_appearance_mode("light")

import platform
if platform.system() == "Linux":
    ctk.set_widget_scaling(1.35)
    ctk.set_window_scaling(1.35)

C = {
    "bg":       "#FFFFFF",
    "white":    "#FFFFFF",
    "card":     "#F5F2EC",
    "olive":    "#8C7B5D",
    "olive_h":  "#AA9874",
    "olive_dk": "#6B5C41",
    "red":      "#BC4B34",
    "red_h":    "#9B3A28",
    "border":   "#E4E7EC",
    "text":     "#1D2939",
    "muted":    "#667085",
    "light":    "#98A2B3",
    "hdr":      "#344054",
    "s_g_bg":   "#ECFDF3", "s_g_fg": "#027A48",
    "s_o_bg":   "#FDF1E6", "s_o_fg": "#BC4B34",
    "s_b_bg":   "#D0E8FD", "s_b_fg": "#1A4A8A",
    "s_p_bg":   "#E8D0FD", "s_p_fg": "#4A1A8A",
    "s_skip_bg":"#F3F4F6", "s_skip_fg":"#6B7280",
    "s_fail_bg":"#FEF3F2", "s_fail_fg":"#B42318",
}

CONFIG_FILE = os.path.join(BASE_DIR, "sync_config.json")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
STATUS_STYLES = {
    "Uploading":  (C["s_g_bg"],    "#039855"),
    "Complete":   (C["s_g_bg"],    "#039855"),
    "Queued":     ("#F1DDCA",      "#BC4B34"),
    "Processing": (C["s_p_bg"],    C["s_p_fg"]),
    "Skipped":    (C["s_skip_bg"], C["s_skip_fg"]),
    "Failed":     (C["s_fail_bg"], C["s_fail_fg"]),
}

class ImagePreviewWindow(ctk.CTkToplevel):
    def __init__(self, parent, image_path, px, fs, delete_callback):
        super().__init__(parent)
        self.title("Image Preview")
        self.image_path = image_path
        self.px = px
        self.fs = fs
        self.delete_callback = delete_callback
        
        self.geometry(f"{px(1000)}x{px(800)}")
        self.minsize(px(600), px(400))
        self.configure(fg_color=C.get("bg", "#F8F9FA"))
        self.attributes("-topmost", True)
        
        from PIL import Image
        self.pil_img = Image.open(image_path)
        
        # Calculate initial zoom to fit within 90% of the window
        win_w, win_h = self.px(1000), self.px(800)
        img_w, img_h = self.pil_img.size
        scale = min(win_w/img_w, win_h/img_h) * 0.9
        self.zoom_factor = scale if scale < 1.0 else 1.0
        
        # Panning variables
        self._drag_data = {"x": 0, "y": 0}
        self.img_offset_x = 0
        self.img_offset_y = 0
        
        self._build_ui()
        self.focus()
        
    def _build_ui(self):
        # The main container for the image
        self.canvas_frame = ctk.CTkFrame(self, fg_color=C.get("bg", "#F8F9FA"), corner_radius=0)
        self.canvas_frame.pack(fill="both", expand=True)
        
        self.img_lbl = ctk.CTkLabel(self.canvas_frame, text="", cursor="fleur")
        self.img_lbl.place(relx=0.5, rely=0.5, x=self.img_offset_x, y=self.img_offset_y, anchor="center")
        
        # Bindings for panning (click and drag)
        self.img_lbl.bind("<ButtonPress-1>", self._on_drag_start)
        self.img_lbl.bind("<B1-Motion>", self._on_drag_motion)
        
        # Bindings for zooming
        self.canvas_frame.bind("<MouseWheel>", self._on_mousewheel)
        self.img_lbl.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas_frame.bind("<Button-4>", self._on_mousewheel)
        self.canvas_frame.bind("<Button-5>", self._on_mousewheel)
        self.img_lbl.bind("<Button-4>", self._on_mousewheel)
        self.img_lbl.bind("<Button-5>", self._on_mousewheel)
        
        # Floating Overlay Toolbar
        self.toolbar = ctk.CTkFrame(self, fg_color=C.get("white", "#FFFFFF"), corner_radius=self.px(25), border_width=2, border_color=C.get("border", "#E4E7EC"))
        self.toolbar.place(relx=0.5, rely=0.92, anchor="center")
        
        btn_font = ctk.CTkFont("Inter", size=self.fs(14), weight="bold")
        
        btn_close = ctk.CTkButton(self.toolbar, text="Close Preview", font=btn_font, 
                                  fg_color="transparent", hover_color=C.get("card", "#F5F2EC"), text_color=C.get("text", "#1D2939"),
                                  width=self.px(140), height=self.px(42), corner_radius=self.px(20),
                                  command=self.destroy)
        btn_close.pack(side="left", padx=self.px(16), pady=self.px(12))
        
        btn_delete = ctk.CTkButton(self.toolbar, text="Delete Image", font=btn_font, 
                                   fg_color="#EF4444", hover_color="#B91C1C", text_color="white",
                                   width=self.px(140), height=self.px(42), corner_radius=self.px(20),
                                   command=self._delete)
        btn_delete.pack(side="right", padx=self.px(16), pady=self.px(12))
        
        self._update_image()
        
    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root

    def _on_drag_motion(self, event):
        dx = event.x_root - self._drag_data["x"]
        dy = event.y_root - self._drag_data["y"]
        self.img_offset_x += dx
        self.img_offset_y += dy
        self.img_lbl.place(relx=0.5, rely=0.5, x=self.img_offset_x, y=self.img_offset_y, anchor="center")
        self._drag_data["x"] = event.x_root
        self._drag_data["y"] = event.y_root
        
    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_factor *= 1.15
        elif event.num == 5 or event.delta < 0:
            self.zoom_factor /= 1.15
            
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))
        self._update_image()
        return "break"
        
    def _update_image(self):
        w, h = self.pil_img.size
        new_w, new_h = max(1, int(w * self.zoom_factor)), max(1, int(h * self.zoom_factor))
        ctk_img = ctk.CTkImage(light_image=self.pil_img, dark_image=self.pil_img, size=(new_w, new_h))
        self.img_lbl.configure(image=ctk_img)
        
    def _delete(self):
        self.delete_callback(self.image_path)
        self.destroy()

class ModernDropdown(ctk.CTkFrame):
    def __init__(self, master, px, values, command=None, width=300, height=44, font=None, **kwargs):
        super().__init__(master, width=width, height=height, fg_color="transparent")
        self.px = px
        self.values = values
        self.command = command
        self.font = font
        self.is_open = False
        self.all_values = values
        self.current_value = ctk.StringVar(value=values[0] if values else "No existing labels found")
        self.buttons_cache = {} # Cache buttons for instant filtering (O(1) layout time)
        
        self.is_loading = kwargs.get("is_loading", False)
        self.pack_propagate(False)
        self.toggle_btn = ctk.CTkButton(self, textvariable=self.current_value, 
                                        width=width, height=height, font=self.font,
                                        fg_color=C.get("white", "#FFFFFF"),
                                        text_color=C.get("text", "#1D2939"),
                                        border_width=2, border_color=C.get("border", "#E4E7EC"),
                                        hover_color=C.get("card", "#F5F2EC"),
                                        anchor="w", command=self.toggle)
        self.toggle_btn.pack(fill="both", expand=True)
        
        # Add chevron
        self.chevron = ctk.CTkLabel(self.toggle_btn, text="▼", font=ctk.CTkFont("Inter", size=self.px(10)), text_color=C.get("muted", "#667085"))
        self.chevron.place(relx=1.0, rely=0.5, anchor="e", x=-self.px(12))
        self.chevron.bind("<Button-1>", lambda e: self.toggle())
        
        self.dropdown_window = ctk.CTkFrame(self.winfo_toplevel(), 
                                            width=width, height=self.px(350),
                                            fg_color=C.get("white", "#FFFFFF"),
                                            border_width=1, border_color=C.get("border", "#E4E7EC"),
                                            corner_radius=self.px(8))
        self.dropdown_window.pack_propagate(False)
                                            
        search_bg = ctk.CTkFrame(self.dropdown_window, fg_color=C.get("card", "#F5F2EC"), corner_radius=self.px(6))
        search_bg.pack(fill="x", padx=self.px(12), pady=(self.px(12), self.px(8)))
        
        self.search_entry = ctk.CTkEntry(search_bg, placeholder_text="Search classes...", height=self.px(36),
                                         font=self.font, fg_color="transparent", border_width=0,
                                         text_color=C.get("text", "#1D2939"))
        self.search_entry.pack(fill="x", padx=self.px(8), pady=self.px(2))
        self.search_entry.bind("<KeyRelease>", self.filter_values)
        
        self.dropdown_frame = ctk.CTkScrollableFrame(self.dropdown_window, fg_color="transparent", corner_radius=0)
        self.dropdown_frame.pack(fill="both", expand=True, padx=self.px(4), pady=self.px(4))
        
        self.winfo_toplevel().bind_all("<Button-1>", self.check_close, add="+")
        self.populate(self.all_values)
        
    def filter_values(self, event=None):
        if getattr(self, "_search_timer", None):
            self.winfo_toplevel().after_cancel(self._search_timer)
            
        self._search_timer = self.winfo_toplevel().after(250, self._apply_filter)

    def _apply_filter(self):
        query = self.search_entry.get().lower()
        if not query:
            filtered = self.all_values
        else:
            filtered = [v for v in self.all_values if query in v.lower()]
            
        self.populate(filtered)
        
    def populate(self, values):
        # Destroy all existing buttons to guarantee a clean layout recalculation
        for btn in self.buttons_cache.values():
            btn.destroy()
        self.buttons_cache.clear()
            
        if getattr(self, "is_loading", False):
            if not hasattr(self, "_loader") or not self._loader.winfo_exists():
                self._loader = ctk.CTkProgressBar(self.dropdown_frame, mode="indeterminate", width=self.px(100), height=self.px(4), progress_color=C.get("olive", "#8C7B5D"))
                self._loader.pack(pady=self.px(30))
                self._loader.start()
            else:
                self._loader.pack(pady=self.px(30))
            return
            
        if hasattr(self, "_loader") and self._loader.winfo_exists():
            self._loader.destroy()
            delattr(self, "_loader")
            
        for val in values:
            btn = ctk.CTkButton(self.dropdown_frame, text=val, height=self.px(38), anchor="w", fg_color="transparent", 
                                text_color=C.get("text", "#1D2939"), hover_color=C.get("card", "#F5F2EC"),
                                font=self.font, corner_radius=self.px(6), command=lambda v=val: self.select(v))
            btn.pack(fill="x", pady=self.px(2), padx=self.px(6))
            self.buttons_cache[val] = btn
            
    def toggle(self):
        if self.is_open:
            self.dropdown_window.place_forget()
            self.is_open = False
        else:
            try:
                try:
                    scale = self.winfo_toplevel()._get_widget_scaling()
                except AttributeError:
                    scale = 1.0
                    
                logical_root_y = (self.winfo_rooty() - self.winfo_toplevel().winfo_rooty()) / scale
                logical_root_x = (self.winfo_rootx() - self.winfo_toplevel().winfo_rootx()) / scale
                
                win_h_logical = self.winfo_toplevel().winfo_height() / scale
                
                margin = 8
                
                # Force open UPWARDS and take up to 90% of top space
                space_above = logical_root_y - margin
                max_dh = int(space_above * 0.90)
                
                num_items = len(self.all_values) if self.all_values else 1
                req_dh = (num_items * 42) + 54  # 42px per item + 54px for search bar
                
                dh = min(req_dh, max_dh)
                dh = max(100, dh) # Keep a reasonable minimum
                
                y = logical_root_y - dh - 4
                longest_str = max(self.all_values, key=len) if self.all_values else "No existing labels found"
                req_w = int(len(longest_str) * 9.5 + 60)
                toggle_w = self.toggle_btn.winfo_width() / scale
                w = max(toggle_w, req_w)
                w = min(w, int(self.winfo_toplevel().winfo_width() / scale * 0.9))
                
                self.search_entry.delete(0, "end")
                self.dropdown_window.configure(width=w, height=dh)
                self.dropdown_window.place(x=logical_root_x, y=y)
                self.dropdown_window.lift()
                self.search_entry.focus_set()
                self.is_open = True
            except Exception as e:
                print("Dropdown toggle error:", e)
            
    def check_close(self, event):
        if self.is_open:
            try:
                x_root, y_root = event.x_root, event.y_root
                wx, wy = self.toggle_btn.winfo_rootx(), self.toggle_btn.winfo_rooty()
                ww, wh = self.toggle_btn.winfo_width(), self.toggle_btn.winfo_height()
                dx, dy = self.dropdown_window.winfo_rootx(), self.dropdown_window.winfo_rooty()
                dw, dh = self.dropdown_window.winfo_width(), self.dropdown_window.winfo_height()
                
                in_btn = (wx <= x_root <= wx+ww) and (wy <= y_root <= wy+wh)
                in_drop = (dx <= x_root <= dx+dw) and (dy <= y_root <= dy+dh)
                
                if not in_btn and not in_drop:
                    self.dropdown_window.place_forget()
                    self.is_open = False
            except Exception:
                pass
                
    def select(self, val):
        self.current_value.set(val)
        self.dropdown_window.place_forget()
        self.is_open = False
        if self.command:
            self.command(val)
            
    def set(self, val):
        self.current_value.set(val)
        
    def get(self):
        return self.current_value.get()
        
    def configure(self, values=None, **kwargs):
        if values is not None:
            self.values = values
            self.all_values = values
            self.is_loading = False
            self.current_value.set(values[0] if values else "No existing labels found")
            self.filter_values()
        
    def cget(self, key):
        if key == "values":
            return self.values
        if hasattr(super(), "cget"):
            return super().cget(key)
        return None

class AddNewClassWindow:
    def __init__(self, app):
        self.app = app
        self.px = getattr(app, "_px", lambda x: x)
        self.fs = getattr(app, "_fs", lambda x: x)
        
        # Start with a placeholder, load asynchronously to prevent UI freeze
        self.known_labels = ["Loading labels..."]

        bg = self.app.cget("fg_color") if hasattr(self.app, "cget") else "#F8F9FA"
        self.top = ctk.CTkToplevel(self.app, fg_color=bg)
        self.top.title("Add New Slide Class")
        
        sw, sh = self.app.winfo_screenwidth(), self.app.winfo_screenheight()
        w, h = int(sw * 0.8), int(sh * 0.8)
        
        if hasattr(self.app, "winfo_x"):
            x = self.app.winfo_x() + (self.app.winfo_width() // 2) - (w // 2)
            y = self.app.winfo_y() + (self.app.winfo_height() // 2) - (h // 2)
            self.top.geometry(f"{w}x{h}+{max(0, x)}+{max(0, y)}")
        else:
            self.top.geometry(f"{w}x{h}")
            
        # Allow maximization
        self.top.minsize(int(sw * 0.5), int(sh * 0.5))
        
        self.image_paths = []
        self.image_widgets = {}
        self.empty_box = None
        self._build_ui()
        
        self.top.update_idletasks()
        
        self._resize_timer = None
        self._last_w = self.top.winfo_width()
        
        try:
            self.top.wait_visibility()
        except Exception:
            pass
        self.top.focus()
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start the background thread ONLY AFTER UI IS BUILT to prevent race condition
        import threading
        threading.Thread(target=self._load_labels_bg, daemon=True).start()

    def _load_labels_bg(self):
        try:
            import torch
            import os
            pt_path = "/home/kk/Desktop/CV projects/Rail-and-Road-Auction-main/DesktopApp/models/val_embeddings.pt"
            if os.path.exists(pt_path):
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                labels = sorted(list(set(data["labels"])))
                if hasattr(self, "combo_label") and self.top.winfo_exists():
                    self.top.after(0, lambda: self.combo_label.configure(values=labels))
        except Exception as e:
            print("Could not load embeddings labels:", e)
            if hasattr(self, "combo_label") and self.top.winfo_exists():
                self.top.after(0, lambda: self.combo_label.configure(values=["No labels found"]))

    def _build_ui(self):
        px, fs = self.px, self.fs
        bg = self.app.cget("fg_color") if hasattr(self.app, "cget") else "#F8F9FA"
        
        # Wrap everything in a main scrollable container to allow overflowing UI elements to be reached
        self.main_container = ctk.CTkScrollableFrame(self.top, fg_color=bg, corner_radius=0)
        self.main_container.pack(fill="both", expand=True)
        
        top_bar = ctk.CTkFrame(self.main_container, fg_color=bg)
        top_bar.pack(fill="x", padx=px(20), pady=px(20))
        
        title_lbl = ctk.CTkLabel(top_bar, text="Upload Images for New Class", font=ctk.CTkFont("Inter", size=fs(18), weight="bold"), text_color=C.get("text", "#000"))
        title_lbl.pack(side="left")
        
        btn_clear = ctk.CTkButton(top_bar, text="Clear All", 
                                  font=ctk.CTkFont("Inter", size=fs(14), weight="bold"), 
                                  fg_color="#EF4444", hover_color="#B91C1C", text_color="white",
                                  height=px(36), corner_radius=px(8), width=px(100),
                                  command=self._clear_all_images)
        btn_clear.pack(side="right", padx=(px(12), 0))
        
        btn_folder = ctk.CTkButton(top_bar, text="Upload Folder", 
                                   font=ctk.CTkFont("Inter", size=fs(14), weight="bold"), 
                                   fg_color=C.get("olive", "#8C7B5D"), hover_color=C.get("olive_h", "#AA9874"), 
                                   height=px(36), corner_radius=px(8),
                                   command=self._upload_folder)
        btn_folder.pack(side="right", padx=(px(12), 0))
        
        btn_images = ctk.CTkButton(top_bar, text="Upload Images", 
                                   font=ctk.CTkFont("Inter", size=fs(14), weight="bold"), 
                                   fg_color="transparent", text_color=C.get("olive", "#8C7B5D"),
                                   border_width=2, border_color=C.get("olive", "#8C7B5D"), 
                                   hover_color=C.get("card", "#F5F2EC"),
                                   height=px(36), corner_radius=px(8),
                                   command=self._upload_images)
        btn_images.pack(side="right")
        
        # More prominent shadow frame behind upload_box for depth effect
        shadow = ctk.CTkFrame(self.main_container, fg_color="#A3A19C", corner_radius=px(18), height=px(500)) 
        shadow.pack_propagate(False) # Force fixed minimum height inside scrollable frame
        self.upload_box = ctk.CTkFrame(shadow, fg_color=C.get("white", "#FFFFFF"), 
                                       border_width=2, border_color=C.get("border", "#E4E7EC"), 
                                       corner_radius=px(16))
        shadow.pack(fill="x", padx=px(106), pady=(px(4), px(54))) # Adjust to fill="x" for scrollable container
        self.upload_box.pack(fill="both", expand=True, padx=px(3), pady=(px(3), px(5)))
        
        self.grid_frame = ctk.CTkScrollableFrame(self.upload_box, fg_color="transparent", corner_radius=0)
        self.grid_frame.bind("<Configure>", self._on_grid_resize, add="+")
        
        bot_bar = ctk.CTkFrame(self.main_container, fg_color=bg)
        bot_bar.pack(fill="x", padx=px(20), pady=(0, px(20)))
        
        lbl_cls = ctk.CTkLabel(bot_bar, text="Class Label:", font=ctk.CTkFont("Inter", size=fs(15), weight="bold"), text_color=C.get("text", "#000"))
        lbl_cls.pack(side="left", padx=(0, px(12)))
        
        self.combo_label = ModernDropdown(bot_bar, px=px, values=["Loading labels..."], 
                                          width=px(300), height=px(44),
                                          font=ctk.CTkFont("Inter", size=fs(14)), is_loading=True,
                                          command=self._on_dropdown_select)
        self.combo_label.pack(side="left")
        
        self.entry_new_label = ctk.CTkEntry(bot_bar, placeholder_text="New label name...", 
                                            width=px(200), height=px(44), corner_radius=px(8),
                                            font=ctk.CTkFont("Inter", size=fs(14)))
        self.entry_new_label.pack(side="left", padx=(px(12), 0))
        self.entry_new_label.bind("<KeyRelease>", self._on_new_label_typing)
        
        self.btn_generate = ctk.CTkButton(bot_bar, text="Generate Embeddings", 
                                          font=ctk.CTkFont("Inter", size=fs(14), weight="bold"), 
                                          fg_color=C.get("olive", "#8C7B5D"), hover_color=C.get("olive_h", "#AA9874"), 
                                          text_color=C.get("white", "#FFFFFF"),
                                          height=px(48), corner_radius=px(8),
                                          state="disabled",
                                          command=self._generate_embeddings_prompt)
        self.btn_generate.pack(side="right")
        
        self._refresh_grid()

    def _show_loader(self):
        if not hasattr(self, "loader_frame"):
            self.loader_frame = ctk.CTkFrame(self.top, fg_color=C.get("white", "#FFFFFF"), corner_radius=self.px(12), border_width=1, border_color=C.get("border", "#E4E7EC"))
            self.loader_lbl = ctk.CTkLabel(self.loader_frame, text="Processing Images...", font=ctk.CTkFont("Inter", size=self.fs(16), weight="bold"), text_color=C.get("text", "#1D2939"))
            self.loader_lbl.pack(pady=(self.px(20), self.px(10)), padx=self.px(40))
            self.loader_bar = ctk.CTkProgressBar(self.loader_frame, mode="indeterminate", width=self.px(200), progress_color=C.get("olive", "#8C7B5D"))
            self.loader_bar.pack(pady=(0, self.px(20)), padx=self.px(40))
        
        self.loader_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.loader_bar.start()
        self.top.update_idletasks()

    def _hide_loader(self):
        if hasattr(self, "loader_frame") and self.loader_frame.winfo_ismapped():
            self.loader_bar.stop()
            self.loader_frame.place_forget()

    def _upload_images(self):
        from customtkinter import filedialog
        paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")], parent=self.top)
        self.top.lift()
        self.top.focus_force()
        if paths:
            new_paths = [p for p in paths if p not in self.image_paths]
            if new_paths:
                for p in new_paths:
                    self.image_paths.append(p)
                # Show loader immediately, then add images on next frame
                self._show_loader()
                self.top.update()  # Force UI redraw so loader is visible before heavy work
                self.top.after(50, lambda: self._add_images_to_grid(new_paths))

    def _upload_folder(self):
        from customtkinter import filedialog
        folder = filedialog.askdirectory(title="Select Folder", parent=self.top)
        self.top.lift()
        self.top.focus_force()
        if folder:
            new_paths = []
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    p = os.path.join(folder, f)
                    if p not in self.image_paths:
                        self.image_paths.append(p)
                        new_paths.append(p)
            if new_paths:
                # Show loader immediately before heavy image processing
                self._show_loader()
                self.top.update()  # Force UI redraw so loader is visible
                self.top.after(50, lambda: self._add_images_to_grid(new_paths))

    def _clear_all_images(self):
        for path, widget in list(self.image_widgets.items()):
            widget.destroy()
        self.image_widgets.clear()
        self.image_paths.clear()
        self._rearrange_grid()
    def _on_dropdown_select(self, val):
        if hasattr(self, "entry_new_label") and self.entry_new_label.winfo_exists():
            self.entry_new_label.delete(0, "end")

    def _on_new_label_typing(self, event):
        val = self.entry_new_label.get().strip()
        if val:
            self.combo_label.set("Creating new label...")

    def _generate_embeddings_prompt(self):
        new_label = self.entry_new_label.get().strip()
        
        if new_label:
            lower_known = [k.lower() for k in self.known_labels]
            if new_label.lower() in lower_known:
                idx = lower_known.index(new_label.lower())
                existing_name = self.known_labels[idx]
                messagebox.showinfo("Label Exists", f"The class '{existing_name}' already exists. It has been auto-selected.", parent=self.top)
                self.combo_label.set(existing_name)
                self.entry_new_label.delete(0, "end")
                return
            label = new_label
        else:
            label = getattr(self.combo_label, "get", lambda: "")()
        
        if not label or label in ["Loading labels...", "No existing labels found", "Creating new label...", ""]:
            messagebox.showerror("Error", "Please select a label from the dropdown or enter a new label name.", parent=self.top)
            return
            
        res = messagebox.askyesno(
            "Confirm Embeddings", 
            f"Are you sure all the images belong to the class '{label}'?\n\nPlease verify before proceeding.",
            parent=self.top
        )
        if res:
            if not self.image_paths:
                messagebox.showwarning("No Images", "Please upload at least one image first.", parent=self.top)
                return
                
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from frontend.interactive_cropper import InteractiveCropper
            
            def on_confirm(crop_results):
                self._extract_and_save_embeddings(label, crop_results)
                
            InteractiveCropper(self.top, self.image_paths, C, self.px, self.fs, on_confirm)
            
    def _extract_and_save_embeddings(self, label, crop_results):
        self._show_loader()
        
        def task():
            import torch
            import os
            from PIL import Image
            import torch.nn.functional as F
            from models.train_phase5 import DINOv2DualStream, clean_transform
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "..", "models")
            
            try:
                # 1. Load Model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = DINOv2DualStream(embedding_dim=256)
                weights_path = os.path.join(models_dir, "phase5_best.pth")
                if os.path.exists(weights_path):
                    model.load_state_dict(torch.load(weights_path, map_location=device))
                model.to(device)
                model.eval()
                
                # 2. Load Existing Embeddings for Deduplication
                val_path = os.path.join(models_dir, "val_embeddings.pt")
                temp_path = os.path.join(models_dir, "temp_embeddings.pt")
                
                existing_embs = []
                if os.path.exists(val_path):
                    existing_embs.append(torch.load(val_path, map_location=device, weights_only=True)["embeddings"])
                
                temp_data = {"embeddings": None, "labels": []}
                if os.path.exists(temp_path):
                    temp_data = torch.load(temp_path, map_location=device, weights_only=True)
                    if temp_data["embeddings"] is not None:
                        existing_embs.append(temp_data["embeddings"])
                        
                if existing_embs:
                    all_existing = torch.cat(existing_embs, dim=0)
                else:
                    all_existing = torch.empty((0, 256)).to(device)
                    
                new_embs = []
                new_labels = []
                
                skipped = 0
                processed = 0
                
                # 3. Process Images
                for path, boxes in crop_results.items():
                    img = Image.open(path).convert("RGB")
                    
                    g_box = boxes["global_box"]
                    l_box = boxes["local_box"]
                    
                    if g_box:
                        global_img = img.crop(g_box)
                    else:
                        global_img = img
                        
                    if l_box:
                        local_img = img.crop(l_box)
                    else:
                        local_img = img
                        
                    g_tensor = clean_transform(global_img).unsqueeze(0).to(device)
                    l_tensor = clean_transform(local_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = model.forward_one(g_tensor, l_tensor) # [1, 256]
                        emb = F.normalize(emb, p=2, dim=1)
                        
                    # Deduplication
                    if len(all_existing) > 0:
                        sims = F.cosine_similarity(emb, all_existing)
                        if sims.max().item() >= 0.9999:
                            skipped += 1
                            continue
                            
                    new_embs.append(emb)
                    new_labels.append(label)
                    
                    # Also append to all_existing so subsequent images in this batch don't duplicate each other
                    all_existing = torch.cat([all_existing, emb], dim=0)
                    processed += 1
                    
                # 4. Save to temp
                if new_embs:
                    stacked_new = torch.cat(new_embs, dim=0)
                    if temp_data["embeddings"] is not None:
                        temp_data["embeddings"] = torch.cat([temp_data["embeddings"], stacked_new], dim=0)
                    else:
                        temp_data["embeddings"] = stacked_new
                    temp_data["labels"].extend(new_labels)
                    
                    torch.save(temp_data, temp_path)
                    
                self.top.after(0, self._hide_loader)
                
                msg = f"Processed {processed} images."
                if skipped > 0:
                    msg += f"\nSkipped {skipped} duplicates."
                    
                self.top.after(0, lambda: messagebox.showinfo("Success", msg, parent=self.top))
                
                # Update dropdown if it's a completely new label
                if label not in self.known_labels:
                    self.known_labels.append(label)
                    self.known_labels.sort()
                    vals = list(self.combo_label.cget("values"))
                    if vals == ["No existing labels found"]: vals = []
                    vals.append(label)
                    vals.sort()
                    self.top.after(0, lambda: self.combo_label.configure(values=vals))
                    self.top.after(0, lambda: self.combo_label.set(label))
                    
                self.top.after(0, lambda: self.entry_new_label.delete(0, "end"))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.top.after(0, self._hide_loader)
                self.top.after(0, lambda: messagebox.showerror("Error", f"Pipeline failed: {str(e)}", parent=self.top))
                
        import threading
        threading.Thread(target=task, daemon=True).start()
            
    def _bind_scroll(self, widget):
        canvas = getattr(self.grid_frame, "_parent_canvas", None)
        if not canvas:
            return
            
        def on_mouse_wheel(event):
            if event.num == 4:
                canvas.yview_scroll(-3, "units")
            elif event.num == 5:
                canvas.yview_scroll(3, "units")
            else:
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"
                
        widget.bind("<Button-4>", on_mouse_wheel, add="+")
        widget.bind("<Button-5>", on_mouse_wheel, add="+")
        widget.bind("<MouseWheel>", on_mouse_wheel, add="+")
        
        for child in widget.winfo_children():
            self._bind_scroll(child)

    def _remove_image(self, path):
        if path not in self.image_paths:
            return
            
        # Find position of removed image BEFORE removing it
        removed_idx = self.image_paths.index(path)
        self.image_paths.remove(path)
        
        if path in self.image_widgets:
            self.image_widgets[path].grid_forget()
            self.image_widgets[path].destroy()
            del self.image_widgets[path]
            
        # If no images left, show empty state
        if not self.image_paths:
            self.btn_generate.configure(state="disabled")
            self._show_empty_state()
            return
            
        self.btn_generate.configure(state="normal")
        
        # Calculate current columns (same logic as _rearrange_grid)
        gw = self.grid_frame.winfo_width()
        if gw < 10:
            gw = self.top.winfo_width() - self.px(224)
        columns = max(1, gw // self.px(160))
        
        # ONLY re-grid images from the removed index onwards
        # Images before removed_idx are untouched - zero visual flicker!
        for i in range(removed_idx, len(self.image_paths)):
            path_i = self.image_paths[i]
            if path_i in self.image_widgets:
                frame = self.image_widgets[path_i]
                new_row = i // columns
                new_col = i % columns
                frame.grid_configure(row=new_row, column=new_col)
                
        # Update cache so _rearrange_grid doesn't re-run on next configure event
        self._last_cols = columns
        self._last_img_cnt = len(self.image_paths)

    def _show_empty_state(self):
        px = self.px
        # Hide grid frame if it is mapped
        if hasattr(self, "grid_frame") and self.grid_frame.winfo_ismapped():
            self.grid_frame.pack_forget()

        if not self.empty_box:
            # Create a beautiful elevated empty-state box
            self.empty_box = ctk.CTkFrame(self.upload_box, 
                                     fg_color=C.get("card", "#F5F2EC"),
                                     corner_radius=px(16),
                                     border_width=2,
                                     border_color=C.get("border", "#E4E7EC"),
                                     width=px(650), height=px(380))
            self.empty_box.pack_propagate(False)
            
            # Inner container to center everything
            inner = ctk.CTkFrame(self.empty_box, fg_color="transparent")
            inner.place(relx=0.5, rely=0.5, anchor="center")
            
            icon_lbl = ctk.CTkLabel(inner, text="📁", font=ctk.CTkFont("Inter", size=self.fs(56)))
            icon_lbl.pack(pady=(0, px(16)))
            
            lbl = ctk.CTkLabel(inner, text="No images uploaded yet", 
                               font=ctk.CTkFont("Inter", size=self.fs(22), weight="bold"), 
                               text_color=C.get("text", "#1D2939"))
            lbl.pack()
            
            sub_lbl = ctk.CTkLabel(inner, text="Click 'Upload Images' or 'Upload Folder' above to get started.", 
                                   font=ctk.CTkFont("Inter", size=self.fs(16)), 
                                   text_color=C.get("muted", "#667085"))
            sub_lbl.pack(pady=(px(8), 0))

        self.empty_box.pack(pady=px(60), padx=px(40), expand=True)

    def _add_images_to_grid(self, new_paths):
        px = self.px
        
        # Clean up empty state if it's there
        if self.empty_box and self.empty_box.winfo_ismapped():
            self.empty_box.pack_forget()

        import threading
        
        def process_images_thread():
            from PIL import Image
            processed_images = []
            for path in new_paths:
                try:
                    img = Image.open(path)
                    img.thumbnail((px(130), px(130)))
                    processed_images.append((path, img, img.size))
                except Exception as e:
                    processed_images.append((path, None, None))
            
            # Update UI on main thread
            self.top.after(0, lambda: self._render_processed_images(processed_images))
            
        threading.Thread(target=process_images_thread, daemon=True).start()

    def _render_processed_images(self, processed_images):
        px = self.px
        for path, img, size in processed_images:
            frame = ctk.CTkFrame(self.grid_frame, fg_color=C.get("card", "#F5F2EC"), border_width=1, border_color=C.get("border", "#E4E7EC"), corner_radius=px(12))
            
            if img is not None:
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
                
                img_lbl = ctk.CTkLabel(frame, image=ctk_img, text="", cursor="hand2")
                img_lbl.pack(pady=px(12), padx=px(12))
                
                # Bind left click to open preview
                img_lbl.bind("<Button-1>", lambda e, p=path: ImagePreviewWindow(self.top, p, self.px, self.fs, self._remove_image))
                
                del_btn = ctk.CTkButton(frame, text="✕", width=px(26), height=px(26), corner_radius=13, 
                                        fg_color="#EF4444", hover_color="#B91C1C", text_color="white", 
                                        font=ctk.CTkFont("Inter", size=self.fs(12), weight="bold"), 
                                        command=lambda p=path: self._remove_image(p))
                del_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-px(6), y=px(6))
                
                # Apply custom scroll binding so mouse scrolling works perfectly over images
                self._bind_scroll(frame)
            else:
                lbl = ctk.CTkLabel(frame, text="Error")
                lbl.pack(pady=px(20))
                
            self.image_widgets[path] = frame
            
        self._rearrange_grid()
        self._hide_loader()
        
    def _rearrange_grid(self):
        if not self.image_paths:
            self.btn_generate.configure(state="disabled")
            self._show_empty_state()
            return
            
        self.btn_generate.configure(state="normal")
            
        if self.empty_box and self.empty_box.winfo_ismapped():
            self.empty_box.pack_forget()
            
        if not self.grid_frame.winfo_ismapped():
            self.grid_frame.pack(fill="both", expand=True, padx=self.px(10), pady=self.px(10))
            
        # Dynamically calculate columns based on grid width
        gw = self.grid_frame.winfo_width()
        if gw < 10:
            gw = self.top.winfo_width() - self.px(224) # Approx if not drawn
            
        col_w = self.px(160)
        columns = max(1, gw // col_w)
        
        # Optimization: Prevent infinite <Configure> loop!
        if getattr(self, "_last_cols", None) == columns and getattr(self, "_last_img_cnt", None) == len(self.image_paths):
            return
        self._last_cols = columns
        self._last_img_cnt = len(self.image_paths)
        
        # Reset previous column weights
        for c in range(20):
            self.grid_frame.grid_columnconfigure(c, weight=0)
            
        # Center the active columns
        for c in range(columns):
            self.grid_frame.grid_columnconfigure(c, weight=1)
            
        for i, path in enumerate(self.image_paths):
            if path in self.image_widgets:
                frame = self.image_widgets[path]
                row = i // columns
                col = i % columns
                # Only move if position actually changed
                try:
                    info = frame.grid_info()
                    if info.get("row") != row or info.get("column") != col:
                        frame.grid_configure(row=row, column=col)
                except Exception:
                    frame.grid(row=row, column=col, padx=self.px(12), pady=self.px(12))

    def _on_grid_resize(self, event):
        # Ignore non-width changing events (prevents infinite recursive grid loops)
        if getattr(self, "_last_grid_w", None) == event.width:
            return
        self._last_grid_w = event.width
        
        # Debounce the resize event to prevent flickering during drag-resize
        if self._resize_timer:
            self.grid_frame.after_cancel(self._resize_timer)
        self._resize_timer = self.grid_frame.after(150, self._rearrange_grid)

    def _refresh_grid(self):
        self._show_empty_state()

    def _on_close(self):
        """Smoothly hide the window instantly for better UX, then destroy it in the background."""
        self.top.withdraw()
        self.top.update_idletasks()
        self.top.after(50, self.top.destroy)

class SyncApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.C = C
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
        self._compute_fonts(w)

        # Bind messagebox to app instance so it inherits scale/font flawlessly
        messagebox.app = self

        # Unified Adaptive Padding: Balanced for both small and large screens
        # 450px for ultra-wide, otherwise scales (12%), with a safe minimum of 80px
        self._init_pad = 450 if w > 3000 else max(80, int(w * 0.12))
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
        self._session_token = None # Persistent ONLY within this session (Memory-only)
        self.last_sync_results = {} # book_id -> pages list

        self._load_config()
        self._build_ui()
        self._poll_log()
        self.after(500, self._debug_scale)
        # Configure binding: Linux only — on Windows it causes hang + wrong padding on maximize/minimize
        import platform
        if platform.system() != "Windows":
            self.bind("<Configure>", self._on_resize)

        # Connect ocr_pipeline logs to our UI activity log
        try:
            import main_mineru_ocr as ocr_pipeline
            ocr_pipeline.LOG_CALLBACK = self._log
        except Exception as e:
            self._log(f"⚠️ Could not hook OCR logs: {e}")

        # Graceful close protocol handler
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)

    def _on_app_close(self):
        """Handle app close event by stopping any running background threads safely first."""
        try:
            self._log("👋 Closing application gracefully...")
            self.sync_running = False
            # Allow a short delay for threads to exit
            self.after(200, self._destroy_app_safely)
        except Exception:
            self.destroy()

    def _destroy_app_safely(self):
        try:
            # Force destroy all widgets and child windows
            for win in list(self._detail_windows.values()):
                try: win.destroy()
                except: pass
            self.destroy()
        except Exception:
            pass

    def _get_os_scale(self):
        """
        Detect the OS-level display scale factor.
        Returns a float >= 1.0.
        """
        import platform, os as _os
        
        # Windows handles scaling automatically via CTK, so we return 1.0 to avoid double-scaling
        if platform.system() == "Windows":
            return 1.0
            
        # Linux specific (GDK/Wayland) where manual scaling is needed to look 'perfect'
        if platform.system() == "Linux":
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
        Calculate font sizes proportional to window width.
        Windows: s = width-based only (CTK handles OS scale).
        Linux:   s = width-based * os_scale.
        """
        import platform
        os_scale = self._get_os_scale()
        
        # Base scale from width
        s_base = (win_w / 1280.0)
        
        if platform.system() == "Windows":
            # On Windows, we just use the width scale, but slightly higher baseline
            s = s_base * 1.05 # 5% fudge factor to feel more 'Linux-like'
        else:
            # On Linux, multiply by our detected OS scale
            s = s_base * os_scale
            
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
        """Update padding on resize (debounced). Font refresh skipped on Windows to prevent hang."""
        if getattr(self, "_resize_lock", False):
            return
        if event and event.widget is not self:
            return

        new_w = self.winfo_width()
        if new_w < 100:
            return

        old_w = getattr(self, "_last_resize_w", 0)
        if abs(new_w - old_w) < 15:
            return

        import platform
        debounce = 600 if platform.system() == "Windows" else 300

        if hasattr(self, "_resize_job") and self._resize_job:
            try: self.after_cancel(self._resize_job)
            except: pass

        self._resize_job = self.after(debounce, lambda: self._do_resize(new_w))

    def _do_resize(self, new_w):
        """Actual resize logic — runs once after debounce settles."""
        if getattr(self, "_resize_lock", False):
            return

        actual_w = self.winfo_width()
        if abs(actual_w - new_w) > 50:
            new_w = actual_w

        self._last_resize_w = new_w
        self._resize_job = None
        self._resize_lock = True

        import platform
        is_windows = platform.system() == "Windows"

        try:
            self._compute_fonts(new_w)
            self._win_w = new_w

            # Padding: use the fixed startup value — never recalculate during resize
            # This prevents margin collapse on maximize/minimize
            pad = self._init_pad
            for widget_name in ("_dashboard_label", "_cards_frame", "_act_frame_outer"):
                widget = getattr(self, widget_name, None)
                if widget:
                    try: widget.grid_configure(padx=pad)
                    except: pass

            # Font refresh: SKIP on Windows (causes hang by updating 100s of widgets at once)
            # On Linux it's fine because the OS handles redraws differently
            if not is_windows:
                old_scale = getattr(self, "_prev_scale", 0)
                if abs(self._scale - old_scale) >= 0.01:
                    self._prev_scale = self._scale
                    self._refresh_fonts()

        finally:
            self._resize_lock = False

    def _refresh_fonts(self):
        """Update fonts on all registered widgets (only if size changed)."""
        font_map = getattr(self, "_font_registry", {})
        font_size_cache = getattr(self, "_font_size_cache", {})
        updated = 0
        for key, (widget, base_size, family, weight) in list(font_map.items()):
            try:
                new_size = max(9, int(round(base_size * self._scale)))
                # Skip if this widget's font size hasn't changed
                if font_size_cache.get(key) == new_size:
                    continue
                font_size_cache[key] = new_size
                widget.configure(font=ctk.CTkFont(family=family, size=new_size, weight=weight))
                updated += 1
            except Exception:
                font_map.pop(key, None)
                font_size_cache.pop(key, None)
        self._font_size_cache = font_size_cache

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
        # --- Fix 12: Threaded Save ---
        def _task():
            try:
                with open(CONFIG_FILE, "w") as f:
                    json.dump(self.config, f, indent=2)
                # Also sync important keys to .env for fallback
                with open(".env", "w") as f:
                    for k, v in self.config.items():
                        if v is not None:
                            f.write(f"{k}={str(v)}\n")
            except Exception as e:
                self.after(0, lambda: self._log(f"⚠️ Error saving config: {e}"))
        threading.Thread(target=_task, daemon=True).start()

    # ─── Build UI ─────────────────────────────────────────────────────────────
    def _open_add_new_class(self):
        if not self.current_user:
            messagebox.showwarning("Authorization Required", "Please enter a valid token to authorize before proceeding.")
            return
        self._add_class_win = AddNewClassWindow(self)

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
        right = ctk.CTkFrame(hdr, fg_color=C["white"])
        right.grid(row=0, column=2, sticky="e", padx=24, pady=16)

        self._reg(ctk.CTkLabel(right, text="Connection Status :",
                     font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="normal"),
                     text_color="#000000", fg_color=C["white"]), 13, "Inter", "normal").pack(side="left", padx=(0, 8))

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
                    import platform
                    from PIL import Image
                    # Windows: CTkImage multiplies by DPI scale, so use smaller base size
                    icon_size = (16, 16) if platform.system() == "Windows" else (20, 20)
                    return ctk.CTkImage(light_image=Image.open(p), size=icon_size)
                except Exception:
                    try:
                        return PhotoImage(file=p)
                    except Exception:
                        return None
            return None
        self._conn_icon_green = _mk_icon(g)
        self._conn_icon_grey  = _mk_icon(gy) or None
        self._conn_icon_red   = _mk_icon(r)  or None

        self.icon_img_lbl = ctk.CTkLabel(self.conn_badge, text="", image=self._conn_icon_grey)
        self.icon_txt_lbl = ctk.CTkLabel(self.conn_badge, text="!",
                                       font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                                       text_color=C["s_skip_fg"])
        self._reg(self.icon_txt_lbl, 13, "Inter", "bold")
        # Initially show grey icon
        self.icon_img_lbl.pack(side="left", padx=(10, 6))

        # --- Connection Spinner (Circular Loader) ---
        self.conn_spinner_canvas = Canvas(self.conn_badge, width=20, height=20, bd=0, 
                                        highlightthickness=0, bg=C["s_skip_bg"])
        self.conn_spinner_arc = self.conn_spinner_canvas.create_arc(3, 3, 17, 17, start=0, 
                                                                  extent=280, style="arc", 
                                                                  outline="#92400E", width=3)
        self._conn_spinner_state = {"angle": 0, "job": None}

        self.conn_text = ctk.CTkLabel(self.conn_badge, text="Token Required",
                                      font=ctk.CTkFont(family="Inter", size=self.F["label"], weight="bold"),
                                      text_color=C["s_skip_fg"])
        self._reg(self.conn_text, 13, "Inter", "bold")
        self.conn_text.pack(side="left", padx=(0, 10))
        self._set_conn_visual("neutral")

        self.btn_settings = ctk.CTkButton(right, text="⚙", width=self._px(36), height=self._px(36),
                      corner_radius=self._px(18),
                      fg_color="transparent",
                      border_width=1, border_color=C["border"],
                      text_color=C["muted"],
                      hover_color="#F5F5F5",
                      font=ctk.CTkFont(size=self.F["heading"]),
                      state="disabled", # Initially disabled
                      command=self._open_settings)
        self._reg(self.btn_settings, 16, "Inter", "normal").pack(side="left", padx=(0, 8))

        self.btn_refresh = ctk.CTkButton(right, text="↻", width=self._px(36), height=self._px(36),
                      corner_radius=self._px(18),
                      fg_color="transparent",
                      border_width=1, border_color=C["border"],
                      text_color=C["muted"],
                      hover_color="#F5F5F5",
                      font=ctk.CTkFont(size=self.F["heading"]),
                      state="disabled", # Initially disabled
                      command=self._check_models_manual)
        self._reg(self.btn_refresh, 16, "Inter", "normal").pack(side="left", padx=(0, 8))

        self.btn_stop = ctk.CTkButton(right, text="⏹", width=self._px(36), height=self._px(36),
                                      corner_radius=self._px(18),
                                      fg_color="transparent",
                                      border_width=1, border_color=C["border"],
                                      text_color=C["s_fail_fg"],
                                      hover_color="#FDE8E8", # Light red hover
                                      font=ctk.CTkFont(size=self.F["heading"]),
                                      command=self._stop_sync)
        self.btn_stop.pack(side="left")
        self.btn_stop.configure(state="disabled")

    def _set_conn_visual(self, state, msg=None):
        try:
            # Hide both status widgets initially
            self.icon_img_lbl.pack_forget()
            self.icon_txt_lbl.pack_forget()
            
            if hasattr(self, "conn_spinner_canvas"):
                self.conn_spinner_canvas.pack_forget()
                if self._conn_spinner_state.get("job"):
                    try: self.after_cancel(self._conn_spinner_state["job"])
                    except: pass
                    self._conn_spinner_state["job"] = None
        except Exception:
            pass

        if state == "neutral":
            self.conn_badge.configure(fg_color=C["s_skip_bg"]) 
            self.conn_text.configure(text="Token Required", text_color=C["s_skip_fg"]) 
            img = getattr(self, "_conn_icon_grey", None)
            if img:
                self.icon_img_lbl.configure(image=img)
                self.icon_img_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
            else:
                self.icon_txt_lbl.configure(text="!", text_color=C["s_skip_fg"])
                self.icon_txt_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
        elif state == "connecting":
            self.conn_badge.configure(fg_color="#FEF9C3") 
            self.conn_text.configure(text=msg if msg else "Connecting...", text_color="#92400E") 
            self.conn_spinner_canvas.configure(bg="#FEF9C3")
            self.conn_spinner_canvas.pack(side="left", padx=(10, 6), before=self.conn_text)
            self._spin_conn_loader()
        elif state == "active":
            self.conn_badge.configure(fg_color=C["s_g_bg"]) 
            self.conn_text.configure(text="Active", text_color=C["s_g_fg"]) 
            img = getattr(self, "_conn_icon_green", None)
            if img:
                self.icon_img_lbl.configure(image=img)
                self.icon_img_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
            else:
                self.icon_txt_lbl.configure(text="✔", text_color=C["s_g_fg"])
                self.icon_txt_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
        elif state in ("invalid", "failed", "offline"):
            self.conn_badge.configure(fg_color=C["s_fail_bg"]) 
            txt = "Offline" if state == "offline" else ("Invalid Token" if state == "invalid" else "Failed")
            self.conn_text.configure(text=txt, text_color=C["s_fail_fg"]) 
            img = getattr(self, "_conn_icon_red", None)
            if img:
                self.icon_img_lbl.configure(image=img)
                self.icon_img_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
            else:
                self.icon_txt_lbl.configure(text="!", text_color=C["s_fail_fg"])
                self.icon_txt_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
        elif state == "idle":
            self.conn_badge.configure(fg_color="#F3F4F6")
            self.conn_text.configure(text="Idle", text_color="#6B7280")
            img = getattr(self, "_conn_icon_grey", None) or getattr(self, "_conn_icon_green", None)
            if img:
                self.icon_img_lbl.configure(image=img)
                self.icon_img_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)
            else:
                self.icon_txt_lbl.configure(text="", text_color="#6B7280")
                self.icon_txt_lbl.pack(side="left", padx=(10, 6), before=self.conn_text)

    def _spin_conn_loader(self):
        if not self.winfo_exists() or not self.conn_spinner_canvas.winfo_exists():
            return
        self._conn_spinner_state["angle"] = (self._conn_spinner_state["angle"] + 20) % 360
        self.conn_spinner_canvas.itemconfigure(self.conn_spinner_arc, start=self._conn_spinner_state["angle"])
        try: self.conn_spinner_canvas.tk_raise()
        except: pass
        self._conn_spinner_state["job"] = self.after(30, self._spin_conn_loader)

    def _check_models_on_start(self):
        def _task():
            ok, msg, missing = model_manager.health_check()
            if self.winfo_exists():
                self.after(0, lambda: self._on_health_check_start_done(ok, msg, missing))
        threading.Thread(target=_task, daemon=True).start()

    def _on_health_check_start_done(self, ok, msg, missing):
        if not self.winfo_exists(): return
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
        def _task():
            ok, msg, missing = model_manager.health_check()
            
            # 3. Connection Health: Check if internet/DB is reachable
            conn_ok = True
            if self.db_connector:
                # Fast ping
                conn_ok = self.db_connector.ping()
            else:
                conn_ok = False

            if self.winfo_exists():
                self.after(0, lambda: self._on_health_check_manual_done(ok, msg, missing, conn_ok))
        threading.Thread(target=_task, daemon=True).start()

    def _on_health_check_manual_done(self, ok, msg, missing, conn_ok=True):
        if not self.winfo_exists(): return
        
        # Show specific internet error ONLY if we actually have a DB connector and it failed.
        # If we don't have a token, we don't care about DB connection here.
        if not conn_ok and self.db_connector:
            messagebox.showwarning("Connection Error", 
                "Could not reach the database.\n\nPlease check your internet connection and try again.")
            self._set_conn_visual("offline")
            return

        if ok:
            messagebox.showinfo("Health Check", "All systems active!")
            # If we have a session token, we should be active. 
            has_token = bool(self._session_token)
            if has_token:
                self._set_conn_visual("active" if self.db_connector and self.db_connector.connected else "offline")
            else:
                self._set_conn_visual("neutral")
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

        # Use the startup padding value (calculated once from screen width)
        init_pad = self._init_pad

        # Dashboard title
        lbl = ctk.CTkLabel(self.scroll, text="Dashboard",
                     font=ctk.CTkFont(family="Inter", size=self.F["section"], weight="bold"),
                     text_color=C["text"], fg_color=C["bg"])
        self._dashboard_label = self._reg(lbl, 20, "Inter", "bold")
        self._dashboard_label.grid(
            row=0, column=0, sticky="w", padx=init_pad, pady=(28, 18))

        # ── 3 Cards — token card (left, wider) + upload pair (right) ────────
        cards = ctk.CTkFrame(self.scroll, fg_color=C["bg"])
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
        upload_pair = ctk.CTkFrame(cards, fg_color=C["bg"])
        upload_pair.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=4)
        upload_pair.grid_columnconfigure(0, weight=1, uniform="up")
        upload_pair.grid_columnconfigure(1, weight=1, uniform="up")
        upload_pair.grid_rowconfigure(0, weight=1)

        self._build_upload_card(upload_pair, col=0,
                                title="Upload Slides",
                                desc="Automatically analyze your scanned slides with AI to extract colors, logos, text, and catalog-ready details.",
                                bg="#8C7B5D", title_color="#FFFFFF", desc_color="#E5E5E5",
                                icon_bg="#AA9874",
                                cmd=lambda: self._open_train_slides_app(mode="folder"),
                                active=True,
                                width=None, height=None, border_color="#E4E7EC",
                                title_size=17, desc_size=14, wrap=218, icon_pady=(0, self._px(48)),
                                icon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "upload_icon.png"))
        self._build_upload_card(upload_pair, col=1,
                                title="Upload Books",
                                desc="Process book cover images to identify titles, authors, and generate complete auction-ready descriptions.",
                                bg="#F5F2EC", title_color="#090909", desc_color="#808080",
                                icon_bg="#AA9874",
                                cmd=lambda: self._browse_folder("books"),
                                show_manual=True,
                                width=None, height=None, border_color="#E4E7EC",
                                title_size=17, desc_size=14, wrap=218, icon_pady=(0, self._px(48)),
                                icon_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons", "upload_icon.png"))

        # Row 1 of cards: All three buttons aligned
        cards.grid_rowconfigure(1, weight=0)

        new_slide_btn = ctk.CTkButton(cards, text="Add a new slide class in model",
                              height=55, corner_radius=12,
                              font=ctk.CTkFont(family="Inter", size=self._fs(9), weight="normal"),
                              fg_color="transparent", border_width=2, border_color="#8C7B5D",
                              text_color="#8C7B5D", hover_color="#F5F2EC",
                              command=self._open_add_new_class)
        new_slide_btn.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(12, 0))

        manual_buttons_frame = ctk.CTkFrame(cards, fg_color=C["bg"])
        manual_buttons_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=0)
        manual_buttons_frame.grid_columnconfigure(0, weight=1, uniform="up")
        manual_buttons_frame.grid_columnconfigure(1, weight=1, uniform="up")

        m_slides_btn = ctk.CTkButton(manual_buttons_frame, text="Process Single Lot of Slides",
                              height=55, corner_radius=12,
                              font=ctk.CTkFont(family="Inter", size=self._fs(9), weight="normal"),
                              fg_color="transparent", border_width=2, border_color="#8C7B5D",
                              text_color="#8C7B5D", hover_color="#F5F2EC",
                              command=lambda: self._open_train_slides_app(mode="files"))
        m_slides_btn.grid(row=0, column=0, sticky="ew", padx=8, pady=(12, 0))

        m_books_btn = ctk.CTkButton(manual_buttons_frame, text="Process Single Book Files",
                              height=55, corner_radius=12,
                              font=ctk.CTkFont(family="Inter", size=self._fs(9), weight="normal"),
                              fg_color="transparent", border_width=2, border_color="#8C7B5D",
                              text_color="#8C7B5D", hover_color="#F5F2EC",
                              command=lambda: self._browse_manual("books"))
        m_books_btn.grid(row=0, column=1, sticky="ew", padx=8, pady=(12, 0))
        
        # Recent Activity on Row 2 of self.scroll
        self._build_activity_section()

    # ── Token Card ─────────────────────────────────────────────────────────────
    def _build_token_card(self, parent):
        card = ctk.CTkFrame(parent, corner_radius=self._px(10),
                            fg_color=C["white"],
                            border_width=2, border_color=C["border"])
        card.grid(row=0, column=0, padx=(0, 8), pady=4, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        inner = ctk.CTkFrame(card, fg_color=C["white"])
        inner.grid(row=0, column=0, padx=self._px(19), pady=self._px(27), sticky="ew")
        inner.grid_columnconfigure(0, weight=1)

        self._reg(ctk.CTkLabel(inner, text="Enter Secure Connection Token",
                     font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                     text_color="#293751",
                     anchor="w"), 16, "Inter", "bold").grid(row=0, column=0, sticky="ew", pady=(0, 8))

        token_desc = self._reg(ctk.CTkLabel(inner,
                     text="Paste the token provided from the web app to securely link this desktop client to your cloud OCR system.",
                     font=ctk.CTkFont(family="Outfit", size=self._fs(14)),
                     text_color="#000000",
                     justify="left", anchor="w"), 14, "Outfit", "normal")
        token_desc.grid(row=1, column=0, sticky="ew", pady=(0, 16))

        _token_last_w = [0]
        def _token_wrap(e=None):
            try:
                w = inner.winfo_width()
                if w <= 1: return
                if w == _token_last_w[0]: return
                _token_last_w[0] = w
                scale = ctk.ScalingTracker.get_widget_scaling(inner)
                token_desc.configure(wraplength=max(100, int(w / scale) - 4))
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

        # Manual button removed from inside card for cleaner UI

        _wrap_last_w = [0]
        def _update_wrap(e=None):
            try:
                w = cf.winfo_width()
                if w <= 1: return
                if w == _wrap_last_w[0]: return
                _wrap_last_w[0] = w
                scale = ctk.ScalingTracker.get_widget_scaling(cf)
                wl = max(100, int(w / scale) - 4)
                _title_lbl.configure(wraplength=wl)
                _desc_lbl.configure(wraplength=wl)
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
        # Use the same unified startup padding
        init_pad = self._init_pad
        self.act_frame = ctk.CTkFrame(self.scroll,
                                      fg_color=C["white"],
                                      border_width=1, border_color=C["border"],
                                      corner_radius=self._px(12))
        self.act_frame.grid(row=3, column=0, sticky="ew", padx=init_pad, pady=(0, 32))
        self._act_frame_outer = self.act_frame
        self.act_frame.grid_columnconfigure(0, weight=1)

        # Section title
        self._reg(ctk.CTkLabel(self.act_frame, text="Recent Activity",
                     font=ctk.CTkFont(family="Outfit", size=self.F["heading"], weight="bold"),
                     text_color=C["text"], fg_color=C["white"]), 16, "Outfit", "bold").pack(anchor="w", padx=20, pady=(16, 0))
        ctk.CTkFrame(self.act_frame, height=1, fg_color=C["border"]).pack(fill="x", padx=20, pady=(8, 0))

        # Table header
        th = ctk.CTkFrame(self.act_frame, fg_color=C["white"], height=self._px(35))
        th.pack(fill="x", padx=20, pady=(8, 0))
        th.grid_columnconfigure(0, weight=4, uniform="th")
        th.grid_columnconfigure(1, weight=3, uniform="th")
        th.grid_columnconfigure(2, weight=2, uniform="th")
        th.grid_columnconfigure(3, weight=2, uniform="th")
        self._table_header = th

        for i, col_name in enumerate(["File Name", "Status", "Type", "Upload Time"]):
            self._reg(ctk.CTkLabel(th, text=col_name,
                         font=ctk.CTkFont(family="Outfit", size=self.F["table_h"], weight="bold"),
                         text_color="#000000", anchor="w", fg_color=C["white"]), 13, "Outfit", "bold").grid(
                row=0, column=i, sticky="ew", padx=(4, 0), pady=8)

        # Header divider
        import tkinter as _tk_
        _sep0 = _tk_.Canvas(self.act_frame, height=1, bd=0, highlightthickness=0, bg="#BABABA")
        _sep0.pack(fill="x", padx=20)

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
        
        # Logs moved to Settings window — not shown on main page for clean client UI
        # Create a hidden log_box for background logging (not visible on main page)
        from tkinter import Text
        self.log_box = Text(self.act_frame)
        self.log_box.pack_forget()  # completely hidden
        self.log_box.configure(state="disabled")
        self.log_visible = False

    def _add_activity_row(self, book_id, status, type_, timestamp, error_msg=None):
        """Add a new row to the activity table — optimized for performance."""
        bg, fg = STATUS_STYLES.get(status, (C["s_o_bg"], C["s_o_fg"]))

        # Row frame
        _pad = self._px(14)
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
                                text_color=C["text"], anchor="w",
                                fg_color="#FFFFFF") # Use solid background to avoid Windows ghosting
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
                                text_color=C["muted"], anchor="w",
                                fg_color="#FFFFFF") # Use solid background to avoid Windows ghosting
        type_lbl.grid(row=0, column=2, sticky="ew", pady=_pad)

        # Time
        time_lbl = ctk.CTkLabel(row, text=timestamp,
                                font=ctk.CTkFont(family="Outfit", size=self.F["table_b"]),
                                text_color=C["muted"], anchor="w",
                                fg_color="#FFFFFF") # Use solid background to avoid Windows ghosting
        time_lbl.grid(row=0, column=3, sticky="ew", pady=_pad)

        # Divider
        import tkinter as _tk_
        _sep = _tk_.Canvas(self.rows_container, height=1, bd=0, highlightthickness=0, bg="#BABABA")
        _sep.pack(fill="x", padx=10)

        # Recursive binding for scroll events
        def _bind_mouse_wheel(widget):
            def dual_scroll(direction):
                # Scroll BOTH inner and outer frames
                self.rows_container._parent_canvas.yview_scroll(direction, "units")
                self.scroll._parent_canvas.yview_scroll(direction, "units")

            # Universal binding for both Linux and Windows/macOS to handle dual scrolling
            if sys.platform.startswith("linux"):
                widget.bind("<Button-4>", lambda e: dual_scroll(-1), add="+")
                widget.bind("<Button-5>", lambda e: dual_scroll(1), add="+")
            else:
                widget.bind("<MouseWheel>", lambda e: dual_scroll(int(-1*(e.delta/120))), add="+")
            
            for child in widget.winfo_children():
                _bind_mouse_wheel(child)
        _bind_mouse_wheel(row)

        # Store error message in row metadata
        self.activity_rows[book_id] = {
            "row": row, "badge": badge, "type_lbl": type_lbl, 
            "time_lbl": time_lbl, "error_msg": error_msg
        }
        self.row_order.append(book_id)

        def _open_detail(_e=None):
            # Check if this is a Train Lot
            current_type = type_lbl.cget("text")
            if current_type == "Train Lot":
                current_status = badge.cget("text")
                if "Failed" in current_status:
                    stored_err = self.activity_rows[book_id].get("error_msg")
                    if stored_err:
                        messagebox.showerror("Processing Failed", f"Lot {book_id} failed.\n\nReason: {stored_err}")
                    else:
                        messagebox.showinfo("Failed", f"Lot {book_id} failed to process.")
                    return
                if current_status != "Complete":
                    self._log(f"ℹ️ Detail view unavailable: {book_id} is still {current_status}")
                    return
                
                # Debounce and loading cursor
                if getattr(self, "_opening_train_lot", False):
                    return
                self._opening_train_lot = True
                self.configure(cursor="watch")
                
                def _do_open():
                    try:
                        import sys
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from frontend.train_slides_ui import open_train_lot_detail
                        open_train_lot_detail(self, book_id)
                    except Exception as ex:
                        messagebox.showerror("Error", f"Failed to open Train Lot details: {str(ex)}")
                    finally:
                        self.configure(cursor="")
                        self.after(500, lambda: setattr(self, "_opening_train_lot", False))
                
                # Schedule the heavy UI creation to allow the cursor to update immediately
                self.after(10, _do_open)
                return

            # Check connection first - User request: show "you are offline" if clicked while disconnected
            if not (self.db_connector and self.db_connector.connected):
                messagebox.showwarning("Offline", "You are offline. Please connect to view details.")
                return

            # Check current status from badge
            current_status = badge.cget("text")
            
            if current_status == "Failed":
                stored_err = self.activity_rows[book_id].get("error_msg")
                if stored_err:
                    messagebox.showerror("Processing Failed", f"Book {book_id} failed.\n\nReason: {stored_err}")
                else:
                    messagebox.showinfo("Failed", f"Book {book_id} failed to produce metadata. Check the logs for details.")
                return

            if current_status not in ["Complete", "Skipped"]:
                self._log(f"ℹ️ Detail view unavailable: {book_id} is {current_status}")
                return

            try:
                coll = self.config.get("collection", "Book Data")
                
                # --- Fix 10: Non-blocking Open ---
                def _fetch_and_open():
                    # Set waiting cursor
                    self.after(0, lambda: self.configure(cursor="watch"))
                    
                    try:
                        # 1. ALWAYS prioritize in-memory cached doc (skipped or freshly synced in this session)
                        if book_id in self.last_sync_results:
                            cached = self.last_sync_results[book_id]
                            if isinstance(cached, dict) and "doc" in cached:
                                self._log(f"⚡ [UI] Book {book_id} fetched directly from Temp List (Cache Hit!)")
                                self.after(0, lambda: _create_window(cached["doc"]))
                                return
                                
                        # 2. Fallback to Database: Fetch the NEWEST document for this book_id (sort by _id descending)
                        self._log(f"🔍 [UI] Book {book_id} not in Temp List. Fetching from Database (Cache Miss)...")
                        doc = self.db_connector.db[coll].find_one(
                            {"book_id": book_id}, 
                            sort=[("_id", -1)]
                        ) if (self.db_connector and self.db_connector.connected) else None
                        
                        self.after(0, lambda: _create_window(doc))
                    except Exception:
                        self.after(0, lambda: _create_window(None))
                    finally:
                        self.after(0, lambda: self.configure(cursor=""))

                def _create_window(doc):
                    # DEBOUNCE: Check again if already open
                    if book_id in self._detail_windows:
                        win = self._detail_windows[book_id]
                        if win.winfo_exists():
                            win.focus_set(); win.lift()
                            if win.state() == "iconic": win.deiconify()
                            return
                    
                    # Store data doc for use in window logic
                    win = ctk.CTkToplevel(self)
                    # Force window to be top-level and not hidden by main
                    win.attributes("-topmost", True)
                    win.after(100, lambda: win.attributes("-topmost", False))
                    
                    self._detail_windows[book_id] = win
                    # Proceed with window building... (rest of the logic remains same)
                    _build_window_content(win, doc)

                threading.Thread(target=_fetch_and_open, daemon=True).start()
                return

            except Exception as e:
                self._log(f"❌ Error opening detail: {e}")

        # Helper to isolate window content building to avoid deep nesting
        def _build_window_content(win, doc):
            def _on_close():
                if book_id in self._detail_windows:
                    del self._detail_windows[book_id]
                win.destroy()
            
            win.protocol("WM_DELETE_WINDOW", _on_close)
            win.title(f"Details — {book_id}")
            win.withdraw()  # Hide immediately to prevent flickering during setup

            # Proportional sizing — 50% of screen
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            w, h   = int(sw * 0.50), int(sh * 0.50)
            x      = max(0, (sw - w) // 2)
            y      = max(0, (sh - h) // 2)
            win.geometry(f"{w}x{h}+{x}+{y}")
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
                            fg_color=C["olive"], hover_color=C["olive_h"],
                            font=ctk.CTkFont(family="Inter", size=self.F["heading"])).pack(side="left", padx=16, pady=12)
            # Pack controls
            # for wdg in topbar.winfo_children(): # This loop is now redundant as the checkbox is packed directly
            #     try:
            #         wdg.pack(side="left", padx=16, pady=12)
            #     except Exception:
            #         pass
            
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

            self._reg(ctk.CTkButton(topbar, text="Close", height=self._px(32), width=self._px(80), 
                          corner_radius=self._px(8),
                          font=ctk.CTkFont(family="Inter", size=self._fs(13), weight="bold"),
                          fg_color=C["olive"], hover_color=C["olive_h"], text_color="white",
                          command=lambda: _close_win()), 13, "Inter", "bold").pack(side="right", padx=24, pady=12)

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

            # --- LOGGING SETUP ---
            # --- Layout Stabilization Loader ---
            loader_container = ctk.CTkFrame(preview, fg_color=C["white"])
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
                                  text_color=C["muted"], anchor="w", justify="left", fg_color=C["bg"]) 
            status_lbl.grid(row=2, column=0, sticky="ew", padx=12, pady=(0,8))
            win._info_scroll_canvas = None

            _status_last_w = [0]
            def _update_status_wrap(_e=None):
                try:
                    w = win.winfo_width()
                    if w == _status_last_w[0]: return
                    _status_last_w[0] = w
                    status_lbl.configure(wraplength=max(240, w - 32))
                except Exception:
                    pass

            _detail_last_bw = [0]
            def _sync_detail_split(_e=None):
                try:
                    bw = max(360, body.winfo_width() - 24)
                except Exception:
                    return
                if bw == _detail_last_bw[0]: return
                _detail_last_bw[0] = bw
                if info_var.get():
                    left = int(bw * 0.62)
                    left = max(220, min(left, bw - 220))
                    right = max(220, bw - left)
                    body.grid_columnconfigure(0, weight=0, minsize=left)
                    body.grid_columnconfigure(1, weight=1, minsize=right) # Changed col1 to weight 1 when visible
                else:
                    body.grid_columnconfigure(0, weight=1, minsize=bw)
                    body.grid_columnconfigure(1, weight=0, minsize=0)

            win._resize_job = None
            def _on_detail_resize(event):
                # Only handle window resize, ignore child widget Configure events
                if event.widget != win: return

                new_w = win.winfo_width()
                old_w = getattr(win, "_last_resize_w", 0)
                if abs(new_w - old_w) < 10:
                    return
                win._last_resize_w = new_w
                
                # Immediately show shroud to hide layout jumping
                if hasattr(win, "_show_shroud"):
                    win._show_shroud()
                
                if win._resize_job: win.after_cancel(win._resize_job)
                win._resize_job = win.after(150, _do_detail_resize)

            def _do_detail_resize():
                win._resize_job = None
                if win.winfo_exists():
                    # Reduced update_idletasks frequency for Windows stability
                    _sync_detail_split()
                    _update_status_wrap()
                    if hasattr(win, "_render_main_cmd"):
                        win._render_main_cmd()
                    
                    # Safer check for existing job
                    final_job = getattr(win, "_final_resize_job", None)
                    if final_job: win.after_cancel(final_job)
                    win._final_resize_job = win.after(400, _do_final_detail_pass)

            def _do_final_detail_pass():
                if win.winfo_exists():
                    if hasattr(win, "_render_main_cmd"):
                        win._render_main_cmd()
                    # Hide shroud after final stable pass
                    if hasattr(win, "_hide_shroud"):
                        win._hide_shroud()

            win.bind("<Configure>", _on_detail_resize, add="+")

            # --- Smooth Transition Shroud (to hide layout jitters) ---
            win._shroud = ctk.CTkFrame(win, fg_color=C["bg"])
            # Dedicated shroud spinner to avoid TclError with parentage
            win._shroud_canvas = ctk.CTkCanvas(win._shroud, width=60, height=60,
                                              bg=C["bg"], highlightthickness=0)
            win._shroud_angle = 0
            win._shroud_job = None
            
            def _rotate_shroud_spinner():
                if not win.winfo_exists(): return
                win._shroud_canvas.delete("all")
                win._shroud_angle = (win._shroud_angle + 15) % 360
                win._shroud_canvas.create_arc(5, 5, 55, 55, start=win._shroud_angle, 
                                             extent=120, outline=C["olive"], width=4, style="arc")
                win._shroud_job = win.after(40, _rotate_shroud_spinner)

            def _show_shroud():
                if win.winfo_exists():
                    win._shroud.place(relx=0, rely=0, relwidth=1, relheight=1)
                    win._shroud_canvas.place(relx=0.5, rely=0.5, anchor="center")
                    win._shroud.lift()
                    try: win._shroud_canvas.tk_raise()
                    except: pass
                    _rotate_shroud_spinner()

            def _hide_shroud():
                if win.winfo_exists():
                    win._shroud.place_forget()
                    if win._shroud_job: 
                        win.after_cancel(win._shroud_job)
                        win._shroud_job = None

            win._show_shroud = _show_shroud
            win._hide_shroud = _hide_shroud

            # Thumbs: Horizontal Scrollable Frame for 100% stability
            thumbs_scroll = ctk.CTkScrollableFrame(body, orientation="horizontal", 
                                                 fg_color=C["white"], height=150)
            thumbs_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0,12))
            # body.grid_rowconfigure(1, weight=0) is fine, it will take 'height' from widget
            
            _sync_detail_split()
            
            # Show window and process images after a short delay to keep UI snappy
            def _deferred_init():
                if not win.winfo_exists(): return
                
                # Center properly
                win.update_idletasks()
                # Center relative to screen instead of main app to guarantee middle placement
                sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                ww = getattr(win, "initial_w", win.winfo_width())
                wh = getattr(win, "initial_h", win.winfo_height())
                nx = max(0, (sw - ww) // 2)
                ny = max(0, (sh - wh) // 2)
                
                # Combine size and position to enforce centering immediately
                win.geometry(f"{ww}x{wh}+{nx}+{ny}")
                
                # --- Initial Loader: Show shroud BEFORE deiconify ---
                if hasattr(win, "_show_shroud"):
                    win._show_shroud()
                
                win.deiconify()
                win.update() # Force shroud to map and render immediately
                # Force a full layout pass AFTER deiconify to fix "jumping" glitch
                win.update()
                
                # Removed grab_set() as it causes minimization issues on Windows and blocks the main app.
                # Detail view is now non-modal, which is better for multi-tasking.

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
                
                # Identify potential search roots from config
                search_roots = []
                for key in ("folder_path", "books_path", "slides_path"):
                    root = (self.config.get(key, "") or "").strip()
                    if root and os.path.isdir(root) and root not in search_roots:
                        search_roots.append(root)

                # 1. First Pass: Try to resolve items from Doc (if available)
                if items:
                    for ent, _ in items:
                        raw = (ent.get("file_path") or ent.get("file_name") or "").strip()
                        if raw and os.path.exists(raw):
                            paths.append(raw)
                        else:
                            unresolved.append((ent, raw))

                # 1.5. NEW: Try to resolve from in-memory session cache (Fixes skipped books missing paths)
                if not paths and book_id in self.last_sync_results:
                    cached = self.last_sync_results[book_id]
                    if isinstance(cached, list) and cached and isinstance(cached[0], tuple):
                        for _, fp in sorted(cached, key=lambda x: x[0]):
                            if fp and os.path.exists(fp):
                                paths.append(fp)
                    elif isinstance(cached, dict) and "files" in cached:
                        files_list = cached["files"]
                        if files_list and isinstance(files_list[0], tuple):
                            for _, fp in sorted(files_list, key=lambda x: x[0]):
                                if fp and os.path.exists(fp):
                                    paths.append(fp)
                        else:
                            for fp in files_list:
                                if fp and os.path.exists(fp):
                                    paths.append(fp)

                # 2. Second Pass: If unresolved OR no items (Offline/No Doc), scan filesystem
                if (unresolved or not items) and search_roots:
                    # Build index of all files in search roots (smart BFS)
                    idx_exact = {} # filename.ext -> full path
                    idx_noext = {} # filename -> full path
                    for root in search_roots:
                        for r, _, files in os.walk(root):
                            for fn in files:
                                full = os.path.join(r, fn)
                                lk = fn.lower()
                                if lk not in idx_exact: idx_exact[lk] = full
                                stem = os.path.splitext(lk)[0]
                                if stem not in idx_noext: idx_noext[stem] = full

                    # If we have specific items that are unresolved, try to match them
                    for ent, raw in unresolved:
                        candidates = []
                        f_name = (ent.get("file_name") or "").strip()
                        p_id = (ent.get("page_id") or "").strip()
                        raw_base = os.path.basename((raw or "").strip())
                        if f_name: candidates.append(f_name)
                        if raw_base: candidates.append(raw_base)
                        if p_id: candidates.append(p_id)

                        found = None
                        for cand in candidates:
                            lk = cand.lower()
                            found = idx_exact.get(lk) or idx_noext.get(os.path.splitext(lk)[0])
                            if found and os.path.exists(found):
                                break
                        
                        if found:
                            paths.append(found)

                    # 3. GLOBAL FALLBACK: If we still have NO paths (or items was empty), 
                    # find ALL files matching book_id pattern (e.g. 756_001.jpg, 756.jpg)
                    if not paths:
                        bid = str(book_id).lower()
                        for fn_l, full in idx_exact.items():
                            stem = os.path.splitext(fn_l)[0]
                            if stem == bid or stem.startswith(bid + "_"):
                                if os.path.exists(full):
                                    paths.append(full)
                        # Sort them naturally so 001 comes before 002
                        paths.sort()

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
                
                # Centered with place() so it never disturbs the grid/pack layout
                main_lbl = ctk.CTkLabel(preview, text="")
                main_lbl.place(relx=0.5, rely=0.5, anchor="center")
                
                # --- Circular Spinner Overlay ---
                spinner_canvas = ctk.CTkCanvas(preview, width=60, height=60, 
                                            bg=C["white"], highlightthickness=0)
                win._spinner_active = False
                win._spinner_angle = 0
                
                def _start_spinner():
                    if win._spinner_active: return
                    win._spinner_active = True
                    # Spinner stays in absolute middle
                    spinner_canvas.place(relx=0.5, rely=0.5, anchor="center")
                    try: 
                        spinner_canvas.tk_raise()
                    except: 
                        pass
                    _rotate_spinner()

                def _stop_spinner():
                    win._spinner_active = False
                    spinner_canvas.place_forget()

                def _rotate_spinner():
                    if not win.winfo_exists() or not win._spinner_active: return
                    spinner_canvas.delete("all")
                    win._spinner_angle = (win._spinner_angle + 15) % 360
                    # Draw a stylish circular arc spinner
                    spinner_canvas.create_arc(5, 5, 55, 55, start=win._spinner_angle, 
                                             extent=120, outline=C["olive"], width=4, style="arc")
                    win.after(40, _rotate_spinner)
                
                win._start_detail_spinner = _start_spinner
                win._stop_detail_spinner = _stop_spinner
                win._spinner_canvas_ref = spinner_canvas
                win._first_load_done = False
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
                # Ensure arrows are placed after grid is ready to stay on top
                def _place_arrows():
                    return # Disabled overlay arrows as per user request
                    prev_btn.place(relx=0.03, rely=0.5, anchor="w")
                    next_btn.place(relx=0.97, rely=0.5, anchor="e")
                    prev_btn.lift()
                    next_btn.lift()
                
                _place_arrows()
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
                ai_doc = doc or {}

                if True:
                    # ── AI Results View (Always Rendered) ──────────────────────────────
                    # Scrollable to handle long content
                    info_wrap = ctk.CTkFrame(info, fg_color="#F8FAFC")
                    info_wrap.pack(fill="both", expand=True, padx=4, pady=4)

                    info_scroll = ctk.CTkScrollableFrame(
                        info_wrap,
                        fg_color="#F8FAFC",
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
                    _label_last_w = {}
                    def _on_label_resize(event, lbl, padding=48):
                        if event and lbl.winfo_exists():
                            key = id(lbl)
                            if _label_last_w.get(key) == event.width: return
                            _label_last_w[key] = event.width
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

                        ctk.CTkLabel(edit_win, text="Edit Book Title", 
                                     font=ctk.CTkFont(family="Outfit", size=26, weight="bold"),
                                     text_color="#1E293B").pack(pady=(30, 10))

                        entry_frame = ctk.CTkFrame(edit_win, fg_color="#F8FAFC", corner_radius=12, border_width=1, border_color="#E2E8F0")
                        entry_frame.pack(fill="x", padx=60, pady=10)
                        
                        entry = ctk.CTkEntry(entry_frame, height=60, border_width=0, fg_color="transparent",
                                            font=ctk.CTkFont(family="Inter", size=20, weight="bold"),
                                            placeholder_text="Type corrected title here...")
                        entry.pack(fill="x", padx=20)
                        entry.insert(0, title_lbl.cget("text"))

                        sub_entry = None
                        if subtitle_lbl and subtitle_lbl.cget("text"):
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
                            sub_entry.insert(0, subtitle_lbl.cget("text"))
                        
                        entry.focus()

                        def _save():
                            val = entry.get().strip()
                            s_val = sub_entry.get().strip() if sub_entry else ""
                            if val:
                                title_lbl.configure(text=val)
                                if subtitle_lbl:
                                    subtitle_lbl.configure(text=s_val)
                                
                                if self.db_connector and self.db_connector.connected:
                                    try:
                                        coll = self.config.get("collection", "Book Data")
                                        update_fields = {"title": val}
                                        if sub_entry: update_fields["subtitle"] = s_val
                                        self.db_connector.db[coll].update_one({"book_id": book_id}, {"$set": update_fields})
                                        self._log(f"✅ Title/Subtitle updated for {book_id}")
                                    except Exception as err: self._log(f"❌ DB Update Error: {err}")
                                edit_win.destroy()

                        btn_row = ctk.CTkFrame(edit_win, fg_color="transparent")
                        btn_row.pack(pady=30)
                        
                        common_font = ctk.CTkFont(family="Inter", size=18, weight="bold")
                        
                        ctk.CTkButton(btn_row, text="Cancel", width=200, height=52, corner_radius=10,
                                     font=common_font,
                                     fg_color="#FFFFFF", text_color="#667085", border_width=1, border_color="#E2E8F0",
                                     hover_color="#E2E8F0",
                                     command=edit_win.destroy).pack(side="left", padx=10)
                                     
                        ctk.CTkButton(btn_row, text="Save Changes", width=200, height=52, corner_radius=10,
                                     fg_color="#0F172A", text_color="white", hover_color="#1E293B",
                                     font=common_font,
                                     command=_save).pack(side="left", padx=10)
                        
                        # Bind Enter key
                        edit_win.bind("<Return>", lambda e: _save())

                    # Removed _on_edit_desc handler as requested.

                    # Edit Button (Title)
                    title_edit = ctk.CTkLabel(title_hdr, text="✎ Edit",
                                           font=ctk.CTkFont(family="Inter", size=22, weight="bold"),
                                           text_color="#2563EB", cursor="hand2")
                    title_edit.pack(side="right", padx=24)
                    title_edit.bind("<Button-1>", _on_edit_title)

                    # Title Body
                    title_body = ctk.CTkFrame(title_card, fg_color="#FFFFFF")
                    title_body.pack(fill="x", padx=24, pady=24)

                    ai_title = ai_doc.get("title", "") or "(Not Found)"
                    title_lbl = ctk.CTkLabel(title_body, text=ai_title,
                                 font=ctk.CTkFont(family="Outfit", size=32, weight="bold"),
                                 text_color="#0F172A", anchor="w",
                                 justify="left",
                                 wraplength=300,
                                 fg_color="#FFFFFF"
                                 )
                    title_lbl.pack(anchor="w", fill="x")

                    ai_subtitle = ai_doc.get("subtitle", "")
                    subtitle_lbl = None
                    if ai_subtitle and ai_subtitle != "N/A":
                        subtitle_lbl = ctk.CTkLabel(title_body, text=ai_subtitle,
                                     font=ctk.CTkFont(family="Inter", size=24, weight="normal"),
                                     text_color="#475569", anchor="w",
                                     justify="left",
                                     wraplength=300,
                                     fg_color="#FFFFFF"
                                     )
                        subtitle_lbl.pack(anchor="w", fill="x", pady=(8, 0))

                    _title_card_last_w = [0]
                    def _on_title_resize(e, l=title_lbl, sl=subtitle_lbl):
                        w = e.width - 64
                        if w == _title_card_last_w[0]: return
                        _title_card_last_w[0] = w
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
                    description_val = ai_doc.get("description", "") or "(Not Found)"
                    desc_lbl = ctk.CTkLabel(desc_card, text=description_val,
                                 font=ctk.CTkFont(family="Inter", size=22),
                                 text_color="#000000", anchor="nw",
                                 justify="left",
                                 wraplength=300,
                                 fg_color="#FFFFFF"
                                 )
                    desc_lbl.pack(anchor="nw", padx=24, pady=24, fill="both", expand=True)
                    desc_card.bind("<Configure>", lambda e, l=desc_lbl: _on_label_resize(e, l, 64), add="+")

                    _ai_last_w = [0]
                    def _refresh_ai_wrap(_e=None):
                        iw = info.winfo_width()
                        if iw < 50:
                            return  # layout not ready yet
                        # wraplength is in raw Tkinter pixels (same unit as winfo_width)
                        # subtract scrollbar (~16px) + padx (24px each side) + safety margin (30px)
                        w = max(150, iw - 120)
                        if w == _ai_last_w[0]:
                            return
                        _ai_last_w[0] = w
                        
                        # CTkFont sizes are in CTk units → multiplied by widget_scaling (1.35) internally
                        # So size=24 renders as ~32px, size=18 → ~24px, size=16 → ~21px
                        title_lbl.configure(wraplength=w, font=ctk.CTkFont(family="Outfit", size=24, weight="bold"))
                        if subtitle_lbl:
                            subtitle_lbl.configure(wraplength=w, font=ctk.CTkFont(family="Inter", size=18))
                        desc_lbl.configure(wraplength=w, font=ctk.CTkFont(family="Inter", size=16))

                    info.bind("<Configure>", _refresh_ai_wrap, add="+")
                    win.after(400, _refresh_ai_wrap)  # Run after layout is fully stable


                    # Section 3: Detected Traits (Gray/White Card)
                    traits_card = ctk.CTkFrame(info_scroll, fg_color="#FFFFFF", border_width=1, border_color="#E5E7EB", corner_radius=12)
                    traits_card.pack(fill="x", padx=12, pady=6)

                    # Traits Header
                    traits_hdr = ctk.CTkFrame(traits_card, fg_color="#FFFFFF", height=48, corner_radius=0)
                    traits_hdr.pack(fill="x")
                    traits_hdr.pack_propagate(False)

                    ctk.CTkLabel(traits_hdr, text="🏷️ Detected Traits",
                                 font=ctk.CTkFont(family="Inter", size=24, weight="bold"),
                                 text_color="#111827", fg_color="#FFFFFF").pack(side="left", padx=24)

                    # Traits Body
                    traits_body = ctk.CTkFrame(traits_card, fg_color="#FFFFFF")
                    traits_body.pack(fill="x", padx=24, pady=(0, 24))

                    # 2. Author Subsection
                    ctk.CTkLabel(traits_body, text="Author",
                                 font=ctk.CTkFont(family="Inter", size=15 if platform.system() == "Windows" else 20, weight="normal"),
                                 text_color="#6B7280", fg_color="#FFFFFF").pack(anchor="w", pady=(0, 6))

                    author_val = ai_doc.get("author", "") or "(Not Found)"
                    author_pill = ctk.CTkFrame(traits_body, fg_color="#EFF6FF", border_width=1, border_color="#BFDBFE", corner_radius=8)
                    author_pill.pack(anchor="w")
                    author_lbl = ctk.CTkLabel(author_pill, text=author_val,
                                 font=ctk.CTkFont(family="Outfit", size=16 if platform.system() == "Windows" else 20, weight="bold"),
                                 text_color="#1D4ED8", padx=16, pady=8,
                                 justify="left", anchor="w", fg_color="#EFF6FF")
                    author_lbl.pack(fill="x")

                    # 3. Edition Subsection
                    ctk.CTkLabel(traits_body, text="Edition",
                                 font=ctk.CTkFont(family="Inter", size=15 if platform.system() == "Windows" else 20, weight="normal"),
                                 text_color="#6B7280", fg_color="#FFFFFF").pack(anchor="w", pady=(12, 6))
                    
                    edition_val = ai_doc.get("edition", "") or "(Not Found)"
                    edition_pill = ctk.CTkFrame(traits_body, fg_color="#FFFBEB", border_width=1, border_color="#FEF3C7", corner_radius=8)
                    edition_pill.pack(anchor="w")
                    edition_lbl = ctk.CTkLabel(edition_pill, text=edition_val,
                                 font=ctk.CTkFont(family="Outfit", size=16 if platform.system() == "Windows" else 20, weight="bold"),
                                 text_color="#B45309", padx=16, pady=8,
                                 justify="left", anchor="w", fg_color="#FFFBEB")
                    edition_lbl.pack(fill="x")

                    # 4. ISBN Subsection
                    ctk.CTkLabel(traits_body, text="ISBN",
                                 font=ctk.CTkFont(family="Inter", size=15 if platform.system() == "Windows" else 20, weight="normal"),
                                 text_color="#6B7280", fg_color="#FFFFFF").pack(anchor="w", pady=(12, 6))
                    
                    isbn_val = ai_doc.get("isbn", "") or "(Not Found)"
                    isbn_pill = ctk.CTkFrame(traits_body, fg_color="#F3F4F6", border_width=1, border_color="#D1D5DB", corner_radius=8)
                    isbn_pill.pack(anchor="w")
                    isbn_lbl = ctk.CTkLabel(isbn_pill, text=isbn_val,
                                 font=ctk.CTkFont(family="Outfit", size=16 if platform.system() == "Windows" else 20, weight="bold"),
                                 text_color="#374151", padx=16, pady=8,
                                 justify="left", anchor="w")
                    isbn_lbl.pack(fill="x")

                    _trait_last_w = [0]
                    def _update_trait_wrap(e=None):
                        import platform
                        wv = max(160, traits_card.winfo_width() - 110)
                        if wv == _trait_last_w[0]: return
                        _trait_last_w[0] = wv
                        try:
                            # Scale font size same way as title/description
                            s = (info.winfo_width() / 1440.0)
                            if platform.system() == "Windows":
                                s *= 0.95
                            else:
                                s *= self._get_os_scale()
                            s = max(0.75, min(s, 1.4))
                            
                            # Much tighter font scaling for traits on Windows
                            val_base = 16 if platform.system() == "Windows" else 18
                            lbl_base = 13 if platform.system() == "Windows" else 14
                            
                            val_size  = max(13, int(val_base * s))
                            lbl_size  = max(11, int(lbl_base * s))
                            
                            author_lbl.configure(wraplength=wv, font=ctk.CTkFont(family="Outfit", size=val_size, weight="bold"))
                            edition_lbl.configure(wraplength=wv, font=ctk.CTkFont(family="Outfit", size=lbl_size, weight="bold"))
                            isbn_lbl.configure(wraplength=wv, font=ctk.CTkFont(family="Outfit", size=lbl_size, weight="bold"))
                        except Exception:
                            pass

                    traits_card.bind("<Configure>", _update_trait_wrap, add="+")
                    _update_trait_wrap()

                    ctk.CTkLabel(info_scroll, text=f"Book ID: {book_id}",
                                 font=ctk.CTkFont(family="Inter", size=18),
                                 text_color="#9CA3AF").pack(anchor="w", padx=24, pady=12)

                else:
                    pass # Legacy Basic Info View completely removed as per user request
                    
                def _render_info(p):
                    # No longer updates basic info labels on image switch
                    pass

                # --- PIL image cache for instant navigation ---
                _pil_cache = {}
                win._loading_path = None

                def _render_main():
                    if not win.winfo_exists(): return
                    i = max(0, min(idx_var.get(), len(paths) - 1))
                    if not paths: return
                    p = paths[i]
                    
                    # Pre-cache adjacent images
                    def _precache():
                        for offset in [-1, 1]:
                            ni = i + offset
                            if 0 <= ni < len(paths):
                                np = paths[ni]
                                if np not in _pil_cache:
                                    try: _pil_cache[np] = Image.open(np)
                                    except: pass

                    bn = os.path.basename(p)
                    status_lbl.configure(text=bn if len(bn) <= 90 else (bn[:89] + "…"))
                    
                    if not PIL_OK:
                        main_lbl.configure(text="Install Pillow for previews (pip install pillow)")
                        return

                    # 1. Check Cache
                    if p in _pil_cache:
                        _do_render(p, _pil_cache[p])
                        threading.Thread(target=_precache, daemon=True).start()
                        return

                    # 2. Async Load if not in cache
                    win._loading_path = p
                    if hasattr(win, "_start_detail_spinner"):
                        win._start_detail_spinner()
                    
                    def _load_task():
                        try:
                            im = Image.open(p)
                            if win.winfo_exists() and win._loading_path == p:
                                _pil_cache[p] = im
                                self.after(0, lambda: _do_render(p, im))
                                _precache()
                        except Exception as e:
                            print(f"DEBUG: Failed to load {p}: {e}")
                            if win.winfo_exists():
                                self.after(0, lambda: main_lbl.configure(text="Preview unavailable"))

                    threading.Thread(target=_load_task, daemon=True).start()

                def _do_render(path, im):
                    if not win.winfo_exists() or (hasattr(win, "_loading_path") and win._loading_path and win._loading_path != path):
                        return
                    
                    if not getattr(win, "_first_load_done", True):
                        win._first_load_done = True
                        if hasattr(win, "_hide_shroud"):
                            # Hide initial loader shroud once first image is ready
                            win.after(100, win._hide_shroud)
                    
                    win._loading_path = None
                    if hasattr(win, "_stop_detail_spinner"):
                        win._stop_detail_spinner()
                    try:
                        # Force window-level layout update to resolve parent grid geometries
                        win.update_idletasks()
                        # Use raw winfo dimensions — CTkImage handles its own DPI scaling internally
                        cur_w = preview.winfo_width()
                        cur_h = preview.winfo_height()
                        if cur_w < 120 or cur_h < 120:
                            win.after(50, lambda: _do_render(path, im))
                            return

                        # Get the real CTk widget scale (this is what CTkImage multiplies by internally)
                        try:
                            _ctk_ws = ctk.ScalingTracker.get_widget_scaling(main_lbl)
                        except Exception:
                            _ctk_ws = 1.0
                        # Divide physical winfo pixels by CTk scale to get CTkImage logical size
                        pw = max(10, int(cur_w / _ctk_ws) - 8)
                        ph = max(10, int(cur_h / _ctk_ws) - 8)
                        w, h = im.size
                        zf = getattr(win, "zoom_factor", 1.0)
                        
                        if zf > 1.0:
                            crop_w = int(w / zf)
                            crop_h = int(h / zf)
                            cx = int(win.pan_x * w)
                            cy = int(win.pan_y * h)
                            x1 = max(0, cx - crop_w // 2)
                            y1 = max(0, cy - crop_h // 2)
                            if x1 + crop_w > w: x1 = w - crop_w
                            if y1 + crop_h > h: y1 = h - crop_h
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(w, x1 + crop_w); y2 = min(h, y1 + crop_h)
                            cropped = im.crop((x1, y1, x2, y2))
                            cw, ch = cropped.size
                            r = min(pw / float(cw), ph / float(ch))
                            sz = (max(1, int(cw * r)), max(1, int(ch * r)))
                            img = ctk.CTkImage(light_image=cropped, size=sz)
                            main_lbl.configure(cursor="fleur")
                        else:
                            if fit_var.get(): r = min(pw/float(w), ph/float(h))
                            else: r = min(1.0, min(pw/float(w), ph/float(h)))
                            sz = (max(1, int(w*r)), max(1, int(h*r)))
                            img = ctk.CTkImage(light_image=im, size=sz)
                            main_lbl.configure(cursor="")
                        
                        main_lbl.configure(image=img, text="", width=sz[0], height=sz[1])
                        main_lbl.image = img
                        _render_info(path)
                    except Exception as e:
                        print(f"DEBUG: Render failed: {e}")
                        main_lbl.configure(text="Preview unavailable")

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
                        win.update_idletasks()
                        # Get the real CTk widget scale
                        try:
                            _ctk_ws2 = ctk.ScalingTracker.get_widget_scaling(main_lbl)
                        except Exception:
                            _ctk_ws2 = 1.0
                        pw = max(10, int(preview.winfo_width() / _ctk_ws2) - 8)
                        ph = max(10, int(preview.winfo_height() / _ctk_ws2) - 8)
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
                        main_lbl.configure(image=img, width=sz[0], height=sz[1])
                        main_lbl.image = img
                    except Exception:
                        pass
                win._render_pan_cmd = _render_pan_only

                # Persistent widget cache for thumbnails to prevent flickering
                win.thumb_widgets = []

                def _render_thumbs():
                    if not win.winfo_exists(): return
                    if not thumbs_var.get():
                        for ch in thumbs_scroll.winfo_children(): ch.destroy()
                        win.thumb_widgets = []
                        thumbs_scroll.grid_forget()
                        return
                    
                    thumbs_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0,12))
                    new_idx = idx_var.get()
                    
                    # 1. INITIAL BUILD
                    if not win.thumb_widgets:
                        for ch in thumbs_scroll.winfo_children(): ch.destroy()
                        try:
                            from PIL import Image as PILImage
                        except Exception:
                            PILImage = None
                        
                        # Pack into the scrollable frame
                        # No need for intermediate container
                        for i, p in enumerate(paths[:60]):
                            is_active = (i == new_idx)
                            cell = ctk.CTkFrame(thumbs_scroll, fg_color=C["card"] if is_active else "transparent", 
                                               corner_radius=8, border_width=2 if is_active else 0, 
                                               border_color=C["olive"])
                            cell.pack(side="left", padx=4, pady=4)
                            
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
                    # Force initial render immediately after paths found
                    _render_main()
                    _render_thumbs()
                    
                    def _refresh(*_):
                        # Use a small delay for traces to allow state to settle
                        win.after(10, _render_main)
                        win.after(10, _render_thumbs)
                    
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

    def update_activity_row(self, book_id, status, type_, timestamp, error_msg=None):
        """Update existing row or create new one. Called from main thread via after()."""
        if not book_id: return
        
        bg, fg = STATUS_STYLES.get(status, (C["s_o_bg"], C["s_o_fg"]))

        if book_id in self.activity_rows:
            w = self.activity_rows[book_id]
            w["badge"].configure(text=status, fg_color=bg, text_color=fg)
            w["time_lbl"].configure(text=timestamp)
            w["type_lbl"].configure(text=type_)
            if error_msg:
                w["error_msg"] = error_msg
        elif book_id not in self._pending_ids:
            # Mark as pending to prevent duplicate rows from rapid updates
            self._pending_ids.add(book_id)
            self._add_activity_row(book_id, status, type_, timestamp, error_msg=error_msg)
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
        self._dev_clicks = 0
        def _on_dev_click(e):
            self._dev_clicks += 1
            if self._dev_clicks >= 3:
                try:
                    log_label.grid()
                    log_card.grid()
                    win.geometry(f"{sw}x{sh}") # Expand to fit logs
                except: pass

        log_header_lbl = ctk.CTkLabel(scroll, text="⚙️  Automation Settings",
                     font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                     text_color=C["text"])
        log_header_lbl.grid(row=0, column=0, sticky="w", padx=24, pady=(24, 12))
        log_header_lbl.bind("<Button-1>", _on_dev_click)

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

        # DB section intentionally hidden — config is locked and not relevant for end-user

        # ── Section: Pipeline Logs (Hidden Developer Section) ──
        log_label = ctk.CTkLabel(scroll, text="📋  Pipeline Logs",
                     font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                     text_color=C["text"])
        log_label.grid(row=2, column=0, sticky="w", padx=24, pady=(20, 12))
        log_label.grid_remove() # Hidden by default

        log_card = self._settings_card(scroll, 3)
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_remove() # Hidden by default

        # Copy log content from main log_box into this settings log viewer
        from tkinter import Text, Scrollbar
        log_frame = ctk.CTkFrame(log_card, fg_color="#FFFBEB", corner_radius=8)
        log_frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=(12, 4))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)

        settings_log_scroll = Scrollbar(log_frame)
        settings_log_scroll.pack(side="right", fill="y")

        settings_log_box = Text(log_frame,
                                bg="#FFFBEB", fg="#1a1a1a",
                                font=("Monospace", 10),
                                borderwidth=0, highlightthickness=0,
                                height=18,
                                yscrollcommand=settings_log_scroll.set)
        settings_log_box.pack(fill="both", expand=True, padx=6, pady=6)
        settings_log_scroll.config(command=settings_log_box.yview)

        # Copy existing log content from the hidden main log_box
        try:
            self._settings_log_box = settings_log_box # Store reference for live updates
            existing_logs = self.log_box.get("1.0", "end-1c")
            if existing_logs.strip():
                settings_log_box.configure(state="normal")
                settings_log_box.insert("end", existing_logs)
                settings_log_box.configure(state="disabled")
                settings_log_box.see("end")
            else:
                settings_log_box.configure(state="normal")
                settings_log_box.insert("end", "No logs yet. Run a sync to see activity here.")
                settings_log_box.configure(state="disabled")
        except Exception:
            pass

        def _on_settings_close():
            self._settings_log_box = None # Clear reference
            win.destroy()
        
        win.protocol("WM_DELETE_WINDOW", _on_settings_close)

        ctk.CTkButton(log_card, text="🗑  Clear Logs",
                      height=self._px(36), corner_radius=self._px(8),
                      font=ctk.CTkFont(family="Inter", size=self.F["btn"], weight="bold"),
                      fg_color="#EF4444", hover_color="#DC2626",
                      text_color=C["white"],
                      command=lambda: [
                          self.log_box.configure(state="normal"),
                          self.log_box.delete("1.0", "end"),
                          self.log_box.configure(state="disabled"),
                          settings_log_box.configure(state="normal"),
                          settings_log_box.delete("1.0", "end"),
                          settings_log_box.configure(state="disabled")
                      ]).grid(row=1, column=0, sticky="w", padx=16, pady=(4, 16))

        # Responsive sizes
        sw = max(560, int(self.winfo_width() * 0.45))
        sh = max(750, int(self.winfo_height() * 0.85))
        
        # Initial small size for client view
        win.geometry(f"{sw}x{self._px(400)}")

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
    def _browse_manual(self, mode="books"):
        if not self.current_user:
            messagebox.showwarning("Authorization Required", "Please enter a valid token to authorize before uploading.")
            return
        
        title = "Select Book Images" if mode == "books" else "Select Slide Images"
        files = filedialog.askopenfilenames(title=title, 
                                            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.JPG *.JPEG *.PNG *.WEBP")])
        if not files: return

        selected_files = list(files)
        # Smart Auto-Pickup: For EVERY selected file, look for siblings with same prefix
        new_selected = set(selected_files)
        
        for f_path in selected_files:
            folder = os.path.dirname(f_path)
            bn = os.path.basename(f_path)
            prefix = bn.split('_')[0] if '_' in bn else (bn.split('-')[0] if '-' in bn else bn.split('.')[0])
            
            # Look for other files in same folder with same prefix
            siblings = [os.path.join(folder, f) for f in os.listdir(folder) 
                        if (f.startswith(prefix + "_") or f.startswith(prefix + "-") or os.path.splitext(f)[0] == prefix) 
                        and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            if siblings:
                new_selected.update(siblings)
                
        selected_files = list(new_selected)
        
        if len(selected_files) == 1 and mode == "books":
            res = messagebox.askyesno("Limited Data Warning", 
                "Only one image found for this book.\n\n"
                "AI might not be able to extract full metadata (ISBN, Author, Edition) from a single page.\n\n"
                "Do you want to continue anyway?")
            if not res: return

        # Group files by prefix
        groups = {}
        for f in selected_files:
            bn = os.path.basename(f)
            prefix = bn.split('_')[0] if '_' in bn else (bn.split('-')[0] if '-' in bn else bn.split('.')[0])
            if prefix not in groups: groups[prefix] = []
            
            import re
            nums = re.findall(r'\d+', bn)
            page_num = int(nums[-1]) if nums else 0
            groups[prefix].append((page_num, f))
        
        if not groups:
            messagebox.showerror("Selection Error", "No valid images could be grouped.")
            return

        self._clear_activity()
        self._log(f"📄 Manual processing: {len(groups)} book(s) identified.")
        
        def _check_and_start_manual():
            self.sync_running = True
            self.after(0, lambda: self.btn_stop.configure(state="normal"))
            self._log("🚀 Manual Sync started!")
            threading.Thread(target=self._manual_worker, args=(groups,), daemon=True).start()

        threading.Thread(target=_check_and_start_manual, daemon=True).start()

    def _open_train_slides_app(self, mode="folder"):
        if not self.current_user:
            messagebox.showwarning("Authorization Required", "Please enter a valid token to authorize before proceeding.")
            return
        import datetime
        import threading
        from pathlib import Path
        
        # Check if custom train slide weights exist before proceeding
        try:
            from backend.train_slides_logic import get_analyzer
            analyzer = get_analyzer()
            missing_weights = []
            if not os.path.exists(analyzer.yolo_parts_model):
                missing_weights.append("best.pt (Locomotive parts detector)")
            if not getattr(analyzer, 'dino_weights', None) or not os.path.exists(analyzer.dino_weights):
                missing_weights.append("phase5_best.pth (DINOv2 Classifier)")
            if not getattr(analyzer, 'val_emb_file', None) or not os.path.exists(analyzer.val_emb_file):
                missing_weights.append("val_embeddings.pt or temp_embeddings.pt (Railroad embeddings)")
                
            if missing_weights:
                missing_list_str = "\n".join([f" • {w}" for w in missing_weights])
                messagebox.showerror(
                    "Missing Custom Weights",
                    "The following custom model weights are required but could not be located:\n\n"
                    f"{missing_list_str}\n\n"
                    "Please download these files and place them in a 'weights/' folder next to the application executable."
                )
                return
        except Exception as e_check:
            messagebox.showerror("Error", f"Failed to verify model weights: {e_check}")
            return

        supported = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        # Natural sort function for filenames
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        def get_lot_id_from_filename(file_path):
            base = os.path.splitext(os.path.basename(file_path))[0]
            if "_" in base:
                parts = base.rsplit('_', 1)
                if parts[1].isdigit():
                    return parts[0]
            return base

        if mode == "folder":
            folder = filedialog.askdirectory(title="Select Train Slides Folder")
            if not folder:
                return
            folder_path = os.path.normpath(folder)
            try:
                image_files = [str(p) for p in Path(folder_path).glob("*.*") if p.suffix.lower() in supported]
            except Exception:
                image_files = []
        else:
            files = filedialog.askopenfilenames(title="Select Train Slide Images", 
                                                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp *.JPG *.JPEG *.PNG *.WEBP *.BMP")])
            if not files:
                return
            image_files = list(files)
            
            # Smart Auto-Pickup: For EVERY selected slide, look for siblings in same folder with same Lot ID
            new_selected = set(image_files)
            for f_path in image_files:
                folder = os.path.dirname(f_path)
                prefix = get_lot_id_from_filename(f_path)
                
                # Find all siblings matching this Lot ID
                siblings = [os.path.join(folder, f) for f in os.listdir(folder)
                            if get_lot_id_from_filename(os.path.join(folder, f)) == prefix
                            and os.path.splitext(f)[1].lower() in supported]
                
                if siblings:
                    new_selected.update(siblings)
                    
            image_files = list(new_selected)

        if not image_files:
            messagebox.showwarning("No Images", "No matching images found to process.")
            return

        self._clear_activity()

        # Group all images into lots by naming convention
        raw_lots = {}
        for f in image_files:
            lid = get_lot_id_from_filename(f)
            if lid not in raw_lots:
                raw_lots[lid] = []
            raw_lots[lid].append(f)

        # Sort the lot files naturally and sort the lot keys naturally
        lots = {}
        sorted_lids = sorted(raw_lots.keys(), key=natural_sort_key)
        for lid in sorted_lids:
            flist = raw_lots[lid]
            flist.sort(key=natural_sort_key)
            lots[lid] = flist

        # Prepare sync results structure but do not pre-insert Queued rows to keep the UI clean
        ts = datetime.datetime.now().strftime("%I:%M %p")

        # Process each lot sequentially in a single background thread
        def _bg_process_all():
            self.sync_running = True
            self.after(0, lambda: self.btn_stop.configure(state="normal"))
            self._log("🚀 Train Slides sync started!")
            
            # Force unload Ollama models from VRAM to make room for PyTorch
            self._log("🧹 Clearing Ollama models from VRAM...")
            try:
                import requests
                requests.post('http://localhost:11434/api/generate', json={'model': 'minicpm-v:latest', 'prompt': '', 'keep_alive': 0}, timeout=2)
                requests.post('http://localhost:11434/api/generate', json={'model': 'llama3.2:1b', 'prompt': '', 'keep_alive': 0}, timeout=2)
                requests.post('http://localhost:11434/api/generate', json={'model': 'llama3:latest', 'prompt': '', 'keep_alive': 0}, timeout=2)
            except Exception:
                pass

            try:
                from backend.train_slides_logic import get_analyzer
                analyzer = get_analyzer()
                analyzer._should_close_popup = False
                analyzer._popup_shown = False
                analyzer._popup_win = None
            except Exception as ex:
                err_str = str(ex)
                for lid in lots:
                    self.after(0, lambda l=lid, err=err_str: self.update_activity_row(l, "Failed", "Train Lot", ts, error_msg=err))
                self.sync_running = False
                self.after(0, lambda: self.btn_stop.configure(state="disabled"))
                return

            # --- CHUNKING LOGIC ---
            BATCH_SIZE = 10
            lot_items = list(lots.items())
            
            for batch_start in range(0, len(lot_items), BATCH_SIZE):
                if not self.sync_running:
                    break
                
                batch_lots = dict(lot_items[batch_start:batch_start + BATCH_SIZE])
                batch_num = (batch_start // BATCH_SIZE) + 1
                total_batches = (len(lot_items) + BATCH_SIZE - 1) // BATCH_SIZE
                self._log(f"📦 Starting Batch {batch_num}/{total_batches} ({len(batch_lots)} lots)...")
                
                # --- PHASE 1: Process batch with PyTorch models ---
                all_lot_results = {}
                for lid, flist in batch_lots.items():
                    if not self.sync_running:
                        break

                    self.after(0, lambda l=lid: self.update_activity_row(l, "Processing", "Train Lot", ts))
                    lot_results = {}
                    try:
                        for idx, img_path in enumerate(flist):
                            if not self.sync_running:
                                break
                            def _model_cb(msg, pct):
                                if "Download" in msg or "Loading" in msg:
                                    short_msg = "Init Models..." if "Loading" in msg else "Downloading..."
                                    self.after(0, lambda l=lid, m=short_msg: self.update_activity_row(l, m, "Train Lot", ts))
                                
                                    if "Download" in msg and not getattr(analyzer, "_popup_shown", False):
                                        analyzer._popup_shown = True
                                        def _show_popup():
                                            if getattr(analyzer, "_should_close_popup", False):
                                                return
                                            w, h = 400, 200
                                            win = ctk.CTkToplevel(self)
                                            win.title("Downloading Models")
                                            win.geometry(f"{self._px(w)}x{self._px(h)}")
                                            win.attributes("-topmost", True)
                                            win.grab_set()
                                        
                                            self.update_idletasks()
                                            px, py = self.winfo_x(), self.winfo_y()
                                            pw, ph = self.winfo_width(), self.winfo_height()
                                            win.geometry(f"+{px + (pw - self._px(w))//2}+{py + (ph - self._px(h))//2}")
                                        
                                            lbl = ctk.CTkLabel(win, text="Loading EasyOCR...", 
                                                            font=ctk.CTkFont(family="Inter", size=self.F["heading"], weight="bold"),
                                                            text_color=C["text"])
                                            lbl.pack(pady=(self._px(40), self._px(20)))
                                        
                                            prog = ctk.CTkProgressBar(win, width=self._px(360), height=self._px(12),
                                                                    progress_color=C["olive"], fg_color=C["border"])
                                            prog.pack(pady=self._px(10))
                                            prog.configure(mode="indeterminate")
                                            prog.start()
                                        
                                            status = ctk.CTkLabel(win, text="Please wait. This is a one-time download...", 
                                                                font=ctk.CTkFont(family="Inter", size=self.F["label"]),
                                                                text_color=C["muted"])
                                            status.pack(pady=self._px(5))
                                            analyzer._popup_win = win
                                        self.after(0, _show_popup)
                                    
                                elif "success" in msg.lower():
                                    analyzer._should_close_popup = True
                                    if getattr(analyzer, "_popup_win", None):
                                        self.after(0, lambda: analyzer._popup_win.destroy() if analyzer._popup_win.winfo_exists() else None)
                                        analyzer._popup_win = None
                        
                            res = analyzer.analyze_image(img_path, log_fn=self._log, progress_callback=_model_cb)
                            lot_results[img_path] = res
                            prog = f"Proc ({idx+1}/{len(flist)})"
                            self.after(0, lambda l=lid, p=prog: self.update_activity_row(l, p, "Train Lot", ts))
                        
                        if not self.sync_running:
                            self.after(0, lambda l=lid: self.update_activity_row(l, "Stopped", "Train Lot", ts))
                    
                        all_lot_results[lid] = lot_results
                    except Exception as ex:
                        err_str = str(ex)
                        self.after(0, lambda l=lid, err=err_str: self.update_activity_row(l, "Failed", "Train Lot", ts, error_msg=err))

                # --- PHASE 2: GPU Cleanup ---
                # Unload PyTorch models ONCE after all lots are processed
                if self.sync_running:
                    try:
                        analyzer.unload_models(log_fn=self._log)
                    except Exception as e_unload:
                        self._log(f"  ⚠️ GPU flush warning: {e_unload}")

                # --- PHASE 3: Run Ollama Summarization for batch ---
                for lid, flist in batch_lots.items():
                    if not self.sync_running:
                        break
                    
                    if lid not in all_lot_results:
                        continue
                    
                    lot_results = all_lot_results[lid]
                    self.after(0, lambda l=lid: self.update_activity_row(l, "AI Summary...", "Train Lot", ts))
                
                    # 1. Compile lot statistics first to build fallback description
                    railroad_counts = {}
                    railroad_type_breakdown = {}

                    for p_img in flist:
                        r_res = lot_results.get(p_img, {})
                        if not r_res:
                            continue
                        rr = r_res.get("railroad")
                        lt = r_res.get("loco_type")
                    
                        if not rr or rr in ["-", "Unprocessed", "Pending Analysis"]:
                            continue
                        if not lt or lt in ["-", "Unprocessed", "Pending Analysis"]:
                            continue
                        
                        railroad_counts[rr] = railroad_counts.get(rr, 0) + 1
                        if rr not in railroad_type_breakdown:
                            railroad_type_breakdown[rr] = {}
                        railroad_type_breakdown[rr][lt] = railroad_type_breakdown[rr].get(lt, 0) + 1

                    ai_desc = "This lot contains detailed train slide analysis. Engine classifications and railroad identities are extracted using deep neural network pipelines."
                
                    if railroad_counts:
                        self._log(f"   -> Analyzing statistics for {len(railroad_counts)} railroads (Lot: {lid})...")
                        total_slides = len(flist)
                        railroads_list = sorted(list(railroad_counts.keys()))
                        railroads_str = ", ".join(railroads_list)
                        suffix = "Railroad" if len(railroads_list) == 1 else "Railroads"
                    
                        desc_prefix = ""
                    
                        # Build a detailed, formatted string of only the detected data
                        breakdown_items = []
                        for rr, types in railroad_type_breakdown.items():
                            lt_names = []
                            for lt in types.keys():
                                name = lt.lower().replace("locomotive", "").strip()
                                if not name: name = "locomotive"
                                if not name.endswith('s'): name += "s"
                                lt_names.append(name)
                        
                            if len(lt_names) > 1:
                                types_str = ", ".join(lt_names[:-1]) + " and " + lt_names[-1]
                            else:
                                types_str = lt_names[0] if lt_names else "locomotives"
                        
                            breakdown_items.append(f"{rr} {types_str}")
                    
                        if len(breakdown_items) > 2:
                            breakdown_str = ", ".join(breakdown_items[:-1]) + ", and " + breakdown_items[-1]
                        elif len(breakdown_items) == 2:
                            breakdown_str = f"{breakdown_items[0]} and {breakdown_items[1]}"
                        elif breakdown_items:
                            breakdown_str = breakdown_items[0]
                        else:
                            breakdown_str = f"{total_slides} locomotive slides"

                        # Generate deterministic fallback
                        fallback_note = f"This lot contains slides of {breakdown_str}."
                        ai_desc = desc_prefix + fallback_note

                        # 2. Synchronously generate dynamic AI Lot description
                        try:
                            import time
                            start_time = time.time()
                        
                            # Force using the 8B model as requested
                            active_model = "llama3:latest"
                        
                            self._log(f"🤖 [Ollama] Querying model '{active_model}' for dynamic lot description (Lot: {lid})...")

                            prompt = (
                                f"You are an expert railway archival cataloguer.\n"
                                f"Please write a single, natural, and professional sentence summarizing the contents of this train lot based ONLY on the data below.\n\n"
                                f"Extracted Data: {breakdown_str}\n\n"
                                "CRITICAL INSTRUCTIONS:\n"
                                "1. Write a fluent, conversational sentence.\n"
                                "2. DO NOT use any numbers or slide counts.\n"
                                "3. DO NOT invent or add any locomotive builders (e.g. EMD, Alco), models, or locations.\n"
                                "4. DO NOT guess, assume, or classify any train/car as 'passenger' or 'freight' unless that word is explicitly present in the Extracted Data.\n"
                                "5. Output ONLY the final sentence. No introductory filler, no quotes, no extra text.\n\n"
                                "Example: This lot features slides of BNSF and Santa Fe diesel locomotives, along with Union Pacific steam engines."
                            )

                            # pyrefly: ignore [missing-import]
                            import ollama
                            response = ollama.chat(
                                model=active_model,
                                messages=[{
                                    "role": "user",
                                    "content": prompt
                                }]
                            )
                            possible_note = response.get("message", {}).get("content", "").strip()
                            if possible_note.startswith("'") and possible_note.endswith("'"):
                                possible_note = possible_note[1:-1]
                            elif possible_note.startswith('"') and possible_note.endswith('"'):
                                possible_note = possible_note[1:-1]
                            
                            # Strip conversational filler from LLM
                            filler_prefixes = ["here is", "sure", "output:", "description:", "the final sentence is", "this lot features", "locomotive notes:", "note:", "this lot contains"]
                            lower_note = possible_note.lower()
                            for _ in range(3):
                                for prefix in filler_prefixes:
                                    if lower_note.startswith(prefix):
                                        possible_note = possible_note[len(prefix):].strip(" :\n\"'")
                                        lower_note = possible_note.lower()
                            if possible_note:
                                possible_note = possible_note[0].upper() + possible_note[1:]
                            else:
                                possible_note = fallback_note
                            
                            possible_desc = desc_prefix + possible_note
                        
                            # Clean up common meta-garbage patterns from bad LLMs
                            bad_patterns = ["archival lot description", "this description", "provides an overview", "focuses on", "the language used", "states that", "without unnecessary", "in the provided stats", "the provided statistics"]
                            is_garbage = any(pat in possible_desc.lower() for pat in bad_patterns)
                        
                            if is_garbage or len(possible_note.strip()) < 8:
                                possible_desc = desc_prefix + fallback_note
                            
                            if possible_desc:
                                ai_desc = possible_desc
                                duration = time.time() - start_time
                                self._log(f"✅ [Ollama] Finished in {duration:.2f}s using '{active_model}'!")
                                self._log(f"   -> Summary: \"{ai_desc}\"")
                        except Exception as e_desc:
                            self._log(f"⚠️ [Ollama] Generation failed: {e_desc}")
                            self._log(f"   -> Caching default fallback lot description instead.")
                    
                    # Save Train Slides Lot to Database
                    slides_coll = "Train Slides Data"
                    lot_doc = {
                        "lot_id": lid,
                        "title": f"Train Slide Lot {lid}",
                        "description": ai_desc,
                        "synced_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "user_id": self.current_user.get("id") if getattr(self, "current_user", None) else None,
                        "slides_data": lot_results
                    }
                
                    if getattr(self, "db_connector", None) and getattr(self.db_connector, "connected", False):
                        try:
                            self.db_connector.insert_book(slides_coll, lot_doc)
                            self._log(f"  ✅ DB Insert: Lot {lid} saved to '{slides_coll}'")
                        except Exception as db_err:
                            self._log(f"  ⚠️ DB Insert Failed for Lot {lid}: {db_err}")

                    # Cache the full complete lot structure under last_sync_results only now!
                    self.last_sync_results[lid] = {
                        "files": flist,
                        "results": lot_results,
                        "ai_description": ai_desc,
                        "doc": lot_doc
                    }
                    self.after(0, lambda l=lid: self.update_activity_row(l, "Complete", "Train Lot", ts))
                
            # --- PHASE 4: Unload Ollama from RAM after ALL lots ---
            if self.sync_running:
                try:
                    import requests as _req
                    for _m in ['llama3:latest', 'llama3.2:1b', 'minicpm-v:latest']:
                        _req.post('http://localhost:11434/api/generate',
                                  json={'model': _m, 'prompt': '', 'keep_alive': 0},
                                  timeout=3)
                    self._log("🧹 [RAM] Ollama models unloaded from RAM at end of sync.")
                except Exception:
                    pass

            self.sync_running = False
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))
            self._log("⏹ Train Slides sync finished or stopped.")

        threading.Thread(target=_bg_process_all, daemon=True).start()

    def _manual_worker(self, books):
        """Processes only the specific books selected manually and then stops."""
        try:
            if OCR_AVAILABLE:
                ocr_pipeline.cleanup_gpu()

            sorted_ids = sorted(books.keys())
            for book_id in sorted_ids:
                if not self.sync_running: break
                
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Reuse the core processing logic
                self._process_single_book_flow(book_id, books[book_id], ts)
            
            self._log("✅ Manual sync complete.")
        finally:
            self.sync_running = False
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))

    def _process_single_book_flow(self, book_id, book_pages, ts):
        """The core logic to process one book from start to finish."""
        coll = self.config.get("collection", "Book Data")
        
        # Save to last_sync_results immediately so local paths are available even if skipped
        self.last_sync_results[book_id] = book_pages
        
        # UI Status: Queued
        self.after(0, lambda: self.update_activity_row(book_id, "Queued", "Book", ts))
        time.sleep(0.1)

        # UI Status: Processing
        self.after(0, lambda: self.update_activity_row(book_id, "Processing", "Book", ts))
        time.sleep(0.2)

        self._log(f"  📖 {book_id}…")

        # 1. Build Base Doc
        grouper = BookGrouper()
        doc = grouper.build_document(book_id, book_pages)
        if self.current_user:
            doc["user_id"] = self.current_user.get("id")

        # 2. OCR Pipeline
        ai_result = None
        if OCR_AVAILABLE:
            try:
                ai_result = self._process_book_ocr(book_id, book_pages, ts)
                if ai_result:
                    if ai_result.get("duplicate"):
                        if "doc" in ai_result:
                            self.last_sync_results[book_id] = {"files": book_pages, "doc": ai_result["doc"]}
                        self.after(0, lambda: self.update_activity_row(book_id, "Skipped", "Book", ts))
                        return
                        
                    extracted_title = ai_result.get("title", "")
                    
                    # 3. DB Check (Title matching) AFTER OCR
                    try:
                        if extracted_title:
                            matched_doc = self.db_connector.book_title_exists(coll, extracted_title, return_doc=True)
                            if matched_doc:
                                self._log(f"  ⏭️  Title '{extracted_title}' matches an existing book (90%+). Skipping.")
                                # Save the matched DB document to memory so the preview page can display it!
                                self.last_sync_results[book_id] = {"files": book_pages, "doc": matched_doc}
                                self.after(0, lambda: self.update_activity_row(book_id, "Skipped", "Book", ts))
                                return
                    except Exception as e:
                        self._log(f"  ⚠️ DB title check error: {e}")
                        
                    doc.update({
                        "title": extracted_title,
                        "subtitle": ai_result.get("subtitle", ""),
                        "author": ai_result.get("author", "Not Found"),
                        "edition": ai_result.get("edition", "Not Specified"),
                        "isbn": ai_result.get("isbn", "N/A"),
                        "description": ai_result.get("description", ""),
                        "ocr_completed": True
                    })
                    self._log(f"  🎉 AI done: {book_id}")
                else:
                    doc["ocr_completed"] = False
                    self._log(f"  ⚠️ OCR returned no results for {book_id}")
            except Exception as e:
                doc["ocr_completed"] = False
                self._log(f"  ⚠️ OCR pipeline error: {e}")
        else:
            self.after(0, lambda: self.update_activity_row(book_id, "Failed", "No Models", ts))
            return

        # 4. Insert to DB
        if ai_result:
            try:
                self.db_connector.insert_book(coll, doc)
                self._log(f"  ✅ Synced: {book_id}")
                # Save full doc to last_sync_results so it can be opened instantly on click
                self.last_sync_results[book_id] = {"files": book_pages, "doc": doc}
                self._log(f"  📝 Saved to Temp List. Total cached items: {len(self.last_sync_results)}")
                self.after(0, lambda: self.update_activity_row(book_id, "Complete", "Book", ts))
            except Exception as e:
                self._log(f"  ❌ DB Error: {e}")
                self.after(0, lambda: self.update_activity_row(book_id, "Failed", "DB Error", ts))
        else:
            self.after(0, lambda: self.update_activity_row(book_id, "Partial", "OCR Fail", ts))

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
                self._set_conn_visual("offline")
                messagebox.showinfo("Folder Linked Successfully",
                                    f"Success! The {mode.lower()} directory has been mapped and saved.\n\n"
                                    "To begin the automated OCR pipeline and synchronization, please establish a connection "
                                    "by verifying your secure token.\n\n"
                                    f"Selected Path: {path}")

    def _test_conn(self, token=None, silent=False):
        if token is None:
            token = self.token_entry.get().strip()
        
        if not token:
            if not silent:
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
            if not silent:
                messagebox.showerror("Missing Database URI",
                                     "Database connection is not configured.")
            return

        self._set_conn_visual("connecting")
        self.update_idletasks()

        # Run DB connection in background to keep UI responsive
        dbname = self.config.get("db_name", "Test")
        def _connect_task():
            conn = DBConnector(uri, dbname)
            ok, msg = conn.connect()
            user = None
            if ok:
                user = conn.find_user_by_token(token)
            if self.winfo_exists():
                self.after(0, lambda: self._on_connect_done(conn, ok, msg, user, token, dbname, silent=silent))
        threading.Thread(target=_connect_task, daemon=True).start()

    def _on_connect_done(self, conn, ok, msg, user, token, dbname, silent=False):
        if not self.winfo_exists(): return
        if ok:
            if not user:
                self._set_conn_visual("invalid")
                try:
                    self.token_entry.configure(border_color=C["s_fail_fg"]) 
                    self.token_entry.focus_set()
                except Exception:
                    pass
                if not silent:
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
            
            # Unlock header features on success
            self.after(0, lambda: self.btn_settings.configure(state="normal"))
            self.after(0, lambda: self.btn_refresh.configure(state="normal"))
            
            # --- Session-Only Persistence: Save only in memory, NOT to file for security ---
            self._session_token = token

            # Auto-start sync only if user selected a folder in this session
            if self._folder_selected_session and not self.sync_running:
                self._start_sync()
        else:
            self._set_conn_visual("failed")
            if not silent:
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
            error_details = f"\nDetails: {OCR_IMPORT_ERROR}" if OCR_IMPORT_ERROR else ""
            messagebox.showwarning("Models Missing", 
                "OCR Pipeline (main_mineru_ocr.py) or dependencies are missing.\n"
                f"Syncing is disabled until models are installed.{error_details}")
            return
            
        # --- Fix 11: Async Ollama Check ---
        def _check_ollama_and_start():
            # USER REQUEST: Do not show 'Checking AI...' in the global connection badge.
            # This status belongs in the activity table or internal logs.
            try:
                import ollama
                ollama.list()
                self.after(0, _proceed_to_sync)
            except Exception:
                # Use 'active' instead of 'connected' which is not a valid state
                self.after(0, lambda: self._set_conn_visual("active"))
                self._log("⚠️ Warning: Ollama server not responding. AI extraction might fail.")
                if not messagebox.askyesno("Ollama Missing", 
                    "Ollama is either not installed or not running.\n"
                    "AI extraction (Title, Color, Description) will fail.\n\n"
                    "Continue anyway?"):
                    return
                self.after(0, _proceed_to_sync)

        def _proceed_to_sync():
            self._set_conn_visual("active")
            self.total_ok = self.total_skip = self.total_fail = 0
            self.sync_running = True
            self.btn_stop.configure(state="normal") # Enable stop button
            self._log("🚀 Sync started!")
            threading.Thread(target=self._worker, daemon=True).start()

        threading.Thread(target=_check_ollama_and_start, daemon=True).start()

    def _stop_sync(self):
        """Request graceful termination of the sync process."""
        if self.sync_running:
            self._log("🛑 Stop requested. Finishing current book and exiting...")
            self.sync_running = False
            self.btn_stop.configure(state="disabled")

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
            self.after(0, self.update_idletasks)
            
            self._log(f"  🔍 Checking for ISBN logic for {book_id}…")
            try:
                # Use normalized log function for isbn_logic
                res = isbn_logic.process_book(book_id, image_paths, log_fn=self._log)
                # Optimization: Do NOT unload here, let ocr_pipeline reuse it.
                
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

        # Convert "N/A" to empty string and handle lists from API
        for var_name, val in [("title_str", title_str), ("author", author), ("edition", edition), ("description", description), ("subtitle", subtitle)]:
            if isinstance(val, list):
                val = " ".join(str(v) for v in val if v)
            if val == "N/A":
                val = ""
            
            if var_name == "title_str": title_str = val
            elif var_name == "author": author = val
            elif var_name == "edition": edition = val
            elif var_name == "description": description = val
            elif var_name == "subtitle": subtitle = val
        
        # ── API Metadata Verification Checkpoint (User Request) ──
        self._log(f"\n  📡 API METADATA CHECKPOINT:")
        self._log(f"    • Title:       {title_str or 'N/A'}")
        self._log(f"    • Subtitle:    {subtitle or 'N/A'}")
        self._log(f"    • Author:      {author or 'N/A'}")
        self._log(f"    • Edition:     {edition or 'N/A'}")
        self._log(f"    • Description: {description or 'N/A'}")
        self._log("")
        
        coll = self.config.get("collection", "Book Data")
        if title_str:
            try:
                if self.db_connector:
                    matched_doc = self.db_connector.book_title_exists(coll, title_str, return_doc=True)
                    if matched_doc:
                        self._log(f"  ⏭️  API Title '{title_str}' matches an existing book (90%+). Skipping heavy OCR.")
                        return {"duplicate": True, "doc": matched_doc}
            except Exception as e:
                self._log(f"  ⚠️ DB title check error (API Title): {e}")

        # Optimization: If all fields are already found via ISBN, skip heavy OCR
        all_meta_found = all([title_str, author, edition, description])
        interior_text = ""
        ocr_data = None

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
            self.after(0, self.update_idletasks)
            time.sleep(0.2)
            
            self._log(f"  ✂️  Cropping {book_id} ({len(images_to_process)} pages)…")
            try:
                pages = ocr_pipeline.crop_book(images_to_process, book_crops_folder)
            except Exception as e:
                raise RuntimeError(f"YOLO Cropping failed: {e}")
            
            if not pages:
                raise RuntimeError("No book content identified by YOLO. Try clearer cover photos.")
            
            self._log(f"  ✅ {len(pages)} page(s) cropped")

            # ── Phase 2: MinerU OCR ───────────────────────
            # Only run MinerU if description or interior text is still needed
            interior_text = ""
            if not description:
                self.after(0, lambda b=book_id, t=ts:
                           self.update_activity_row(b, "Processing", "OCR…", t))
                self.after(0, self.update_idletasks)
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
                    else:
                        raise RuntimeError("OCR returned empty result (check if EasyOCR models are missing/corrupt).")
                except Exception as e:
                    raise RuntimeError(f"OCR Phase failed: {e}")
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
                if title_str:
                    try:
                        coll = self.config.get("collection", "Book Data")
                        if self.db_connector:
                            matched_doc = self.db_connector.book_title_exists(coll, title_str, return_doc=True)
                            if matched_doc:
                                self._log(f"  ⏭️  Extracted Title '{title_str}' matches an existing book (90%+). Skipping heavy AI.")
                                return {"duplicate": True, "doc": matched_doc}
                    except Exception as e:
                        self._log(f"  ⚠️ DB title check error (Extracted Title): {e}")
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
                    edition = ocr_pipeline.extract_edition_from_cover(target_images, isbn=official_isbn)
                
                if not edition:
                    # FAST FALLBACK: Use Regex on accumulated OCR text before AI
                    combined_text = "\n".join(ocr_data.get("interior_texts", [])) + "\n" + "\n".join(isbn_ocr_texts)
                    edition = ocr_pipeline.find_edition_via_regex(combined_text)
                    if edition:
                        self._log(f"  ⚡ Fast-Match Edition: {edition}")
                
                if not edition:
                    # FALLBACK 1: Use the already-extracted OCR text from MinerU
                    if interior_text.strip():
                        self._log("  🔍 Vision failed → trying text-based edition search (interior text)…")
                        edition = ocr_pipeline.extract_edition_from_text(interior_text, title_str, isbn=official_isbn)
                
                if not edition and isbn_ocr_texts:
                    # FALLBACK 2: Use the ISBN OCR text (copyright page text captured during ISBN phase)
                    combined_isbn_text = "\n".join(isbn_ocr_texts)
                    self._log("  🔍 Trying text-based edition search (ISBN page text)…")
                    edition = ocr_pipeline.extract_edition_from_text(combined_isbn_text, title_str, isbn=official_isbn)
                    
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
                        
                        # Save to last_sync_results immediately so local paths are available even if skipped
                        self.last_sync_results[book_id] = books[book_id]

                        # Show this book as Queued first
                        self.after(0, lambda b=book_id, t=ts:
                                   self.update_activity_row(b, "Queued", "Book", t))
                        self.after(0, self.update_idletasks)
                        time.sleep(0.15)

                        # Now mark as Processing
                        self.after(0, lambda b=book_id, t=ts:
                                   self.update_activity_row(b, "Processing", "Book", t))
                        self.after(0, self.update_idletasks)
                        time.sleep(0.3)  # Delay for Ubuntu to process the update

                        self._log(f"  📖 {book_id}…")

                        # Build base document
                        doc = grouper.build_document(book_id, books[book_id])
                        if self.current_user:
                            doc["user_id"] = self.current_user.get("id")

                        # ── OCR Pipeline ──────────────────────────────────
                        ai_result = None
                        last_error = None
                        if OCR_AVAILABLE:
                            try:
                                ai_result = self._process_book_ocr(
                                    book_id, books[book_id], ts)
                                if ai_result:
                                    if ai_result.get("duplicate"):
                                        if "doc" in ai_result:
                                            self.last_sync_results[book_id] = {"files": books[book_id], "doc": ai_result["doc"]}
                                        self.total_skip += 1
                                        synced.add(book_id)
                                        self.after(0, lambda b=book_id, t=ts:
                                                   self.update_activity_row(b, "Skipped", "Book", t))
                                        self.after(0, self.update_idletasks)
                                        time.sleep(0.15)
                                        continue
                                        
                                    extracted_title = ai_result.get("title", "")
                                    
                                    # Check DB based on extracted title
                                    try:
                                        if extracted_title and self.db_connector:
                                            matched_doc = self.db_connector.book_title_exists(coll, extracted_title, return_doc=True)
                                            if matched_doc:
                                                self._log(f"  ⏭️  Title '{extracted_title}' matches an existing book (90%+). Skipping.")
                                                self.last_sync_results[book_id] = {"files": books[book_id], "doc": matched_doc}
                                                self._log(f"  📝 Skipped book saved to Temp List. Total cached items: {len(self.last_sync_results)}")
                                                self.total_skip += 1
                                                synced.add(book_id)
                                                self.after(0, lambda b=book_id, t=ts:
                                                           self.update_activity_row(b, "Skipped", "Book", t))
                                                self.after(0, self.update_idletasks)
                                                time.sleep(0.15)
                                                continue
                                    except Exception as e:
                                        self._log(f"  ⚠️ DB title check error: {e}")
                                        
                                    doc["title"]          = extracted_title
                                    doc["subtitle"]       = ai_result.get("subtitle", "")
                                    doc["author"]         = ai_result.get("author", "Not Found")
                                    doc["edition"]        = ai_result.get("edition", "Not Specified")
                                    doc["isbn"]           = ai_result.get("isbn", "N/A")
                                    doc["description"]    = ai_result.get("description", "")
                                    doc["ocr_completed"]  = True
                                    self._log(f"  🎉 AI done: {book_id}")
                                else:
                                    doc["ocr_completed"] = False
                                    last_error = "OCR/AI process failed to extract metadata. Check app logs."
                                    self._log(f"  ⚠️ OCR returned no results for {book_id}")
                            except Exception as e:
                                doc["ocr_completed"] = False
                                last_error = str(e)
                                self._log(f"  ⚠️ OCR pipeline error: {e}")
                        else:
                            self._log(f"  ❌ ERROR: OCR pipeline not available (models missing). Skipping book.")
                            # Mark as failed in UI
                            self.after(0, lambda b=book_id, t=ts:
                                       self.update_activity_row(b, "Failed", "No Models", t))
                            self.after(0, self.update_idletasks)
                            continue

                        # ── Insert to DB ──────────────────────────────────
                        if ai_result:
                            try:
                                self.db_connector.insert_book(coll, doc)
                                self.total_ok += 1
                                synced.add(book_id)
                                self._log(f"  ✅ Synced: {book_id}")
                                # Save full doc to last_sync_results
                                self.last_sync_results[book_id] = {"files": books[book_id], "doc": doc}
                                self._log(f"  📝 Synced book saved to Temp List. Total cached items: {len(self.last_sync_results)}")
                                
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
                                self.after(0, lambda b=book_id, t=ts, err=str(e):
                                           self.update_activity_row(b, "Failed", "DB Error", t, error_msg=err))
                                continue
                        else:
                            # If we reached here without ai_result, skip sync
                            self._log(f"  ⚠️ Skipping sync for {book_id} (No metadata extracted)")
                            self.total_fail += 1
                            self.after(0, lambda b=book_id, t=ts, err=last_error:
                                       self.update_activity_row(b, "Failed", "No Metadata", t, error_msg=err))
                            continue

                        self.after(0, self.update_idletasks)
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
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))
            self.after(0, lambda: self._set_conn_visual("active")) # Reset status
            self._log("🏁 Sync finished.")

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
            # 1. Update the hidden master log box
            if self.log_box and hasattr(self.log_box, "winfo_exists") and self.log_box.winfo_exists():
                try:
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", batch)
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                except Exception: pass
            
            # 2. Update the Settings log box IF open (LIVE MIRRORING)
            s_log = getattr(self, "_settings_log_box", None)
            if s_log and hasattr(s_log, "winfo_exists") and s_log.winfo_exists():
                try:
                    s_log.configure(state="normal")
                    # If this was the first log after "No logs yet", clear that placeholder
                    if s_log.get("1.0", "end-1c").startswith("No logs yet"):
                        s_log.delete("1.0", "end")
                    s_log.insert("end", batch)
                    s_log.see("end")
                    s_log.configure(state="disabled")
                except Exception: pass
        
        self.after(150, self._poll_log) # Faster polling


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SyncApp()
    app.mainloop()