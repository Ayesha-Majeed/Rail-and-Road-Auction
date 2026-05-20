import os
import threading
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import Canvas

class ZoomableCanvas(Canvas):
    def __init__(self, master, bg="#FFFFFF", **kwargs):
        super().__init__(master, bg=bg, highlightthickness=0, **kwargs)
        self.bind("<MouseWheel>", self.zoom)
        self.bind("<Button-4>", self.zoom)
        self.bind("<Button-5>", self.zoom)
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.pan)
        self.bind("<Configure>", self.on_resize)
        self.original_image = None
        self.scale = 1.0
        self.x = 0
        self.y = 0
        self.tk_image = None
        self._after_id = None
        self.loading_text = "Loading image..."
        self.spinner_angle = 0
        self.spinner_job = None
        self.is_loading = False
        
        # Professional Overlay: Zoom Percentage & Reset View Button
        self.overlay_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        self.zoom_lbl = ctk.CTkLabel(self.overlay_frame, text="100%", 
                                     font=ctk.CTkFont(family="Inter", size=10, weight="bold"),
                                     text_color="#475569", fg_color="#F1F5F9", corner_radius=6,
                                     width=40, height=22)
        self.zoom_lbl.pack(side="left", padx=2)
        
        self.reset_btn = ctk.CTkButton(self.overlay_frame, text="Reset View",
                                       font=ctk.CTkFont(family="Inter", size=10, weight="bold"),
                                       text_color="white", fg_color="#8C7B5D", hover_color="#6B5C41", # Primary branding olive
                                       width=70, height=22, corner_radius=6,
                                       command=self.reset_view)
        self.reset_btn.pack(side="left", padx=2)
        
    def show_text(self, text):
        self.is_loading = False
        if self.spinner_job:
            try: self.after_cancel(self.spinner_job)
            except: pass
            self.spinner_job = None
        if hasattr(self, "overlay_frame"):
            self.overlay_frame.place_forget()
        self.original_image = None
        self.loading_text = text
        self.redraw()

    def show_loading(self):
        if self.is_loading: return
        self.original_image = None
        self.is_loading = True
        if hasattr(self, "overlay_frame"):
            self.overlay_frame.place_forget()
        self.animate_spinner()

    def animate_spinner(self):
        if not self.is_loading:
            return
        self.delete("all")
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 500, 400
            
        self.spinner_angle = (self.spinner_angle + 12) % 360
        cx, cy = cw / 2, ch / 2
        r = 20
        
        self.create_arc(cx - r, cy - r, cx + r, cy + r, 
                        start=self.spinner_angle, extent=280, 
                        style="arc", outline="#8C7B5D", width=4)
        
        self.spinner_job = self.after(30, self.animate_spinner)

    def set_image(self, pil_image):
        self.is_loading = False
        if self.spinner_job:
            try: self.after_cancel(self.spinner_job)
            except: pass
            self.spinner_job = None
            
        self.original_image = pil_image
        self.update_idletasks()
        
        # Display overlay on top-right
        if hasattr(self, "overlay_frame"):
            self.overlay_frame.place(relx=0.98, rely=0.02, anchor="ne")
            
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 500, 500
        iw, ih = pil_image.size
        self.scale = min(cw / iw, ch / ih)
        self.x = (cw - iw * self.scale) / 2
        self.y = (ch - ih * self.scale) / 2
        self.update_zoom_label()
        self.redraw()

    def reset_view(self):
        if not self.original_image:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = 500, 500
        iw, ih = self.original_image.size
        self.scale = min(cw / iw, ch / ih)
        self.x = (cw - iw * self.scale) / 2
        self.y = (ch - ih * self.scale) / 2
        self.update_zoom_label()
        self.redraw()

    def update_zoom_label(self):
        if hasattr(self, "zoom_lbl") and self.original_image:
            cw, ch = self.winfo_width(), self.winfo_height()
            if cw <= 1 or ch <= 1: cw, ch = 500, 500
            iw, ih = self.original_image.size
            fit_scale = min(cw / iw, ch / ih)
            if fit_scale > 0:
                pct = int((self.scale / fit_scale) * 100)
                self.zoom_lbl.configure(text=f"{pct}%")
            else:
                self.zoom_lbl.configure(text="100%")

    def zoom(self, event):
        if not self.original_image: return
        # Check delta
        if event.num == 4 or getattr(event, 'delta', 0) > 0:
            scale_factor = 1.1
        elif event.num == 5 or getattr(event, 'delta', 0) < 0:
            scale_factor = 1.0 / 1.1
        else:
            return
            
        x = event.x
        y = event.y
        new_scale = self.scale * scale_factor
        if new_scale < 0.05 or new_scale > 20.0:
            return
            
        self.x = x - (x - self.x) * scale_factor
        self.y = y - (y - self.y) * scale_factor
        self.scale = new_scale
        self.update_zoom_label()
        self.redraw()

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.x += dx
        self.y += dy
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.redraw()
        
    def on_resize(self, event):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(50, self.redraw)

    def redraw(self):
        if self.is_loading:
            return
        self.delete("all")
        if not self.original_image:
            if hasattr(self, "loading_text") and self.loading_text:
                cw, ch = self.winfo_width(), self.winfo_height()
                self.create_text(cw/2, ch/2, text=self.loading_text, fill="#333333", font=("Inter", 14))
            return
            
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw <= 1 or ch <= 1: return
            
        iw, ih = self.original_image.size
        x0 = int((0 - self.x) / self.scale)
        y0 = int((0 - self.y) / self.scale)
        x1 = int((cw - self.x) / self.scale)
        y1 = int((ch - self.y) / self.scale)
        
        crop_x0 = max(0, min(iw, x0))
        crop_y0 = max(0, min(ih, y0))
        crop_x1 = max(0, min(iw, x1))
        crop_y1 = max(0, min(ih, y1))
        
        if crop_x0 >= crop_x1 or crop_y0 >= crop_y1: return
            
        cropped = self.original_image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        draw_w = int((crop_x1 - crop_x0) * self.scale)
        draw_h = int((crop_y1 - crop_y0) * self.scale)
        
        if draw_w <= 0 or draw_h <= 0: return
            
        resized = cropped.resize((draw_w, draw_h), PILImage.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        screen_x = crop_x0 * self.scale + self.x
        screen_y = crop_y0 * self.scale + self.y
        
        self.create_image(screen_x, screen_y, image=self.tk_image, anchor="nw")

def open_train_lot_detail(parent_app, lot_id):
    # Prevent opening multiple details windows at the same time
    if hasattr(parent_app, "_active_detail_window") and parent_app._active_detail_window:
        try:
            if parent_app._active_detail_window.winfo_exists():
                active_lot = getattr(parent_app, "_active_detail_lot_id", None)
                if active_lot == lot_id:
                    parent_app._active_detail_window.deiconify()
                    parent_app._active_detail_window.attributes("-topmost", True)
                    parent_app._active_detail_window.after(100, lambda: parent_app._active_detail_window.attributes("-topmost", False))
                    parent_app._active_detail_window.focus_force()
                else:
                    from tkinter import messagebox
                    messagebox.showwarning(
                        "Details Window Open",
                        "Please close the currently open Lot Details window before opening another lot."
                    )
                    parent_app._active_detail_window.deiconify()
                    parent_app._active_detail_window.focus_force()
                return
        except:
            parent_app._active_detail_window = None

    # Fetch lot data
    data = parent_app.last_sync_results.get(lot_id)
    if not data:
        from tkinter import messagebox
        messagebox.showerror("Error", f"No data found for Lot: {lot_id}")
        return

    paths = data.get("files", [])
    results = data.get("results", {})

    if not paths:
        from tkinter import messagebox
        messagebox.showerror("Error", "No image files in this Lot.")
        return

    C = parent_app.C
    F = parent_app.F

    # Create detailed top-level window
    win = ctk.CTkToplevel(parent_app)
    parent_app._active_detail_window = win
    parent_app._active_detail_lot_id = lot_id
    win.withdraw() # Hide immediately during setup to prevent flickering & rendering issues on Linux
    win.title(f"🚂 Train Lot Details — {lot_id}")
    win.attributes("-topmost", True)
    win.after(100, lambda: win.attributes("-topmost", False))

    # Size: 70% of screen
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    w, h = int(sw * 0.70), int(sh * 0.70)
    win.geometry(f"{w}x{h}")
    win.minsize(600, 450)

    # Center window
    ax, ay = parent_app.winfo_x(), parent_app.winfo_y()
    aw, ah = parent_app.winfo_width(), parent_app.winfo_height()
    nx = max(0, ax + (aw - w) // 2)
    ny = max(0, ay + (ah - h) // 2)
    win.geometry(f"{w}x{h}+{nx}+{ny}")

    # Layout config
    win.grid_columnconfigure(0, weight=1)
    win.grid_rowconfigure(1, weight=1)

    # Topbar
    topbar = ctk.CTkFrame(win, fg_color=C["white"], corner_radius=0, height=60)
    topbar.grid(row=0, column=0, sticky="ew")

    title_lbl = ctk.CTkLabel(topbar, text=f"🚂 Train Lot: {lot_id}", 
                             font=ctk.CTkFont(family="Outfit", size=18, weight="bold"),
                             text_color=C["text"])
    title_lbl.pack(side="left", padx=20, pady=15)

    close_btn = ctk.CTkButton(topbar, text="Close", height=32, width=80,
                              corner_radius=8,
                              font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
                              fg_color=C["olive"], hover_color=C["olive_h"], text_color="white",
                              command=win.destroy)
    close_btn.pack(side="right", padx=20, pady=15)

    # Body
    body = ctk.CTkFrame(win, fg_color=C["bg"], corner_radius=0)
    body.grid(row=1, column=0, sticky="nsew")
    body.grid_columnconfigure(0, weight=60) # Left Image panel (60% width)
    body.grid_columnconfigure(1, weight=40) # Right Results panel (40% width)
    body.grid_rowconfigure(0, weight=1)

    # Left Preview Panel
    left_panel = ctk.CTkFrame(body, fg_color=C["white"], corner_radius=10)
    left_panel.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
    left_panel.grid_columnconfigure(0, weight=1)
    left_panel.grid_rowconfigure(0, weight=8) # Main image
    left_panel.grid_rowconfigure(1, weight=2) # Thumbnails

    # Large Image Area
    preview_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
    preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    main_lbl = ZoomableCanvas(preview_frame, bg=C["white"])
    main_lbl.pack(expand=True, fill="both")
    main_lbl.show_loading()

    # Navigation overlay/buttons (Initially hidden to prevent raw clipped buttons while loading)
    nav_frame = ctk.CTkFrame(preview_frame, fg_color="transparent", height=40)



    idx_var = tk.IntVar(value=0)

    # Next / Prev buttons
    def _prev():
        if idx_var.get() > 0:
            idx_var.set(idx_var.get() - 1)
            _render_main()
            _render_thumbs()

    def _next():
        if idx_var.get() < len(paths) - 1:
            idx_var.set(idx_var.get() + 1)
            _render_main()
            _render_thumbs()

    prev_btn = ctk.CTkButton(nav_frame, text="◀ Previous", command=_prev, width=100,
                             fg_color=C["olive"], hover_color=C["olive_h"], text_color="white")
    prev_btn.pack(side="left", padx=10)

    counter_lbl = ctk.CTkLabel(nav_frame, text="0 / 0", font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
                               text_color=C["text"])
    counter_lbl.pack(side="left", expand=True)

    next_btn = ctk.CTkButton(nav_frame, text="Next ▶", command=_next, width=100,
                             fg_color=C["olive"], hover_color=C["olive_h"], text_color="white")
    next_btn.pack(side="right", padx=10)

    # Thumbnails Frame
    scale = getattr(parent_app, "_scale", 1.0)
    thumbs_scroll = ctk.CTkScrollableFrame(left_panel, orientation="horizontal", 
                                          fg_color=C["bg"], height=int(150 * scale))

    # Hide horizontal scrollbar completely for a modern clean touch-dragging look
    try:
        thumbs_scroll._scrollbar.grid_forget()
    except:
        pass

    # --- Modern Mouse Drag-to-Scroll & Horizontal Scroll Wheel Panning ---
    canvas = thumbs_scroll._parent_canvas
    drag_data = {"x": 0, "dragged": False}
    
    def on_drag_start(event):
        drag_data["x"] = event.x_root
        drag_data["dragged"] = False
        
    def on_drag_move(event):
        dx = drag_data["x"] - event.x_root
        if abs(dx) > 4:
            drag_data["dragged"] = True
        try:
            canvas.xview_scroll(int(dx / 5), "units")
            update_scroll_cues()
        except:
            pass
        drag_data["x"] = event.x_root

    def on_mouse_wheel(event):
        try:
            if event.num == 4 or getattr(event, 'delta', 0) > 0:
                canvas.xview_scroll(-2, "units")
            elif event.num == 5 or getattr(event, 'delta', 0) < 0:
                canvas.xview_scroll(2, "units")
            update_scroll_cues()
        except:
            pass

    # Bind scroll events to the container canvas
    canvas.bind("<ButtonPress-1>", on_drag_start, add="+")
    canvas.bind("<B1-Motion>", on_drag_move, add="+")
    canvas.bind("<MouseWheel>", on_mouse_wheel, add="+")
    canvas.bind("<Button-4>", on_mouse_wheel, add="+")
    canvas.bind("<Button-5>", on_mouse_wheel, add="+")

    # --- Live navigation scroll cues ---
    left_cue = ctk.CTkLabel(left_panel, text="◀", font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
                            text_color="#94A3B8", fg_color="#F1F5F9", corner_radius=12, width=24, height=24)
    right_cue = ctk.CTkLabel(left_panel, text="▶", font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
                             text_color="#94A3B8", fg_color="#F1F5F9", corner_radius=12, width=24, height=24)

    def update_scroll_cues(*args):
        try:
            x_start, x_end = canvas.xview()
            if x_start <= 0.01:
                left_cue.place_forget()
            else:
                left_cue.place(in_=thumbs_scroll, relx=0.01, rely=0.5, anchor="w")
                
            if x_end >= 0.99:
                right_cue.place_forget()
            else:
                right_cue.place(in_=thumbs_scroll, relx=0.99, rely=0.5, anchor="e")
        except:
            pass

    # Chain our scroll listener to xscrollcommand
    try:
        canvas.configure(xscrollcommand=lambda *args: (thumbs_scroll._scrollbar.set(*args), update_scroll_cues()))
    except:
        pass

    # Bind layout resize/startup to update cues
    canvas.bind("<Configure>", lambda e: win.after(100, update_scroll_cues), add="+")

    def bind_thumbnail_scroll(w):
        w.bind("<ButtonPress-1>", on_drag_start, add="+")
        w.bind("<B1-Motion>", on_drag_move, add="+")
        w.bind("<MouseWheel>", on_mouse_wheel, add="+")
        w.bind("<Button-4>", on_mouse_wheel, add="+")
        w.bind("<Button-5>", on_mouse_wheel, add="+")

    # --- Layout Stabilization Loader ---
    # Unified single circular loader inside ZoomableCanvas is used.
    spinner_state = {"angle": 0, "job": None}

    def _spin_loader():
        pass

    def _stop_loader():
        pass

    def _close_win():
        job = spinner_state.get("job")
        if job:
            try: win.after_cancel(job)
            except: pass
        if hasattr(parent_app, "_active_detail_window"):
            parent_app._active_detail_window = None
        if hasattr(parent_app, "_active_detail_lot_id"):
            parent_app._active_detail_lot_id = None
        win.withdraw()
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _close_win)
    close_btn.configure(command=_close_win)

    # Keyboard Navigation Shortcuts
    def _on_key_press(event):
        if event.keysym == "Left":
            _prev()
        elif event.keysym == "Right":
            _next()

    win.bind("<Left>", _on_key_press)
    win.bind("<Right>", _on_key_press)

    _spin_loader()

    # Right Results Panel (Scrollable Card Layout matching Book OCR)
    right_panel = ctk.CTkFrame(body, fg_color="#F8FAFC", corner_radius=10,
                               border_width=1, border_color=C["border"])
    right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
    right_panel.grid_rowconfigure(0, weight=1)
    right_panel.grid_columnconfigure(0, weight=1)
    right_panel.grid_propagate(False)

    res_scroll = ctk.CTkScrollableFrame(right_panel, fg_color="#F8FAFC",
                                        scrollbar_fg_color="#E5E7EB",
                                        scrollbar_button_color=C["olive_dk"],
                                        scrollbar_button_hover_color=C["olive"])
    res_scroll.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Section 1: Railroads Name Card (Blue Theme) ──
    card1 = ctk.CTkFrame(res_scroll, fg_color="#FFFFFF", border_width=2, border_color="#CBD5E1", corner_radius=12)
    card1.pack(fill="x", padx=12, pady=(12, 6))

    card1_hdr = ctk.CTkFrame(card1, fg_color="#EFF6FF", height=48, corner_radius=0)
    card1_hdr.pack(fill="x")
    card1_hdr.pack_propagate(False)

    card1_badge_wrap = ctk.CTkFrame(card1_hdr, fg_color="transparent")
    card1_badge_wrap.pack(side="left", padx=16)

    ctk.CTkLabel(card1_badge_wrap, text="Ai", width=28, height=28,
                 corner_radius=6, fg_color="#DBEAFE", text_color="#2563EB",
                 font=ctk.CTkFont(family="Inter", size=11, weight="bold")
                 ).pack(side="left")

    ctk.CTkLabel(card1_badge_wrap, text="Railroads",
                 font=ctk.CTkFont(family="Inter", size=int(15 * scale), weight="bold"),
                 text_color="#1E3A8A").pack(side="left", padx=(10, 0))

    railroads_val_lbl = ctk.CTkLabel(card1, text="",
                                     font=ctk.CTkFont(family="Inter", size=int(14 * scale), weight="bold"),
                                     text_color="#94A3B8", anchor="nw", justify="left", wraplength=350)
    railroads_val_lbl.pack(anchor="nw", padx=24, pady=20, fill="both", expand=True)

    # ── Section 2: Locomotive Engine Types Card (Purple Theme) ──
    card2 = ctk.CTkFrame(res_scroll, fg_color="#FFFFFF", border_width=2, border_color="#CBD5E1", corner_radius=12)
    card2.pack(fill="x", padx=12, pady=6)

    card2_hdr = ctk.CTkFrame(card2, fg_color="#FAF5FF", height=48, corner_radius=0)
    card2_hdr.pack(fill="x")
    card2_hdr.pack_propagate(False)

    card2_badge_wrap = ctk.CTkFrame(card2_hdr, fg_color="transparent")
    card2_badge_wrap.pack(side="left", padx=16)

    ctk.CTkLabel(card2_badge_wrap, text="Ai", width=28, height=28,
                 corner_radius=6, fg_color="#F3E8FF", text_color="#9C25EB",
                 font=ctk.CTkFont(family="Inter", size=11, weight="bold")
                 ).pack(side="left")

    ctk.CTkLabel(card2_badge_wrap, text="Locomotive Engine Types",
                 font=ctk.CTkFont(family="Inter", size=int(15 * scale), weight="bold"),
                 text_color="#581C87").pack(side="left", padx=(10, 0))

    engine_types_val_lbl = ctk.CTkLabel(card2, text="",
                                         font=ctk.CTkFont(family="Inter", size=int(14 * scale), weight="bold"),
                                         text_color="#94A3B8", anchor="nw", justify="left", wraplength=350)
    engine_types_val_lbl.pack(anchor="nw", padx=24, pady=20, fill="both", expand=True)

    # ── Section 3: Description Card (Green Theme) ──
    card3 = ctk.CTkFrame(res_scroll, fg_color="#FFFFFF", border_width=2, border_color="#CBD5E1", corner_radius=12)
    card3.pack(fill="x", padx=12, pady=6)

    card3_hdr = ctk.CTkFrame(card3, fg_color="#F0FDF4", height=48, corner_radius=0)
    card3_hdr.pack(fill="x")
    card3_hdr.pack_propagate(False)

    card3_badge_wrap = ctk.CTkFrame(card3_hdr, fg_color="transparent")
    card3_badge_wrap.pack(side="left", padx=16)

    ctk.CTkLabel(card3_badge_wrap, text="Ai", width=28, height=28,
                 corner_radius=6, fg_color="#DCFCE7", text_color="#15803D",
                 font=ctk.CTkFont(family="Inter", size=11, weight="bold")
                 ).pack(side="left")

    ctk.CTkLabel(card3_badge_wrap, text="Description",
                 font=ctk.CTkFont(family="Inter", size=int(15 * scale), weight="bold"),
                 text_color="#14532D").pack(side="left", padx=(10, 0))


    initial_desc = data.get("ai_description")
    if not initial_desc:
        # Calculate a deterministic fallback description from results
        railroad_counts = {}
        railroad_type_breakdown = {}
        for p_img in paths:
            r_res = results.get(p_img, {})
            if not r_res:
                continue
            rr = r_res.get("railroad")
            lt = r_res.get("loco_type")
            if rr and rr not in ["-", "Unprocessed", "Pending Analysis"] and lt and lt not in ["-", "Unprocessed", "Pending Analysis"]:
                railroad_counts[rr] = railroad_counts.get(rr, 0) + 1
                if rr not in railroad_type_breakdown:
                    railroad_type_breakdown[rr] = {}
                railroad_type_breakdown[rr][lt] = railroad_type_breakdown[rr].get(lt, 0) + 1

        if railroad_counts:
            total_slides = len(paths)
            railroads_list = sorted(list(railroad_counts.keys()))
            railroads_str = ", ".join(railroads_list)
            suffix = "Railroad" if len(railroads_list) == 1 else "Railroads"
            desc_prefix = ""
            
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
                
            fallback_note = f"This lot contains slides of {breakdown_str}."
            initial_desc = desc_prefix + fallback_note
        else:
            initial_desc = "This lot contains detailed train slide analysis. Engine classifications and railroad identities are extracted using deep neural network pipelines."

    desc_val_lbl = ctk.CTkLabel(card3, text=initial_desc,
                                 font=ctk.CTkFont(family="Inter", size=int(13 * scale)),
                                 text_color="#4B5563", anchor="nw", justify="left", wraplength=350)
    desc_val_lbl.pack(anchor="nw", padx=24, pady=16, fill="both", expand=True)

    # ── Section 4: Active Slide Metadata (Slate Theme) ──
    card4 = ctk.CTkFrame(res_scroll, fg_color="#FFFFFF", border_width=2, border_color="#CBD5E1", corner_radius=12)
    card4.pack(fill="x", padx=12, pady=(6, 12))

    card4_hdr = ctk.CTkFrame(card4, fg_color="#F1F5F9", height=48, corner_radius=0)
    card4_hdr.pack(fill="x")
    card4_hdr.pack_propagate(False)

    card4_badge_wrap = ctk.CTkFrame(card4_hdr, fg_color="transparent")
    card4_badge_wrap.pack(side="left", padx=16)

    ctk.CTkLabel(card4_badge_wrap, text="Slide", width=42, height=28,
                 corner_radius=6, fg_color="#E2E8F0", text_color="#475569",
                 font=ctk.CTkFont(family="Inter", size=11, weight="bold")
                 ).pack(side="left")

    ctk.CTkLabel(card4_badge_wrap, text="Active Slide Metadata",
                 font=ctk.CTkFont(family="Inter", size=int(14 * scale), weight="bold"),
                 text_color="#334155").pack(side="left", padx=(10, 0))

    card4_body = ctk.CTkFrame(card4, fg_color="transparent")
    card4_body.pack(fill="both", expand=True, padx=20, pady=16)

    def _create_local_field(parent, label_text):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        
        lbl = ctk.CTkLabel(row, text=label_text, 
                           font=ctk.CTkFont(family="Inter", size=int(12 * scale), weight="bold"),
                           text_color="#64748B", anchor="w")
        lbl.pack(side="left")
        
        val = ctk.CTkLabel(row, text="", 
                           font=ctk.CTkFont(family="Inter", size=int(12 * scale)),
                           text_color="#334155", anchor="w", justify="left")
        val.pack(side="left", padx=(10, 0), fill="x", expand=True)
        return val

    lbl_local_railroad = _create_local_field(card4_body, "Railroad:")
    lbl_local_conf = _create_local_field(card4_body, "Confidence:")
    lbl_local_type = _create_local_field(card4_body, "Loco Type:")
    lbl_local_parts = _create_local_field(card4_body, "Parts:")

    # Dynamic Wrapping helper for clean professional look (prevents text clipping/overflow)
    def _update_label_wrapping():
        if not win.winfo_exists(): return
        try:
            rw = right_panel.winfo_width()
            if rw < 100:
                rw = 300 # safe fallback
            
            # Card values (Railroads, Engine Types, Description)
            card_wp = max(120, rw - 90)
            railroads_val_lbl.configure(wraplength=card_wp)
            engine_types_val_lbl.configure(wraplength=card_wp)
            desc_val_lbl.configure(wraplength=card_wp)
            
            # Metadata fields (Card 4 values)
            meta_wp = max(120, rw - 180)
            lbl_local_railroad.configure(wraplength=meta_wp)
            lbl_local_conf.configure(wraplength=meta_wp)
            lbl_local_type.configure(wraplength=meta_wp)
            lbl_local_parts.configure(wraplength=meta_wp)
        except Exception:
            pass

    def _render_main():
        """Asynchronously load and render the main image, then update UI components on the main thread."""
        if not paths:
            return
        idx = idx_var.get()
        p = paths[idx]

        # Show loading spinner in the canvas
        main_lbl.show_loading()

        # Hide navigation and scroll cues while loading for a clean look
        try:
            nav_frame.pack_forget()
            left_cue.place_forget()
            right_cue.place_forget()
        except:
            pass

        def _update_ui(image_obj=None, error_msg=None):
            """Update UI components after image processing is finished.
            This runs on the main thread via win.after.
            """
            # Show navigation controls regardless of success or failure
            try:
                nav_frame.pack(side="bottom", fill="x", pady=5)
                update_scroll_cues()
            except:
                pass

            # Update counter and navigation button states
            counter_lbl.configure(text=f"{idx + 1} / {len(paths)}")
            prev_btn.configure(state="normal" if idx > 0 else "disabled")
            next_btn.configure(state="normal" if idx < len(paths) - 1 else "disabled")

            if error_msg:
                # Display error text in the canvas
                main_lbl.show_text(f"Error loading image:\n{error_msg}")
            else:
                # Set the processed image
                main_lbl.set_image(image_obj)

            # Recalculate global stats across all slides
            all_railroads = set()
            all_types = set()
            for p_path, r_res in results.items():
                if r_res:
                    rr = r_res.get("railroad")
                    lt = r_res.get("loco_type")
                    if rr and rr not in ["-", "Unprocessed"]:
                        all_railroads.add(rr)
                    if lt and lt not in ["-", "Unprocessed"]:
                        all_types.add(lt)
            if all_railroads:
                unique_rr_str = ", ".join(sorted(list(all_railroads)))
                railroads_val_lbl.configure(text=unique_rr_str, text_color="#1E293B")
            else:
                railroads_val_lbl.configure(text="Pending Analysis", text_color="#94A3B8")
            if all_types:
                unique_type_str = ", ".join(sorted(list(all_types)))
                engine_types_val_lbl.configure(text=unique_type_str, text_color="#0D0D0D")
            else:
                engine_types_val_lbl.configure(text="Pending Analysis", text_color="#94A3B8")

            # Update local slide metadata for the currently displayed image
            res = results.get(p, {})
            if res:
                lbl_local_railroad.configure(text=res.get('railroad', '-'), text_color="#1E3A8A")
                conf = res.get('confidence', 0)
                lbl_local_conf.configure(text=f"{conf:.1%}" if isinstance(conf, float) else f"{conf}")
                lbl_local_type.configure(text=res.get('loco_type', '-'), text_color="#581C87")
                parts = res.get('parts_detected', [])
                parts_str = ", ".join(parts) if parts else "None"
                lbl_local_parts.configure(text=parts_str)
            else:
                lbl_local_railroad.configure(text="Unprocessed", text_color="#64748B")
                lbl_local_conf.configure(text="-")
                lbl_local_type.configure(text="-", text_color="#64748B")
                lbl_local_parts.configure(text="-")

            # Adjust wrapping based on current window size
            _update_label_wrapping()

        def _worker():
            """Background thread to load the image and draw any bounding boxes."""
            try:
                im = PILImage.open(p)
                # Draw bounding boxes if present in results
                res = results.get(p, {})
                if res and "boxes" in res:
                    im_drawn = im.copy()
                    draw = ImageDraw.Draw(im_drawn)
                    boxes = res["boxes"]
                    # Load a larger font for nice labels
                    try:
                        from PIL import ImageFont
                        try:
                            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
                        except:
                            try:
                                font = ImageFont.truetype("ubuntu-font-family/Ubuntu-B.ttf", 36)
                            except:
                                try:
                                    font = ImageFont.load_default(size=36)
                                except:
                                    font = ImageFont.load_default()
                    except:
                        font = None

                    def draw_nice_bbox(box, label, color):
                        if not box:
                            return
                        draw.rectangle(box, outline=color, width=10)
                        try:
                            if hasattr(font, "getbbox"):
                                _, _, tw, th = font.getbbox(label)
                            else:
                                tw, th = draw.textsize(label, font=font)
                        except:
                            tw, th = len(label) * 15, 20
                        pad_x, pad_y = 12, 8
                        label_y1 = box[1] - th - pad_y * 2
                        if label_y1 < 0:
                            label_y1 = box[1]
                        label_box = [box[0], label_y1, box[0] + tw + pad_x * 2, label_y1 + th + pad_y * 2]
                        draw.rectangle(label_box, fill=color)
                        draw.text((label_box[0] + pad_x, label_box[1] + pad_y), label, fill="#000000", font=font)

                    # Global Train crop box (green)
                    g_box = boxes.get("global_train")
                    if g_box:
                        draw_nice_bbox(g_box, "Locomotive Crop", "#10B981")
                    # Local emblem box (blue)
                    l_box = boxes.get("local_emblem")
                    if l_box:
                        draw_nice_bbox(l_box, "Emblem / Logo Crop", "#3B82F6")
                    # Parts boxes with premium colors
                    PART_COLORS = {
                        "pantograph": "#FFFFFF",
                        "fan": "#FBBF24",
                        "fuel_tank": "#F472B6",
                        "side_rods": "#34D399",
                        "chimney": "#FB923C",
                    }
                    for p_item in boxes.get("parts", []):
                        p_box = p_item.get("box")
                        p_label = p_item.get("label", "")
                        if p_box and p_label:
                            color = PART_COLORS.get(p_label.lower(), "#FFFFFF")
                            nice_label = p_label.replace("_", " ").title()
                            draw_nice_bbox(p_box, nice_label, color)
                    im = im_drawn
                # Schedule UI update on the main thread if window still exists
                try:
                    if win.winfo_exists():
                        win.after(0, lambda: _update_ui(image_obj=im))
                except Exception:
                    pass
            except Exception as e:
                try:
                    if win.winfo_exists():
                        win.after(0, lambda: _update_ui(error_msg=str(e)))
                except Exception:
                    pass

        # Start background worker thread
        threading.Thread(target=_worker, daemon=True).start()

    thumb_widgets = []
    def _render_thumbs():
        nonlocal thumb_widgets
        new_idx = idx_var.get()
        
        # Grid thumbs_scroll now that thumbnails are ready!
        try:
            if not thumbs_scroll.winfo_viewable():
                thumbs_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        except:
            pass
        
        if not thumb_widgets:
            for ch in thumbs_scroll.winfo_children():
                ch.destroy()
            
            tsize_h = int(100 * scale)
            tsize_w = int(tsize_h * 1.33)
            
            for i, p in enumerate(paths):
                is_active = (i == new_idx)
                cell = ctk.CTkFrame(thumbs_scroll, fg_color=C["white"] if is_active else "transparent",
                                     corner_radius=8, border_width=2 if is_active else 0,
                                     border_color=C["olive"])
                cell.pack(side="left", padx=4, pady=4)
                
                # Placeholder label for image with fixed dimensions to prevent layout shifting
                img_lbl = ctk.CTkLabel(cell, text="Loading...", width=tsize_w, height=tsize_h,
                                       fg_color="#F1F5F9", text_color="#94A3B8",
                                       font=ctk.CTkFont(family="Inter", size=9))
                img_lbl.pack(padx=2, pady=2)
                
                fname = os.path.basename(p)
                if len(fname) > 10:
                    fname = fname[:7] + "..."
                lab = ctk.CTkLabel(cell, text=fname, font=ctk.CTkFont(family="Inter", size=int(10 * scale)),
                                   text_color=C["olive"] if is_active else C["muted"])
                lab.pack(padx=4, pady=(0, 2))
                
                thumb_widgets.append({"cell": cell, "img_lbl": img_lbl, "label": lab})
                
                # Bind dragging and scrolling events
                bind_thumbnail_scroll(cell)
                bind_thumbnail_scroll(img_lbl)
                bind_thumbnail_scroll(lab)
                
                def _load_thumb_async(idx=i, path=p, label=img_lbl):
                    def _loader():
                        try:
                            im = PILImage.open(path)
                            im.thumbnail((tsize_w, tsize_h))
                            
                            def _update():
                                try:
                                    if not win.winfo_exists():
                                        return
                                    timg = ctk.CTkImage(light_image=im, dark_image=im, size=im.size)
                                    label.configure(image=timg, text="")
                                    label.image = timg  # keep reference
                                except Exception:
                                    pass
                            
                            try:
                                if win.winfo_exists():
                                    win.after(0, _update)
                            except Exception:
                                pass
                        except Exception:
                            def _update_error():
                                try:
                                    if win.winfo_exists():
                                        label.configure(text="Error", fg_color="#FEE2E2", text_color="#EF4444")
                                except Exception:
                                    pass
                            try:
                                if win.winfo_exists():
                                    win.after(0, _update_error)
                            except Exception:
                                pass
                            
                    threading.Thread(target=_loader, daemon=True).start()
                
                # Stagger thread starts to prevent UI stuttering
                win.after(i * 15, _load_thumb_async)
                
                def _mk_cb(idx=i):
                    def _callback(event=None):
                        if not drag_data.get("dragged", False):
                            idx_var.set(idx)
                            _render_main()
                            _render_thumbs()
                    return _callback
                
                cell.bind("<ButtonRelease-1>", _mk_cb(), add="+")
                img_lbl.bind("<ButtonRelease-1>", _mk_cb(), add="+")
                lab.bind("<ButtonRelease-1>", _mk_cb(), add="+")
        else:
            for i, widgets in enumerate(thumb_widgets):
                is_active = (i == new_idx)
                widgets["cell"].configure(
                    fg_color=C["white"] if is_active else "transparent",
                    border_width=2 if is_active else 0,
                    border_color=C["olive"]
                )
                widgets["label"].configure(
                    text_color=C["olive"] if is_active else C["muted"]
                )

    _detail_last_bw = [0]
    def _sync_detail_split():
        try:
            bw = max(360, body.winfo_width() - 24)
        except Exception:
            return
        if bw == _detail_last_bw[0]: return
        _detail_last_bw[0] = bw
        left = int(bw * 0.55)
        left = max(220, min(left, bw - 220))
        right = max(220, bw - left)
        body.grid_columnconfigure(0, weight=0, minsize=left)
        body.grid_columnconfigure(1, weight=1, minsize=right)

    # Defer render until layout is stabilized (Matches Book OCR)
    def _deferred_init():
        if not win.winfo_exists(): return
        
        # Enforce centered layout immediately
        win.update_idletasks()
        ax, ay = parent_app.winfo_x(), parent_app.winfo_y()
        aw, ah = parent_app.winfo_width(), parent_app.winfo_height()
        nx = max(0, ax + (aw - w) // 2)
        ny = max(0, ay + (ah - h) // 2)
        win.geometry(f"{w}x{h}+{nx}+{ny}")
        
        # deiconify and force layout mapping to prevent transparency bugs on Linux
        win.deiconify()
        win.update()
        
        def _wait_layout_ready(attempt=0):
            if not win.winfo_exists():
                return
            win.update_idletasks()
            _sync_detail_split()
            if preview_frame.winfo_width() >= 100 and preview_frame.winfo_height() >= 100:
                _stop_loader()
                _render_main()
                _render_thumbs()
                return
            if attempt < 30:
                win.after(40, lambda: _wait_layout_ready(attempt + 1))
            else:
                _stop_loader()
                _render_main()
                _render_thumbs()

        win.after(100, _wait_layout_ready)

    _deferred_init()

    # Debounced Resize handler with dynamic parts text wrapping to prevent infinite layout recursion loops
    win._resize_job = None
    win._last_resize_w = 0
    resize_loader = [None]
    
    def _show_resize_loader():
        if not win.winfo_exists() or resize_loader[0] is not None:
            return
        overlay = ctk.CTkFrame(right_panel, fg_color="#F8FAFC", corner_radius=10)
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        spinner = Canvas(overlay, width=48, height=48, bd=0, highlightthickness=0, bg="#F8FAFC")
        spinner.place(relx=0.5, rely=0.5, anchor="center")
        
        arc = spinner.create_arc(6, 6, 42, 42, start=0, extent=280, style="arc", outline=C["olive"], width=4)
        
        state = {"angle": 0, "active": True}
        def _spin():
            if not win.winfo_exists() or not overlay.winfo_exists() or not state["active"]:
                return
            state["angle"] = (state["angle"] + 15) % 360
            spinner.itemconfigure(arc, start=state["angle"])
            win.after(30, _spin)
            
        _spin()
        resize_loader[0] = (overlay, state)
        
    def _hide_resize_loader():
        if resize_loader[0]:
            overlay, state = resize_loader[0]
            state["active"] = False
            try: overlay.destroy()
            except: pass
            resize_loader[0] = None

    def _on_resize(event):
        if event.widget != win: return
        
        new_w = win.winfo_width()
        if abs(new_w - win._last_resize_w) < 10:
            return
        win._last_resize_w = new_w
        
        _show_resize_loader()
        
        if win._resize_job:
            try: win.after_cancel(win._resize_job)
            except: pass
            
        win._resize_job = win.after(200, _do_resize)

    def _do_resize():
        win._resize_job = None
        if not win.winfo_exists():
            return
        _sync_detail_split()
        _render_main()
        _hide_resize_loader()

    win.bind("<Configure>", _on_resize)
