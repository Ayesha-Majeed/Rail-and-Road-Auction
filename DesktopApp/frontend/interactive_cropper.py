import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import threading

class InteractiveCropper(ctk.CTkToplevel):
    def __init__(self, master, image_paths, C, px, fs, on_confirm_callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Interactive Human-in-the-Loop Cropping")
        self.geometry(f"{px(1000)}x{px(800)}")
        self.minsize(px(800), px(600))
        
        self.transient(master)
        self.grab_set()
        
        self.image_paths = image_paths
        self.C = C
        self.px = px
        self.fs = fs
        self.on_confirm_callback = on_confirm_callback
        
        self.current_idx = 0
        # Results: {image_path: {"global_box": [x1,y1,x2,y2], "local_box": [x1,y1,x2,y2]}}
        self.crop_results = {p: {"global_box": None, "local_box": None} for p in image_paths}
        
        # UI State
        self.active_box = None  # "global" or "local"
        self.drag_mode = None   # "move", "resize_tl", "resize_br", etc.
        self.start_x = 0
        self.start_y = 0
        self.display_image = None
        self.photo_image = None
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0
        
        # Prediction Thread State
        self.predicting = False
        self.predictions_cache = {}
        
        self._build_ui()
        
        # Start background predictions
        self._start_predictions()
        
        self._load_current_image()
        
    def _build_ui(self):
        # Top Bar
        top_bar = ctk.CTkFrame(self, fg_color=self.C.get("olive", "#8C7B5D"), height=self.px(60), corner_radius=0)
        top_bar.pack(fill="x", side="top")
        
        title = ctk.CTkLabel(top_bar, text="Human-Centric Bounding Box Editor", 
                             font=ctk.CTkFont("Inter", size=self.fs(18), weight="bold"),
                             text_color=self.C.get("white", "#FFFFFF"))
        title.pack(side="left", padx=self.px(20), pady=self.px(10))
        
        self.lbl_progress = ctk.CTkLabel(top_bar, text=f"Image 1 / {len(self.image_paths)}",
                                         font=ctk.CTkFont("Inter", size=self.fs(14)),
                                         text_color=self.C.get("white", "#FFFFFF"))
        self.lbl_progress.pack(side="right", padx=self.px(20))
        
        # Main Canvas Area
        self.canvas_frame = ctk.CTkFrame(self, fg_color=self.C.get("card", "#F5F2EC"))
        self.canvas_frame.pack(fill="both", expand=True, padx=self.px(20), pady=self.px(20))
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#E4E7EC", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        
        # Bottom Controls
        bot_bar = ctk.CTkFrame(self, fg_color="transparent")
        bot_bar.pack(fill="x", side="bottom", padx=self.px(20), pady=(0, self.px(20)))
        
        self.btn_prev = ctk.CTkButton(bot_bar, text="Previous", 
                                      font=ctk.CTkFont("Inter", size=self.fs(15), weight="bold"),
                                      height=self.px(44), width=self.px(120), corner_radius=self.px(8),
                                      fg_color="transparent", text_color=self.C.get("olive", "#8C7B5D"),
                                      border_width=2, border_color=self.C.get("olive", "#8C7B5D"),
                                      hover_color=self.C.get("card", "#F5F2EC"),
                                      command=self._prev_image)
        self.btn_prev.pack(side="left")
        
        self.btn_discard = ctk.CTkButton(bot_bar, text="Discard Image", 
                                      font=ctk.CTkFont("Inter", size=self.fs(15), weight="bold"),
                                      height=self.px(44), width=self.px(140), corner_radius=self.px(8),
                                      fg_color="#EF4444", hover_color="#B91C1C", text_color="white",
                                      command=self._discard_image)
        self.btn_discard.pack(side="left", padx=(self.px(12), 0))
        
        # Instructions
        inst = ctk.CTkLabel(bot_bar, text="🟢 Green = Global (Train)    🔴 Red = Local (Name)",
                            font=ctk.CTkFont("Inter", size=self.fs(14), weight="bold"), text_color=self.C.get("text", "#000"))
        inst.pack(side="left", expand=True)
        
        self.btn_next = ctk.CTkButton(bot_bar, text="Next", 
                                      font=ctk.CTkFont("Inter", size=self.fs(15), weight="bold"),
                                      height=self.px(44), width=self.px(120), corner_radius=self.px(8),
                                      fg_color=self.C.get("olive", "#8C7B5D"), hover_color=self.C.get("olive_h", "#AA9874"),
                                      text_color=self.C.get("white", "#FFFFFF"),
                                      command=self._next_image)
        self.btn_next.pack(side="right")
        
    def _start_predictions(self):
        # We spawn a background thread to run auto_cropper on all images.
        def run_predictor():
            import sys, os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from models.auto_cropper import predict_crops
            import time
            
            total_start = time.time()
            total_imgs = len(self.image_paths)
            print(f"\n🚀 Started background AI processing for {total_imgs} images...")
            
            for i, path in enumerate(self.image_paths):
                try:
                    res = predict_crops(path)
                    self.predictions_cache[path] = res
                    # If this image is currently displayed and user hasn't modified it yet, update it!
                    if getattr(self, "image_paths", None) and self.current_idx < len(self.image_paths) and self.image_paths[self.current_idx] == path:
                        self.after(0, self._apply_predictions_if_current, path)
                    print(f"✅ Processed {i+1}/{total_imgs} background images.")
                except Exception as e:
                    print(f"Failed to auto-crop {path}: {e}")
                    
            print(f"🎉 All {total_imgs} images processed in {time.time()-total_start:.1f}s\n")
                    
        threading.Thread(target=run_predictor, daemon=True).start()

    def _apply_predictions_if_current(self, path):
        if self.image_paths[self.current_idx] == path:
            # Only apply if we haven't already saved human corrections
            if self.crop_results[path]["global_box"] is None:
                self._load_current_image()

    def _load_current_image(self):
        path = self.image_paths[self.current_idx]
        self.lbl_progress.configure(text=f"Image {self.current_idx + 1} / {len(self.image_paths)}")
        
        self.btn_prev.configure(state="normal" if self.current_idx > 0 else "disabled")
        if self.current_idx == len(self.image_paths) - 1:
            self.btn_next.configure(text="Confirm & Generate")
        else:
            self.btn_next.configure(text="Next")
            
        self.canvas.delete("all")
        
        # Load image
        img = Image.open(path).convert("RGB")
        self.orig_w, self.orig_h = img.size
        
        # Wait until canvas is drawn to get size
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw < 10 or ch < 10:
            cw, ch = 800, 600
            
        # Calculate scale
        self.scale_factor = min(cw / self.orig_w, ch / self.orig_h)
        new_w = int(self.orig_w * self.scale_factor)
        new_h = int(self.orig_h * self.scale_factor)
        
        self.x_offset = (cw - new_w) // 2
        self.y_offset = (ch - new_h) // 2
        
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized)
        # Determine boxes
        boxes_ready = False
        if self.crop_results[path]["global_box"] is not None:
            g_box = self.crop_results[path]["global_box"]
            l_box = self.crop_results[path]["local_box"]
            boxes_ready = True
        elif path in self.predictions_cache:
            g_box = self.predictions_cache[path]["global_box"]
            l_box = self.predictions_cache[path]["local_box"]
            boxes_ready = True
            
        if boxes_ready:
            self._hide_loader()
            # Models finished: draw the image and set up boxes
            self.canvas.create_image(self.x_offset, self.y_offset, anchor="nw", image=self.photo_image)
            self.current_g_box = g_box.copy() if g_box else None
            self.current_l_box = l_box.copy() if l_box else None
            self._draw_boxes()
        else:
            # Predictions still running - show a blank canvas with a centered loader
            self.current_g_box = None
            self.current_l_box = None
            self._draw_boxes()
            self._show_loader()

    def _show_loader(self):
        if not hasattr(self, "loader_frame"):
            self.loader_frame = ctk.CTkFrame(self.canvas_frame, fg_color=self.C.get("white", "#FFFFFF"), corner_radius=self.px(16), border_width=2, border_color=self.C.get("border", "#E4E7EC"))
            self.loader_lbl = ctk.CTkLabel(self.loader_frame, text="Initializing Smart Cropping Engine...", font=ctk.CTkFont("Inter", size=self.fs(18), weight="bold"), text_color=self.C.get("text", "#1D2939"))
            self.loader_lbl.pack(pady=(self.px(24), self.px(16)), padx=self.px(40))
            self.loader_bar = ctk.CTkProgressBar(self.loader_frame, mode="indeterminate", width=self.px(250), progress_color=self.C.get("olive", "#8C7B5D"))
            self.loader_bar.pack(pady=(0, self.px(30)), padx=self.px(40))
        
        self.loader_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.loader_bar.start()
        self.update_idletasks()

    def _hide_loader(self):
        if hasattr(self, "loader_frame") and self.loader_frame.winfo_ismapped():
            self.loader_bar.stop()
            self.loader_frame.place_forget()

    def _draw_boxes(self):
        self.canvas.delete("boxes")
        self.canvas.delete("loading_txt")
        
        def draw(box, color, tag):
            if box is None: return
            x1, y1, x2, y2 = box
            # To canvas coords
            cx1 = x1 * self.scale_factor + self.x_offset
            cy1 = y1 * self.scale_factor + self.y_offset
            cx2 = x2 * self.scale_factor + self.x_offset
            cy2 = y2 * self.scale_factor + self.y_offset
            
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=3, tags=("boxes", tag))
            
            # Draw handles
            r = 5
            for hx, hy in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]:
                self.canvas.create_rectangle(hx-r, hy-r, hx+r, hy+r, fill=color, outline="white", tags=("boxes", tag))
                
        draw(self.current_g_box, "green", "global")
        draw(self.current_l_box, "red", "local")

    def _on_press(self, event):
        x, y = event.x, event.y
        # Convert canvas coords back to image coords
        ix = (x - self.x_offset) / self.scale_factor
        iy = (y - self.y_offset) / self.scale_factor
        
        # Check handles or body for local first (it's smaller, inside global)
        self.active_box, self.drag_mode = self._get_drag_mode(ix, iy, self.current_l_box, "local")
        if not self.active_box:
            self.active_box, self.drag_mode = self._get_drag_mode(ix, iy, self.current_g_box, "global")
            
        self.start_ix = ix
        self.start_iy = iy
        
    def _get_drag_mode(self, x, y, box, box_type):
        if box is None: 
            return None, None
            
        x1, y1, x2, y2 = box
        pad = 20 / self.scale_factor # 20px hit area
        
        if abs(x - x1) < pad and abs(y - y1) < pad: return box_type, "tl"
        if abs(x - x2) < pad and abs(y - y1) < pad: return box_type, "tr"
        if abs(x - x1) < pad and abs(y - y2) < pad: return box_type, "bl"
        if abs(x - x2) < pad and abs(y - y2) < pad: return box_type, "br"
        
        # Body drag
        if x1 < x < x2 and y1 < y < y2:
            return box_type, "move"
            
        return None, None

    def _on_drag(self, event):
        if not self.active_box: return
        
        x, y = event.x, event.y
        ix = (x - self.x_offset) / self.scale_factor
        iy = (y - self.y_offset) / self.scale_factor
        
        dx = ix - self.start_ix
        dy = iy - self.start_iy
        
        box = self.current_g_box if self.active_box == "global" else self.current_l_box
        
        if self.drag_mode == "move":
            box[0] += dx
            box[1] += dy
            box[2] += dx
            box[3] += dy
        elif self.drag_mode == "tl":
            box[0] += dx; box[1] += dy
        elif self.drag_mode == "tr":
            box[2] += dx; box[1] += dy
        elif self.drag_mode == "bl":
            box[0] += dx; box[3] += dy
        elif self.drag_mode == "br":
            box[2] += dx; box[3] += dy
            
        # Constrain
        box[0] = max(0, min(box[0], self.orig_w - 1))
        box[1] = max(0, min(box[1], self.orig_h - 1))
        box[2] = max(0, min(box[2], self.orig_w))
        box[3] = max(0, min(box[3], self.orig_h))
        
        if box[0] > box[2]: box[0], box[2] = box[2], box[0]
        if box[1] > box[3]: box[1], box[3] = box[3], box[1]
        
        self.start_ix = ix
        self.start_iy = iy
        
        self._draw_boxes()

    def _on_release(self, event):
        self.active_box = None
        
    def _save_current(self):
        path = self.image_paths[self.current_idx]
        if self.current_g_box is not None:
            self.crop_results[path]["global_box"] = [int(v) for v in self.current_g_box]
        if self.current_l_box is not None:
            self.crop_results[path]["local_box"] = [int(v) for v in self.current_l_box]

    def _prev_image(self):
        self._save_current()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_current_image()
            
    def _next_image(self):
        self._save_current()
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self._load_current_image()
        else:
            self.destroy()
            self.on_confirm_callback(self.crop_results)
            
    def _discard_image(self):
        path = self.image_paths[self.current_idx]
        
        # Remove from crop_results so it won't be processed for embeddings
        if path in self.crop_results:
            del self.crop_results[path]
            
        # Remove from the image queue
        self.image_paths.pop(self.current_idx)
        
        if not self.image_paths:
            # If they discarded the very last image in the batch, close the UI
            self.destroy()
            self.on_confirm_callback(self.crop_results)
            return
            
        # If we discarded the last image but there are others before it, shift the index back
        if self.current_idx >= len(self.image_paths):
            self.current_idx = len(self.image_paths) - 1
            
        self._load_current_image()
