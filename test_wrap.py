import customtkinter as ctk
app = ctk.CTk()
ctk.set_widget_scaling(1.35)
app.geometry("800x600")

def _wrap(e=None):
    w = f.winfo_width()
    scale = ctk.ScalingTracker.get_widget_scaling(f)
    print(f"w={w}, scale={scale}, wl={int(w / scale)}")
    l.configure(wraplength=int(w / scale))

f = ctk.CTkFrame(app)
f.pack(fill="both", expand=True, padx=20, pady=20)
l = ctk.CTkLabel(f, text="This is a very long text to see how it wraps in custom tkinter when widget scaling is applied. Let's see if it wraps correctly or if it wraps too early.", anchor="w", justify="left", bg_color="red")
l.pack(fill="x")

f.bind("<Configure>", _wrap)
app.after(1000, app.destroy)
app.mainloop()
