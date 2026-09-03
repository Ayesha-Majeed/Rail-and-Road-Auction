import customtkinter as ctk

app = ctk.CTk()
app.geometry("800x600")

def _create_dialog(title, message, mtype):
    top = ctk.CTkToplevel(app)
    top.title(title)
    
    px = lambda x: x
    fs = 14
    
    w, h = px(360), px(180)
    
    top.geometry(f"{w}x{h}")
    top.resizable(False, False)
    top.transient(app)
    top.grab_set()
    top.attributes("-topmost", True)
    
    colors = {"info": "#3B82F6", "error": "#EF4444", "warning": "#F59E0B", "question": "#3B82F6"}
    color = colors.get(mtype, "#3B82F6")
    
    frame = ctk.CTkFrame(top, fg_color="transparent")
    frame.pack(fill="both", expand=True, padx=px(20), pady=px(20))
    
    lbl = ctk.CTkLabel(frame, text=message, font=ctk.CTkFont("Inter", size=fs), wraplength=w - px(40))
    lbl.pack(expand=True, fill="both", pady=(0, px(20)))
    
    result = [False]
    def _close(res):
        result[0] = res
        top.destroy()
        
    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.pack(fill="x")
    
    btn_font = ctk.CTkFont("Inter", size=fs, weight="bold")
    
    if mtype == "question":
        btn_frame.grid_columnconfigure((0, 1), weight=1)
        btn_no = ctk.CTkButton(btn_frame, text="No", font=btn_font, fg_color="gray", command=lambda: _close(False), width=px(80), height=px(36))
        btn_no.grid(row=0, column=0, padx=px(10), sticky="e")
        
        btn_yes = ctk.CTkButton(btn_frame, text="Yes", font=btn_font, fg_color=color, command=lambda: _close(True), width=px(80), height=px(36))
        btn_yes.grid(row=0, column=1, padx=px(10), sticky="w")
    else:
        btn_ok = ctk.CTkButton(btn_frame, text="OK", font=btn_font, fg_color=color, command=lambda: _close(True), width=px(100), height=px(36))
        btn_ok.pack(anchor="center")
        
    top.update_idletasks()
    top.update()
    top.wait_window()
    return result[0]

btn = ctk.CTkButton(app, text="Open", command=lambda: _create_dialog("Test", "This is a test message", "info"))
btn.pack(pady=50)

app.after(1000, lambda: _create_dialog("Test Auto", "This should appear", "info"))
app.after(3000, app.destroy)
app.mainloop()
