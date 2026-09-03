import customtkinter as ctk

app = ctk.CTk()

def open_msg():
    top = ctk.CTkToplevel(app)
    top.title("Test")
    top.geometry("300x200")
    # top.grab_set() BEFORE wait_visibility causes issues on some WMs
    # top.grab_set()
    lbl = ctk.CTkLabel(top, text="Hello World")
    lbl.pack(expand=True, fill="both")
    top.update_idletasks()
    top.update()
    
    # Try with grab_set after wait_visibility
    top.wait_visibility()
    top.grab_set()
    top.wait_window()

btn = ctk.CTkButton(app, text="Open", command=open_msg)
btn.pack()
app.after(500, open_msg)
app.after(3000, app.destroy)
app.mainloop()
