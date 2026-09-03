import customtkinter as ctk

app = ctk.CTk()
sf = ctk.CTkScrollableFrame(app)
sf.pack(fill="both", expand=True)

btns = {}
for i in range(10):
    b = ctk.CTkButton(sf, text=f"Button {i}")
    b.pack(pady=2)
    btns[i] = b

def toggle():
    for b in btns.values():
        b.pack_forget()
    for i in [1, 3, 5]:
        btns[i].pack(pady=2)

ctk.CTkButton(app, text="Toggle", command=toggle).pack()
app.after(2000, toggle)
app.after(4000, app.destroy)
app.mainloop()
