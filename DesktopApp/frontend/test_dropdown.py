import customtkinter as ctk

app = ctk.CTk()
app.geometry("400x500")

dropdown_window = ctk.CTkFrame(app, width=300, height=300, fg_color="white")
dropdown_window.pack_propagate(False)
dropdown_window.place(x=50, y=50)

search_bg = ctk.CTkFrame(dropdown_window, fg_color="gray")
search_bg.pack(fill="x", padx=12, pady=12)

search_entry = ctk.CTkEntry(search_bg, placeholder_text="Search...")
search_entry.pack(fill="x", padx=8, pady=2)

dropdown_frame = ctk.CTkScrollableFrame(dropdown_window, fg_color="blue", corner_radius=0)
dropdown_frame.pack(fill="both", expand=True, padx=4, pady=4)
dropdown_frame.grid_columnconfigure(0, weight=1)

for idx in range(10):
    btn = ctk.CTkButton(dropdown_frame, text=f"Item {idx}", height=38, fg_color="transparent", text_color="black")
    btn.grid(row=idx, column=0, sticky="ew", pady=2, padx=6)

app.after(3000, app.destroy)
app.mainloop()
