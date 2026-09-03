import os
import sys
import traceback
import ssl

def setup_global_error_handler():
    # ─── Global Exception Catcher for Windows Debugging (IMMEDIATE) ───
    def _global_log_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            print("\n👋 Closing application gracefully...")
            sys.exit(0)
            
        try:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(sys.argv[0] or "."))
        except Exception:
            base_dir = "."
            
        log_path = os.path.join(base_dir, "syna_error.log")
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        tb_text = "".join(tb_lines)
        
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== Uncaught Application Exception ===\n")
                f.write(tb_text)
        except Exception:
            pass
            
        try:
            from tkinter import messagebox
            messagebox.showerror(
                "Application Error",
                f"An unexpected error occurred during execution.\n\n"
                f"A detailed crash report has been saved to:\n{log_path}\n\n"
                f"Error: {exc_value}\n\n"
                f"Traceback:\n{tb_text[:1000]}"
            )
        except Exception:
            pass
            
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _global_log_exception

    # ─── SSL Certificate Fix for Windows / PyInstaller (IMMEDIATE) ───
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    # ─── Stream Redirection for Windows --windowed mode ───
    class SafeStream:
        def __init__(self, original): self._s = original
        def write(self, data):
            if self._s: self._s.write(data)
        def flush(self):
            if self._s: self._s.flush()
        @property
        def encoding(self): return getattr(self._s, 'encoding', 'utf-8') or 'utf-8'

    if sys.stdout is None: sys.stdout = SafeStream(None)
    if sys.stderr is None: sys.stderr = SafeStream(None)
