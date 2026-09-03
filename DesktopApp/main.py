import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import multiprocessing
from backend.error_handler import setup_global_error_handler

if __name__ == "__main__":
    multiprocessing.freeze_support()
    setup_global_error_handler()

    # Import and run the UI
    from frontend.sync_app_ui import SyncApp
    app = SyncApp()
    app.mainloop()
