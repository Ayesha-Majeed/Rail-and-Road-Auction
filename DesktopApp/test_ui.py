import sys
try:
    from frontend.sync_app_ui import SyncAppUI
    app = SyncAppUI(None, {"id": "test"}, None)
    app._show_detail("748", doc=None)
except Exception as e:
    import traceback
    traceback.print_exc()
