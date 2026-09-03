#!/bin/bash
# Launcher for Rail-and-Road Auction Desktop App
# Forces venv's bundled cuDNN 9.20 to load BEFORE any system cuDNN libraries.
# This prevents CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH errors.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_CUDNN_LIB="$SCRIPT_DIR/../venv310/lib/python3.10/site-packages/nvidia/cudnn/lib"

if [ -d "$VENV_CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$VENV_CUDNN_LIB:$LD_LIBRARY_PATH"
    echo "✅ cuDNN lib path set: $VENV_CUDNN_LIB"
else
    echo "⚠️  venv cuDNN lib not found at: $VENV_CUDNN_LIB"
fi

source "$SCRIPT_DIR/../venv310/bin/activate"
python "$SCRIPT_DIR/main.py" "$@"
