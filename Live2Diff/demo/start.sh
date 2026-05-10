#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_NODE_BIN="$PROJECT_ROOT/.tools/node-v20.20.1-linux-x64/bin"

if [ -d "$LOCAL_NODE_BIN" ]; then
    export PATH="$LOCAL_NODE_BIN:$PATH"
fi

CUDA_RUNTIME_LIB="$(python - <<'PY'
import os

try:
    import nvidia.cuda_runtime
    lib_dir = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
    if os.path.exists(os.path.join(lib_dir, "libcudart.so.12")):
        print(lib_dir)
except Exception:
    pass
PY
)"

if [ -n "$CUDA_RUNTIME_LIB" ]; then
    export LD_LIBRARY_PATH="$CUDA_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ ! -e "$CUDA_RUNTIME_LIB/libcudart.so" ] && [ -f "$CUDA_RUNTIME_LIB/libcudart.so.12" ]; then
        ln -s "$CUDA_RUNTIME_LIB/libcudart.so.12" "$CUDA_RUNTIME_LIB/libcudart.so"
    fi
fi

cd "$SCRIPT_DIR/frontend"
npm install
npm run build
echo -e "\033[1;32m\nfrontend build success \033[0m"

cd "$SCRIPT_DIR"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python main.py --port 7860 --host 0.0.0.0 --acceleration none
# tensorrt xformer none
