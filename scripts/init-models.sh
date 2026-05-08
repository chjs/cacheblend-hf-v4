#!/usr/bin/env bash
# Pod init — package install + mydata clone + HF model download
# Run once after Pod create (Phase 1+ entry).

set -euo pipefail

echo "===== Step 1: HF cache to network volume ====="
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
export HF_DATASETS_CACHE=/workspace/hf_cache/datasets
export TRANSFORMERS_CACHE=/workspace/hf_cache/hub
mkdir -p "$HF_HOME"

echo "===== Step 2: Install rsync (Pod image lacks it) [L23] ====="
apt-get update -qq && apt-get install -y -qq rsync git

echo "===== Step 3: mydata clone + SHA verification ====="
MYDATA_DIR=/workspace/cacheblend-hf-v4/external/mydata
if [ ! -d "$MYDATA_DIR" ]; then
    git clone --depth 1 https://github.com/chjs/mydata.git "$MYDATA_DIR"
fi

EXPECTED="791e1cf50d984f27b314c8abd49f25e3b27a0a1598a6cfcf53e28d13868a3e21"
ACTUAL=$(sha256sum "$MYDATA_DIR/cacheblend_fig12/prompts.jsonl" | awk '{print $1}')
if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "ERROR: prompts.jsonl SHA mismatch"
    echo "  Expected: $EXPECTED"
    echo "  Actual:   $ACTUAL"
    exit 1
fi
echo "✓ mydata prompts.jsonl SHA verified"

echo "===== Step 4: Pin packages (parity with Mac venv) ====="
# Skip torch (image's 2.4.1+cu124 already installed and correct)
grep -v -E '^torch(\s|=|$)' /workspace/cacheblend-hf-v4/requirements.txt > /tmp/reqs-no-torch.txt
pip install -q -r /tmp/reqs-no-torch.txt 2>&1 | tail -5
pip install -e /workspace/cacheblend-hf-v4

echo "===== Step 5: Verify torch CUDA build ====="
python -c "
import torch
v = torch.__version__
assert v.startswith('2.4.1') and 'cu' in v, f'torch downgraded to {v}!'
print(f'torch {v} OK, cuda available: {torch.cuda.is_available()}')
"

echo "===== Step 6: HF auth ====="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set in environment"
    exit 1
fi
python -c "
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('HF auth OK')
"

echo "===== Step 7: Download Mistral-7B (Phase 1+) ====="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-7B-Instruct-v0.2', cache_dir='$HF_HUB_CACHE')
print('Mistral-7B downloaded')
"

if [ "${DOWNLOAD_LLAMA8B:-false}" = "true" ]; then
    echo "===== Step 7b: Download Llama-3.1-8B ====="
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', cache_dir='$HF_HUB_CACHE')
print('Llama-3.1-8B downloaded')
"
fi

if [ "${DOWNLOAD_LLAMA70B:-false}" = "true" ]; then
    echo "===== Step 7c: Download Llama-3.1-70B + bitsandbytes ====="
    pip install -q "bitsandbytes>=0.43"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Llama-3.1-70B-Instruct', cache_dir='$HF_HUB_CACHE')
print('Llama-3.1-70B downloaded')
"
fi

echo ""
echo "===== Pod init complete ====="
echo "HF_HOME: $HF_HOME"
echo "Disk usage:"
df -h /workspace | tail -1
