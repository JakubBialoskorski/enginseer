#!/usr/bin/env bash
# Sets up the local environment: venv, dependencies and storage directories.
#
# By default everything lives inside the repo: data/ (docs and the pretraining
# corpus), out/ (weights, GGUFs, checkpoints) and cache/ (Hugging Face downloads).
# They grow to tens of GB. To keep them on another disk, create a .env file next
# to this script, e.g.:
#   ENGINSEER_STORAGE=/mnt/hddgry/enginseer
# data/, out/ and cache/ then become symlinks into that directory.
#
# Other optional .env settings:
#   PYTHON=python3.12                                       interpreter for the venv
#   TORCH_INDEX=https://download.pytorch.org/whl/cu130      torch wheel index (match your CUDA driver)
#   LLAMA_CPP=/opt/llama.cpp                                gguf-py is installed from here if present
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && . ./.env
PYTHON=${PYTHON:-python3.12}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
LLAMA_CPP=${LLAMA_CPP:-/opt/llama.cpp}

for dir in data out cache; do
    if [ -n "${ENGINSEER_STORAGE:-}" ]; then
        if [ -e "$dir" ] && [ ! -L "$dir" ]; then
            echo "$dir/ is a real directory; move its contents to $ENGINSEER_STORAGE/$dir and remove it first" >&2
            exit 1
        fi
        mkdir -p "$ENGINSEER_STORAGE/$dir"
        ln -sfn "$ENGINSEER_STORAGE/$dir" "$dir"
    else
        mkdir -p "$dir"
    fi
done

[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url "$TORCH_INDEX"
.venv/bin/pip install -q -r requirements.txt
if [ -d "$LLAMA_CPP/gguf-py" ]; then
    # same gguf version as the convert_hf_to_gguf.py we will run
    .venv/bin/pip install -q "$LLAMA_CPP/gguf-py"
else
    .venv/bin/pip install -q gguf
fi

# HF CLI tools (hf download etc.) use the repo cache too; scripts set it themselves in scripts/paths.py
if ! grep -q HF_HUB_CACHE .venv/bin/activate; then
    cat >> .venv/bin/activate <<EOF

export HF_HUB_CACHE="$PWD/cache/hf/hub"
export HF_DATASETS_CACHE="$PWD/cache/hf/datasets"
EOF
fi

.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
