# Enginseer

A small English-only assistant for DevOps, Linux (Gentoo), Pi-hole/unbound and MeshCore/Meshtastic, published as GGUF. The project is built in two stages:

1. **enginseer-4b**: a QLoRA fine-tune of [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). It is trained on Q&A pairs generated only from real, openly licensed documentation. It includes examples of answering "I don't know" instead of inventing flags and commands.
2. **enginseer-tiny**: a ~20–30M parameter Llama-compatible GPT (RMSNorm, RoPE, SwiGLU) with its own BPE tokenizer, written and pretrained from scratch on TinyStories. Its weights are exported to the Hugging Face `LlamaForCausalLM` format and converted to GGUF with the stock llama.cpp converter.

Status: work in progress. The environment is ready, and documentation collection is next.

## Setup

Requirements: Linux, an NVIDIA GPU (developed on an RTX 3060 12 GB), Python 3.12 and a llama.cpp checkout for GGUF conversion.

```sh
./setup.sh
```

This creates `.venv/`, installs pinned dependencies and creates the working directories. All of them are gitignored:

| Directory | Contents |
|---|---|
| `data/` | source docs and generated datasets |
| `out/` | checkpoints, merged weights, GGUF files |
| `cache/` | Hugging Face downloads |

By default they live inside the repo. To keep `out/` and `cache/` on another disk, create a `.env` file before running `setup.sh`:

```sh
ENGINSEER_STORAGE=/path/to/big/disk/enginseer
# optional overrides
PYTHON=python3.12
TORCH_INDEX=https://download.pytorch.org/whl/cu130   # pick the wheel index matching your CUDA driver
LLAMA_CPP=/opt/llama.cpp
```

Check that CUDA and 4-bit loading work:

```sh
.venv/bin/python scripts/smoke_test.py
```
