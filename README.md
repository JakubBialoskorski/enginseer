# Enginseer

A small English-only language model for DevOps (Kubernetes, Docker, OpenTofu, GitHub Actions, Prometheus, nginx), OpenRC, unbound and MeshCore, written and trained from scratch and published as GGUF.

- **Architecture:** Llama-compatible (RMSNorm, RoPE, SwiGLU), ~125M parameters, written in plain PyTorch. The weights are exported to the Hugging Face `LlamaForCausalLM` format and converted to GGUF with the stock llama.cpp converter.
- **Pipeline:** own BPE tokenizer, pretraining with a hand-written training loop, then instruction tuning on Q&A pairs generated only from real documentation, including examples of answering "I don't know" instead of inventing flags and commands.
- **Data:** permissively licensed sources only (CC-BY, Apache, MIT, BSD, MPL, ODC-BY), with every source and its license recorded.

Status: work in progress. The environment is ready, and data collection is next.

## Setup

Requirements: Linux, an NVIDIA GPU (developed on an RTX 3060 12 GB), Python 3.12 and a llama.cpp checkout for GGUF conversion.

```sh
./setup.sh
```

This creates `.venv/`, installs pinned dependencies and creates the working directories. All of them are gitignored:

| Directory | Contents |
|---|---|
| `data/` | source docs, pretraining corpus, generated datasets |
| `out/` | checkpoints, merged weights, GGUF files |
| `cache/` | Hugging Face downloads |

By default they live inside the repo. They grow to tens of GB, so to keep them on another disk, create a `.env` file before running `setup.sh`:

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
