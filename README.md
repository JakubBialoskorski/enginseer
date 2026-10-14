# Enginseer

A small English-only language model for DevOps (Kubernetes, Docker, OpenTofu, GitHub Actions, Prometheus, nginx), OpenRC, unbound and MeshCore, written and trained from scratch and published as GGUF.

- **Architecture:** Llama-compatible (RMSNorm, RoPE, SwiGLU), ~150M parameters, written in plain PyTorch. The weights are exported to the Hugging Face `LlamaForCausalLM` format and converted to GGUF with the stock llama.cpp converter.
- **Pipeline:** own SentencePiece BPE tokenizer (32k vocab), pretraining with a hand-written training loop, then instruction tuning on Q&A pairs generated only from real documentation, including examples of answering "I don't know" instead of inventing flags and commands.
- **Data:** permissively licensed sources only (CC-BY, Apache, MIT, BSD, MPL, ODC-BY), with every source and its license recorded.

Status: work in progress. The environment, the data (documentation plus the general corpus) and the tokenizer are ready. The model and pretraining are next.

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

## Tokenizer

SentencePiece BPE with byte fallback and split digits, trained on a ~340 MB sample of the pretraining mix (FineWeb-Edu, Cosmopedia and the domain docs counted twice, so terms like `kubectl` and `unbound` get their own tokens). Whitespace is kept as is, so YAML and code indentation survives. There is no dummy prefix. BPE was chosen over unigram because llama.cpp's SentencePiece tokenizer merges pieces by score, which reproduces BPE exactly.

```sh
.venv/bin/python scripts/train_tokenizer.py sample
.venv/bin/python scripts/train_tokenizer.py train 16384 32768
.venv/bin/python scripts/train_tokenizer.py eval --gpt2
.venv/bin/python scripts/train_tokenizer.py hf sp32k
```

Compression on held-out text, in bytes per token (higher is better):

| Tokenizer | FineWeb-Edu | Cosmopedia | Domain docs |
|---|---|---|---|
| ours, 16k | 4.12 | 4.71 | 3.53 |
| **ours, 32k** | **4.40** | **5.04** | **3.73** |
| GPT-2, 50k | 4.62 | 5.12 | 3.45 |

The 32k vocabulary produces 6–7% fewer tokens than 16k on every part of the mix, and it beats GPT-2 on the documentation despite the smaller vocabulary. `hf` writes `tokenizer.model` and `tokenizer_config.json` for the Hugging Face export, then checks that transformers encodes all held-out docs exactly like SentencePiece. A vocab-only `convert_hf_to_gguf.py --vocab-only` run on these files gives a `llama` (SPM) vocab with `add_space_prefix = false` and the chat tokens `<|im_start|>` / `<|im_end|>` marked as control tokens.
