# Enginseer

A small English-only language model for DevOps (Kubernetes, Docker, OpenTofu, GitHub Actions, Prometheus, nginx), OpenRC, unbound and MeshCore, written and trained from scratch and published as GGUF.

- **Architecture:** Llama-compatible (RMSNorm, RoPE, SwiGLU), ~150M parameters, written in plain PyTorch. The weights are exported to the Hugging Face `LlamaForCausalLM` format and converted to GGUF with the stock llama.cpp converter.
- **Pipeline:** own SentencePiece BPE tokenizer (32k vocab), pretraining with a hand-written training loop, then instruction tuning on Q&A pairs generated only from real documentation, including examples of answering "I don't know" instead of inventing flags and commands.
- **Data:** permissively licensed sources only (CC-BY, Apache, MIT, BSD, MPL, ODC-BY), with every source and its license recorded.

Status: work in progress. The environment, the data (documentation plus the general corpus), the tokenizer and the pretraining code are ready, verified end to end on a small test model. The full pretraining run is next.

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

## Pretraining

The mix is tokenized once into flat `uint16` files under `data/tokens/` (each document as `<s> text </s>`), with a small validation split per source that the tokenizer never saw either:

```sh
.venv/bin/python scripts/fetch_corpus.py
.venv/bin/python scripts/tokenize_corpus.py
```

| Source | Train tokens | Val tokens |
|---|---|---|
| FineWeb-Edu (4 files of `sample-10BT`) | 3,149M | 3.0M |
| Cosmopedia | 516M | 0.5M |
| Domain docs | 5.7M | 0.4M (held-out docs) |

`scripts/model.py` is the model: RMSNorm, rotary embeddings, SwiGLU, optional grouped-query attention and an output head tied to the embedding. Module names and the RoPE layout match Hugging Face's `LlamaForCausalLM`, so the two give bit-identical logits on the same weights. `scripts/train.py` is the training loop: AdamW, linear warmup and cosine decay, bf16 autocast, gradient accumulation and `torch.compile`. Each micro-batch row is a block from one source, drawn by the mix weights. For the last 10% of the steps, while the learning rate decays, the mix shifts towards textbooks and documentation. Validation loss is reported in bits per byte, which compares across tokenizers.

Runs are resumable. Ctrl+C, SIGTERM or `--until HH:MM` saves a checkpoint and exits, and the same command continues with exactly the batches an uninterrupted run would have seen:

```sh
.venv/bin/python scripts/train.py configs/base.toml --until 07:30
```

| Config | Parameters | Shape | Tokens | Time on an RTX 3060 |
|---|---|---|---|---|
| `configs/test.toml` | 13M | 256 wide, 6 layers | 100M | 10 min |
| `configs/base.toml` | 140M | 640 wide, 24 layers | 3.1B | ~47 h |

The base model is 24 layers deep but only 640 wide, because at this size depth helps more than width (MobileLLM) for nearly the same throughput (~67M tokens/h). 3.1B tokens are about 22 per parameter, close to the Chinchilla-optimal ratio. In this mix the domain docs come back about 8 times, Cosmopedia once and FineWeb-Edu is not repeated.

### Export and GGUF

```sh
.venv/bin/python scripts/export_hf.py test
.venv/bin/python /opt/llama.cpp/convert_hf_to_gguf.py out/runs/test/hf --outtype f16 --outfile out/runs/test/test-f16.gguf
.venv/bin/python scripts/check_gguf.py test out/runs/test/test-f16.gguf
```

`export_hf.py` writes a `LlamaForCausalLM` directory and checks its logits against our model. `check_gguf.py` loads the GGUF in `llama-server` and compares it with SentencePiece and PyTorch. On the test model, all 1409 tokenizer eval docs get identical token ids, the top-1 next token matches on 32 of 32 held-out prefixes, and the top-10 log-probabilities differ by 0.002 on average (f16 vs fp32).
