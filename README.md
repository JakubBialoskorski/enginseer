# Enginseer

A small English-only language model for DevOps (Kubernetes, Docker, OpenTofu, GitHub Actions, Prometheus, nginx), OpenRC, unbound and MeshCore, written and trained from scratch and published as GGUF.

- **Architecture:** Llama-compatible (RMSNorm, RoPE, SwiGLU), ~150M parameters, written in plain PyTorch. The weights are exported to the Hugging Face `LlamaForCausalLM` format and converted to GGUF with the stock llama.cpp converter.
- **Pipeline:** own SentencePiece BPE tokenizer (32k vocab), pretraining with a hand-written training loop, then instruction tuning on Q&A pairs generated only from real documentation, including examples of answering "I don't know" instead of inventing flags and commands.
- **Data:** permissively licensed sources only (CC-BY, Apache, MIT, BSD, MPL, ODC-BY), with every source and its license recorded.

Status: work in progress. The environment, the data (documentation plus the general corpus), the tokenizer, the pretraining code and the instruction-tuning data and code are ready, verified end to end on a small test model. The full pretraining run is next.

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
| `configs/base.toml` | 140M | 640 wide, 24 layers | 3.1B | 44 h |

The base model is 24 layers deep but only 640 wide, because at this size depth helps more than width (MobileLLM) for nearly the same throughput (~67M tokens/h). 3.1B tokens are about 22 per parameter, close to the Chinchilla-optimal ratio. In this mix the domain docs come back about 8 times, Cosmopedia once and FineWeb-Edu is not repeated.

The base run took 44 h of GPU time, spread over four sessions with pauses in between, at ~19.5k tokens/s and 9.5 GiB of VRAM. Validation bits per byte:

| Step | Tokens | Docs | FineWeb-Edu | Cosmopedia |
|---|---|---|---|---|
| 1000 | 0.5B | 0.996 | 1.121 | 0.780 |
| 2000 | 1.0B | 0.884 | 1.045 | 0.693 |
| 3000 | 1.6B | 0.834 | 1.007 | 0.651 |
| 4000 | 2.1B | 0.798 | 0.982 | 0.623 |
| 5000 | 2.6B | 0.778 | 0.966 | 0.604 |
| 5913 (final) | 3.1B | 0.754 | 0.961 | 0.594 |

The anneal phase (the last 591 steps) brought the docs from 0.776 to 0.754, while FineWeb-Edu, now a smaller share of the mix, still improved slightly. The model writes fluent, on-topic English but gets facts wrong, as expected at this size. A sample from the final checkpoint, prompted with "Photosynthesis is":

> Photosynthesis is the process by which plants convert sunlight into carbohydrates (food) and oxygen. Photosynthesize to create various compounds that make up the food chain.

### Export and GGUF

```sh
.venv/bin/python scripts/export_hf.py test
.venv/bin/python /opt/llama.cpp/convert_hf_to_gguf.py out/runs/test/hf --outtype f16 --outfile out/runs/test/test-f16.gguf
.venv/bin/python scripts/check_gguf.py test out/runs/test/test-f16.gguf
```

`export_hf.py` writes a `LlamaForCausalLM` directory and checks its logits against our model. `check_gguf.py` loads the GGUF in `llama-server` and compares it with SentencePiece and PyTorch. On the test model, all 1409 tokenizer eval docs get identical token ids, the top-1 next token matches on 32 of 32 held-out prefixes, and the top-10 log-probabilities differ by 0.002 on average (f16 vs fp32). On the 140M base model the export is again bit-identical, the tokenizer check passes on all 1409 docs, the top-1 token matches on 32 of 32 prefixes and the top-10 log-probabilities differ by 0.007 on average (max 0.039).

## Instruction tuning

The chat data is built from three parts, all in ChatML (`<|im_start|>user … <|im_end|>`):

1. **Grounded Q&A from the domain docs.** A local teacher, Qwen3.6-35B-A3B (Apache-2.0) in llama.cpp, reads one documentation chunk at a time and writes the key facts in it as JSON (the schema is enforced by the server). Each fact gets one answer and 3–4 differently worded questions, so the model learns the fact and not one phrasing of it. Each chunk also gets one trick question about an option that does not exist, answered with "I'm not aware of…", and about every tenth chunk gets one question translated to German, French, Spanish, Italian, Portuguese or Dutch, still with the English answer. Chunks from generated API references get at most three facts.
2. **General chat:** English threads from [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) (Apache-2.0), following the top-ranked reply at each turn. Deleted, unreviewed, synthetic and toxic messages are dropped, and so are threads where the assistant calls itself Open Assistant.
3. **Identity:** a few dozen "Who are you?" examples, so the model calls itself Enginseer and admits that it is small.

```sh
scripts/run_teacher.sh --until 07:30   # starts llama-server with TEACHER_GGUF from .env; resumable
.venv/bin/python scripts/prep_oasst.py
.venv/bin/python scripts/build_sft.py
```

Generation took ~23 h on the RTX 3060 (the MoE experts partly run on the CPU, ~40 output tokens/s over 4 parallel requests). It covered all 5,127 chunks with at least 80 tokens: 18,373 facts with 73,368 questions, 5,127 trick questions and 517 translated questions. `build_sft.py` drops answers that leak the setup ("the excerpt", "the documentation says"). That removes 23 fact answers but 12% of the trick answers, which tend to explain what the docs do list instead. Validation is a 2% split by source document, so no paraphrase of a val fact is trained on, plus oasst2's own validation split.

| Kind | Train examples | Train answer tokens | Val examples |
|---|---|---|---|
| Q&A | 72,188 | 4.50M | 1,088 |
| Trick | 4,465 | 0.25M | 60 |
| Translated | 513 | 0.03M | 4 |
| oasst2 | 4,963 | 1.77M | 265 |
| Identity | 24 | <0.01M | – |

Only the answers and their closing `<|im_end|>` are trained on, and generation stops at `<|im_end|>`. `scripts/sft.py` starts from the pretrained checkpoint and groups examples into length buckets of 128/256/512/1024 tokens with a fixed token budget per micro-batch, so short Q&A pairs are not padded to the full context. The loss is averaged over all answer tokens of a step, and validation loss is reported per kind.

```sh
.venv/bin/python scripts/sft.py configs/sft.toml
```

On the 13M test model (`configs/sft-test.toml`, one epoch, 202 steps, 5 minutes), validation loss fell from 4.83 to 3.72 on Q&A, from 4.56 to 2.91 on trick questions and from 4.08 to 3.86 on oasst2.

On the 140M base model (`configs/sft.toml`) one epoch is 203 steps and takes 14 minutes. Validation loss per answer token:

| Kind | Base model | 1 epoch | 3 epochs |
|---|---|---|---|
| Q&A | 3.002 | 2.202 | 2.761 |
| Trick | 3.059 | 1.700 | 1.784 |
| Translated | 2.675 | 1.727 | 2.109 |
| oasst2 | 2.326 | 2.112 | 2.173 |

A first run for three epochs overfit: Q&A validation loss was lowest after about half an epoch (2.23) and rose to 2.76 by the end, while training loss fell below 1. The model memorised the trained answers instead of the facts, so the config trains for one epoch and repeats the identity examples six times instead. After SFT the model calls itself Enginseer, answers in a few sentences, stops at `<|im_end|>` and declines some invented options ("I am not aware of a `--auto-heal` flag for `docker run`"), but the commands and keys it gives are often made up.
