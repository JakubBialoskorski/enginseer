# Enginseer

A small English language model, written and trained from scratch on a single RTX 3060 to learn and show the whole pipeline: tokenizer, model, pretraining, instruction tuning, evaluation and GGUF export. It is not an expert in anything. It writes fluent English, but at 140M parameters it gets most facts wrong, so treat it as a demonstration, not as an assistant.

- **Model:** Llama-compatible decoder (RMSNorm, RoPE, SwiGLU), 140M parameters, 24 layers × 640 wide, 1024-token context, written in plain PyTorch.
- **Tokenizer:** own SentencePiece BPE, 32k vocabulary.
- **Pretraining:** 3.1B tokens in 44 h, almost all general English (FineWeb-Edu and Cosmopedia). About 1.5% is documentation for DevOps tools, mostly Kubernetes and Docker.
- **Instruction tuning:** 82k chat examples. Most are Q&A that a local teacher model generated from that documentation, including trick questions about options that do not exist, answered with "I'm not aware of…". The rest is general chat from oasst2.
- **Output:** Hugging Face `LlamaForCausalLM` weights, converted to GGUF (f16, Q8_0, Q4_K_M) with the stock llama.cpp converter.
- **Data:** permissively licensed sources only (CC-BY, Apache, MIT, BSD, MPL, ODC-BY), each recorded with its license in [`sources.toml`](sources.toml).

How each stage works, with all measurements: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Results

Bits per byte on held-out text (lower is better) and zero-shot benchmarks, all measured with the same code:

| Model | Training tokens | bpb held-out docs | bpb FineWeb-Edu | HellaSwag | WinoGrande | LAMBADA |
|---|---|---|---|---|---|---|
| GPT-2 124M | WebText, 40 GB of text | 1.051 | 0.983 | 0.312 | 0.516 | 0.326 |
| SmolLM2-135M | 2T | 0.768 | 0.842 | 0.432 | 0.530 | 0.429 |
| **Enginseer base, 140M** | 3.1B | **0.718** | 0.949 | 0.324 | 0.528 | 0.257 |

Answers to documentation questions, graded by the teacher model, on documents that neither pretraining nor instruction tuning saw:

| Model | Facts correct / partial | Invented options declined (regex) |
|---|---|---|
| SmolLM2-135M-Instruct | 0.3% / 1.0% | 1.3% |
| Enginseer SFT, Q8_0 | 3.3% / 1.7% | 87.0% |

- The base model predicts the held-out documentation better than GPT-2 and SmolLM2, since it trained on the rest of the same documentation. It also beats GPT-2 on general text and on two of the three benchmarks, but stays behind SmolLM2, which saw 650 times more tokens.
- After instruction tuning it declines most questions about options that do not exist, where SmolLM2-135M-Instruct accepts almost every one. It still gets 95% of the facts wrong.
- Q8_0 is lossless (perplexity 7.048 vs 7.047 in f16) and Q4_K_M costs 0.7%.

## Running the model

The chat model runs in any llama.cpp-based tool. `llama-server` applies the ChatML template stored in the GGUF:

```sh
llama-server -m sft-q8_0.gguf -c 1024 --temp 0.7 --repeat-penalty 1.1 --port 8080
curl -s localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "What does kubectl drain do?"}]}'
```

Without the repeat penalty, longer answers tend to loop. Expect fluent, on-topic English with made-up details: most commands, flags and config keys it gives do not exist.

## Reproducing

Requirements: Linux, an NVIDIA GPU with 12 GB (developed on an RTX 3060), Python 3.12 and a llama.cpp checkout for GGUF conversion.

```sh
./setup.sh
.venv/bin/python scripts/smoke_test.py configs/base.toml
```

`setup.sh` creates `.venv/`, installs the pinned dependencies and creates the gitignored working directories `data/`, `out/` and `cache/`. They grow to tens of GB, so to keep them on another disk, create a `.env` file first:

```sh
ENGINSEER_STORAGE=/path/to/big/disk/enginseer
# optional overrides
PYTHON=python3.12
TORCH_INDEX=https://download.pytorch.org/whl/cu130   # pick the wheel index matching your CUDA driver
LLAMA_CPP=/opt/llama.cpp
TEACHER_GGUF=/path/to/Qwen3.6-35B-A3B.gguf               # teacher for Q&A generation and grading
```

The smoke test runs a few compiled bf16 training steps of the base model on random tokens (19.6k tokens/s and 7.7 GiB on the RTX 3060). The full pipeline, in order:

| Step | Commands |
|---|---|
| Documentation | `collect_docs.py`, `chunk_docs.py` |
| General corpus | `fetch_corpus.py` |
| Tokenizer | `train_tokenizer.py sample`, `train`, `eval`, `hf sp32k` |
| Pretraining | `tokenize_corpus.py`, `train.py configs/base.toml`, `export_hf.py base` |
| Instruction data | `run_teacher.sh` (runs `gen_qa.py`), `chunk_heldout.py` and `run_teacher.sh` on the held-out chunks, `prep_oasst.py`, `build_sft.py` |
| Instruction tuning | `sft.py configs/sft.toml`, `export_hf.py sft --chat`, then `convert_hf_to_gguf.py` and `llama-quantize` |
| Evaluation | `eval_lm.py`, `eval_judge.py`, `llama-perplexity` |

All scripts are in `scripts/`. The exact commands and their arguments are in [docs/TECHNICAL.md](docs/TECHNICAL.md).
