# Enginseer

A small English chat model, written and trained from scratch on a single RTX 3060 as a proof of concept: tokenizer, model code, training loop, pretraining, instruction data, evaluation and GGUF export are all built in this repository. No existing model was fine-tuned or copied. It is a general chat model, not an expert in anything: at 140M parameters it writes fluent English, but it may get facts wrong, so check anything important.

- **English only.** It was trained on English text only. Messages in other languages get a short English reply asking to write in English.
- **Model:** Llama-compatible decoder (RMSNorm, RoPE, SwiGLU), 140M parameters, 24 layers × 640 wide, 1024-token context, written in plain PyTorch.
- **Tokenizer:** own SentencePiece BPE, 32k vocabulary.
- **Pretraining:** 3.1B tokens in 44 h, almost all general English (FineWeb-Edu and Cosmopedia), plus about 1.5% documentation for DevOps tools, mostly Kubernetes and Docker.
- **Instruction tuning:** 33k chat examples. 58% are short conversations written by the teacher, Qwen3.6-35B-A3B (Apache-2.0) running locally in llama.cpp: everyday questions answered from passages the model read in pretraining, small talk, short creative tasks, questions about invented things it should not make up, and messages in other languages. 34% are Q&A the same teacher wrote about the DevOps documentation, and 9% are general chat from oasst2. The teacher also grades the answers in the evaluation.
- **Output:** Hugging Face `LlamaForCausalLM` weights, converted to GGUF (f16, Q8_0, Q4_K_M) with the stock llama.cpp converter.
- **Data:** permissively licensed sources only (CC-BY, Apache, MIT, BSD, MPL, ODC-BY), each recorded with its license in [`sources.toml`](sources.toml).

How each stage works, with all measurements: [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Results

Bits per byte on held-out text (lower is better) and zero-shot benchmarks of the pretrained model, all measured with the same code:

| Model | Training tokens | bpb held-out docs | bpb FineWeb-Edu | HellaSwag | WinoGrande | LAMBADA |
|---|---|---|---|---|---|---|
| GPT-2 124M | WebText, 40 GB of text | 1.051 | 0.983 | 0.312 | 0.516 | 0.326 |
| SmolLM2-135M | 2T | 0.768 | 0.842 | 0.432 | 0.530 | 0.429 |
| **Enginseer base, 140M** | 3.1B | **0.718** | 0.949 | 0.324 | 0.528 | 0.257 |

Chat answers graded by the teacher, on conversations held out from instruction tuning. The first version of the chat model was tuned mostly on documentation Q&A; the current one on general chat:

| Model | Everyday questions correct / partial | Invented things declined | Other languages → asks for English |
|---|---|---|---|
| SmolLM2-135M-Instruct | 3.0% / 7.0% | 2.2% | 0.8% |
| Enginseer, first version | 0.3% / 0.3% | 0% | 1.7% |
| **Enginseer, Q8_0** | **10.0% / 8.0%** | **31.1%** | **99.2%** |

- The base model predicts the held-out documentation better than GPT-2 and SmolLM2, since it trained on the rest of the same documentation. It also beats GPT-2 on general text and on two of the three benchmarks, but stays behind SmolLM2, which saw 650 times more tokens.
- The chat model answers in a few fluent sentences, introduces itself as a small English-only model and replies to other languages in English. Most of its answers still contain a wrong detail, so the judge marks them wrong.
- On documentation it has not seen it gets almost no exact facts right (0.7%, SmolLM2-135M-Instruct 0.3%), but declines 89% of questions about options that do not exist (SmolLM2 1%).
- Q8_0 is lossless (perplexity 6.694 vs 6.691 in f16) and Q4_K_M costs 0.7%.

## Running the model

The chat model runs in any llama.cpp-based app, such as [Jan](https://jan.ai) or LM Studio. Load `sft-q8_0.gguf` and set the context to 1024 tokens. The GGUF stores the ChatML template and the sampling defaults (temperature 0.7, top-p 0.9, repeat penalty 1.1; without the penalty longer answers tend to loop). The template drops system prompts, because the model never saw one and is too small to follow it.

With `llama-server`:

```sh
llama-server -m sft-q8_0.gguf -c 1024 --port 8080
curl -s localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Give me three tips for sleeping better."}]}'
```

It has no internet access and no memory between chats. Expect fluent, short English answers with made-up details now and then.

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
TEACHER_GGUF=/path/to/Qwen3.6-35B-A3B.gguf               # teacher for data generation and grading
```

The smoke test runs a few compiled bf16 training steps of the base model on random tokens (19.6k tokens/s and 7.7 GiB on the RTX 3060). The full pipeline, in order:

| Step | Commands |
|---|---|
| Documentation | `collect_docs.py`, `chunk_docs.py` |
| General corpus | `fetch_corpus.py` |
| Tokenizer | `train_tokenizer.py sample`, `train`, `eval`, `hf sp32k` |
| Pretraining | `tokenize_corpus.py`, `train.py configs/base.toml`, `export_hf.py base` |
| Instruction data | `run_teacher.sh scripts/gen_chat.py`, `run_teacher.sh` (runs `gen_qa.py`), `chunk_heldout.py` and `run_teacher.sh` on the held-out chunks, `prep_oasst.py`, `build_sft.py` |
| Instruction tuning | `sft.py configs/sft.toml`, `export_hf.py sft --chat`, then `convert_hf_to_gguf.py` and `llama-quantize` |
| Evaluation | `eval_lm.py`, `eval_judge.py`, `llama-perplexity` |

All scripts are in `scripts/`. The exact commands and their arguments are in [docs/TECHNICAL.md](docs/TECHNICAL.md).
