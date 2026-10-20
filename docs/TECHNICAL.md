# Enginseer: technical details

How each stage was built and measured. The [README](../README.md) has the overview and the main results.

## Documentation

[`sources.toml`](../sources.toml) lists the 12 documentation sources, each with its git repository, the paths taken from it and its license.

```sh
.venv/bin/python scripts/collect_docs.py
.venv/bin/python scripts/chunk_docs.py
```

`collect_docs.py` makes shallow, sparse clones and records the commit of every source in `data/raw/manifest.json`. `chunk_docs.py` cleans each file with a cleaner for its format (`doc_cleaners.py`) and writes `data/docs.jsonl` (2,762 documents) and `data/chunks.jsonl` (heading-aware chunks of the training documents, the input for Q&A generation). 5% of the documents (153) are held out by a hash of source and path. They never reach pretraining or instruction tuning and are used only for evaluation. Chunk sizes are counted with the Qwen3 tokenizer, because chunking came before our own tokenizer.

| Product | Documents | MB of text | Share | SFT facts |
|---|---|---|---|---|
| Kubernetes | 1,172 | 12.17 | 51.4% | 8,659 |
| Docker | 759 | 4.87 | 20.6% | 4,125 |
| GitHub Actions | 208 | 1.82 | 7.7% | 1,502 |
| nginx | 152 | 1.39 | 5.9% | 1,167 |
| OpenTofu | 303 | 1.36 | 5.7% | 1,191 |
| Prometheus | 83 | 1.27 | 5.4% | 1,140 |
| unbound | 41 | 0.44 | 1.9% | 317 |
| MeshCore | 16 | 0.22 | 0.9% | 174 |
| OpenRC | 28 | 0.14 | 0.6% | 98 |

The small products are in the data, but a few dozen pages repeated about 8 times in pretraining are too little for a 140M model to learn: asked what MeshCore is, it calls it a driver for container orchestration.

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

The base model is 24 layers deep but only 640 wide, because at this size depth helps more than width (MobileLLM) for nearly the same throughput (~70M tokens/h). 3.1B tokens are about 22 per parameter, close to the Chinchilla-optimal ratio. In this mix the domain docs come back about 8 times, Cosmopedia once and FineWeb-Edu is not repeated.

The base run took 44 h of GPU time, spread over four sessions with pauses in between, at ~19.5k tokens/s and 9.5 GiB of VRAM. Validation bits per byte:

| Step | Tokens | Docs | FineWeb-Edu | Cosmopedia |
|---|---|---|---|---|
| 1000 | 0.5B | 0.996 | 1.121 | 0.780 |
| 2000 | 1.0B | 0.884 | 1.045 | 0.693 |
| 3000 | 1.6B | 0.834 | 1.007 | 0.651 |
| 4000 | 2.1B | 0.798 | 0.982 | 0.623 |
| 5000 | 2.6B | 0.778 | 0.966 | 0.604 |
| 5913 (final) | 3.1B | 0.754 | 0.961 | 0.594 |

The anneal phase (the last 591 steps) brought the docs from 0.775 to 0.754, while FineWeb-Edu, now a smaller share of the mix, still improved slightly. The model writes fluent, on-topic English but gets facts wrong, as expected at this size. A sample from the final checkpoint, prompted with "Photosynthesis is":

> Photosynthesis is the process by which plants convert sunlight into carbohydrates (food) and oxygen. Photosynthesize to create various compounds that make up the food chain.

## Export and GGUF

```sh
.venv/bin/python scripts/export_hf.py base
.venv/bin/python /opt/llama.cpp/convert_hf_to_gguf.py out/runs/base/hf --outtype f16 --outfile out/runs/base/base-f16.gguf
.venv/bin/python scripts/check_gguf.py base out/runs/base/base-f16.gguf
```

`export_hf.py` writes a `LlamaForCausalLM` directory and checks its logits against our model. `check_gguf.py` loads the GGUF in `llama-server` and compares it with SentencePiece and PyTorch. On the test model, all 1409 tokenizer eval docs get identical token ids, the top-1 next token matches on 32 of 32 held-out prefixes, and the top-10 log-probabilities differ by 0.002 on average (f16 vs fp32). On the 140M base model the export is again bit-identical, the tokenizer check passes on all 1409 docs, the top-1 token matches on 32 of 32 prefixes and the top-10 log-probabilities differ by 0.007 on average (max 0.039).

## Instruction tuning

The chat data is built from three parts, all in ChatML (`<|im_start|>user … <|im_end|>`):

1. **Grounded Q&A from the domain docs.** A local teacher, Qwen3.6-35B-A3B (Apache-2.0) in llama.cpp, reads one documentation chunk at a time and writes the key facts in it as JSON (the schema is enforced by the server). Each fact gets one answer and 3–4 differently worded questions, so the model learns the fact and not one phrasing of it. Each chunk also gets one trick question about an option that does not exist, answered with "I'm not aware of…", and about every tenth chunk gets one question translated to German, French, Spanish, Italian, Portuguese or Dutch, still with the English answer. Chunks from generated API references get at most three facts.
2. **General chat:** English threads from [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) (Apache-2.0), following the top-ranked reply at each turn. Deleted, unreviewed, synthetic and toxic messages are dropped, and so are threads where the assistant calls itself Open Assistant.
3. **Identity:** a few dozen "Who are you?" examples, so the model calls itself Enginseer and admits that it is small.

```sh
scripts/run_teacher.sh --until 07:30   # starts llama-server with TEACHER_GGUF from .env; resumable
.venv/bin/python scripts/chunk_heldout.py
scripts/run_teacher.sh --chunks data/eval/chunks-heldout.jsonl --out data/eval/qa-heldout.jsonl
.venv/bin/python scripts/prep_oasst.py
.venv/bin/python scripts/build_sft.py
```

Generation took ~23 h on the RTX 3060 (the MoE experts partly run on the CPU, ~40 output tokens/s over 4 parallel requests). It covered all 5,127 chunks with at least 80 tokens: 18,373 facts with 73,368 questions, 5,127 trick questions and 517 translated questions. `build_sft.py` drops answers that leak the setup ("the excerpt", "the documentation says"). That removes 23 fact answers but 12% of the trick answers, which tend to explain what the docs do list instead. Validation is a 2% split by source document, so no paraphrase of a val fact is trained on. Those documents were still part of pretraining (where the docs repeat about 8 times), so this split measures recall of text the model has read, not knowledge of new documentation. For that, validation also includes questions generated the same way from the held-out docs that neither pretraining nor SFT saw (`heldout`, see [Evaluation](#evaluation)), plus oasst2's own validation split.

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
.venv/bin/python scripts/export_hf.py sft --chat
.venv/bin/python /opt/llama.cpp/convert_hf_to_gguf.py out/runs/sft/hf --outtype f16 --outfile out/runs/sft/sft-f16.gguf
/opt/llama.cpp/build/bin/llama-quantize out/runs/sft/sft-f16.gguf out/runs/sft/sft-q8_0.gguf Q8_0
/opt/llama.cpp/build/bin/llama-quantize out/runs/sft/sft-f16.gguf out/runs/sft/sft-q4_k_m.gguf Q4_K_M
```

`--chat` adds the ChatML template to the export and makes `<|im_end|>` the end-of-sequence token, so llama.cpp stops there.

On the 13M test model (`configs/sft-test.toml`, one epoch, 202 steps, 5 minutes), validation loss fell from 4.83 to 3.72 on Q&A, from 4.56 to 2.91 on trick questions and from 4.08 to 3.86 on oasst2.

On the 140M base model (`configs/sft.toml`) one epoch is 203 steps and takes 14 minutes. Validation loss per answer token:

| Kind | Base model | 1 epoch | 3 epochs |
|---|---|---|---|
| Q&A | 3.002 | 2.202 | 2.761 |
| Trick | 3.059 | 1.700 | 1.784 |
| Translated | 2.675 | 1.727 | 2.109 |
| oasst2 | 2.326 | 2.112 | 2.173 |
| Held-out Q&A | 3.115 | 2.257 | 2.805 |
| Held-out trick | 3.164 | 1.677 | 1.730 |

A first run for three epochs overfit: Q&A validation loss was lowest after about half an epoch (2.23) and rose to 2.76 by the end, while training loss fell below 1. The model memorised the trained answers instead of the facts, so the config trains for one epoch and repeats the identity examples six times instead. After SFT the model calls itself Enginseer, answers in a few sentences, stops at `<|im_end|>` and declines some invented options ("I am not aware of a `--auto-heal` flag for `docker run`"), but the commands and keys it gives are often made up.

## Evaluation

Three kinds of measurement, all run on the same machine with the same code for every model:

```sh
.venv/bin/python scripts/eval_lm.py base sft openai-community/gpt2 HuggingFaceTB/SmolLM-135M HuggingFaceTB/SmolLM2-135M
.venv/bin/python scripts/eval_judge.py answer sft --gguf out/runs/sft/sft-q8_0.gguf
scripts/run_teacher.sh scripts/eval_judge.py judge sft
.venv/bin/python scripts/eval_judge.py report
/opt/llama.cpp/build/bin/llama-perplexity -m out/runs/sft/sft-q8_0.gguf -f data/eval/heldout.txt -c 1024 -ngl 99
```

**Language modelling** (`eval_lm.py`). Bits per byte on our validation text, held-out docs plus FineWeb-Edu and Cosmopedia documents nobody trained on here. Each model reads the same raw text with its own tokenizer, so the numbers compare across tokenizers. Zero-shot HellaSwag, WinoGrande and LAMBADA are scored by log-likelihood as in lm-evaluation-harness. For GPT-2 the script reproduces the harness's published numbers exactly (HellaSwag acc_norm 0.312, WinoGrande 0.516, LAMBADA 0.326 / perplexity 40.1). Only benchmarks under permissive licenses are used, so ARC is left out.

| Model | Training tokens | bpb docs | bpb FineWeb-Edu | bpb Cosmopedia | HellaSwag | WinoGrande | LAMBADA acc | LAMBADA ppl |
|---|---|---|---|---|---|---|---|---|
| GPT-2 124M | WebText, 40 GB of text | 1.051 | 0.983 | 0.829 | 0.312 | 0.516 | 0.326 | 40.1 |
| SmolLM-135M | 600B | 0.831 | 0.838 | 0.479* | 0.427 | 0.533 | 0.377 | 32.3 |
| SmolLM2-135M | 2T | 0.768 | 0.842 | 0.513* | 0.432 | 0.530 | 0.429 | 19.3 |
| Enginseer test, 13M | 0.1B | 1.306 | 1.439 | 1.128 | 0.253 | 0.514 | 0.008 | 28,412 |
| Enginseer base, 140M | 3.1B | **0.718** | 0.949 | 0.576 | 0.324 | 0.528 | 0.257 | 100.0 |
| Enginseer SFT, 140M | 3.1B + SFT | 0.760 | 0.968 | 0.606 | 0.320 | 0.527 | 0.260 | 101.5 |

\* SmolLM was trained on Cosmopedia itself, so its Cosmopedia numbers are not a fair comparison. The public models may also have seen web copies of the documentation.

With 200–650 times fewer training tokens than SmolLM, the base model has the lowest bits per byte on the held-out docs of all the models here and beats GPT-2 on every column except LAMBADA, but it stays behind SmolLM on general English and on the benchmarks. LAMBADA needs the last word of a long narrative passage, and with only 3.1B tokens, mostly educational web text, that is its weakest point. SFT costs a little language modelling (0.04 bpb on the docs) and nothing measurable on the benchmarks.

**GGUF quantization** (`llama-perplexity -f data/eval/heldout.txt -c 1024`, held-out docs):

| Model | File | Size | Perplexity |
|---|---|---|---|
| base | f16 | 281 MB | 6.380 ± 0.039 |
| SFT | f16 | 281 MB | 7.047 ± 0.044 |
| SFT | Q8_0 | 150 MB | 7.048 ± 0.044 |
| SFT | Q4_K_M | 112 MB | 7.098 ± 0.044 |

Q8_0 is lossless within noise and Q4_K_M costs 0.7%. Because the model is 640 wide, which is not a multiple of 256, 168 of 218 tensors fall back from k-quants to simpler types in Q4_K_M. `check_gguf.py` on the SFT f16 file gives the same token ids on all 1409 docs, top-1 agreement 32 of 32 and a mean top-10 log-probability difference of 0.007.

**Answer quality** (`eval_judge.py`). The chat model answers questions written by the teacher in three groups, which separate memorization from generalization:

- *sft*: questions the model was trained on.
- *pretrain*: questions about documents that were in pretraining but had no questions in SFT.
- *heldout*: questions about the 153 held-out documents that neither stage saw, generated the same way (`scripts/chunk_heldout.py`, then `gen_qa.py --chunks`).

Each group has up to 300 fact questions and 300 trick questions about options that do not exist. The teacher grades each answer against the reference as correct, partial or wrong and gives a one-sentence reason. On trick questions, correct means the model says there is no such option and makes nothing up afterwards. A regex also counts "I'm not aware"-style answers, independent of the judge.

The baseline is SmolLM2-135M-Instruct, a model of the same size trained on 2T tokens, converted to Q8_0 GGUF and asked the same questions. The pretrain group has only 273 fact and 75 trick questions, as that is all the SFT validation documents have:

| Model | Group | Facts correct / partial | Tricks correct / partial | "Not aware" on tricks (regex) |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | sft | 1.0% / 0.7% | 0.0% / 0.3% | 0.3% |
| SmolLM2-135M-Instruct | pretrain | 0.4% / 0.7% | 0.0% / 0.0% | 0.0% |
| SmolLM2-135M-Instruct | heldout | 0.3% / 1.0% | 0.0% / 0.3% | 1.3% |
| Enginseer SFT, Q8_0 | sft | 8.3% / 5.3% | 10.0% / 42.0% | 86.0% |
| Enginseer SFT, Q8_0 | pretrain | 2.6% / 1.8% | 5.3% / 41.3% | 85.3% |
| Enginseer SFT, Q8_0 | heldout | 3.3% / 1.7% | 10.7% / 36.7% | 87.0% |

At this size a general model knows almost none of these specific facts, and it accepts every invented flag as real: in the judge's words, it "hallucinates a non-existent command `pod-suspend` and falsely claims the `--force-suspend` flag exists". Here SmolLM2's "sft" and "pretrain" groups are just question sets, since it never trained on them.

Enginseer answers 8% of the trained fact questions correctly, but only 3% about documents it read in pretraining or never saw, so SFT taught it mostly the trained answers, not to recall what it read. It is still more than SmolLM2 gets, because it knows the products and the shape of a right answer (`kubectl proxy --port=8080`, the `labelSelector` query parameter). On trick questions it learned to decline, on held-out docs as well as trained ones: 87% of its answers start with "I am not aware of…", against 1% for SmolLM2. Most of those are only partial, though, as the model goes on to suggest a related option that does not fit, as in "I am not aware of a `--force-renew` flag in Kubernetes. You might be thinking of the `--certificate-authority` flag…".
