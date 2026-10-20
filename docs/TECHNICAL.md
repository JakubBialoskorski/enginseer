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

The model is tuned as a small general chat model. The chat data is built from four parts, all in ChatML (`<|im_start|>user … <|im_end|>`). The teacher for parts 1 and 2 is Qwen3.6-35B-A3B (Apache-2.0), run locally in llama.cpp, with the JSON schema of each reply enforced by the server.

1. **General chat** (`gen_chat.py`). For 4,969 passages of 800–3,000 characters from the FineWeb-Edu and Cosmopedia training data (never from the validation set), the teacher writes three short conversations a user might have about the topic, answered only with facts from the passage or very common knowledge, so the model is taught facts it has read. Then 770 requests for five conversations each of one kind: greetings and small talk, requests an offline model cannot fulfil (the date, the weather, a website), short creative writing, editing a pasted text, everyday advice, simple explanations, questions about invented people, books or products that it should decline, harmful or medical requests, questions about itself, and simple arithmetic. The prompt tells the teacher what the model is: small, English-only, offline, and it may get facts wrong. Last, 130 requests for ten short messages each in another language (20 in Polish, 110 spread over 17 other languages), answered with one of five fixed English replies asking to write in English. All 5,869 requests took 14 h (~41 output tokens/s over 4 parallel requests, as the MoE experts partly run on the CPU).
2. **Q&A from the domain docs** (`gen_qa.py`). The teacher reads one documentation chunk at a time and writes the key facts in it, each with one answer and 3–4 differently worded questions, plus one trick question about an option that does not exist, answered with "I'm not aware of…". Generation covered all 5,127 chunks with at least 80 tokens in ~23 h: 18,373 facts with 73,368 questions. To keep the documentation from dominating a general model, `build_sft.py` keeps only half of the facts (by hash) with their first question, and half of the trick questions.
3. **oasst2:** English threads from [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) (Apache-2.0), following the top-ranked reply at each turn. Deleted, unreviewed, synthetic and toxic messages are dropped, and so are threads where the assistant calls itself Open Assistant. Each thread is cut before the first answer longer than 1,200 characters, since long answers teach a small model to ramble.
4. **Identity:** "Who are you?"-style questions answered with "I'm Enginseer, a small English-only chat model … I may get facts wrong", and questions about the languages it speaks, repeated six times per epoch.

```sh
scripts/run_teacher.sh scripts/gen_chat.py   # starts llama-server with TEACHER_GGUF from .env; resumable
scripts/run_teacher.sh --until 07:30         # runs gen_qa.py; resumable
.venv/bin/python scripts/chunk_heldout.py
scripts/run_teacher.sh --chunks data/eval/chunks-heldout.jsonl --out data/eval/qa-heldout.jsonl
.venv/bin/python scripts/prep_oasst.py
.venv/bin/python scripts/build_sft.py
```

`build_sft.py` drops answers that leak the setup ("the excerpt", "the documentation says", "the passage"). Chat validation is a 3% split of the grounded requests and 10% of the others, by request, so no conversation from the same passage is trained on. Doc Q&A validation is a 2% split by source document. Those documents were still part of pretraining (where the docs repeat about 8 times), so validation also includes questions generated the same way from the held-out docs that neither pretraining nor SFT saw (`heldout`, see [Evaluation](#evaluation)), plus oasst2's own validation split.

| Kind | Train examples | Train answer tokens | Val examples |
|---|---|---|---|
| General chat | 14,493 | 0.80M | 398 |
| Skills | 3,074 | 0.15M | 275 |
| Invented things | 455 | 0.03M | 45 |
| Other languages | 1,180 | 0.03M | 120 |
| Doc Q&A | 8,986 | 0.56M | 133 |
| Doc trick | 2,199 | 0.12M | 35 |
| oasst2 | 2,861 | 0.55M | 148 |
| Identity | 36 | <0.01M | – |

Only the answers and their closing `<|im_end|>` are trained on, and generation stops at `<|im_end|>`. `scripts/sft.py` starts from the pretrained checkpoint and groups examples into length buckets of 128/256/512/1024 tokens with a fixed token budget per micro-batch, so short examples are not padded to the full context. The loss is averaged over all answer tokens of a step, and validation loss is reported per kind.

```sh
.venv/bin/python scripts/sft.py configs/sft.toml
.venv/bin/python scripts/export_hf.py sft --chat
.venv/bin/python /opt/llama.cpp/convert_hf_to_gguf.py out/runs/sft/hf --outtype f16 --outfile out/runs/sft/sft-f16.gguf
/opt/llama.cpp/build/bin/llama-quantize out/runs/sft/sft-f16.gguf out/runs/sft/sft-q8_0.gguf Q8_0
/opt/llama.cpp/build/bin/llama-quantize out/runs/sft/sft-f16.gguf out/runs/sft/sft-q4_k_m.gguf Q4_K_M
```

`--chat` adds the ChatML template to the export and makes `<|im_end|>` the end-of-sequence token, so llama.cpp stops there. The template drops system messages: the model never saw one, and chat apps such as Jan send a long default system prompt that would otherwise end up in front of the first question. It also writes sampling defaults (temperature 0.7, top-p 0.9, repeat penalty 1.1) to `generation_config.json`, which the converter stores in the GGUF and llama.cpp applies unless told otherwise.

One epoch on the 140M base model (`configs/sft.toml`) is 79 steps and takes about 6 minutes. Validation loss per answer token:

| Kind | Base model | After SFT |
|---|---|---|
| General chat | 3.449 | 2.534 |
| Skills | 3.592 | 2.312 |
| Invented things | 3.541 | 2.043 |
| Other languages | 3.731 | 0.096 |
| oasst2 | 2.562 | 2.257 |
| Doc Q&A | 2.941 | 2.225 |
| Held-out doc Q&A | 3.115 | 2.370 |
| Held-out doc trick | 3.164 | 1.747 |

The first version of the chat model was tuned on 82k examples, 88% of them doc Q&A with every paraphrased question (203 steps). In a three-epoch run of that version, Q&A validation loss was lowest after about half an epoch and rose from there while training loss fell below 1, so both versions train for one epoch. The first version reached a held-out doc Q&A loss of 2.257, a little lower than now, but the judge marked 99% of its answers to everyday questions wrong (see [Evaluation](#evaluation)). After the current SFT the model calls itself Enginseer, answers in one to four sentences, replies to other languages in English and declines some invented things ("I am not aware of a movie titled 'Neural Horizons' starring Keanu Reeves"), though many answers still contain a wrong detail.

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
| Enginseer SFT, first version | 3.1B + SFT | 0.760 | 0.968 | 0.606 | 0.320 | 0.527 | 0.260 | 101.5 |
| Enginseer SFT, 140M | 3.1B + SFT | 0.740 | 0.968 | 0.606 | 0.326 | 0.526 | 0.253 | 108.9 |

\* SmolLM was trained on Cosmopedia itself, so its Cosmopedia numbers are not a fair comparison. The public models may also have seen web copies of the documentation.

With 200–650 times fewer training tokens than SmolLM, the base model has the lowest bits per byte on the held-out docs of all the models here and beats GPT-2 on every column except LAMBADA, but it stays behind SmolLM on general English and on the benchmarks. LAMBADA needs the last word of a long narrative passage, and with only 3.1B tokens, mostly educational web text, that is its weakest point. SFT costs a little language modelling (0.02 bpb on the docs, 0.04 for the first version) and nothing measurable on the benchmarks.

**GGUF quantization** (`llama-perplexity -f data/eval/heldout.txt -c 1024`, held-out docs):

| Model | File | Size | Perplexity |
|---|---|---|---|
| base | f16 | 281 MB | 6.380 ± 0.039 |
| SFT, first version | f16 | 281 MB | 7.047 ± 0.044 |
| SFT | f16 | 281 MB | 6.691 ± 0.040 |
| SFT | Q8_0 | 150 MB | 6.694 ± 0.040 |
| SFT | Q4_K_M | 112 MB | 6.736 ± 0.041 |

Q8_0 is lossless within noise and Q4_K_M costs 0.7%. Because the model is 640 wide, which is not a multiple of 256, 168 of 218 tensors fall back from k-quants to simpler types in Q4_K_M. `check_gguf.py` on the SFT f16 file gives the same token ids on all 1409 docs, top-1 agreement 31 of 32 and a mean top-10 log-probability difference of 0.006 (max 0.034).

**Answer quality** (`eval_judge.py`). The chat model answers questions written by the teacher, and the teacher grades each answer against its own reference as correct, partial or wrong, with a one-sentence reason. There are four groups:

- *general*: the held-out conversations of `gen_chat.py`, with three kinds: everyday questions (grounded, advice, explanations, arithmetic), questions about invented things, and messages in other languages. On invented things, correct means the model says it does not know it and makes nothing up; on other languages, that it replies in English and asks for English.
- *sft*: doc questions the model was trained on.
- *pretrain*: questions about documents that were in pretraining but had no questions in SFT.
- *heldout*: questions about the 153 held-out documents that neither stage saw, generated the same way (`scripts/chunk_heldout.py`, then `gen_qa.py --chunks`).

Each doc group has up to 300 fact questions and 300 trick questions about options that do not exist. On trick questions, correct means the model says there is no such option and makes nothing up afterwards. A regex also counts "I'm not aware"-style answers, independent of the judge.

The baseline is SmolLM2-135M-Instruct, a model of the same size trained on 2T tokens, converted to Q8_0 GGUF and asked the same questions. All models answer greedily (temperature 0). General chat:

| Model | Everyday correct / partial | Invented correct / partial | Other languages correct / partial |
|---|---|---|---|
| SmolLM2-135M-Instruct | 3.0% / 7.0% | 2.2% / 0.0% | 0.8% / 10.0% |
| Enginseer SFT, first version | 0.3% / 0.3% | 0.0% / 0.0% | 1.7% / 21.7% |
| **Enginseer SFT, Q8_0** | **10.0% / 8.0%** | **31.1% / 2.2%** | **99.2% / 0.8%** |

The 300 everyday questions, 45 invented and 120 foreign-language messages come from the same generator as the training data, but from requests no conversation was trained on. The judge is strict: an answer with one wrong claim that matters is wrong, and most of the model's answers have one. It answers "tips for organizing my notes?" well, but also says that Jupiter "is a gas giant with no atmosphere", and it cannot add up three prices. It declines about a third of the invented things and makes up the rest ("The Sporkle gadget from the 1950s is a classic example of a modern kitchen tool"). It now and then mistakes an odd English title for a foreign language.

Documentation (the pretrain group has only 273 fact and 75 trick questions, as that is all the SFT validation documents have; the sft group of each version is the questions it trained on):

| Model | Group | Facts correct / partial | Tricks correct / partial | "Not aware" on tricks (regex) |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | sft | 1.0% / 0.7% | 0.0% / 0.3% | 0.3% |
| SmolLM2-135M-Instruct | pretrain | 0.4% / 0.7% | 0.0% / 0.0% | 0.0% |
| SmolLM2-135M-Instruct | heldout | 0.3% / 1.0% | 0.0% / 0.3% | 1.3% |
| Enginseer SFT, first version | sft | 8.3% / 5.3% | 10.0% / 42.0% | 86.0% |
| Enginseer SFT, first version | pretrain | 2.6% / 1.8% | 5.3% / 41.3% | 85.3% |
| Enginseer SFT, first version | heldout | 3.3% / 1.7% | 10.7% / 36.7% | 87.0% |
| Enginseer SFT, Q8_0 | sft | 3.3% / 4.3% | 4.3% / 54.0% | 90.3% |
| Enginseer SFT, Q8_0 | pretrain | 1.1% / 0.7% | 2.7% / 44.0% | 92.0% |
| Enginseer SFT, Q8_0 | heldout | 0.7% / 3.0% | 5.7% / 48.0% | 89.3% |

At this size a general model knows almost none of these specific facts, and SmolLM2 accepts every invented flag as real: in the judge's words, it "hallucinates a non-existent command `pod-suspend` and falsely claims the `--force-suspend` flag exists". Its "sft" and "pretrain" groups are just question sets, since it never trained on them.

The first version, tuned mostly on doc Q&A, got 8% of its trained fact questions right but only 3% about documents it never saw, so SFT taught it mostly the trained answers, not to recall what it read. The current version trained on an eighth as many doc questions and gets fewer facts right, as expected. Both learned to decline questions about options that do not exist, on held-out docs as well as trained ones (87–92% against 1% for SmolLM2). Most of those are only partial, though, as the model goes on to suggest a related option that does not fit, as in "I am not aware of a `--force-renew` flag in Kubernetes. You might be thinking of the `--certificate-authority` flag…".
