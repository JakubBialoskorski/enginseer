"""Train the SentencePiece BPE tokenizer on a sample of the pretraining mix and measure its compression.

  python scripts/train_tokenizer.py sample             # write out/tokenizer/sample.txt and eval.jsonl
  python scripts/train_tokenizer.py train 16384 32768  # train out/tokenizer/sp<k>k.model per vocab size
  python scripts/train_tokenizer.py eval               # bytes per token of every trained model (+ GPT-2)
  python scripts/train_tokenizer.py hf sp32k           # out/tokenizer/hf/ for the HF export, checked against sentencepiece

SentencePiece BPE (not unigram, not HF byte-level BPE) because llama.cpp's SPM tokenizer merges pieces by score,
which reproduces BPE exactly, and convert_hf_to_gguf.py loads tokenizer.model without a pre-tokenizer hash check.
Settings follow Llama 2: byte fallback, split digits, no normalization, whitespace runs kept for code and YAML.
Unlike Llama 2 there is no dummy prefix, so the HF tokenizer_config.json must say add_prefix_space: false.
The training data wraps every document in <s> ... </s>, so inference adds BOS (add_bos_token: true).
"""

import argparse
import hashlib
import json
import random
import shutil
import time

import paths  # noqa: F401  (sets the HF cache location)
import pyarrow.parquet as pq
import sentencepiece as spm
from paths import DATA, OUT

TOK = OUT / "tokenizer"
SPECIAL = ["<|im_start|>", "<|im_end|>"]
MAX_LINE = 16384  # bytes; sentencepiece drops longer sentences


def is_eval(text, fraction):
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < fraction


def parquet_texts(pattern, chars, rng):
    """Yield texts from randomly chosen row groups until `chars` characters were taken."""
    files = sorted((DATA / "corpus").glob(pattern))
    groups = [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]
    rng.shuffle(groups)
    taken = 0
    for f, g in groups:
        for text in pq.ParquetFile(f).read_row_group(g, columns=["text"]).column("text").to_pylist():
            yield text
            taken += len(text)
            if taken >= chars:
                return


def cmd_sample(args):
    rng = random.Random(0)
    TOK.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(line) for line in open(DATA / "docs.jsonl")]
    sources = {
        "fineweb-edu": parquet_texts("fineweb-edu/**/*.parquet", args.general_chars * 0.75, rng),
        "cosmopedia": parquet_texts("cosmopedia/**/*.parquet", args.general_chars * 0.25, rng),
        "docs": (d["text"] for d in docs if d["split"] == "train" for _ in range(args.docs_weight)),
    }
    stats = {}
    with open(TOK / "sample.txt", "w") as sample, open(TOK / "eval.jsonl", "w") as evals:
        for name, texts in sources.items():
            s = stats[name] = {"docs": 0, "chars": 0, "eval_docs": 0}
            for text in texts:
                if name != "docs" and is_eval(text, 0.02):
                    evals.write(json.dumps({"source": name, "text": text}) + "\n")
                    s["eval_docs"] += 1
                    continue
                s["docs"] += 1
                s["chars"] += len(text)
                for line in text.split("\n"):
                    if line.strip() and len(line.encode()) <= MAX_LINE:
                        sample.write(line + "\n")
        for d in docs:
            if d["split"] == "heldout":
                evals.write(json.dumps({"source": "docs", "text": d["text"]}) + "\n")
                stats["docs"]["eval_docs"] += 1
    for name, s in stats.items():
        print(f"{name:12} {s['docs']:8,} docs {s['chars'] / 1e6:8.1f}M chars  {s['eval_docs']:6,} eval docs")


def cmd_train(args):
    for vocab in args.vocab:
        start = time.time()
        spm.SentencePieceTrainer.train(
            input=str(TOK / "sample.txt"), model_prefix=str(TOK / f"sp{vocab // 1024}k"),
            model_type="bpe", vocab_size=vocab, character_coverage=0.9995, byte_fallback=True,
            split_digits=True, normalization_rule_name="identity", remove_extra_whitespaces=False,
            add_dummy_prefix=False, allow_whitespace_only_pieces=True, max_sentence_length=MAX_LINE,
            input_sentence_size=args.sentences, shuffle_input_sentence=True,
            user_defined_symbols=SPECIAL, unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
            num_threads=args.threads, minloglevel=1,
        )
        print(f"sp{vocab // 1024}k: {time.time() - start:.0f} s")


def cmd_eval(args):
    evals = [json.loads(line) for line in open(TOK / "eval.jsonl")]
    encoders = {}
    for path in sorted(TOK.glob("sp*k.model"), key=lambda p: int(p.stem[2:-1])):
        sp = spm.SentencePieceProcessor(model_file=str(path))
        encoders[path.stem] = lambda texts, sp=sp: [len(ids) for ids in sp.encode(texts, num_threads=args.threads)]
    if args.gpt2:
        from transformers import AutoTokenizer
        gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
        encoders["gpt2"] = lambda texts: [len(ids) for ids in gpt2(texts)["input_ids"]]

    names = sorted({e["source"] for e in evals})
    print(f"{'bytes/token':12}" + "".join(f"{n:>13}" for n in names))
    for enc_name, encode in encoders.items():
        row = []
        for name in names:
            texts = [e["text"] for e in evals if e["source"] == name]
            row.append(sum(len(t.encode()) for t in texts) / sum(encode(texts)))
        print(f"{enc_name:12}" + "".join(f"{v:13.3f}" for v in row))


def cmd_hf(args):
    from transformers import AutoTokenizer
    hf = TOK / "hf"
    hf.mkdir(exist_ok=True)
    shutil.copy(TOK / f"{args.model}.model", hf / "tokenizer.model")
    sp = spm.SentencePieceProcessor(model_file=str(hf / "tokenizer.model"))
    specials = ["<unk>", "<s>", "</s>", *SPECIAL]
    config = {
        "tokenizer_class": "LlamaTokenizer", "add_prefix_space": False, "legacy": False,
        "add_bos_token": True, "add_eos_token": False, "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "added_tokens_decoder": {str(sp.piece_to_id(t)): {"content": t, "special": True} for t in specials},
    }
    (hf / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(hf)
    texts = [json.loads(line)["text"] for line in open(TOK / "eval.jsonl")]
    mismatches = sum(sp.encode(t) != tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts)
    print(f"{hf}: {len(tokenizer)} tokens, {mismatches} of {len(texts)} eval docs encode differently in HF")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--general-chars", type=float, default=300e6, help="characters of FineWeb-Edu + Cosmopedia")
    p.add_argument("--docs-weight", type=int, default=2, help="repeat the domain docs this many times")
    p = sub.add_parser("train")
    p.add_argument("vocab", type=int, nargs="+")
    p.add_argument("--sentences", type=int, default=0, help="sample this many lines (0 = all)")
    p.add_argument("--threads", type=int, default=6)
    p = sub.add_parser("eval")
    p.add_argument("--gpt2", action="store_true", help="also measure the GPT-2 tokenizer as a reference")
    p.add_argument("--threads", type=int, default=6)
    p = sub.add_parser("hf")
    p.add_argument("model", help="e.g. sp32k")
    args = parser.parse_args()
    {"sample": cmd_sample, "train": cmd_train, "eval": cmd_eval, "hf": cmd_hf}[args.cmd](args)


if __name__ == "__main__":
    main()
