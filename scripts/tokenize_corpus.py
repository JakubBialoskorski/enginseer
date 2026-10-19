"""Encode the pretraining mix with our tokenizer into flat uint16 token files for the training loop.

  python scripts/tokenize_corpus.py            # every source
  python scripts/tokenize_corpus.py --only docs

Outputs data/tokens/{train,val}/<source>.bin, each document wrapped as <s> text </s>, and data/tokens/meta.json
with token and byte counts per file (bytes are needed for bits-per-byte, which compares across tokenizers).
The val split of the general corpus is the same sha256 rule as the tokenizer eval set with a smaller fraction,
so no val document was seen by the tokenizer either. The val split of the domain docs is their held-out split.
"""

import argparse
import json
import time

import numpy as np
import paths  # noqa: F401  (sets the HF cache location)
import pyarrow.parquet as pq
import sentencepiece as spm
from paths import DATA, OUT
from train_tokenizer import is_eval

TOKENS = DATA / "tokens"
VAL_FRACTION = 0.001  # of the general corpus; a subset of the 2% tokenizer eval set
BATCH = 4096


def parquet_docs(pattern):
    for f in sorted((DATA / "corpus").glob(pattern)):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g, columns=["text"]).column("text").to_pylist()


def doc_texts(split):
    for line in open(DATA / "docs.jsonl"):
        doc = json.loads(line)
        if doc["split"] == split:
            yield doc["text"]


class Writer:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "wb")
        self.docs = self.tokens = self.bytes = 0

    def write(self, sp, texts, threads):
        if not texts:
            return
        for ids in sp.encode(texts, num_threads=threads):
            np.array([1, *ids, 2], dtype=np.uint16).tofile(self.file)
            self.tokens += len(ids) + 2
        self.docs += len(texts)
        self.bytes += sum(len(t.encode()) for t in texts)

    def close(self):
        self.file.close()
        return {"docs": self.docs, "tokens": self.tokens, "bytes": self.bytes}


def encode_general(sp, name, pattern, threads):
    train, val = Writer(TOKENS / "train" / f"{name}.bin"), Writer(TOKENS / "val" / f"{name}.bin")
    batch = {train: [], val: []}
    for text in parquet_docs(pattern):
        out = val if is_eval(text, VAL_FRACTION) else train
        batch[out].append(text)
        if len(batch[out]) == BATCH:
            out.write(sp, batch[out], threads)
            batch[out] = []
            if out is train and train.docs % (BATCH * 64) == 0:
                print(f"  {train.docs:,} docs, {train.tokens / 1e9:.2f}B tokens", flush=True)
    for out, texts in batch.items():
        out.write(sp, texts, threads)
    return train.close(), val.close()


def encode_docs(sp, threads):
    stats = []
    for split, path in (("train", "train"), ("heldout", "val")):
        out = Writer(TOKENS / path / "docs.bin")
        out.write(sp, list(doc_texts(split)), threads)
        stats.append(out.close())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="sp32k")
    parser.add_argument("--only", nargs="*", help="encode only these sources")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{args.tokenizer}.model"))
    assert sp.vocab_size() <= 2**16 and sp.bos_id() == 1 and sp.eos_id() == 2
    meta_path = TOKENS / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"train": {}, "val": {}}
    meta["tokenizer"] = args.tokenizer
    jobs = {
        "fineweb-edu": lambda: encode_general(sp, "fineweb-edu", "fineweb-edu/**/*.parquet", args.threads),
        "cosmopedia": lambda: encode_general(sp, "cosmopedia", "cosmopedia/**/*.parquet", args.threads),
        "docs": lambda: encode_docs(sp, args.threads),
    }
    for name, job in jobs.items():
        if args.only and name not in args.only:
            continue
        print(name, flush=True)
        start = time.time()
        meta["train"][name], meta["val"][name] = job()
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        for split in ("train", "val"):
            s = meta[split][name]
            print(f"  {split:5} {s['docs']:9,} docs {s['tokens']:14,} tokens  {s['bytes'] / s['tokens']:.3f} bytes/token")
        print(f"  {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
