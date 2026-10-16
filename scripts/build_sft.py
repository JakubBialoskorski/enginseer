"""Build the tokenized SFT set from the teacher Q&A (data/sft/qa.jsonl), oasst2 threads and identity examples.

  python scripts/build_sft.py

Every example is a ChatML conversation, starting with <s>:
  <|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>\\n
Only the answer tokens and their closing <|im_end|> are trained on; the model stops generating at <|im_end|>.

Each fact's 3-4 paraphrased questions become separate examples with the same answer. Q&A val is a 2% split by
source document, so the paraphrases of a val fact never show up in training; the documents themselves were in
pretraining, though. Val also gets Q&A about the held-out docs, if data/eval/qa-heldout.jsonl exists (kinds
"heldout" and "heldout-trick"), which neither pretraining nor SFT saw. oasst2 keeps its own val split.
Answers that leak the generation setup ("the excerpt", "the documentation says") are dropped.

Output: data/sft/{train,val}.npz with the token ids (uint16), loss mask (bool), example offsets and kinds.
"""

import hashlib
import json
import re
from collections import Counter

import numpy as np
import sentencepiece as spm
from paths import DATA, OUT

MAX_LEN = 1024
VAL_FRACTION = 0.02
LEAK = re.compile(r"\b(excerpt|the (provided |given )?(text|passage|documentation|docs) (says|states|mentions|"
                  r"recommends|describes|notes|does not)|according to the (documentation|docs|text)|"
                  r"provided (text|documentation|docs))\b", re.I)

IDENTITY_QUESTIONS = [
    "Who are you?", "What are you?", "What is your name?", "Introduce yourself.", "What can you help me with?",
    "Are you ChatGPT?", "Which model are you?", "Who made you?", "What do you know about?", "hi, who am I talking to?",
    "What are your limitations?", "Can I trust your answers?",
]
IDENTITY_ANSWERS = [
    "I'm Enginseer, a small language model trained from scratch. I know the most about DevOps tools such as "
    "Kubernetes, Docker, OpenTofu, GitHub Actions, Prometheus and nginx, and also about OpenRC, Unbound and "
    "MeshCore. I'm small, so I can get exact flags and config keys wrong; check the official documentation "
    "before you rely on them.",
    "I'm Enginseer, a compact assistant for DevOps and Linux infrastructure: Kubernetes, Docker, OpenTofu, "
    "GitHub Actions, Prometheus, nginx, OpenRC, Unbound and MeshCore. I answer in English. Because I'm a small "
    "model, please double-check commands and options against the official docs.",
]


class Encoder:
    def __init__(self, sp):
        self.sp = sp

    def __call__(self, messages):
        """Token ids and answer mask of one conversation, or None if an answer does not encode on its own
        (the ids of prompt + answer must start with the ids of the prompt that inference will see)."""
        text, spans, checks = "", [], []
        for m in messages:
            head = f"<|im_start|>{m['role']}\n"
            if m["role"] == "assistant":
                prompt = self.sp.encode(text + head)
                full = self.sp.encode(text + head + m["content"] + "<|im_end|>")
                if full[:len(prompt)] != prompt:
                    return None
                spans.append((len(prompt), len(full)))
                checks.append(full)
            text += head + m["content"] + "<|im_end|>\n"
        ids = self.sp.encode(text)
        if any(ids[:len(c)] != c for c in checks):
            return None
        mask = [False] * (len(ids) + 1)
        for a, b in spans:
            mask[1 + a:1 + b] = [True] * (b - a)
        return [1] + ids, mask


def qa_examples():
    for row in map(json.loads, open(DATA / "sft" / "qa.jsonl")):
        # salted: the unsalted hash of the doc path is chunk_docs.py's held-out rule, so these docs all pass it
        digest = hashlib.sha256(f"sft-val/{row['doc']}".encode()).hexdigest()
        split = "val" if int(digest[:8], 16) / 2**32 < VAL_FRACTION else "train"
        for fact in row["facts"]:
            if LEAK.search(fact["answer"]):
                continue
            for q in fact["questions"]:
                yield split, "qa", [q, fact["answer"]]
        if not LEAK.search(row["trick"]["answer"]):
            yield split, "trick", [row["trick"]["question"], row["trick"]["answer"]]
        if "translated" in row and not LEAK.search(row["facts"][0]["answer"]):
            yield split, "translated", [row["translated"], row["facts"][0]["answer"]]


def heldout_examples():
    """Val only: Q&A about the held-out docs (data/eval/qa-heldout.jsonl), which pretraining never saw either."""
    path = DATA / "eval" / "qa-heldout.jsonl"
    if not path.exists():
        return
    for row in map(json.loads, open(path)):
        for fact in row["facts"]:
            if not LEAK.search(fact["answer"]):
                for q in fact["questions"]:
                    yield "val", "heldout", [q, fact["answer"]]
        if not LEAK.search(row["trick"]["answer"]):
            yield "val", "heldout-trick", [row["trick"]["question"], row["trick"]["answer"]]


def oasst_examples():
    for row in map(json.loads, open(DATA / "sft" / "oasst2.jsonl")):
        yield row["split"], "oasst", [m["content"] for m in row["messages"]]


def identity_examples():
    for q in IDENTITY_QUESTIONS:
        for a in IDENTITY_ANSWERS:
            yield "train", "identity", [q, a]


def main():
    meta = json.loads((DATA / "tokens" / "meta.json").read_text())
    enc = Encoder(spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model")))
    splits = {s: {"ids": [], "mask": [], "offsets": [0], "kinds": []} for s in ("train", "val")}
    stats = Counter()

    for gen in (qa_examples(), heldout_examples(), oasst_examples(), identity_examples()):
        for split, kind, turns in gen:
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t.strip()}
                        for i, t in enumerate(turns)]
            out = enc(messages)
            # long threads lose their last exchanges until they fit
            while out and len(out[0]) > MAX_LEN and len(messages) > 2:
                messages = messages[:-2]
                out = enc(messages)
            if out is None or len(out[0]) > MAX_LEN:
                stats[f"dropped {kind}"] += 1
                continue
            s = splits[split]
            s["ids"] += out[0]
            s["mask"] += out[1]
            s["offsets"].append(len(s["ids"]))
            s["kinds"].append(kind)
            stats[f"{split} {kind}"] += 1
            stats[f"{split} {kind} answer tokens"] += sum(out[1])

    for split, s in splits.items():
        np.savez(DATA / "sft" / f"{split}.npz", ids=np.array(s["ids"], dtype=np.uint16),
                 mask=np.array(s["mask"], dtype=bool), offsets=np.array(s["offsets"], dtype=np.int64),
                 kinds=np.array(s["kinds"]))
        print(f"{split}: {len(s['kinds']):,} examples, {len(s['ids']):,} tokens, {sum(s['mask']):,} answer tokens")
    for k, v in sorted(stats.items()):
        print(f"  {k:32} {v:,}")


if __name__ == "__main__":
    main()
