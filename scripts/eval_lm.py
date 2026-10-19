"""Language-model evaluation of our runs against small public models, on equal terms.

  python scripts/eval_lm.py base sft openai-community/gpt2 HuggingFaceTB/SmolLM-135M
  python scripts/eval_lm.py test --tasks bpb,lambada --limit 200       # quick pipeline check

A model is a run name under out/runs/ or a Hugging Face repo id. Tasks:
  bpb         bits per byte on our validation text (held-out docs, FineWeb-Edu, Cosmopedia), decoded from
              data/tokens/val/ so every model reads the same raw text with its own tokenizer; long documents
              are scored in 1024-token windows with a 512-token stride
  hellaswag   validation, 10,042 items, 4 endings (MIT)
  winogrande  winogrande_xl validation, 1,267 items, partial scoring of the text after the blank (CC-BY)
  lambada     LAMBADA (OpenAI version) test, 5,153 items: greedy last-word accuracy (MIT)
Multiple choice is zero-shot by log-likelihood, as in lm-evaluation-harness: acc compares sums, acc_norm divides
by the length of the ending in bytes. Every model is limited to a 1024-token context. Our models get <s> before
the context, as in training; the public ones get no prefix, as in lm-evaluation-harness, so their numbers match
the published ones (GPT-2: HellaSwag acc_norm 0.312, WinoGrande 0.519, LAMBADA acc 0.326). ARC (CC-BY-SA) is left out.

Results: out/eval/lm/<model>.json, plus a table of all models evaluated so far.
"""

import argparse
import json
import math
import re

import numpy as np
import paths  # noqa: F401  (sets the HF cache location)
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from model import Llama, ModelConfig
from paths import DATA, OUT

CONTEXT = 1024
STRIDE = 512
BATCH_TOKENS = 8192
TASKS = ("bpb", "hellaswag", "winogrande", "lambada")
DATASETS = {
    "hellaswag": ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
    "winogrande": ("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet"),
    "lambada": ("EleutherAI/lambada_openai", "data/lambada_test.jsonl"),
}


class Ours:
    def __init__(self, run, device):
        ckpt = torch.load(OUT / "runs" / run / "final.pt", map_location="cpu")
        self.model = Llama(ModelConfig(**ckpt["model_config"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        meta = json.loads((DATA / "tokens" / "meta.json").read_text())
        self.sp = spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model"))
        self.prefix = 1  # <s>, as every pretraining document starts with it
        self.task_prefix = [1]

    def encode(self, text):
        return self.sp.encode(text)

    def logits(self, x):
        return self.model(x)


class HF:
    def __init__(self, repo, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(device).eval()
        self.prefix = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        self.task_prefix = []  # like lm-evaluation-harness, which adds no BOS for these models

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def logits(self, x):
        return self.model(x).logits


@torch.no_grad()
def score(lm, requests, device):
    """requests: (ids, start) pairs. Returns (sum of log-probs of ids[start:], all of them greedy) per request."""
    out = [None] * len(requests)
    order = sorted(range(len(requests)), key=lambda i: len(requests[i][0]))
    batch = []

    def flush():
        width = len(requests[batch[-1]][0])
        x = torch.zeros((len(batch), width), dtype=torch.long)
        for r, i in enumerate(batch):
            ids = requests[i][0]
            x[r, :len(ids)] = torch.tensor(ids)
        # fp32 on purpose: under bf16 autocast GPT-2's LAMBADA accuracy drops from 0.326 to 0.308
        x = x.to(device)
        logp = F.log_softmax(lm.logits(x).float(), dim=-1)
        for r, i in enumerate(batch):
            ids, start = requests[i]
            target = x[r, start:len(ids)]
            lp = logp[r, start - 1:len(ids) - 1]
            out[i] = (lp.gather(1, target[:, None]).sum().item(), bool((lp.argmax(-1) == target).all()))
        batch.clear()

    for i in order:
        if batch and len(requests[i][0]) * (len(batch) + 1) > BATCH_TOKENS:
            flush()
        batch.append(i)
    if batch:
        flush()
    return out


def windows(ids):
    """Split one long sequence into (window, first scored index) pairs; every token after the prefix is scored once."""
    pos, out = 1, []
    while pos < len(ids):
        end = min(len(ids), CONTEXT if pos == 1 else pos + STRIDE)
        begin = max(0, end - CONTEXT)
        out.append((ids[begin:end], pos - begin))
        pos = end
    return out


def val_texts(limit):
    meta = json.loads((DATA / "tokens" / "meta.json").read_text())
    sp = spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model"))
    for name in meta["val"]:
        ids = np.fromfile(DATA / "tokens" / "val" / f"{name}.bin", dtype=np.uint16)
        starts = np.flatnonzero(ids == 1)
        docs = [sp.decode(ids[a + 1:b - 1].tolist()) for a, b in zip(starts, list(starts[1:]) + [len(ids)])]
        yield name, docs[:limit] if limit else docs


def bpb(lm, device, limit):
    out = {}
    for name, docs in val_texts(limit):
        requests = [w for text in docs for w in windows([lm.prefix] + lm.encode(text))]
        nll = -sum(lp for lp, _ in score(lm, requests, device))
        out[name] = round(nll / math.log(2) / sum(len(t.encode()) for t in docs), 4)
    return out


def rows(task, limit):
    repo, file = DATASETS[task]
    path = hf_hub_download(repo, file, repo_type="dataset")
    data = [json.loads(line) for line in open(path)] if file.endswith(".jsonl") else pq.read_table(path).to_pylist()
    return data[:limit] if limit else data


def hellaswag_clean(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def multiple_choice(lm, device, items):
    """items: (contexts, continuations, gold); one context for all endings or one per ending (WinoGrande)."""
    requests, spans = [], []
    for contexts, conts, _ in items:
        spans.append(len(requests))
        for ctx, cont in zip(contexts if len(contexts) > 1 else contexts * len(conts), conts):
            whole, head = lm.encode(ctx + cont), lm.encode(ctx)
            ids = (lm.task_prefix + whole)[-CONTEXT:]
            requests.append((ids, max(1, len(ids) - (len(whole) - len(head)))))
    scores = score(lm, requests, device)
    acc = acc_norm = 0
    for (_, conts, gold), a in zip(items, spans):
        lps = [s[0] for s in scores[a:a + len(conts)]]
        acc += int(np.argmax(lps)) == gold
        acc_norm += int(np.argmax([lp / len(c.encode()) for lp, c in zip(lps, conts)])) == gold
    return {"acc": round(acc / len(items), 4), "acc_norm": round(acc_norm / len(items), 4), "n": len(items)}


def hellaswag(lm, device, limit):
    items = []
    for r in rows("hellaswag", limit):
        ctx = hellaswag_clean(f"{r['activity_label']}: {r['ctx_a']} {r['ctx_b'].capitalize()}")
        items.append(([ctx], [" " + hellaswag_clean(e) for e in r["endings"]], int(r["label"])))
    return multiple_choice(lm, device, items)


def winogrande(lm, device, limit):
    items = []
    for r in rows("winogrande", limit):
        i = r["sentence"].index("_")
        cont = " " + r["sentence"][i + 1:].strip()
        items.append(([r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]], [cont, cont],
                      int(r["answer"]) - 1))
    res = multiple_choice(lm, device, items)
    return {"acc": res["acc"], "n": res["n"]}  # both options share the continuation, so acc_norm == acc


def lambada(lm, device, limit):
    requests = []
    for r in rows("lambada", limit):
        ctx, word = r["text"].rsplit(" ", 1)
        whole, head = lm.encode(ctx + " " + word), lm.encode(ctx)
        ids = (lm.task_prefix + whole)[-CONTEXT:]
        requests.append((ids, len(ids) - (len(whole) - len(head))))
    scores = score(lm, requests, device)
    return {"acc": round(sum(g for _, g in scores) / len(scores), 4),
            "ppl": round(math.exp(-sum(lp for lp, _ in scores) / len(scores)), 2), "n": len(scores)}


def table(results):
    cols = [("bpb", k) for k in ("docs", "fineweb-edu", "cosmopedia")] + [
        ("hellaswag", "acc_norm"), ("winogrande", "acc"), ("lambada", "acc"), ("lambada", "ppl")]
    lines = ["| Model | " + " | ".join(f"{t} {k}" for t, k in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in results.items():
        cells = [f"{r[t][k]}" if t in r and k in r[t] else "–" for t, k in cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="run names under out/runs/ or Hugging Face repo ids")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--limit", type=int, help="items (documents for bpb) per task, for quick checks")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = OUT / "eval" / "lm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        lm = HF(name, args.device) if "/" in name else Ours(name, args.device)
        path = out_dir / f"{name.replace('/', '--')}.json"
        res = json.loads(path.read_text()) if path.exists() and not args.limit else {}
        for task in args.tasks.split(","):
            res[task] = globals()[task](lm, args.device, args.limit)
            print(f"{name:32} {task:10} {res[task]}", flush=True)
        if not args.limit:
            path.write_text(json.dumps(res, indent=2) + "\n")
        del lm
        torch.cuda.empty_cache()

    results = {p.stem.replace("--", "/"): json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))}
    if results:
        print(table(results))


if __name__ == "__main__":
    main()
