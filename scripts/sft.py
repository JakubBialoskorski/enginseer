"""Supervised fine-tuning of the pretrained model on the chat set built by build_sft.py.

  python scripts/sft.py configs/sft.toml

Starts from out/runs/<base>/final.pt and writes out/runs/<name>/final.pt plus log.jsonl and samples.txt.
The loss covers only answer tokens (and the closing <|im_end|>), averaged over all answer tokens of a step.

Batches are built per length bucket (128/256/512/1024 tokens, right-padded) with a fixed token budget per
micro-batch, so short Q&A pairs are not padded to the full context and torch.compile sees only four shapes.
Every example kind (chat, skill, invented, language, qa, trick, oasst, identity) can be repeated per epoch with [repeat].
"""

import argparse
import json
import os
import shutil
import time
import tomllib

import numpy as np
import paths  # noqa: F401  (sets the HF cache location)
import sentencepiece as spm
import torch
import torch.nn.functional as F
from model import Llama, ModelConfig
from paths import DATA, OUT
from train import lr_at

BUCKETS = (128, 256, 512, 1024)
SAMPLE_QUESTIONS = [
    "Why is the sky blue?",
    "Give me three tips for sleeping better.",
    "Who wrote the novel 'The Silver Orchard of Kaltberg'?",
    "Cześć, jak się masz?",
    "Who are you?",
    "What does `kubectl drain` do?",
]


class Split:
    def __init__(self, name):
        d = np.load(DATA / "sft" / f"{name}.npz")
        self.ids, self.mask, self.offsets, self.kinds = d["ids"], d["mask"], d["offsets"], d["kinds"]
        self.lengths = np.diff(self.offsets)

    def __len__(self):
        return len(self.kinds)

    def batches(self, order, budget, drop_last=False):
        """Micro-batches (lists of example indices) of one bucket each, in the given example order.
        drop_last skips the partial batch of every bucket, so training only sees four batch shapes."""
        open_, out = {b: [] for b in BUCKETS}, []
        for i in order:
            bucket = next(b for b in BUCKETS if self.lengths[i] <= b)
            open_[bucket].append(i)
            if len(open_[bucket]) == budget // bucket:
                out.append(open_[bucket])
                open_[bucket] = []
        return out if drop_last else out + [rows for rows in open_.values() if rows]

    def tensors(self, rows, device):
        bucket = next(b for b in BUCKETS if self.lengths[rows].max() <= b)
        x = np.zeros((len(rows), bucket), dtype=np.int64)
        y = np.full((len(rows), bucket), -100, dtype=np.int64)
        for r, i in enumerate(rows):
            a, b = self.offsets[i], self.offsets[i + 1]
            ids, mask = self.ids[a:b].astype(np.int64), self.mask[a:b]
            x[r, :b - a - 1] = ids[:-1]
            y[r, :b - a - 1] = np.where(mask[1:], ids[1:], -100)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def epoch_order(split, repeat, rng):
    order = np.concatenate([np.repeat(np.flatnonzero(split.kinds == k), n) for k, n in repeat.items()])
    return rng.permutation(order)


def step_loss(model, split, micro_batches, device):
    """Backward over the micro-batches of one step; the loss is the mean over all their answer tokens."""
    data = [split.tensors(rows, device) for rows in micro_batches]
    n_total = sum(int((y != -100).sum()) for _, y in data)
    loss_sum = 0.0
    for x, y in data:
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100,
                               reduction="sum") / n_total
        loss.backward()
        loss_sum += loss.detach()
    return loss_sum, n_total


@torch.no_grad()
def evaluate(model, split, budget, device):
    """Mean answer-token loss per example kind (run on the uncompiled model: partial batches change shapes)."""
    model.eval()
    out = {}
    for kind in np.unique(split.kinds):
        total, n = 0.0, 0
        for rows in split.batches(np.flatnonzero(split.kinds == kind), budget):
            x, y = split.tensors(rows, device)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = model(x)
            total += F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100,
                                     reduction="sum").item()
            n += int((y != -100).sum())
        out[str(kind)] = round(total / n, 4)
    model.train()
    return out


@torch.no_grad()
def samples(model, sp, step, device):
    model.eval()
    lines = [f"--- step {step}"]
    for q in SAMPLE_QUESTIONS:
        prompt = [1, *sp.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")]
        with torch.autocast(device, dtype=torch.bfloat16):
            ids = model.generate(torch.tensor([prompt], device=device), 160, temperature=0, eos_id=4)[0]
        lines.append(f"Q: {q}\nA: {sp.decode([i for i in ids[len(prompt):].tolist() if i != 4])}")
    model.train()
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--name", help="run name (default: the config file name)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    config = tomllib.loads(open(args.config).read())
    tc, repeat, device = config["train"], config["repeat"], args.device
    run = OUT / "runs" / (args.name or os.path.splitext(os.path.basename(args.config))[0])
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run / "config.toml")
    meta = json.loads((DATA / "tokens" / "meta.json").read_text())
    sp = spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model"))

    ckpt = torch.load(OUT / "runs" / tc["base"] / "final.pt", map_location="cpu")
    cfg = ModelConfig(**ckpt["model_config"])
    torch.manual_seed(tc["seed"])
    torch.set_float32_matmul_precision("high")
    raw = Llama(cfg)
    raw.load_state_dict(ckpt["model"])
    raw.to(device)
    del ckpt
    decay = [p for p in raw.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")

    train, val = Split("train"), Split("val")
    rng = np.random.default_rng(tc["seed"])
    budget, accum = tc["micro_tokens"], tc["grad_accum"]
    epochs = [train.batches(epoch_order(train, repeat, rng), budget, drop_last=True) for _ in range(tc["epochs"])]
    for micro_batches in epochs:
        rng.shuffle(micro_batches)
    steps = [mb[i:i + accum] for mb in epochs for i in range(0, len(mb), accum)]
    total = len(steps)
    print(f"{raw.num_params() / 1e6:.1f}M params from {tc['base']}, {len(train):,} train examples, "
          f"{tc['epochs']} epochs, {total:,} steps")

    model = raw if args.no_compile else torch.compile(raw)
    log = open(run / "log.jsonl", "w")
    t0 = time.time()
    record = {"step": 0, "val": evaluate(raw, val, budget, device)}
    print("  val " + "  ".join(f"{k} {v:.3f}" for k, v in record["val"].items()), flush=True)
    log.write(json.dumps(record) + "\n")
    with open(run / "samples.txt", "w") as f:
        f.write(samples(raw, sp, 0, device))

    for step, micro_batches in enumerate(steps, 1):
        lr = lr_at(step - 1, total, tc)
        for group in opt.param_groups:
            group["lr"] = lr
        loss, n_tokens = step_loss(model, train, micro_batches, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)

        record = None
        if step % tc["log_every"] == 0 or step == total:
            record = {"step": step, "loss": round(loss.item(), 4), "answer_tokens": n_tokens, "lr": lr,
                      "grad_norm": round(grad_norm.item(), 3), "hours": round((time.time() - t0) / 3600, 3)}
            eta = (total - step) * (time.time() - t0) / step / 60
            print(f"step {step:5}/{total}  loss {record['loss']:.4f}  lr {lr:.2e}  "
                  f"gnorm {record['grad_norm']:.2f}  eta {eta:.0f} min", flush=True)
        if step % tc["eval_every"] == 0 or step == total:
            record = record or {"step": step}
            record["val"] = evaluate(raw, val, budget, device)
            print("  val " + "  ".join(f"{k} {v:.3f}" for k, v in record["val"].items()), flush=True)
            with open(run / "samples.txt", "a") as f:
                f.write(samples(raw, sp, step, device))
        if record:
            log.write(json.dumps(record) + "\n")
            log.flush()

    torch.save({"model": raw.state_dict(), "model_config": vars(cfg)}, run / "final.pt")
    print(f"done: {run / 'final.pt'}")


if __name__ == "__main__":
    main()
