"""Pretrain the model on the tokenized mix (data/tokens/), resumable from out/runs/<name>/ckpt.pt.

  python scripts/train.py configs/test.toml
  python scripts/train.py configs/base.toml --until 07:30   # save and stop at 07:30, run the same command to resume

The run directory holds a copy of the config, log.jsonl (train and val loss), samples.txt and the checkpoints.
Ctrl+C or SIGTERM finishes the current step, saves a checkpoint and exits.

Data: each micro-batch row is a seq_len + 1 token block from one source, the source drawn by the mix weights
with an RNG seeded by the step, so a resumed run sees exactly the batches an uninterrupted run would have.
Every source is walked through a shuffled permutation of its blocks (each epoch with a new offset), so the
general corpus is not repeated within the run and the small domain docs come back for several epochs.
"""

import argparse
import json
import math
import os
import shutil
import signal
import time
import tomllib
import zlib
from datetime import datetime, timedelta

import numpy as np
import paths  # noqa: F401  (sets the HF cache location)
import sentencepiece as spm
import torch
from model import Llama, ModelConfig, flops_per_token
from paths import DATA, OUT

TOKENS = DATA / "tokens"
SAMPLE_PROMPTS = ["Kubernetes is", "To restart a service in OpenRC,", "The unbound resolver", "Photosynthesis is"]


class Mix:
    def __init__(self, split, names, seq_len, seed):
        self.seq_len, self.seed = seq_len, seed
        # the token files are loaded into RAM: random block reads from a spinning disk would stall the GPU
        self.data = {n: np.fromfile(TOKENS / split / f"{n}.bin", dtype=np.uint16) for n in names}
        self.cursor = {n: [0, 0] for n in names}  # epoch, position in that epoch's permutation
        self._perm = {}

    def tokens(self, name):
        return len(self.data[name])

    def permutation(self, name, epoch):
        if self._perm.get(name, (None,))[0] != epoch:
            rng = np.random.default_rng([self.seed, zlib.crc32(name.encode()), epoch])
            offset = int(rng.integers(self.seq_len))
            blocks = (len(self.data[name]) - offset - 1) // self.seq_len
            self._perm[name] = (epoch, offset, rng.permutation(blocks))
        return self._perm[name][1:]

    def block(self, name):
        epoch, pos = self.cursor[name]
        offset, perm = self.permutation(name, epoch)
        if pos == len(perm):
            self.cursor[name] = [epoch + 1, 0]
            return self.block(name)
        self.cursor[name][1] += 1
        start = offset + int(perm[pos]) * self.seq_len
        return self.data[name][start:start + self.seq_len + 1]

    def batch(self, step, size, weights):
        names = list(weights)
        p = np.array([weights[n] for n in names], dtype=np.float64)
        picks = np.random.default_rng([self.seed, step]).choice(len(names), size=size, p=p / p.sum())
        rows = np.stack([self.block(names[i]) for i in picks]).astype(np.int64)
        x = torch.from_numpy(rows[:, :-1]).pin_memory().cuda(non_blocking=True)
        y = torch.from_numpy(rows[:, 1:]).pin_memory().cuda(non_blocking=True)
        return x, y


def lr_at(step, total, tc):
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    progress = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    min_lr = tc["lr"] * tc["min_lr_ratio"]
    return min_lr + 0.5 * (tc["lr"] - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, val, meta, seq_len, eval_tokens, micro_batch):
    """Mean loss and bits per byte on fixed, evenly spaced blocks of every val file."""
    model.eval()
    out = {}
    for name, data in val.items():
        n = min(eval_tokens // seq_len, (len(data) - 1) // seq_len) // micro_batch * micro_batch  # one batch shape
        starts = np.linspace(0, len(data) - seq_len - 1, n).astype(np.int64)
        losses = []
        for i in range(0, n, micro_batch):
            rows = np.stack([data[s:s + seq_len + 1] for s in starts[i:i + micro_batch]]).astype(np.int64)
            rows = torch.from_numpy(rows).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(rows[:, :-1], rows[:, 1:])
            losses.append(loss.item() * len(rows))
        loss = sum(losses) / n
        m = meta["val"][name]
        out[name] = {"loss": round(loss, 4), "bpb": round(loss / math.log(2) * m["tokens"] / m["bytes"], 4)}
    model.train()
    return out


@torch.no_grad()
def samples(model, sp, step):
    model.eval()
    lines = [f"--- step {step}"]
    for prompt in SAMPLE_PROMPTS:
        idx = torch.tensor([[1, *sp.encode(prompt)]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(idx, 48)[0, 1:].tolist()
        lines.append(repr(sp.decode([i for i in ids if i != 2])))
    model.train()
    return "\n".join(lines) + "\n"


def save(path, raw, opt, step, mix, cfg, tc):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": raw.state_dict(), "optim": opt.state_dict(), "step": step, "cursor": mix.cursor,
                "model_config": vars(cfg), "train_config": tc}, tmp)
    os.replace(tmp, path)


def deadline(args):
    if args.hours:
        return time.time() + args.hours * 3600
    if args.until:
        h, m = map(int, args.until.split(":"))
        now = datetime.now()
        at = now.replace(hour=h, minute=m, second=0, microsecond=0)
        return (at if at > now else at + timedelta(days=1)).timestamp()
    return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--name", help="run name (default: the config file name)")
    parser.add_argument("--until", help="HH:MM local time: save and stop then")
    parser.add_argument("--hours", type=float, help="save and stop after this many hours")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    config = tomllib.loads(open(args.config).read())
    cfg, tc = ModelConfig(**config["model"]), config["train"]
    run = OUT / "runs" / (args.name or os.path.splitext(os.path.basename(args.config))[0])
    run.mkdir(parents=True, exist_ok=True)
    ckpt_path = run / "ckpt.pt"
    meta = json.loads((TOKENS / "meta.json").read_text())
    sp = spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model"))
    assert sp.vocab_size() == cfg.vocab_size

    torch.manual_seed(tc["seed"])
    torch.set_float32_matmul_precision("high")
    raw = Llama(cfg).cuda()
    decay = [p for p in raw.parameters() if p.dim() >= 2]
    no_decay = [p for p in raw.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(0.9, 0.95), eps=1e-8, fused=True)

    seq_len, micro, accum = cfg.max_seq_len, tc["micro_batch"], tc["grad_accum"]
    step_tokens = micro * accum * seq_len
    total = math.ceil(tc["total_tokens"] / step_tokens)
    anneal_start = total - int(total * tc.get("anneal_fraction", 0.0))
    mix_w, anneal_w = config["mix"], config.get("anneal", config["mix"])
    mix = Mix("train", sorted(set(mix_w) | set(anneal_w)), seq_len, tc["seed"])
    val = {n: np.fromfile(TOKENS / "val" / f"{n}.bin", dtype=np.uint16) for n in meta["val"]}

    step = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cuda")
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        step, mix.cursor = ckpt["step"], ckpt["cursor"]
        print(f"resumed {ckpt_path} at step {step}")
        del ckpt
    else:
        shutil.copy(args.config, run / "config.toml")

    print(f"{raw.num_params() / 1e6:.1f}M params, {step_tokens:,} tokens/step, {total:,} steps, "
          f"{total * step_tokens / 1e9:.2f}B tokens, {flops_per_token(cfg, seq_len) * total * step_tokens:.2e} FLOP")
    for name in mix.data:
        drawn = sum(w.get(name, 0) / sum(w.values()) * n * step_tokens
                    for w, n in ((mix_w, anneal_start), (anneal_w, total - anneal_start)))
        print(f"  {name:12} {mix.tokens(name) / 1e6:9.1f}M tokens, {drawn / mix.tokens(name):5.2f} epochs")

    model = raw if args.no_compile else torch.compile(raw)
    stop_at = deadline(args)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: stop.append("SIGINT"))
    log = open(run / "log.jsonl", "a")
    last_save = t0 = time.time()
    t0_step = step

    while step < total:
        lr = lr_at(step, total, tc)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = mix.batch(step, micro * accum, anneal_w if step >= anneal_start else mix_w)
        loss_sum = 0.0
        for i in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x[i * micro:(i + 1) * micro], y[i * micro:(i + 1) * micro])
            (loss / accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        record = None
        if step % tc["log_every"] == 0 or step == total:
            elapsed = time.time() - t0
            tps = (step - t0_step) * step_tokens / elapsed
            record = {"step": step, "tokens": step * step_tokens, "loss": round(loss_sum.item() / accum, 4),
                      "lr": lr, "grad_norm": round(grad_norm.item(), 3), "tok_s": round(tps),
                      "hours": round(elapsed / 3600, 3)}
            eta = (total - step) * step_tokens / tps / 3600
            print(f"step {step:6}/{total}  loss {record['loss']:.4f}  lr {lr:.2e}  gnorm {record['grad_norm']:.2f}  "
                  f"{tps / 1e3:.1f}k tok/s  eta {eta:.1f} h", flush=True)
        if step % tc["eval_every"] == 0 or step == total:
            record = record or {"step": step, "tokens": step * step_tokens}
            record["val"] = evaluate(model, val, meta, seq_len, tc["eval_tokens"], micro)
            print("  val " + "  ".join(f"{n} {v['loss']:.3f} ({v['bpb']:.3f} bpb)" for n, v in record["val"].items()),
                  flush=True)
            with open(run / "samples.txt", "a") as f:
                f.write(samples(raw, sp, step))
        if record:
            log.write(json.dumps(record) + "\n")
            log.flush()

        reason = stop[0] if stop else "deadline" if time.time() >= stop_at else None
        if reason or step == total or time.time() - last_save >= tc["ckpt_minutes"] * 60:
            save(ckpt_path, raw, opt, step, mix, cfg, tc)
            last_save = time.time()
            print(f"  saved {ckpt_path} at step {step}", flush=True)
        if reason:
            print(f"stopped ({reason}); run the same command to resume")
            return

    torch.save({"model": raw.state_dict(), "model_config": vars(cfg)}, run / "final.pt")
    print(f"done: {run / 'final.pt'}")


if __name__ == "__main__":
    main()
