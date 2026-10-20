"""Stage 0 smoke test: CUDA works, and a few compiled bf16 training steps of the model fit on the GPU."""

import argparse
import time
import tomllib

import torch

from model import Llama, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/test.toml")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    mcfg = ModelConfig(**cfg["model"])
    batch = args.micro_batch or cfg["train"]["micro_batch"]
    model = Llama(mcfg).cuda()
    print(f"{args.config}: {model.num_params() / 1e6:.1f}M parameters, micro-batch {batch} x {mcfg.max_seq_len}")
    step = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    # random tokens: this checks speed and memory, not learning
    x = torch.randint(0, mcfg.vocab_size, (batch, mcfg.max_seq_len + 1), device="cuda")
    for i in range(args.steps + 1):
        if i == 1:  # the first step includes compilation
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = step(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    tok_s = args.steps * batch * mcfg.max_seq_len / (time.time() - t0)
    print(f"loss {loss.item():.2f}, {tok_s / 1e3:.1f}k tokens/s, peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
