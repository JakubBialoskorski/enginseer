"""Check a GGUF file against our tokenizer and model by running it in llama-server (CPU only).

  python scripts/check_gguf.py test out/runs/test/test-f16.gguf

1. Tokenizer: every tokenizer eval doc must get exactly the same ids from llama.cpp as from SentencePiece,
   and the ChatML markers must stay single control tokens.
2. Model: next-token log-probabilities after held-out prefixes, llama.cpp vs our PyTorch model in fp32.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request

import numpy as np
import paths  # noqa: F401  (sets the HF cache location)
import sentencepiece as spm
import torch
from model import Llama, ModelConfig
from paths import DATA, OUT

PORT = 8095


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def wait_ready(proc):
    for _ in range(120):
        if proc.poll() is not None:
            raise SystemExit("llama-server exited, see its output above")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise SystemExit("llama-server did not start")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("gguf")
    parser.add_argument("--prefixes", type=int, default=32)
    args = parser.parse_args()

    llama_cpp = os.environ.get("LLAMA_CPP", "/opt/llama.cpp")
    server = subprocess.Popen([f"{llama_cpp}/build/bin/llama-server", "-m", args.gguf, "--port", str(PORT),
                               "-ngl", "0", "-c", "1024", "-np", "1"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready(server)
        meta = json.loads((DATA / "tokens" / "meta.json").read_text())
        sp = spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model"))

        texts = [json.loads(line)["text"] for line in open(OUT / "tokenizer" / "eval.jsonl")]
        bad = sum(post("/tokenize", {"content": t, "parse_special": False})["tokens"] != sp.encode(t) for t in texts)
        print(f"tokenizer: {bad} of {len(texts)} eval docs differ")
        chat = "<|im_start|>user\nhi<|im_end|>\n"
        assert post("/tokenize", {"content": chat})["tokens"] == sp.encode(chat), "chat markers differ"
        assert post("/tokenize", {"content": "hi", "add_special": True})["tokens"][0] == 1, "no BOS added"

        ckpt = torch.load(OUT / "runs" / args.run / "final.pt", map_location="cpu")
        model = Llama(ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model"])
        model.eval()
        data = np.fromfile(DATA / "tokens" / "val" / "docs.bin", dtype=np.uint16)
        top1, diffs = 0, []
        for start in np.linspace(0, len(data) - 600, args.prefixes).astype(int):
            prefix = [int(t) for t in data[start:start + 64 + start % 448]]
            res = post("/completion", {"prompt": prefix, "n_predict": 1, "temperature": -1, "n_probs": 10,
                                       "cache_prompt": False})
            theirs = {p["id"]: p["logprob"] for p in res["completion_probabilities"][0]["top_logprobs"]}
            with torch.no_grad():
                ours = torch.log_softmax(model(torch.tensor([prefix]))[0, -1], -1)
            top1 += int(ours.argmax()) == max(theirs, key=theirs.get)
            diffs += [abs(float(ours[i]) - lp) for i, lp in theirs.items()]
        print(f"model: top-1 agreement {top1}/{args.prefixes}, top-10 logprob diff mean {np.mean(diffs):.4f} "
              f"max {np.max(diffs):.4f}")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
