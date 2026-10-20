"""Export a training run to a Hugging Face LlamaForCausalLM directory and check it against our own model.

  python scripts/export_hf.py test               # out/runs/test/final.pt (or ckpt.pt) -> out/runs/test/hf/
  python scripts/export_hf.py sft --chat         # chat model: ChatML template, generation stops at <|im_end|>
  python /opt/llama.cpp/convert_hf_to_gguf.py out/runs/test/hf --outtype f16 --outfile out/runs/test/test-f16.gguf

Our parameter names already match HF's, so the export only prefixes them with "model."; the output head is tied
to the embedding, which LlamaForCausalLM handles with tie_word_embeddings. The check runs both models in fp32 on
held-out tokens and compares the logits.

--chat sets eos to <|im_end|> (config.json, which the GGUF converter reads, and tokenizer_config.json) and adds
the ChatML template that build_sft.py trained on, starting with <s> (llama.cpp drops it when it adds its own).
The model never saw a system role and is too small to follow one, so the template drops system messages: chat
apps such as Jan send a long default system prompt, which would otherwise end up in front of the first question.
It also writes greedy sampling defaults to generation_config.json, which the GGUF converter stores in the GGUF and
llama.cpp uses unless told otherwise: without a repeat penalty, longer answers tend to loop.
"""

import argparse
import json
import shutil

import numpy as np
import paths  # noqa: F401  (sets the HF cache location)
import torch
from model import Llama, ModelConfig
from paths import DATA, OUT
from transformers import LlamaConfig, LlamaForCausalLM

IM_END = 4
# The system branch reads the content without printing it: llama.cpp treats a template that never reads it as
# having no system role and prepends the system prompt (and tool descriptions) to the first user message.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}{% if m['role'] != 'system' %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] | trim + '<|im_end|>\\n' }}"
    "{% elif m['content'] %}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
# greedy, as in the evaluation; without the repeat penalty longer answers tend to loop
SAMPLING = {"do_sample": False, "temperature": 0.0, "repetition_penalty": 1.1}


def hf_config(cfg, eos=2):
    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.dim, intermediate_size=cfg.ffn_dim,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim, max_position_embeddings=cfg.max_seq_len, rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta, hidden_act="silu", tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=eos, torch_dtype="float32",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="run name under out/runs/")
    parser.add_argument("--chat", action="store_true", help="SFT run: ChatML template and <|im_end|> as eos")
    args = parser.parse_args()

    run = OUT / "runs" / args.run
    src = run / "final.pt" if (run / "final.pt").exists() else run / "ckpt.pt"
    ckpt = torch.load(src, map_location="cpu")
    cfg = ModelConfig(**ckpt["model_config"])
    ours = Llama(cfg)
    ours.load_state_dict(ckpt["model"])
    ours.eval()

    hf = LlamaForCausalLM(hf_config(cfg, IM_END if args.chat else 2))
    missing, unexpected = hf.load_state_dict({f"model.{k}": v for k, v in ckpt["model"].items()}, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    hf.tie_weights()
    hf.eval()

    out = run / "hf"
    hf.save_pretrained(out)
    for f in (OUT / "tokenizer" / "hf").iterdir():
        shutil.copy(f, out / f.name)
    if args.chat:
        tc = json.loads((out / "tokenizer_config.json").read_text())
        tc.update(eos_token="<|im_end|>", chat_template=CHAT_TEMPLATE)
        (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2) + "\n")
        gc = json.loads((out / "generation_config.json").read_text())
        gc.update(SAMPLING)
        (out / "generation_config.json").write_text(json.dumps(gc, indent=2) + "\n")
    print(f"{src} -> {out} ({ours.num_params() / 1e6:.1f}M params)")

    data = np.fromfile(DATA / "tokens" / "val" / "docs.bin", dtype=np.uint16)
    rows = torch.from_numpy(np.stack([data[i:i + cfg.max_seq_len] for i in (0, len(data) // 2)]).astype(np.int64))
    with torch.no_grad():
        a, b = ours(rows), hf(rows).logits
    diff = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"logits max abs diff {diff:.2e}, argmax agreement {agree:.4%}")
    assert diff < 1e-3, "exported model does not match"
    (out / "export.json").write_text(json.dumps({"source": src.name, "step": ckpt.get("step"),
                                                 "max_abs_logit_diff": diff}, indent=2) + "\n")


if __name__ == "__main__":
    main()
