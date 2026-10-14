"""Stage 0 smoke test: CUDA works and a small model loads in 4-bit NF4 (the QLoRA path)."""

import argparse

import paths  # noqa: F401  (sets the HF cache location)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="cuda:0")
    print(f"loaded {args.model}: {model.get_memory_footprint() / 2**20:.0f} MiB")

    messages = [{"role": "user", "content": "In one sentence, what is a Kubernetes pod?"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
