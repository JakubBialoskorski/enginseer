"""Generate grounded Q&A pairs for SFT from the doc chunks (data/chunks.jsonl) with a local teacher model.

  python scripts/gen_qa.py                                   # teacher behind llama-swap on :8081
  python scripts/gen_qa.py --url http://127.0.0.1:8096 --workers 4 --until 14:00

Any OpenAI-compatible llama-server works. The teacher sees one chunk and must use only that chunk. Per chunk it
writes several key facts (commands, flags, config keys, defaults), each as 3-4 paraphrased questions sharing one
answer, plus one trick question about an option that does not exist, answered with "I'm not aware of that".
Every 10th chunk also gets one fact's question in another language (answer stays English).

Output: data/sft/qa.jsonl, one line per chunk. Re-running skips chunks already done, so the run is resumable.
Chunks are processed in priority order: hand-written docs first, generated API references last.
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from paths import DATA

PROMPT_VERSION = 1
PRODUCTS = {
    "kubernetes": "Kubernetes", "docker": "Docker", "opentofu": "OpenTofu", "github-actions": "GitHub Actions",
    "prometheus": "Prometheus", "prometheus-server": "Prometheus", "nginx": "nginx", "openrc": "OpenRC",
    "unbound": "Unbound (DNS resolver)", "unbound-man": "Unbound (DNS resolver)", "meshcore": "MeshCore",
    "meshcore-wiki": "MeshCore",
}
LANGUAGES = ["German", "French", "Spanish", "Italian", "Portuguese", "Dutch"]
MIN_TOKENS = 80
MAX_CHARS = 14000

SYSTEM = """You write training data for a small technical assistant. You get one excerpt of {product} \
documentation. Use ONLY facts stated in the excerpt; never add knowledge from elsewhere.

Write exactly {n} key facts. A key fact is something a user would ask about: a command, flag, config key or \
directive, default value, API field, behavior, limit or procedure. Pick the most useful, concrete and distinct facts.
For each fact:
- "questions": 3 or 4 different phrasings of the same question, as real users would type them (a how-to, a direct \
"what does X do", a short terse one, a scenario). Every question must be self-contained: name {product} or the \
specific tool/resource, and never refer to "the text", "the excerpt" or "the documentation".
- "answer": one correct answer to all of the questions, 1-4 sentences, in the voice of a knowledgeable assistant. \
Include the exact command, flag, key or a short config/code snippet from the excerpt when relevant (Markdown \
backticks or a fenced block). Do not say "according to the documentation".

Then write one "trick" item: a question about a plausible-sounding but clearly invented option, flag, directive or \
feature of {product} that is NOT in the excerpt and does not exist. Its answer must say you are not aware of such \
an option, then point to the real, related one from the excerpt.

No answer may mention "the excerpt", "the text", "the provided documentation" or similar: answer as if you simply \
know {product}.
{extra}
Everything is in English. Return JSON only."""

TRANSLATE = """
Finally, write "translated": the first question of the first fact translated into {language}. Its answer stays \
the English answer of that fact."""


def schema(n, translated):
    s = {"type": "string", "minLength": 8}
    props = {
        "facts": {"type": "array", "minItems": n, "maxItems": n, "items": {
            "type": "object", "required": ["questions", "answer"], "properties": {
                "questions": {"type": "array", "minItems": 3, "maxItems": 4, "items": s}, "answer": s}}},
        "trick": {"type": "object", "required": ["question", "answer"], "properties": {"question": s, "answer": s}},
    }
    if translated:
        props["translated"] = s
    return {"type": "object", "required": list(props), "properties": props}


def low_value(chunk):
    """Generated API references and old release notes: fewer facts, processed last."""
    return any(p in chunk["doc"] for p in ("reference/kubernetes-api", "/generated/", "/previous-versions/"))


def n_facts(chunk):
    low = low_value(chunk)
    return max(1, min(3 if low else 6, round(chunk["tokens"] / 250)))


def priority(chunk):
    return (low_value(chunk), hashlib.sha256(chunk["id"].encode()).hexdigest())


def request(args, chunk):
    n = n_facts(chunk)
    rng = random.Random(chunk["id"])
    language = rng.choice(LANGUAGES) if rng.random() < 0.1 else None
    product = PRODUCTS[chunk["source"]]
    system = SYSTEM.format(product=product, n=n, extra=TRANSLATE.format(language=language) if language else "")
    user = f"Page: {chunk['title']}\nSection: {chunk['heading']}\n\n{chunk['text'][:MAX_CHARS]}"
    body = {
        "model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 600 + 220 * n,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "qa", "schema": schema(n, language)}},
    }
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        res = json.load(r)
    choice = res["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError(f"finish_reason {choice['finish_reason']}")
    qa = json.loads(choice["message"]["content"])
    if language:
        qa["language"] = language
    return {"id": chunk["id"], "doc": chunk["doc"], "source": chunk["source"], "license": chunk["license"],
            "product": product, "prompt_version": PROMPT_VERSION, "teacher": args.model, **qa,
            "usage": res["usage"]}


def deadline(until):
    if not until:
        return float("inf")
    h, m = map(int, until.split(":"))
    now = datetime.now()
    at = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (at if at > now else at + timedelta(days=1)).timestamp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--workers", type=int, default=1, help="parallel requests (match the server's -np)")
    parser.add_argument("--limit", type=int, help="stop after this many chunks (pilot runs)")
    parser.add_argument("--until", help="HH:MM local time: send no new requests after that")
    parser.add_argument("--out", default=str(DATA / "sft" / "qa.jsonl"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    chunks = [json.loads(line) for line in open(DATA / "chunks.jsonl")]
    todo = sorted((c for c in chunks if c["tokens"] >= MIN_TOKENS and c["id"] not in done), key=priority)
    todo = todo[:args.limit] if args.limit else todo
    print(f"{len(done)} chunks done, {len(todo)} to go", flush=True)

    stop_at = deadline(args.until)
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "out_tokens": 0}
    start = time.time()

    def work(chunk):
        if time.time() > stop_at:
            return
        for attempt in range(3):
            try:
                row = request(args, chunk)
                break
            except (ValueError, json.JSONDecodeError, OSError) as e:
                err = e
        else:
            with lock:
                stats["failed"] += 1
                print(f"  failed {chunk['id']}: {err}", flush=True)
            return
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            stats["out_tokens"] += row["usage"]["completion_tokens"]
            if stats["ok"] % 20 == 0:
                dt = time.time() - start
                eta = (len(todo) - stats["ok"] - stats["failed"]) * dt / stats["ok"] / 3600
                print(f"{stats['ok']}/{len(todo)} chunks, {stats['failed']} failed, "
                      f"{stats['out_tokens'] / dt:.1f} out tok/s, eta {eta:.1f} h", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    print(f"done: {stats['ok']} written, {stats['failed']} failed, {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
