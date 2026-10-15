"""Chunk the held-out documents of data/docs.jsonl the same way chunk_docs.py chunks the train ones.

  python scripts/chunk_heldout.py

Output: data/eval/chunks-heldout.jsonl, the input for eval Q&A generation:
  scripts/run_teacher.sh --chunks data/eval/chunks-heldout.jsonl --out data/eval/qa-heldout.jsonl
Neither pretraining nor SFT ever sees these documents, so questions about them measure generalization.
"""

import json

import paths  # noqa: F401  (sets the HF cache location)
from chunk_docs import pack, split_sections
from paths import DATA
from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    count = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    out = DATA / "eval" / "chunks-heldout.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = chunks = 0
    with open(out, "w") as f:
        for doc in map(json.loads, open(DATA / "docs.jsonl")):
            if doc["split"] != "heldout":
                continue
            docs += 1
            for i, (head, body, n) in enumerate(pack(split_sections(doc["text"]), count, 1500)):
                f.write(json.dumps({
                    "id": f"{doc['id']}#{i}", "doc": doc["id"], "source": doc["source"], "domain": doc["domain"],
                    "license": doc["license"], "title": doc["title"], "heading": " > ".join(head or [doc["title"]]),
                    "tokens": n, "text": body,
                }, ensure_ascii=False) + "\n")
                chunks += 1
    print(f"{docs} held-out docs, {chunks} chunks -> {out}")


if __name__ == "__main__":
    main()
