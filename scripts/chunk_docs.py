"""Clean every fetched documentation file and split it into chunks.

Outputs:
  data/docs.jsonl    one cleaned document per line (for pretraining), with a train/heldout split
  data/chunks.jsonl  heading-aware chunks of the train documents (input for SFT Q&A generation)

The split is per document and deterministic (hash of source + path), so held-out docs never leak into training.
Token counts use the Qwen3 tokenizer, because chunking came before our own tokenizer (stage 2); they only set chunk sizes.
"""

import argparse
import hashlib
import json
import re
import tomllib
from collections import defaultdict

import paths  # noqa: F401  (sets the HF cache location)
from doc_cleaners import clean
from paths import DATA, ROOT
from transformers import AutoTokenizer

EXTENSIONS = {".md", ".mdx", ".rst", ".xml"}
SKIP = re.compile(r"(^|/)(release-notes?|changelog)(\.|/|$)|(^|/)README\.md$|/_[^/]+/", re.I)
MIN_CHARS = 300


def doc_files(src, repo):
    for entry in src.get("docs", src["paths"]) or ["."]:
        base = repo / entry
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        for path in files:
            rel = path.relative_to(repo).as_posix()
            is_man = entry == "man" and re.fullmatch(r"\.[1-8]", path.suffix)
            if (path.suffix in EXTENSIONS or is_man) and not SKIP.search(rel) and ".git/" not in rel:
                yield path, rel


def heldout(name, rel, fraction):
    digest = hashlib.sha256(f"{name}/{rel}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < fraction


def split_sections(text):
    """Split at markdown headings; each section keeps its heading path for context."""
    sections, path, buf = [], [], []
    in_fence = False
    for line in text.split("\n"):
        if re.match(r"^(```|~~~)", line):
            in_fence = not in_fence
        m = None if in_fence else re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            if any(b.strip() for b in buf):
                sections.append((list(path), "\n".join(buf).strip()))
            level = len(m.group(1))
            path = path[:level - 1] + [m.group(2).strip()]
            buf = [line]
        else:
            buf.append(line)
    if any(b.strip() for b in buf):
        sections.append((list(path), "\n".join(buf).strip()))
    return sections


def pack(sections, count, max_tokens):
    """Greedily merge consecutive sections up to max_tokens; split oversized sections by paragraphs."""
    chunks, cur, cur_tokens, cur_head = [], [], 0, None
    for head, body in sections:
        pieces = [body]
        if count(body) > max_tokens:
            # paragraphs, never breaking inside a code fence
            pieces, buf, in_fence = [], [], False
            for para in re.split(r"\n\s*\n", body):
                in_fence ^= para.count("```") % 2 == 1
                buf.append(para)
                if not in_fence:
                    pieces.append("\n\n".join(buf))
                    buf = []
            if buf:
                pieces.append("\n\n".join(buf))
        for piece in pieces:
            n = count(piece)
            if cur and cur_tokens + n > max_tokens:
                chunks.append((cur_head, "\n\n".join(cur), cur_tokens))
                cur, cur_tokens = [], 0
            if not cur:
                cur_head = head
            cur.append(piece)
            cur_tokens += n
    if cur:
        chunks.append((cur_head, "\n\n".join(cur), cur_tokens))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--heldout", type=float, default=0.05, help="fraction of documents held out for eval")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    count = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    sources = tomllib.loads((ROOT / "sources.toml").read_text())["source"]
    stats = defaultdict(lambda: defaultdict(int))

    with open(DATA / "docs.jsonl", "w") as docs_out, open(DATA / "chunks.jsonl", "w") as chunks_out:
        for src in sources:
            repo = DATA / "raw" / src["name"]
            for path, rel in doc_files(src, repo):
                title, text = clean(src, repo, path)
                if len(text) < MIN_CHARS:
                    stats[src["name"]]["skipped_short"] += 1
                    continue
                split = "heldout" if heldout(src["name"], rel, args.heldout) else "train"
                tokens = count(text)
                doc_id = f"{src['name']}/{rel}"
                record = {"id": doc_id, "source": src["name"], "domain": src["domain"], "license": src["license"],
                          "path": rel, "title": title, "split": split, "tokens": tokens, "text": text}
                docs_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                s = stats[src["name"]]
                s["docs"] += 1
                s[f"{split}_tokens"] += tokens
                if split != "train":
                    continue
                for i, (head, body, n) in enumerate(pack(split_sections(text), count, args.max_tokens)):
                    chunks_out.write(json.dumps({
                        "id": f"{doc_id}#{i}", "doc": doc_id, "source": src["name"], "domain": src["domain"],
                        "license": src["license"], "title": title, "heading": " > ".join(head or [title]),
                        "tokens": n, "text": body,
                    }, ensure_ascii=False) + "\n")
                    s["chunks"] += 1

    total = defaultdict(int)
    print(f"{'source':20} {'docs':>6} {'chunks':>7} {'train tok':>11} {'heldout tok':>12} {'short':>6}")
    for name, s in stats.items():
        print(f"{name:20} {s['docs']:6} {s['chunks']:7} {s['train_tokens']:11,} {s['heldout_tokens']:12,} {s['skipped_short']:6}")
        for k, v in s.items():
            total[k] += v
    print(f"{'TOTAL':20} {total['docs']:6} {total['chunks']:7} {total['train_tokens']:11,} {total['heldout_tokens']:12,} {total['skipped_short']:6}")


if __name__ == "__main__":
    main()
