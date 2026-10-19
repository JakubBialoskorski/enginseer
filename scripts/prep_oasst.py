"""Turn oasst2 ([[sft]] in sources.toml) into English chat threads for SFT: data/sft/oasst2.jsonl.

A thread is a path from a root prompt down the message tree where every assistant reply is the top-ranked one
among its siblings, cut after the last answered prompt. Every message must be English,
not deleted, passed review and not too toxic; threads where the assistant talks about being Open Assistant are
dropped, since the model has its own identity. The oasst2 validation split stays a separate split.
"""

import json
import re
import tomllib
from collections import defaultdict
from datetime import date

import paths  # noqa: F401  (sets the HF cache location)
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from paths import DATA, ROOT

IDENTITY = re.compile(r"open[ -]?assistant|laion", re.I)
MAX_TOXICITY = 0.3


def ok(m):
    tox = (m["detoxify"] or {}).get("toxicity") or 0
    return m["lang"] == "en" and not m["deleted"] and m["review_result"] and not m["synthetic"] and tox < MAX_TOXICITY


def threads(rows):
    children = defaultdict(list)
    for m in rows:
        if m["parent_id"] and ok(m):
            children[m["parent_id"]].append(m)

    def walk(m, path):
        path = path + [m]
        kids = children[m["message_id"]]
        if m["role"] == "prompter":
            ranked = [k for k in kids if k["rank"] is not None]
            kids = [min(ranked, key=lambda k: k["rank"])] if ranked else kids[:1]
        found = False
        for k in kids:
            for t in walk(k, path):
                found = True
                yield t
        # a follow-up prompt without an answer still leaves a complete thread up to this reply
        if not found and m["role"] == "assistant":
            yield path

    for m in rows:
        if m["parent_id"] is None and ok(m):
            yield from walk(m, [])


def main():
    sft = {s["name"]: s for s in tomllib.loads((ROOT / "sources.toml").read_text())["sft"]}["oasst2"]
    revision = HfApi().dataset_info(sft["dataset"]).sha
    out = DATA / "sft" / "oasst2.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out, "w") as f:
        for file in sft["files"]:
            split = "val" if "validation" in file else "train"
            rows = pq.read_table(hf_hub_download(sft["dataset"], file, repo_type="dataset",
                                                 revision=revision)).to_pylist()
            n = 0
            for path in threads(rows):
                if any(m["role"] == "assistant" and IDENTITY.search(m["text"]) for m in path):
                    continue
                messages = [{"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"].strip()}
                            for m in path]
                f.write(json.dumps({"id": path[-1]["message_id"], "split": split, "messages": messages},
                                   ensure_ascii=False) + "\n")
                n += 1
            counts[split] = n
            print(f"{split}: {n:,} threads")
    manifest = {"dataset": sft["dataset"], "revision": revision, "license": sft["license"], "files": sft["files"],
                "threads": counts, "prepared": date.today().isoformat()}
    (DATA / "sft" / "oasst2.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
