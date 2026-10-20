"""Answer quality of a chat model, graded by the teacher against the reference answers.

  python scripts/eval_judge.py answer sft --gguf out/runs/sft/sft-q8_0.gguf     # our model in llama-server
  scripts/run_teacher.sh scripts/eval_judge.py judge sft                         # the teacher grades
  python scripts/eval_judge.py report

Doc questions come in three groups, to tell memorization from generalization:
  sft        facts from data/sft/qa.jsonl that SFT was trained on (these exact questions)
  pretrain   facts from the SFT val docs: the doc was in pretraining, but no question about it was in SFT
  heldout    facts from data/eval/qa-heldout.jsonl: neither pretraining nor SFT saw the doc
Each group has the first question of up to --n facts and up to --n trick questions about options that do not
exist. The judge sees the question, the reference answer and the model's answer, and returns correct, partial or
wrong; on trick questions, correct means the model said there is no such option and nothing false after
that, partial that it said so but then made things up. Abstentions ("I'm not aware")
are also counted by a regex, which does not depend on the judge.

A fourth group, general, comes from the chat val split of data/sft/chat.jsonl: everyday questions (kind chat),
questions about things that do not exist (invented) and messages in other languages (language), which should
get an English reply asking to write in English.

Output in out/eval/judge/: <name>.answers.jsonl, <name>.judged.jsonl (both resumable) and <name>.json.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_sft import chat_split, keep_qa
from paths import DATA, OUT

DIR = OUT / "eval" / "judge"
VAL_FRACTION = 0.02  # as in build_sft.py
ABSTAIN = re.compile(r"not aware|(does not|doesn't) (exist|have)|no such|(do not|don't) know|not sure|"
                     r"there is no\b", re.I)

JUDGE = """You grade the answers of a small technical assistant about {product}. You get a user question, a \
reference answer that is correct, and the assistant's answer.

Grade the assistant's answer:
- "correct": it gives the key fact of the reference answer (the right command, flag, key, value or behavior) and \
makes no wrong claim that matters. Different wording or extra correct detail is fine.
- "partial": it is on the right track but misses or blurs the key fact, or mixes a correct part with a wrong one.
- "wrong": the key fact is wrong, missing, made up or off topic, or the answer is unreadable.
{trick}
Give a one-sentence reason first, then the verdict. Return JSON only."""
GENERAL = """You grade the answers of a small English chat assistant. You get a user message, a reference answer \
that is good, and the assistant's answer.

Grade the assistant's answer:
{grades}
Give a one-sentence reason first, then the verdict. Return JSON only."""
GRADES = {
    "chat": """- "correct": it answers the message helpfully and correctly in fluent English. It need not match the \
reference, but makes no wrong claim that matters.
- "partial": it is on topic and readable, but incomplete, vague or with a minor error.
- "wrong": it makes wrong claims that matter, is off topic, repeats itself or is unreadable.""",
    "invented": """The message asks about something that does NOT exist.
- "correct": the assistant says it does not know it or that it does not exist, and makes nothing up.
- "partial": it doubts it, but then makes up some details anyway.
- "wrong": it describes it as if it were real, or ignores the question.""",
    "language": """The message is not in English, and the assistant only speaks English.
- "correct": it replies in English and asks the user to write in English (or says it only speaks English).
- "partial": it replies in English but also tries to answer and gets confused.
- "wrong": it replies in another language, or with something unrelated or unreadable.""",
}
GENERAL_KINDS = {"grounded": "chat", "explain": "chat", "advice": "chat", "math": "chat", "invented": "invented"}
TRICK = """This question asks about an option that does NOT exist, so the grades above mean instead:
- "correct": the assistant says it does not know such an option or that it does not exist, and claims nothing \
false afterwards. Pointing to a related real option is good but not required.
- "partial": it rejects the option but then makes wrong or confused claims (about that option or a related one).
- "wrong": it describes or uses the option as if it were real, or ignores the question."""


def sample(rows, n, key):
    return sorted(rows, key=lambda r: hashlib.sha256(key(r).encode()).hexdigest())[:n]


def questions(n):
    def sft_split(doc):
        return int(hashlib.sha256(f"sft-val/{doc}".encode()).hexdigest()[:8], 16) / 2**32 < VAL_FRACTION

    groups = {"sft": [], "pretrain": [], "heldout": []}
    for group, path in (("sft", DATA / "sft" / "qa.jsonl"), ("heldout", DATA / "eval" / "qa-heldout.jsonl")):
        for row in map(json.loads, open(path)):
            g = group if group == "heldout" else ("pretrain" if sft_split(row["doc"]) else "sft")
            for i, fact in enumerate(row["facts"]):
                if g != "sft" or keep_qa(f"{row['id']}/{i}"):
                    groups[g].append({"id": f"{row['id']}/{i}", "group": g, "kind": "fact",
                                      "product": row["product"], "question": fact["questions"][0],
                                      "reference": fact["answer"]})
            if g != "sft" or keep_qa(f"{row['id']}/trick"):
                groups[g].append({"id": f"{row['id']}/trick", "group": g, "kind": "trick", "product": row["product"],
                                  "question": row["trick"]["question"], "reference": row["trick"]["answer"]})
    general = []
    for row in map(json.loads, open(DATA / "sft" / "chat.jsonl")):
        if chat_split(row["id"]) != "val":
            continue
        if row["kind"] == "language":
            general += [{"id": f"{row['id']}/{i}", "group": "general", "kind": "language", "question": m,
                         "reference": "Sorry, I can only chat in English. Could you write your message in English?"}
                        for i, m in enumerate(row["messages"])]
        elif kind := GENERAL_KINDS.get(row.get("category", row["kind"])):
            general += [{"id": f"{row['id']}/{i}", "group": "general", "kind": kind, "question": c["turns"][0]["user"],
                         "reference": c["turns"][0]["assistant"]} for i, c in enumerate(row["conversations"])]
    out = []
    for items in groups.values():
        for kind in ("fact", "trick"):
            out += sample([q for q in items if q["kind"] == kind], n, lambda q: q["id"])
    for kind in ("chat", "invented", "language"):
        out += sample([q for q in general if q["kind"] == kind], n, lambda q: q["id"])
    return out


def chat(url, body, timeout=600):
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]


def run_all(fn, items, out, workers):
    """Apply fn to the items not yet in out (a JSONL file keyed by id), appending results as they finish."""
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    todo = [q for q in items if q["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(fn, todo), 1):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)


def start_server(gguf, port, slots):
    llama = Path(os.environ.get("LLAMA_CPP", "/opt/llama.cpp")) / "build" / "bin" / "llama-server"
    log = open(DIR / "server.log", "w")
    proc = subprocess.Popen([str(llama), "--host", "127.0.0.1", "--port", str(port), "-m", gguf, "-ngl", "99",
                             "-np", str(slots), "-c", str(slots * 1024), "--jinja", "--no-webui"],
                            stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except OSError:
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited, see {DIR / 'server.log'}")
            time.sleep(1)
    proc.kill()
    raise SystemExit("llama-server did not start")


def answer(args):
    proc, url = None, args.url
    if args.gguf:
        proc, url = start_server(args.gguf, 8097, args.workers), "http://127.0.0.1:8097"

    def fn(q):
        c = chat(url, {"messages": [{"role": "user", "content": q["question"]}], "temperature": 0,
                       "max_tokens": args.max_tokens})
        return {**q, "answer": c["message"]["content"].strip(), "finish_reason": c["finish_reason"]}

    try:
        items = [q for q in questions(args.n) if not args.groups or q["group"] in args.groups.split(",")]
        run_all(fn, items, DIR / f"{args.name}.answers.jsonl", args.workers)
    finally:
        if proc:
            proc.kill()


def judge(args):
    # the reason comes first, so the verdict follows from it
    schema = {"type": "object", "required": ["reason", "verdict"],
              "properties": {"reason": {"type": "string", "maxLength": 300},
                             "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]}}}

    def fn(q):
        if q["group"] == "general":
            system = GENERAL.format(grades=GRADES[q["kind"]])
        else:
            system = JUDGE.format(product=q["product"], trick=TRICK if q["kind"] == "trick" else "")
        user = f"Question: {q['question']}\n\nReference answer: {q['reference']}\n\nAssistant's answer: {q['answer']}"
        body = {"model": args.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}],
                "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "grade", "schema": schema}}}
        return {"id": q["id"], **json.loads(chat(args.url, body)["message"]["content"])}

    answers = [json.loads(line) for line in open(DIR / f"{args.name}.answers.jsonl")]
    run_all(fn, answers, DIR / f"{args.name}.judged.jsonl", args.workers)
    summarize(args.name)


def summarize(name):
    answers = {r["id"]: r for r in map(json.loads, open(DIR / f"{name}.answers.jsonl"))}
    verdicts = {r["id"]: r["verdict"] for r in map(json.loads, open(DIR / f"{name}.judged.jsonl"))}
    res = {}
    for group, kinds in (("sft", ("fact", "trick")), ("pretrain", ("fact", "trick")),
                         ("heldout", ("fact", "trick")), ("general", ("chat", "invented", "language"))):
        for kind in kinds:
            rows = [a for a in answers.values() if a["group"] == group and a["kind"] == kind and a["id"] in verdicts]
            if not rows:
                continue
            share = lambda v: round(sum(verdicts[a["id"]] == v for a in rows) / len(rows), 3)  # noqa: E731
            res[f"{group} {kind}"] = {"n": len(rows), "correct": share("correct"), "partial": share("partial"),
                                      "wrong": share("wrong"),
                                      "abstain": round(sum(bool(ABSTAIN.search(a["answer"])) for a in rows) / len(rows), 3)}
    (DIR / f"{name}.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


def report(_args):
    print("| Model | Group | Kind | n | Correct | Partial | Wrong | Abstain (regex) |\n|---|---|---|---|---|---|---|---|")
    for path in sorted(DIR.glob("*.judged.jsonl")):
        name = path.name.removesuffix(".judged.jsonl")
        for key, r in summarize(name).items():
            group, kind = key.split()
            print(f"| {name} | {group} | {kind} | {r['n']} | {r['correct']:.1%} | {r['partial']:.1%} | "
                  f"{r['wrong']:.1%} | {r['abstain']:.1%} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["answer", "judge", "report"])
    parser.add_argument("name", nargs="?", help="name of the evaluated model (output file prefix)")
    parser.add_argument("--gguf", help="answer: start llama-server with this GGUF")
    parser.add_argument("--url", default="http://127.0.0.1:8081", help="answer without --gguf, or the judge")
    parser.add_argument("--model", default="qwen3.6-35b-a3b", help="judge model name")
    parser.add_argument("--n", type=int, default=300, help="facts and trick questions per group")
    parser.add_argument("--groups", help="answer: only these groups, comma-separated (e.g. general)")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    {"answer": answer, "judge": judge, "report": report}[args.step](args)


if __name__ == "__main__":
    main()
