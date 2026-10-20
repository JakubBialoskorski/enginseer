"""Build the tokenized SFT set from the teacher's general chat (data/sft/chat.jsonl), doc Q&A (data/sft/qa.jsonl),
oasst2 threads and identity examples.

  python scripts/build_sft.py

Every example is a ChatML conversation, starting with <s>:
  <|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>\\n
Only the answer tokens and their closing <|im_end|> are trained on; the model stops generating at <|im_end|>.

The model is a general chat model, so only half of the doc facts and trick questions are kept, each with its
first question. Messages in other languages get one of the ENGLISH_ONLY replies. Chat val is a split by
request. oasst2 threads are cut before the first long answer. Q&A val is a 2% split by source document; the
documents themselves were in pretraining, though. Val also gets Q&A about the held-out docs, if
data/eval/qa-heldout.jsonl exists (kinds
"heldout" and "heldout-trick"), which neither pretraining nor SFT saw. oasst2 keeps its own val split.
Answers that leak the generation setup ("the excerpt", "the documentation says") are dropped.

Output: data/sft/{train,val}.npz with the token ids (uint16), loss mask (bool), example offsets and kinds.
"""

import hashlib
import json
import re
from collections import Counter

import numpy as np
import sentencepiece as spm
from paths import DATA, OUT

MAX_LEN = 1024
VAL_FRACTION = 0.02
CHAT_VAL_FRACTION = 0.03     # of grounded requests; skill and language requests are few, so they give 10%
QA_FRACTION = 0.5
OASST_MAX_ANSWER = 1200      # characters; longer oasst2 answers teach long, rambling replies
LEAK = re.compile(r"\b(excerpt|the (provided |given )?(text|passage|documentation|docs) (says|states|mentions|"
                  r"recommends|describes|notes|does not)|according to the (documentation|docs|text)|"
                  r"provided (text|documentation|docs))\b", re.I)
CHAT_LEAK = re.compile(r"\b(the|this) (passage|article|excerpt)\b", re.I)
ENGLISH_ONLY = [
    "Sorry, I can only chat in English. Could you write your message in English?",
    "I only understand English, so I can't reply to that. If you ask again in English, I'll do my best to help.",
    "I'm a small model that only speaks English. Please write to me in English and I'll try to help.",
    "I'm sorry, I don't speak that language. I can only chat in English. What would you like to ask?",
    "I can only read and write English. Could you ask me again in English?",
]

IDENTITY_QUESTIONS = [
    "Who are you?", "What are you?", "What is your name?", "Introduce yourself.", "What can you help me with?",
    "Are you ChatGPT?", "Which model are you?", "Who made you?", "What do you know about?", "hi, who am I talking to?",
    "What are your limitations?", "Can I trust your answers?",
]
IDENTITY_ANSWERS = [
    "I'm Enginseer, a small language model with 140 million parameters, trained from scratch by Jakub "
    "Białoskórski as a proof of concept. I can chat in English about everyday topics, but I may get facts wrong, "
    "so please check anything important.",
    "I'm Enginseer, a small English-only chat model. Jakub Białoskórski built and trained me from scratch on a "
    "single GPU to show how a language model is made. I run offline, have no internet access, and I may get "
    "facts wrong.",
]
LANGUAGE_QUESTIONS = [
    "What languages do you speak?", "Can you speak Polish?", "Do you speak German?", "Can we talk in Spanish?",
    "Can you translate this into French for me?", "Which languages do you understand?",
]
LANGUAGE_ANSWERS = [
    "I only speak English. I was trained on English text only, so I can't understand or write other languages "
    "well. I'm happy to help in English, though.",
    "Sorry, I only know English. I was trained on English text only, so for other languages or translations "
    "please use a different tool.",
]


def hashed(key, fraction):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32 < fraction


def keep_qa(key):
    """Doc facts and trick questions kept for SFT, keyed "<chunk id>/<fact index>" or "<chunk id>/trick"."""
    return hashed(f"sft-keep/{key}", QA_FRACTION)


def chat_split(request_id):
    fraction = CHAT_VAL_FRACTION if request_id.startswith("grounded/") else 0.1
    return "val" if hashed(f"chat-val/{request_id}", fraction) else "train"


class Encoder:
    def __init__(self, sp):
        self.sp = sp

    def __call__(self, messages):
        """Token ids and answer mask of one conversation, or None if an answer does not encode on its own
        (the ids of prompt + answer must start with the ids of the prompt that inference will see)."""
        text, spans, checks = "", [], []
        for m in messages:
            head = f"<|im_start|>{m['role']}\n"
            if m["role"] == "assistant":
                prompt = self.sp.encode(text + head)
                full = self.sp.encode(text + head + m["content"] + "<|im_end|>")
                if full[:len(prompt)] != prompt:
                    return None
                spans.append((len(prompt), len(full)))
                checks.append(full)
            text += head + m["content"] + "<|im_end|>\n"
        ids = self.sp.encode(text)
        if any(ids[:len(c)] != c for c in checks):
            return None
        mask = [False] * (len(ids) + 1)
        for a, b in spans:
            mask[1 + a:1 + b] = [True] * (b - a)
        return [1] + ids, mask


def qa_examples():
    for row in map(json.loads, open(DATA / "sft" / "qa.jsonl")):
        # salted: the unsalted hash of the doc path is chunk_docs.py's held-out rule, so these docs all pass it
        digest = hashlib.sha256(f"sft-val/{row['doc']}".encode()).hexdigest()
        split = "val" if int(digest[:8], 16) / 2**32 < VAL_FRACTION else "train"
        for i, fact in enumerate(row["facts"]):
            if keep_qa(f"{row['id']}/{i}") and not LEAK.search(fact["answer"]):
                yield split, "qa", [fact["questions"][0], fact["answer"]]
        if keep_qa(f"{row['id']}/trick") and not LEAK.search(row["trick"]["answer"]):
            yield split, "trick", [row["trick"]["question"], row["trick"]["answer"]]


def chat_examples():
    """Kinds: chat (grounded in a pretraining passage), invented (things that do not exist), skill (the other
    categories of gen_chat.py) and language (a message in another language, answered in English)."""
    path = DATA / "sft" / "chat.jsonl"
    if not path.exists():
        return
    for row in map(json.loads, open(path)):
        split = chat_split(row["id"])
        if row["kind"] == "language":
            for m in row["messages"]:
                yield split, "language", [m, ENGLISH_ONLY[int(hashlib.sha256(m.encode()).hexdigest(), 16) % len(ENGLISH_ONLY)]]
            continue
        kind = "chat" if row["kind"] == "grounded" else "invented" if row["category"] == "invented" else "skill"
        for c in row["conversations"]:
            turns = [t for turn in c["turns"] for t in (turn["user"], turn["assistant"])]
            if not any(CHAT_LEAK.search(t) for t in turns):
                yield split, kind, turns


def heldout_examples():
    """Val only: Q&A about the held-out docs (data/eval/qa-heldout.jsonl), which pretraining never saw either."""
    path = DATA / "eval" / "qa-heldout.jsonl"
    if not path.exists():
        return
    for row in map(json.loads, open(path)):
        for fact in row["facts"]:
            if not LEAK.search(fact["answer"]):
                for q in fact["questions"]:
                    yield "val", "heldout", [q, fact["answer"]]
        if not LEAK.search(row["trick"]["answer"]):
            yield "val", "heldout-trick", [row["trick"]["question"], row["trick"]["answer"]]


def oasst_examples():
    for row in map(json.loads, open(DATA / "sft" / "oasst2.jsonl")):
        turns = [m["content"] for m in row["messages"]]
        long = [i for i in range(1, len(turns), 2) if len(turns[i]) > OASST_MAX_ANSWER]
        turns = turns[:long[0] - 1] if long else turns
        if turns:
            yield row["split"], "oasst", turns


def identity_examples():
    for questions, answers in ((IDENTITY_QUESTIONS, IDENTITY_ANSWERS), (LANGUAGE_QUESTIONS, LANGUAGE_ANSWERS)):
        for q in questions:
            for a in answers:
                yield "train", "identity", [q, a]


def main():
    meta = json.loads((DATA / "tokens" / "meta.json").read_text())
    enc = Encoder(spm.SentencePieceProcessor(model_file=str(OUT / "tokenizer" / f"{meta['tokenizer']}.model")))
    splits = {s: {"ids": [], "mask": [], "offsets": [0], "kinds": []} for s in ("train", "val")}
    stats = Counter()

    for gen in (chat_examples(), qa_examples(), heldout_examples(), oasst_examples(), identity_examples()):
        for split, kind, turns in gen:
            messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": t.strip()}
                        for i, t in enumerate(turns)]
            out = enc(messages)
            # long threads lose their last exchanges until they fit
            while out and len(out[0]) > MAX_LEN and len(messages) > 2:
                messages = messages[:-2]
                out = enc(messages)
            if out is None or len(out[0]) > MAX_LEN:
                stats[f"dropped {kind}"] += 1
                continue
            s = splits[split]
            s["ids"] += out[0]
            s["mask"] += out[1]
            s["offsets"].append(len(s["ids"]))
            s["kinds"].append(kind)
            stats[f"{split} {kind}"] += 1
            stats[f"{split} {kind} answer tokens"] += sum(out[1])

    for split, s in splits.items():
        np.savez(DATA / "sft" / f"{split}.npz", ids=np.array(s["ids"], dtype=np.uint16),
                 mask=np.array(s["mask"], dtype=bool), offsets=np.array(s["offsets"], dtype=np.int64),
                 kinds=np.array(s["kinds"]))
        print(f"{split}: {len(s['kinds']):,} examples, {len(s['ids']):,} tokens, {sum(s['mask']):,} answer tokens")
    for k, v in sorted(stats.items()):
        print(f"  {k:32} {v:,}")


if __name__ == "__main__":
    main()
