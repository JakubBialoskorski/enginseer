"""Generate general chat SFT data with a local teacher model, so the model is a small general chat model rather
than a documentation Q&A bot.

  scripts/run_teacher.sh scripts/gen_chat.py              # dedicated llama-server for the teacher
  python scripts/gen_chat.py --url http://127.0.0.1:8096 --workers 4 --until 14:00

Three kinds of requests:
  grounded   a FineWeb-Edu or Cosmopedia passage from the pretraining corpus; the teacher writes 3 short
             conversations answered from that passage, so the facts are ones the model has seen
  skill      5 conversations of one category (small talk, things an offline model cannot do, short creative
             writing, invented people or books it should not make up, harmful requests, ...)
  language   10 messages in another language (Polish included); the answer, picked from fixed English
             replies in build_sft.py, asks the user to write in English, since the model only speaks English

Output: data/sft/chat.jsonl, one line per request. Re-running skips requests already done, so the run is
resumable. Requests are processed in hash order, so a partial run still mixes all kinds.
"""

import argparse
import hashlib
import json
import random
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from gen_qa import deadline
from paths import DATA
from train_tokenizer import is_eval

PROMPT_VERSION = 1
CORPUS = DATA / "corpus"
GROUNDED = {"fineweb-edu": 3000, "cosmopedia": 2000}
MIN_CHARS, MAX_CHARS = 800, 3000

ABOUT = """The assistant is Enginseer, a small English-only language model trained from scratch as a proof of \
concept by Jakub Białoskórski. It runs offline on the user's computer: it has no internet access, no clock, no \
memory of earlier chats, and it cannot open files, links or images. It may get facts wrong."""

SYSTEM_GROUNDED = """You write training data for a small, friendly English chat assistant. You get a passage of \
educational text. Write 3 short conversations in which a user asks something that the passage answers.

- The user message is natural and self-contained, as typed into a chat app: a question, a request to explain \
something simply, to give a few tips or a short list, or to compare two things. Vary the style: some short and \
casual, some lowercase. Ask about general facts, ideas and advice that make sense to someone who has never \
seen the passage, not about the particular people, quotes or events of one story or article. Never refer to \
"the passage", "the text" or "the article".
- The assistant answers with facts from the passage or very common knowledge only. The answer is correct, in \
plain and simple English, 1-4 sentences or a list of at most 4 short items, at most 80 words. No "according to", \
no disclaimers.
- One of the 3 conversations has a second exchange: a short follow-up from the user and the assistant's answer.
Everything is in English. Return JSON only."""

SYSTEM_SKILL = ABOUT + """

You write training data for this assistant. Write 5 different conversations of this kind: {category}
Where they fit, use these topics for variety: {topics}.

- User messages are natural and self-contained, as typed into a chat app; vary length and tone.
- Assistant answers are friendly, honest and in plain English, at most 80 words (a poem: at most 8 short lines). \
They never claim abilities the assistant does not have, such as having searched, checked or looked something up.
Everything is in English. Return JSON only."""

SYSTEM_LANGUAGE = """Write 10 different short messages in {language} that people might send to a chat \
assistant: greetings, everyday questions, requests for help or advice. Vary length, topic and tone; write as \
native speakers type. Return JSON only."""

SKILLS = {
    "greeting": (60, "greetings, small talk, thanks and goodbyes. The assistant replies warmly and briefly, and "
                     "offers to help."),
    "offline": (60, "requests that need live information or abilities the assistant lacks: today's date or "
                    "time, the weather, news, sports results, prices, opening hours, visiting a website, setting "
                    "a reminder, reading a file or an image. It briefly says it cannot do that and suggests "
                    "where to look."),
    "creative": (100, "short creative requests: a poem of 4-8 lines, a very short story under 80 words, a "
                      "joke, a riddle with its answer, ideas for a name or a slogan."),
    "writing": (80, "help with a short piece of writing that the user pastes into the message: fix the grammar, "
                    "make it more polite or formal, shorten it, or write a short message such as a thank-you "
                    "note, an apology or an invitation."),
    "advice": (100, "everyday practical questions: cooking basics, sleep, studying, exercise, saving money, "
                    "packing for a trip, cleaning, plants, pets. Common and safe advice only."),
    "explain": (100, "requests to explain a well-known school or everyday concept simply (science, history, "
                     "geography, language, nature), with widely known facts only."),
    "invented": (100, "questions about a person, book, film, place, product or event that you invent and that "
                      "does not exist, asked as if it were real and well known. The assistant says it is not "
                      "aware of it, makes nothing up, and asks for more context or suggests a reliable source."),
    "safety": (60, "requests for something harmful, dangerous or illegal, and questions that need a doctor, a "
                   "lawyer or a financial adviser. It declines harmful requests briefly and kindly; for medical, "
                   "legal or money questions it gives general, safe information and suggests a professional."),
    "self": (50, "questions about the assistant itself: whether it has feelings or opinions, its favorite "
                 "things, whether it is a human, how it works, what it is good and bad at. It answers honestly, "
                 "as described above."),
    "math": (60, "simple arithmetic or unit conversions with small numbers (one or two steps). The answer shows "
                 "the working in one line and gives the correct result."),
}
TOPICS = """school, work, family, friends, cooking, baking, coffee, tea, sleep, running, cycling, swimming, \
football, chess, music, guitar, piano, painting, photography, books, films, travel, trains, the sea, mountains, \
forests, rain, snow, winter, summer, autumn, spring, birthdays, holidays, gardening, houseplants, cats, dogs, \
birds, fish, horses, bees, the moon, the sun, stars, planets, volcanoes, rivers, deserts, dinosaurs, ancient \
Rome, ancient Egypt, castles, inventions, electricity, magnets, water, plants, the human body, the heart, teeth, \
vitamins, recycling, money, shopping, cars, bicycles, airports, maps, languages, writing, poetry, history \
homework, fractions, percentages, time zones, weather, clouds, rainbows, oceans, whales, robots, computers, \
phones, the internet, games, puzzles, museums, cities, villages, farms, bread, cheese, apples, soup, picnics, \
weddings, moving house, a new job, exams, motivation, friendship, kindness, patience""".split(", ")
LANGUAGES = {
    "Polish": 20, "German": 8, "French": 8, "Spanish": 8, "Italian": 8, "Portuguese": 8, "Dutch": 8, "Czech": 8,
    "Slovak": 6, "Ukrainian": 6, "Russian": 6, "Swedish": 6, "Turkish": 6, "Chinese": 6, "Japanese": 6,
    "Korean": 4, "Arabic": 4, "Hindi": 4,
}

S = {"type": "string", "minLength": 2}
CONVERSATIONS = {"type": "object", "required": ["conversations"], "properties": {"conversations": {
    "type": "array", "minItems": 0, "maxItems": 0, "items": {"type": "object", "required": ["turns"], "properties": {
        "turns": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
            "type": "object", "required": ["user", "assistant"], "properties": {"user": S, "assistant": S}}}}}}}}


def conversations_schema(n):
    s = json.loads(json.dumps(CONVERSATIONS))
    s["properties"]["conversations"].update(minItems=n, maxItems=n)
    return s


def grounded_requests():
    """Passages from the train split of the pretraining corpus (never the val split), spread over the files."""
    for name, n in GROUNDED.items():
        files = sorted((CORPUS / name).glob("**/*.parquet"))
        per_file = -(-n // len(files))
        for path in files:
            taken = 0
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=4096):
                for text in batch.column("text").to_pylist():
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    if int(digest[:8], 16) % 100 or is_eval(text, 0.001) or len(text) < MIN_CHARS:
                        continue
                    yield {"id": f"grounded/{name}/{digest[:16]}", "kind": "grounded", "source": name,
                           "passage": text[:text.rfind("\n", 0, MAX_CHARS)] if len(text) > MAX_CHARS else text}
                    taken += 1
                    if taken == per_file:
                        break
                if taken == per_file:
                    break


def skill_requests():
    for category, (n, _) in SKILLS.items():
        for i in range(n):
            yield {"id": f"skill/{category}/{i}", "kind": "skill", "category": category}


def language_requests():
    for language, n in LANGUAGES.items():
        for i in range(n):
            yield {"id": f"language/{language}/{i}", "kind": "language", "language": language}


def messages(req):
    if req["kind"] == "grounded":
        return SYSTEM_GROUNDED, req["passage"], conversations_schema(3), 1400
    if req["kind"] == "skill":
        topics = ", ".join(random.Random(req["id"]).sample(TOPICS, 5))
        system = SYSTEM_SKILL.format(category=SKILLS[req["category"]][1], topics=topics)
        return system, f"Write the 5 conversations ({req['id']}).", conversations_schema(5), 1400
    schema = {"type": "object", "required": ["messages"], "properties": {
        "messages": {"type": "array", "minItems": 10, "maxItems": 10, "items": S}}}
    return SYSTEM_LANGUAGE.format(language=req["language"]), f"Write the 10 messages ({req['id']}).", schema, 1000


def request(args, req):
    system, user, schema, max_tokens = messages(req)
    body = {
        "model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.8, "top_p": 0.9, "top_k": 40, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "chat", "schema": schema}},
    }
    r = urllib.request.Request(f"{args.url}/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        res = json.load(resp)
    choice = res["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError(f"finish_reason {choice['finish_reason']}")
    out = json.loads(choice["message"]["content"])
    meta = {k: v for k, v in req.items() if k != "passage"}
    return {**meta, "prompt_version": PROMPT_VERSION, "teacher": args.model, **out, "usage": res["usage"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--workers", type=int, default=1, help="parallel requests (match the server's -np)")
    parser.add_argument("--limit", type=int, help="stop after this many requests (pilot runs)")
    parser.add_argument("--until", help="HH:MM local time: send no new requests after that")
    parser.add_argument("--out", default=str(DATA / "sft" / "chat.jsonl"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out)} if out.exists() else set()
    reqs = [*grounded_requests(), *skill_requests(), *language_requests()]
    todo = sorted((r for r in reqs if r["id"] not in done), key=lambda r: hashlib.sha256(r["id"].encode()).digest())
    todo = todo[:args.limit] if args.limit else todo
    print(f"{len(done)} requests done, {len(todo)} to go", flush=True)

    stop_at = deadline(args.until)
    lock = threading.Lock()
    stats = {"ok": 0, "failed": 0, "out_tokens": 0}
    start = time.time()

    def work(req):
        if time.time() > stop_at:
            return
        for attempt in range(3):
            try:
                row = request(args, req)
                break
            except (ValueError, json.JSONDecodeError, OSError) as e:
                err = e
        else:
            with lock:
                stats["failed"] += 1
                print(f"  failed {req['id']}: {err}", flush=True)
            return
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            stats["out_tokens"] += row["usage"]["completion_tokens"]
            if stats["ok"] % 20 == 0:
                dt = time.time() - start
                eta = (len(todo) - stats["ok"] - stats["failed"]) * dt / stats["ok"] / 3600
                print(f"{stats['ok']}/{len(todo)} requests, {stats['failed']} failed, "
                      f"{stats['out_tokens'] / dt:.1f} out tok/s, eta {eta:.1f} h", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    print(f"done: {stats['ok']} written, {stats['failed']} failed, {(time.time() - start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
