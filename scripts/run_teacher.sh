#!/usr/bin/env bash
# Runs SFT data generation with a dedicated llama-server for the teacher, then stops the server.
#
#   scripts/run_teacher.sh --until 14:00     # extra arguments go to gen_qa.py; the same command resumes
#   scripts/run_teacher.sh scripts/eval_judge.py judge sft     # any other script that takes --url and --workers
#
# Needs in .env:
#   TEACHER_GGUF=/path/to/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
# Optional in .env:
#   LLAMA_CPP=/opt/llama.cpp   TEACHER_NP=4   TEACHER_NCMOE=20
#
# Free the GPU first (e.g. unload llama-swap: curl localhost:8081/unload). On a 12 GB card the MoE experts of
# the first TEACHER_NCMOE layers stay on the CPU, which is the bottleneck: 4 slots give ~40 output tok/s
# in total, vs ~30 for a single request.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] && . ./.env
LLAMA_CPP=${LLAMA_CPP:-/opt/llama.cpp}
NP=${TEACHER_NP:-4}
NCMOE=${TEACHER_NCMOE:-20}
PORT=8096
: "${TEACHER_GGUF:?set TEACHER_GGUF in .env}"

SCRIPT=scripts/gen_qa.py
if [[ "${1:-}" == *.py ]]; then SCRIPT=$1; shift; fi

mkdir -p data/sft
"$LLAMA_CPP/build/bin/llama-server" --host 127.0.0.1 --port $PORT -m "$TEACHER_GGUF" \
    -ngl 99 --n-cpu-moe "$NCMOE" -fa on -ctk q8_0 -ctv q8_0 -np "$NP" -c $((NP * 6144)) -t "$(nproc)" \
    --jinja --no-webui > data/sft/teacher-server.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for _ in $(seq 300); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "llama-server exited, see data/sft/teacher-server.log"; exit 1; }
    sleep 2
done

.venv/bin/python "$SCRIPT" --url "http://127.0.0.1:$PORT" --workers "$NP" "$@"
