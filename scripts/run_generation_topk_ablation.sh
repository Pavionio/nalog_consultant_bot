#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export RAG_RERANKER_MODEL_CACHE="${RAG_RERANKER_MODEL_CACHE:-0}"

DATASET="${DATASET:-eval_superhard_dataset.jsonl}"
METHOD="${METHOD:-reranker}"
CONTEXT_ORDER="${CONTEXT_ORDER:-rerank_123}"
PROMPT_VARIANT="${PROMPT_VARIANT:-strict_citations_examples}"

TOP_K_VALUES="${TOP_K_VALUES:-4 6 8 12}"
MAX_TOP_K="${MAX_TOP_K:-12}"
RERANKER_FETCH_K="${RERANKER_FETCH_K:-50}"

GENERATOR_MODEL="${GENERATOR_MODEL:-qwen/qwen3-14b}"
GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-http://172.18.96.1:1234}"
GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0}"
GENERATOR_MAX_TOKENS="${GENERATOR_MAX_TOKENS:-1500}"

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-20b}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://172.18.96.1:1234}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-medium}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1600}"

PREFIX="${PREFIX:-data/metrics/generation_eval_superhard_reranker_prompt_strict_examples_fetch50}"
RETRIEVAL_CACHE="${RETRIEVAL_CACHE:-${PREFIX}_topk${MAX_TOP_K}_retrieval_cache.jsonl}"
PAUSE_BETWEEN_STAGES="${PAUSE_BETWEEN_STAGES:-1}"

pause_between_stages() {
  local message="$1"
  if [[ "${PAUSE_BETWEEN_STAGES}" == "1" ]]; then
    read -r -p "${message}"
  fi
}

echo "==> Retrieval/rerank once: top_k=${MAX_TOP_K}, reranker_fetch_k=${RERANKER_FETCH_K}"
uv run python3 scripts/build_generation_eval_dataset.py \
  --dataset "${DATASET}" \
  --out "${PREFIX}_topk${MAX_TOP_K}_retrieval_only.placeholder.jsonl" \
  --method "${METHOD}" \
  --top-k "${MAX_TOP_K}" \
  --reranker-fetch-k "${RERANKER_FETCH_K}" \
  --context-order "${CONTEXT_ORDER}" \
  --retrieval-cache-out "${RETRIEVAL_CACHE}" \
  --retrieval-only \
  --force

pause_between_stages "Retrieval/rerank complete. Press Enter to start LLM generation..."

echo "==> LLM generation from one retrieval cache"
for top_k in ${TOP_K_VALUES}; do
  out="${PREFIX}_topk${top_k}.jsonl"
  echo "==> Generating top_k=${top_k}: ${out}"
  uv run python3 scripts/build_generation_eval_dataset.py \
    --dataset "${DATASET}" \
    --out "${out}" \
    --method "${METHOD}" \
    --top-k "${top_k}" \
    --reranker-fetch-k "${RERANKER_FETCH_K}" \
    --context-order "${CONTEXT_ORDER}" \
    --retrieval-cache-in "${RETRIEVAL_CACHE}" \
    --prompt-variant "${PROMPT_VARIANT}" \
    --generator-model "${GENERATOR_MODEL}" \
    --generator-base-url "${GENERATOR_BASE_URL}" \
    --generator-temperature "${GENERATOR_TEMPERATURE}" \
    --generator-max-tokens "${GENERATOR_MAX_TOKENS}" \
    --force
done

pause_between_stages "LLM generation complete. Press Enter to start judge validation..."

echo "==> Judge validation"
for top_k in ${TOP_K_VALUES}; do
  input="${PREFIX}_topk${top_k}.jsonl"
  out="${PREFIX}_topk${top_k}_judged.jsonl"
  summary="${PREFIX}_topk${top_k}_judged_summary.json"
  echo "==> Judging top_k=${top_k}: ${summary}"
  uv run python3 scripts/evaluate_generation_judges.py \
    --input "${input}" \
    --out "${out}" \
    --summary-out "${summary}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --judge-reasoning-effort "${JUDGE_REASONING_EFFORT}" \
    --judge-temperature "${JUDGE_TEMPERATURE}" \
    --judge-max-tokens "${JUDGE_MAX_TOKENS}" \
    --judge-examples off \
    --validators precision completeness format \
    --force
done

echo "Done."
