#!/usr/bin/env bash
set -euo pipefail

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-20b}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://172.18.96.1:1234}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-medium}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1600}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"

run_judge() {
  local name="$1"
  local input="$2"
  local out="$3"
  local summary="$4"

  echo "==> Judging ${name}"
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
}

run_judge \
  "default" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_default.jsonl" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_default_judged.jsonl" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_default_judged_summary.json"

run_judge \
  "strict_citations" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_strict_citations.jsonl" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_strict_citations_judged.jsonl" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_strict_citations_judged_summary.json"

run_judge \
  "strict_citations_examples" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_strict_examples.jsonl" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_strict_examples_judged.jsonl" \
  "data/metrics/generation_eval_superhard_reranker_ctx_123_prompt_strict_examples_judged_summary.json"

echo "Done."
