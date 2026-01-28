#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Math Evaluation Script (MobileLLM-R1 methodology)
#
# This script evaluates reasoning models on GSM8K and MATH500 using the exact
# methodology from the MobileLLM-R1 paper:
#   https://github.com/facebookresearch/MobileLLM-R1/tree/main/evaluation
#
# It now delegates to eval/run_eval_vllm_multi.sh in MODE=math_eval.
#
# Usage:
#   ./eval/run_math_eval.sh
#   MODEL_ID=./my_model ./eval/run_math_eval.sh
#   TASKS=gsm8k,math500 ./eval/run_math_eval.sh
#   LIMIT=100 ./eval/run_math_eval.sh
###############################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_MULTI="${REPO_ROOT}/eval/run_eval_vllm_multi.sh"

if [[ ! -x "$RUN_MULTI" ]]; then
  echo "ERROR: Missing or non-executable: $RUN_MULTI" >&2
  exit 1
fi

# Activate venv if available
if [[ -d "${REPO_ROOT}/.venv" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export MODE="math_eval"

# Defaults (can be overridden via env vars)
export MODEL_ID="${MODEL_ID:-facebook/MobileLLM-R1-140M}"
export TASKS="${TASKS:-gsm8k}"
export OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/evals}"

export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-0.95}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export SEED="${SEED:-0}"
export DTYPE="${DTYPE:-bfloat16}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
export SAVE_SAMPLES="${SAVE_SAMPLES:-1}"

if [[ -n "${LIMIT:-}" ]]; then
  export LIMIT
fi

echo "== Math Evaluation (MobileLLM-R1 methodology) =="
echo "MODEL_ID:     $MODEL_ID"
echo "TASKS:        $TASKS"
echo "TEMPERATURE:  $TEMPERATURE"
echo "TOP_P:        $TOP_P"
echo "MAX_TOKENS:   $MAX_TOKENS"
echo "SEED:         $SEED"
echo "OUT_DIR:      $OUT_DIR"
if [[ -n "${LIMIT:-}" ]]; then
  echo "LIMIT:        $LIMIT"
fi
echo

bash "$RUN_MULTI"
