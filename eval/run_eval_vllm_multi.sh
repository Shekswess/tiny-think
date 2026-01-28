#!/usr/bin/env bash
set -euo pipefail

# Multi-model evaluation wrapper.
#
# Modes:
#   MODE=lm_eval   -> uses eval/run_eval_vllm.sh (default)
#   MODE=math_eval -> runs MobileLLM-R1 math methodology (gsm8k, math500)
#
# Examples:
#   ./eval/run_eval_vllm_multi.sh
#   OUT_DIR=./experiments/evals ./eval/run_eval_vllm_multi.sh
#   LIMIT=50 ./eval/run_eval_vllm_multi.sh
#
# Math eval:
#   MODE=math_eval ./eval/run_eval_vllm_multi.sh
#   MODE=math_eval MODEL_ID=facebook/MobileLLM-R1-140M ./eval/run_eval_vllm_multi.sh
#   MODE=math_eval TASKS=gsm8k LIMIT=100 ./eval/run_eval_vllm_multi.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-lm_eval}"
RUN_ONE="${REPO_ROOT}/eval/run_eval_vllm.sh"
MATH_EVAL="${REPO_ROOT}/eval/math_eval_vllm.py"

# Model IDs to evaluate.
MODELS=(
  "facebook/MobileLLM-R1-140M"
  "google/gemma-3-270m-it"
  "HuggingFaceTB/SmolLM2-135M-Instruct"
  "Shekswess/tiny-think-sft-math-stem-loss-dft-bf16-e2-bs8"
  "Shekswess/tiny-think-sft-math-stem-loss-dft-bf16-lr2e-5-e2-bs8"
  "Shekswess/tiny-think-sft-math-stem-loss-dft-bf16-lr5e-5-e2-bs8"
  "Shekswess/tiny-think-sft-math-stem-loss-nll-bf16-e2-bs8"
  "Shekswess/tiny-think-sft-math-stem-loss-nll-bf16-lr2e-5-e2-bs8"
  "Shekswess/tiny-think-sft-math-stem-loss-nll-bf16-lr5e-5-e2-bs8"
  "Shekswess/tiny-think-dpo-math-stem-apo_zero-beta0_3-lr3e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-apo_zero-beta1-lr3e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-apo_zero-beta0_5-lr3e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-dpo-beta1-lr3e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-dpo-beta1-lr5e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-dpo-beta1-lr1e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-dpo-beta2-lr2e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-dpo-beta1-lr2e-6-e1-bs8"
  "Shekswess/tiny-think-dpo-math-stem-dpo-beta0-5-lr2e-6-e1-bs8"
)

# Optional overrides: MODEL_ID (single) or MODELS_OVERRIDE (comma-separated).
if [[ -n "${MODEL_ID:-}" ]]; then
  MODELS=("$MODEL_ID")
elif [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_OVERRIDE"
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/evals}"
mkdir -p "$OUT_DIR"

if [[ "$MODE" == "math_eval" ]]; then
  if [[ ! -f "$MATH_EVAL" ]]; then
    echo "ERROR: Missing: $MATH_EVAL" >&2
    exit 1
  fi

  # Math eval defaults (MobileLLM-R1 methodology)
  TASKS="${TASKS:-gsm8k,math500}"
  TEMPERATURE="${TEMPERATURE:-0.6}"
  TOP_P="${TOP_P:-0.95}"
  SEED="${SEED:-0}"
  DTYPE="${DTYPE:-bfloat16}"
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
  SAVE_SAMPLES="${SAVE_SAMPLES:-1}"

  DEFAULT_MAX_TOKENS="${MAX_TOKENS_DEFAULT:-4096}"
  PERSONAL_MAX_TOKENS="${MAX_TOKENS_PERSONAL:-1024}"

  for model_id in "${MODELS[@]}"; do
    safe_name="${model_id//\//__}"
    model_out_root="${OUT_DIR}"

    # Per-model max token safety (unless explicitly overridden).
    if [[ -n "${MAX_TOKENS:-}" ]]; then
      resolved_max_tokens="$MAX_TOKENS"
    elif [[ "$model_id" == Shekswess/* ]]; then
      resolved_max_tokens="$PERSONAL_MAX_TOKENS"
    else
      resolved_max_tokens="$DEFAULT_MAX_TOKENS"
    fi

    echo "============================================================"
    echo "Model: $model_id"
    echo "Out:   $model_out_root"
    echo "Tasks:$TASKS"
    echo "============================================================"

    cmd=(python "$MATH_EVAL"
      --model "$model_id"
      --tasks "$TASKS"
      --output_dir "$model_out_root"
      --temperature "$TEMPERATURE"
      --top_p "$TOP_P"
      --max_tokens "$resolved_max_tokens"
      --seed "$SEED"
      --dtype "$DTYPE"
      --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    )

    if [[ "$SAVE_SAMPLES" == "1" ]]; then
      cmd+=(--save_samples)
    fi
    if [[ -n "${LIMIT:-}" ]]; then
      cmd+=(--limit "$LIMIT")
    fi

    "${cmd[@]}"
  done

  echo "All math models done. Results are under: $OUT_DIR"
  exit 0
fi

if [[ ! -x "$RUN_ONE" ]]; then
  echo "ERROR: Missing or non-executable: $RUN_ONE" >&2
  exit 1
fi

for model_id in "${MODELS[@]}"; do
  safe_name="${model_id//\//__}"
  model_out="${OUT_DIR}/${safe_name}"
  mkdir -p "$model_out"

  echo "============================================================"
  echo "Model: $model_id"
  echo "Out:   $model_out"
  echo "============================================================"

  MODEL_ID="$model_id" OUT_DIR="$model_out" bash "$RUN_ONE"
done

echo "All models done. Results are under: $OUT_DIR"
