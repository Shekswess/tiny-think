#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# LM-Eval (vLLM offline engine) -> local GPU
#
# Prereqs:
#   uv pip install "lm-eval[vllm]"
#
# Useful:
#   List tasks available in your lm-eval install:
#     lm_eval --tasks list
#
# IMPORTANT: For math evaluation (GSM8K, MATH500) on MobileLLM-R1 style models,
# use run_math_eval.sh instead! This script uses lm-evaluation-harness which
# does NOT properly handle \boxed{} answer extraction that MobileLLM-R1 requires.
#
# Use this script for: MMLU, ARC, PIQA, IFEval, BBH, GPQA, CommonsenseQA
# Use run_math_eval.sh for: GSM8K, MATH500 (math reasoning with \boxed{} format)
###############################################################################

# Prefer the console script if available; fall back to `python -m lm_eval`.
if command -v lm_eval >/dev/null 2>&1; then
  LM_EVAL_CMD=(lm_eval)
elif command -v python >/dev/null 2>&1; then
  LM_EVAL_CMD=(python -m lm_eval)
elif [[ -x ".venv/bin/python" ]]; then
  LM_EVAL_CMD=(".venv/bin/python" -m lm_eval)
else
  echo "ERROR: lm_eval not found. Activate .venv or install lm-eval." >&2
  exit 1
fi

# ----------------------------
# Model configuration
# ----------------------------

# HF model id (local path or hub). Keep facebook/MobileLLM-R1-140M-base unless explicitly changed.
MODEL_ID="${MODEL_ID:-google/gemma-3-270m-it}"

# Optional tokenizer override (defaults to MODEL_ID).
TOKENIZER_ID="${TOKENIZER_ID:-$MODEL_ID}"

# dtype for vLLM (auto/float16/bfloat16).
DTYPE="${DTYPE:-auto}"

# Single-GPU defaults (no distributed).
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"

# vLLM memory/sequence controls.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# ----------------------------
# Eval configuration
# ----------------------------

# Seed for vLLM sampling/reproducibility (used even with greedy).
SEED="${SEED:-1234}"

# Where to store outputs (each task gets its own subfolder)
OUT_DIR="${OUT_DIR:-./experiments/evals}"

# Log per-sample generations (can get big). Set to "0" to disable.
LOG_SAMPLES="${LOG_SAMPLES:-1}"

# Optional: cap the number of eval examples for quick smoke tests (empty = full).
# Example: LIMIT=50 ./run_eval_vllm.sh
LIMIT="${LIMIT:-}"

# Optional: generation controls.
# Defaults align to the math_eval.py settings you referenced.
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
DO_SAMPLE="${DO_SAMPLE:-True}"
N_SAMPLING="${N_SAMPLING:-1}"

# vLLM uses max_tokens; keep unset by default to avoid overriding task limits.
MAX_TOKENS="${MAX_TOKENS:-}"

# vLLM works best with auto batch sizing.
BATCH_SIZE="${BATCH_SIZE:-auto}"

# Few-shot setting (0 = zero-shot).
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

# If GEN_KWARGS is explicitly set, it wins. Otherwise build from the defaults above.
if [[ -n "${GEN_KWARGS:-}" ]]; then
  GEN_KWARGS="$GEN_KWARGS"
else
  GEN_KWARGS="temperature=${TEMPERATURE},top_p=${TOP_P},do_sample=${DO_SAMPLE},n=${N_SAMPLING}"
  if [[ -n "$MAX_TOKENS" ]]; then
    GEN_KWARGS+=",max_tokens=${MAX_TOKENS}"
  fi
fi

# ----------------------------
# Hugging Face auth
# ----------------------------

# Optional: provide an HF token for gated models.
# Example: HF_TOKEN=hf_... ./run_eval_vllm.sh 
if [[ -n "$HF_TOKEN" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

# Humaneval requires explicit opt-in to executing generated code.
# Set to "1" to allow code_eval; keep overridable from the environment.
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

# lm-eval requires explicit confirmation for unsafe tasks (e.g., Humaneval).
# Set to "1" to pass --confirm_run_unsafe_code; keep overridable from the environment.
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-1}"

# Chat templating is supported in lm-eval's vLLM backend; enable by default.
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"

# Optional: pass chat template args as JSON (e.g. {"enable_thinking": false}).
CHAT_TEMPLATE_ARGS="${CHAT_TEMPLATE_ARGS:-}"

# Optional: override thinking tags behavior for postprocessing.
ENABLE_THINKING="${ENABLE_THINKING:-}"
THINK_END_TOKEN="${THINK_END_TOKEN:-}"

# ----------------------------
# Post-training tasks (align with additional_knowledge/model.md)
# ----------------------------

# Post-training suite (reported settings: all 0-shot).
# Task names can vary by lm-eval version; verify with:
#   lm_eval --tasks list | grep -i aime
#
# NOTE: GSM8K and MATH500 are NOT included here because they require
# \boxed{} answer extraction. Use run_math_eval.sh for those tasks.
TASKS=(
  "mmlu" # Broad knowledge
  "openbookqa" # STEM QA
  "arc_easy" # STEM QA
  "arc_challenge" # STEM QA
  "commonsense_qa" # STEM-ish QA
  "gpqa_diamond_zeroshot" # STEM - Reasoning
  "piqa" # Reasoning
  "bbh_zeroshot" # Reasoning
  "ifeval" # Instruction following
)

# Optional: override tasks as a comma-separated list, e.g.:
#   TASKS_OVERRIDE="gsm8k,math_500" ./run_eval_vllm.sh
TASKS_OVERRIDE="${TASKS_OVERRIDE:-}"

# Resolve task list.
if [[ -n "$TASKS_OVERRIDE" ]]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_OVERRIDE"
fi

mkdir -p "$OUT_DIR"

MODEL_ARGS="pretrained=${MODEL_ID},tokenizer=${TOKENIZER_ID},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},seed=${SEED}"
if [[ -n "$MAX_MODEL_LEN" ]]; then
  MODEL_ARGS+=",max_model_len=${MAX_MODEL_LEN}"
fi
if [[ -n "$CHAT_TEMPLATE_ARGS" ]]; then
  MODEL_ARGS+=",chat_template_args=${CHAT_TEMPLATE_ARGS}"
fi
if [[ -n "$ENABLE_THINKING" ]]; then
  MODEL_ARGS+=",enable_thinking=${ENABLE_THINKING}"
fi
if [[ -n "$THINK_END_TOKEN" ]]; then
  MODEL_ARGS+=",think_end_token=${THINK_END_TOKEN}"
fi

echo "== lm-eval vLLM (offline engine) =="
echo "MODEL_ID:              $MODEL_ID"
echo "TOKENIZER_ID:          $TOKENIZER_ID"
echo "DTYPE:                 $DTYPE"
echo "TENSOR_PARALLEL_SIZE:  $TENSOR_PARALLEL_SIZE"
echo "DATA_PARALLEL_SIZE:    $DATA_PARALLEL_SIZE"
echo "GPU_MEMORY_UTILIZATION:$GPU_MEMORY_UTILIZATION"
echo "MAX_MODEL_LEN:         ${MAX_MODEL_LEN:-<unset>}"
echo "BATCH_SIZE:            $BATCH_SIZE"
echo "NUM_FEWSHOT:           $NUM_FEWSHOT"
echo "SEED:                  $SEED"
echo "OUT_DIR:               $OUT_DIR"
if [[ -n "$HF_TOKEN" ]]; then
  echo "HF_TOKEN:              <set>"
fi
if [[ -n "$TASKS_OVERRIDE" ]]; then
  echo "TASKS_OVERRIDE:        $TASKS_OVERRIDE"
fi
echo

run_one () {
  local task="$1"
  local out_task_dir="${OUT_DIR}/${task}"
  mkdir -p "$out_task_dir"

  echo "---- Running task: ${task} ----"
  local fewshot="$NUM_FEWSHOT"
  echo "NUM_FEWSHOT (task):     $fewshot"

  # Build command (space-separated args; no comma-separated --tasks list).
  cmd=("${LM_EVAL_CMD[@]}"
    --model vllm
    --tasks "$task"
    --model_args "$MODEL_ARGS"
    --batch_size "$BATCH_SIZE"
    --num_fewshot "$fewshot"
    --output_path "$out_task_dir"
    --gen_kwargs "$GEN_KWARGS"
  )

  if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
    cmd+=(--apply_chat_template)
  fi

  # Optional flags
  if [[ "$LOG_SAMPLES" == "1" ]]; then
    cmd+=(--log_samples)
  fi
  if [[ "$CONFIRM_RUN_UNSAFE_CODE" == "1" ]]; then
    cmd+=(--confirm_run_unsafe_code)
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  # Run; if a task name doesn't exist in your lm-eval version, skip it.
  if "${cmd[@]}"; then
    echo "✅ Done: ${task} -> ${out_task_dir}"
  else
    echo "⚠️  Skipping '${task}' (failed). If it's 'task not found', check:"
    echo "    lm_eval --tasks list | grep -i ${task%%_*}"
    echo
    return 0
  fi
  echo

}

# Run each task in order.
for task in "${TASKS[@]}"; do
  run_one "$task"
done

echo "All done. Results are under: $OUT_DIR"
