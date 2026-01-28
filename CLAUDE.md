# Tiny Think Instruction

## Purpose

This repository is designed for **full fine-tuning of Tiny Models** on **a single local Blackwell GPU** using:

- Torch & Hugging Face Transformers
- TRL (SFT, DPO)
- vLLM (CUDA 12.8 backend)
- llm-evaluation-harness (evaluation)

---

## Hardware (CRITICAL)

All experiments run on **one local machine only**:

- GPU: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, Blackwell)
- CPU: AMD Ryzen 7 9700X
- RAM: 32 GB
- GPUs: 1 (no distributed or multi-GPU training)

Assume:
- No DeepSpeed
- No FSDP
- No multi-node or multi-GPU setups
- Everything must fit in **16 GB VRAM**

---

## Python Environment (STRICT)

This repository **does NOT support arbitrary installs**.

You **must** use:
- Python **3.12**
- `uv`
- A local `.venv`
- A **strict installation order**

### Environment Rules

1. **If `.venv` exists**
   - Activate it
   - Do NOT recreate it
   - Do NOT reinstall packages unless debugging

2. **If `.venv` does NOT exist**
   - Create it
   - Install dependencies **step by step in the exact order below**

---

## Environment Setup (AUTHORITATIVE)

### Step 1: Create or Activate venv

```bash
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  uv venv .venv --python=3.12 --seed
  source .venv/bin/activate
fi
```

---

### Step 2: Install Dependencies (ORDER MATTERS)

```bash
uv pip install "lm-eval[api]"
uv pip install langdetect immutabledict
uv pip install sympy math_verify antlr4-python3-runtime==4.11
uv pip install -U vllm --torch-backend=cu128
uv pip install trl
uv pip install liger-kernel
uv pip install kernels
uv pip install wandb
```

Do **not**:
- Collapse this into `requirements.txt`
- Change the order
- Downgrade Python
- Mix pip/conda installs

If something breaks, assume **install order or CUDA backend mismatch first**.

---

## Supported Training Approaches

This repository supports 

### Full Fine-Tuning
- Full weight updates (no adapters)

Full fine-tuning is:
- Allowed
- Experimental
- Heavily constrained by VRAM

Because of the nature of the model full fine-tuning is **preferable by default** for quality!

---

## Supported Post-Training Algorithms

- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)

All training must:
- Run on **a single GPU**
- Be memory-aware
- Avoid distributed assumptions

---

## Documentation (AUTHORITATIVE)

Use these links (and links from them) as the **single source of truth** for implementation decisions, troubleshooting, understanding, and documentation.

### TRL Documentation
- SFT Trainer: https://huggingface.co/docs/trl/en/sft_trainer.md
- DPO Trainer: https://huggingface.co/docs/trl/en/dpo_trainer.md
- Liger Kernels: https://huggingface.co/docs/trl/en/liger_kernel_integration.md
- Kernels Hub: https://huggingface.co/docs/trl/en/kernels_hub.md
- Chat Template Utils: https://huggingface.co/docs/trl/en/chat_template_utils.md
- Data Utils: https://huggingface.co/docs/trl/en/data_utils.md
- Model Utils: https://huggingface.co/docs/trl/en/model_utils.md

### llm-evaluation-harness
- Documentation: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs#readme

### vLLM
- Online Serving Parameters: https://docs.vllm.ai/en/latest/cli/serve/

---

## Evaluation + Contamination Rules

- Use `eval/run_eval_vllm_multi.sh` as the main entrypoint:
  - Default `MODE=lm_eval` uses `eval/run_eval_vllm.sh` (lm-eval, offline vLLM)
  - `MODE=math_eval` uses `eval/math_eval_vllm.py` (GSM8K, MATH500; boxed-answer parsing)
- Never train on benchmark test data (questions or answers) used in eval tasks
- Do not edit evaluation scripts or task lists unless explicitly asked


---

## Model Selection 

- All the experiments must use `facebook/MobileLLM-R1-140M-base` as the base model.

---

## Agent Rules

- Assume **staff-level ML engineering context**
- Prefer correctness over convenience
- Never introduce distributed complexity
- Never reorder installs
- If it doesn’t fit on **RTX 5060 Ti (16 GB)** → it’s out of scope

---
