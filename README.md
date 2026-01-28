<p align="center">
  <img src="assets/logo.png" alt="Tiny Think Logo" width="350" />
</p>

<p align="center">
  <strong>Tiny Think</strong><br/>
  Reasoning-first post-training for tiny language models (140M) on a single GPU.
</p>

<p align="center">
  <a href="assets/paper.pdf">
    <img src="https://img.shields.io/badge/paper-PDF-blue?style=for-the-badge" alt="Paper PDF">
  </a>
  <a href="https://huggingface.co/collections/Shekswess/tiny-think">
    <img src="https://img.shields.io/badge/Hugging%20Face-collection-ffcc00?style=for-the-badge" alt="HF Collection">
  </a>
</p>

---

## About

**Tiny Think** is the official research codebase for the paper:

> **Tiny Think: Reasoning-First Post-Training for Tiny Math and STEM Language Models**

The goal of this repo is simple:

> **Understand what post-training actually does to reasoning in very small language models under strict hardware constraints.**

Everything here is designed to be:
- minimal
- reproducible
- runnable on a single consumer GPU

---

## Key ideas (from the paper)

- Full fine-tuning of a **140M parameter model** can yield non-trivial math reasoning ability.
- Reasoning-focused **SFT alone** already reaches strong GSM8K performance for this scale.
- **Preference optimization (DPO / APO)** mainly acts as *calibration*, not free capability gain.
- Improving math accuracy often comes with a **tradeoff in general reasoning** (the “general reasoning tax”).

---

## Constraints (by design)

- **Single machine only**
- **Single GPU** (RTX 5060 Ti, 16 GB VRAM)
- **No distributed training**
- **No LoRA / PEFT**
- **Full fine-tuning only**

These constraints are intentional and reflect the experimental setup in the paper.

---

## Base model

```
facebook/MobileLLM-R1-140M-base
```

---

## Training overview

Tiny Think uses a simple **two-stage post-training recipe**:

**Stage A — Supervised Fine-Tuning (SFT)**
- Math + STEM data with explicit reasoning traces (`<think>`)
- ~60M tokens
- NLL or DFT objectives

**Stage B — Preference Optimization**
- Math/STEM preference pairs
- ~10M tokens
- DPO or APO-zero

Stage B sharpens solution selection but may reduce broad reasoning ability.

---

## Evaluation

Evaluation is done with:
- **vLLM** for inference
- **lm-eval** for benchmarks

Benchmarks include GSM8K, MATH500, BBH, IFEval, and several STEM tasks.

All evaluation settings match those used in the paper.

---

## Repository layout

```
assets/                # logo + paper PDF
configs/               # YAML configs used in experiments
  sft/
  dpo/
data/                  # dataset preparation utilities
  sources/
train/                 # training scripts
  sft.py
  dpo.py
eval/                  # evaluation entrypoints (vLLM + lm-eval)
```

---

## Setup

This repo uses **Python 3.12 + uv**.

```bash
uv venv .venv --python=3.12 --seed
source .venv/bin/activate
```

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

---

## Running

**SFT**
```bash
python train/sft.py --config-path configs/sft/math_stem_nll_bf16.yaml
```

**Preference optimization**
```bash
python train/dpo.py --config-path configs/dpo/math_stem_dpo_beta1_lr3e_6_e1_bs8.yaml
```

**Evaluation**
```bash
./eval/run_eval_vllm_multi.sh
```

Math-only mode:
```bash
MODE=math_eval ./eval/run_eval_vllm_multi.sh
```

---

## What this repo is (and isn’t)

✔ Research code for controlled experiments on tiny models

✘ Production system
✘ General-purpose chatbot
✘ Multi-GPU training framework

---

## Citation

If you use this repository, please cite the paper.

```bibtex
@article{jakimovski2026tinythink,
  title={Tiny Think: Reasoning-First Post-Training for Tiny Math and STEM Language Models},
  author={Jakimovski, Bojan and Ilijoski, Bojan},
  year={2026}
}
```

---

## License

Apache-2.0. See `LICENSE`.
