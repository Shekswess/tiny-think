"""Config-driven Supervised Fine-Tuning (SFT) with TRL + Transformers."""

import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

DEFAULT_ATTN_IMPLEMENTATION = "kernels-community/flash-attn2"


def _load_chat_template(config: Dict) -> str:
    """Resolve the chat template from config or the default Jinja file."""
    configured = config.get("chat_template")
    if configured:
        logger.info("Using chat template from config.")
        return configured

    template_path = Path(__file__).with_name("chat_template.jinja")
    try:
        template = template_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.exception("Failed reading default chat template at: {}", template_path)
        raise RuntimeError(
            f"Failed reading default chat template at {template_path}"
        ) from exc

    if not template.strip():
        raise ValueError(f"Default chat template is empty: {template_path}")

    logger.info("Using default chat template from: {}", template_path)
    return template


def _apply_special_tokens(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict
) -> None:
    """Add special tokens and resize embeddings when needed."""
    special_tokens_cfg = dict(config.get("special_tokens") or {})
    extra_tokens = list(special_tokens_cfg.get("additional_special_tokens") or [])

    tokens_to_add = sorted(set(extra_tokens))
    if tokens_to_add:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("Special tokens ensured: {}", tokens_to_add)

    eos_override = special_tokens_cfg.get("eos_token")
    if eos_override:
        eos_added = tokenizer.add_special_tokens({"eos_token": eos_override})
        if eos_added > 0:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("Overriding eos_token to: {}", eos_override)

    pad_override = special_tokens_cfg.get("pad_token")
    if pad_override:
        pad_added = tokenizer.add_special_tokens({"pad_token": pad_override})
        if pad_added > 0:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("Overriding pad_token to: {}", pad_override)

    if len(tokenizer) % 32 != 0:
        logger.warning(
            "Tokenizer length ({}) is not divisible by 32. "
            "Consider padding special tokens to a multiple of 32.",
            len(tokenizer),
        )


def _resolve_attn_implementation(config: Dict, model_kwargs: Dict) -> None:
    """Ensure attention implementation is set for kernel hub FlashAttention2."""
    configured = model_kwargs.get("attn_implementation") or config.get(
        "attn_implementation"
    )
    attn_impl = configured or DEFAULT_ATTN_IMPLEMENTATION
    model_kwargs["attn_implementation"] = attn_impl
    if attn_impl != DEFAULT_ATTN_IMPLEMENTATION:
        logger.warning(
            "Using attn_implementation={} instead of kernels-community/flash-attn2.",
            attn_impl,
        )
    else:
        logger.info("Using attn_implementation={}.", attn_impl)


def _coerce_torch_dtype(value: object) -> torch.dtype:
    """Normalize a torch dtype value from config."""
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half"}:
            return torch.float16
        if normalized in {"fp32", "float32"}:
            return torch.float32
    raise ValueError(f"Unsupported torch_dtype value: {value!r}")


def _resolve_torch_dtype(trainer_cfg: Dict, model_kwargs: Dict) -> torch.dtype:
    """Choose torch_dtype from model_kwargs or trainer precision flags."""
    configured = model_kwargs.get("dtype")
    if configured is not None:
        dtype = _coerce_torch_dtype(configured)
        model_kwargs["dtype"] = dtype
        logger.info("Using dtype from model_kwargs: {}", dtype)
        return dtype

    if trainer_cfg.get("bf16"):
        dtype = torch.bfloat16
    elif trainer_cfg.get("fp16"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_kwargs["dtype"] = dtype
    logger.info("Using dtype inferred from trainer precision flags: {}", dtype)
    return dtype


def _limit_dataset(
    data: Dataset,
    dataset_cfg: Dict,
    seed: Optional[int],
) -> Dataset:
    """Optionally shuffle and truncate the dataset for quick sanity runs."""
    max_samples = dataset_cfg.get("max_samples") or dataset_cfg.get("max_rows")
    if max_samples is None:
        return data

    if dataset_cfg.get("streaming"):
        raise RuntimeError("max_samples is not supported with streaming datasets.")

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")

    if dataset_cfg.get("shuffle"):
        shuffle_seed = dataset_cfg.get("shuffle_seed", seed or 42)
        data = data.shuffle(seed=int(shuffle_seed))

    if data.num_rows > max_samples:
        data = data.select(range(max_samples))

    logger.info(
        "Limited dataset to {} samples (shuffle={}).",
        min(max_samples, data.num_rows),
        bool(dataset_cfg.get("shuffle")),
    )
    return data


if __name__ == "__main__":

    # Parsing arguments (the config path)
    parser = argparse.ArgumentParser(description="Run SFT using a YAML config.")
    parser.add_argument(
        "--config-path", required=True, help="Path to YAML config file."
    )
    args = parser.parse_args()

    # Loading config file
    config_path = Path(args.config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    logger.info("Using config file at: {}", config_path)

    # Setting up seed
    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))

    # Getting model name and kwargs
    model_name = config.get("model_name")
    tokenizer_name = config.get("tokenizer_name") or model_name
    model_kwargs = dict(config.get("model_kwargs") or {})
    tokenizer_kwargs = dict(config.get("tokenizer_kwargs") or {})
    _resolve_attn_implementation(config, model_kwargs)

    # Loading the model and tokenizer
    logger.info("Loading model from: {}", model_name)
    trainer_cfg = dict(config.get("trainer") or {})
    _resolve_torch_dtype(trainer_cfg, model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, **model_kwargs, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    tokenizer.chat_template = _load_chat_template(config)

    _apply_special_tokens(model, tokenizer, config)

    # Setting pad token if not set
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if trainer_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Loading and preparing dataset
    try:
        dataset_cfg = config.get("dataset") or {}
        subset = dataset_cfg.get("subset")
        num_proc = dataset_cfg.get("num_proc")
        if num_proc is None and not dataset_cfg.get("streaming"):
            num_proc = multiprocessing.cpu_count()
        data = load_dataset(
            dataset_cfg.get("name"),
            subset if subset else None,
            split=dataset_cfg.get("split"),
            num_proc=num_proc,
            streaming=dataset_cfg.get("streaming"),
        )
        data = _limit_dataset(data, dataset_cfg, seed)
        logger.info("Loaded dataset: {}", dataset_cfg.get("name"))
        if dataset_cfg.get("streaming"):
            logger.info("Dataset streaming enabled; skipping sample log.")
        else:
            logger.info("Dataset sample: {}", data[0])
    except Exception as exc:
        logger.exception("Failed loading dataset: {}", dataset_cfg.get("name"))
        raise RuntimeError(
            f"Failed loading dataset: {dataset_cfg.get('name')}"
        ) from exc

    # Check the vocab size of the model and tokenizer
    model_vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    tokenizer_size = len(tokenizer)
    logger.info("Model config vocab size: {}", model_vocab_size)
    logger.info("Model embedding rows: {}", embedding_rows)
    logger.info("Tokenizer length: {}", tokenizer_size)
    if model_vocab_size and model_vocab_size != tokenizer_size:
        logger.warning(
            "Tokenizer size ({}) != model config vocab size ({}).",
            tokenizer_size,
            model_vocab_size,
        )
    if embedding_rows != tokenizer_size:
        logger.warning(
            "Tokenizer size ({}) != embedding rows ({}).",
            tokenizer_size,
            embedding_rows,
        )

    # Setting up trainer and training
    if "use_liger_kernel" not in trainer_cfg:
        trainer_cfg["use_liger_kernel"] = True
        logger.info("use_liger_kernel not set; defaulting to True.")
    elif not trainer_cfg.get("use_liger_kernel"):
        logger.warning("use_liger_kernel is disabled; Liger kernels are recommended.")

    adam_epsilon = trainer_cfg.get("adam_epsilon")
    if isinstance(adam_epsilon, str):
        try:
            trainer_cfg["adam_epsilon"] = float(adam_epsilon)
            logger.warning(
                "Coerced adam_epsilon string to float: {}",
                trainer_cfg["adam_epsilon"],
            )
        except ValueError as exc:
            raise ValueError(
                f"trainer.adam_epsilon must be a float, got {adam_epsilon!r}"
            ) from exc

    training_arguments = SFTConfig(**trainer_cfg)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=training_arguments,
    )

    trainer.train()
    logger.info("Training done.")
