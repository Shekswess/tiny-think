"""Create DPO datasets from Dolci-Think-DPO-7B using source groups file."""

import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DATASET_NAME = "allenai/Dolci-Think-DPO-7B"
TOKENIZER_NAME = "facebook/MobileLLM-R1-140M-base"
OUTPUT_ROOT = Path("data")
MAX_TOKENS = 4096
PER_DATASET_TOKEN_BUDGET = 10_000_000
TOKEN_BUDGET_STRATEGY = "equal"  # "equal" or "proportional"
TOKEN_BUDGETS: Optional[Dict[str, int]] = None  # Optional explicit per-dataset budgets.
MAX_ROWS: Optional[int] = None  # Optional hard cap; leave None to prioritize token budgets.
SHUFFLE_SEED = 42
SFT_SOURCE_FILE = Path(__file__).resolve().parent / "sources" / "cleaned_source_values_sft.txt"
DPO_SOURCE_FILE = Path(__file__).resolve().parent / "sources" / "cleaned_source_values_dpo.txt"
SIMILAR_SOURCES_FILE = Path(__file__).resolve().parent / "sources" / "similar_sources.csv"

NUM_PROC = os.cpu_count() or 1
_TOKENIZER: Optional[PreTrainedTokenizerBase] = None

DPO_SECTION_TO_DATASET = {
    "MATH & STEM DATA": "Shekswess/tiny-think-dpo-math-n-stem",
}
MATH_N_STEM_DATASET = DPO_SECTION_TO_DATASET["MATH & STEM DATA"]
MATH_N_STEM_PRIMARY_SOURCE = "OpenThoughts3-full-filtered-science-no-cot"
MATH_N_STEM_MIN_OTHER_SHARE = 0.65  # Minimum token share from non-primary sources.
MATH_N_STEM_MAX_PRIMARY_SHARE = 0.35  # Maximum token share allowed for the primary source.
MATH_N_STEM_DIVERSITY_POLICY = "strict"  # "strict" (raise) or "adaptive" (shrink budget).
REFERENCE_DATASET_NAME = "Shekswess/tiny-think-dpo-math-n-stem"
USE_REFERENCE_PROPORTIONS = True
REFERENCE_PROPORTIONS_STRICT = True


def detect_source_column(columns: Iterable[str]) -> str:
    """
    Pick the source column from known candidates.

    Args:
        columns: Iterable of column names from the dataset.

    Returns:
        The detected source column name.

    Raises:
        ValueError: If neither `dataset_source` nor `source` is present.
    """
    column_list = list(columns)
    if "dataset_source" in column_list:
        return "dataset_source"
    if "source" in column_list:
        return "source"
    raise ValueError(
        f"Expected a 'dataset_source' or 'source' column, found {column_list}."
    )


def load_sft_source_values(source_file: Path) -> Set[str]:
    """
    Load SFT source values from the cleaned source values file.

    Args:
        source_file: Path to the cleaned SFT source values file.

    Returns:
        Set of SFT source strings.

    Raises:
        RuntimeError: If the source file is missing or malformed.
    """
    if not source_file.exists():
        raise RuntimeError(f"SFT source file not found: {source_file}")

    sources: Set[str] = set()
    current_section = None
    for raw_line in source_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_section = line[:-1].strip()
            continue
        if line.startswith("- "):
            if current_section is None:
                raise RuntimeError(
                    "Encountered a source entry before any section header."
                )
            source = line[2:].strip()
            if source:
                sources.add(source)
            continue
        raise RuntimeError(f"Unrecognized line in SFT source file: {raw_line!r}")

    if not sources:
        raise RuntimeError(f"No sources found in SFT source file: {source_file}")
    return sources


def load_source_aliases(source_file: Path) -> Dict[str, str]:
    """
    Load source alias mappings from a CSV file.

    Args:
        source_file: Path to the CSV file containing alias mappings.

    Returns:
        Mapping of DPO source to SFT source.
    """
    if not source_file.exists():
        logger.warning("Alias file not found: {}", source_file)
        return {}

    with source_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["dpo_source", "sft_source"]:
            raise RuntimeError(
                "Alias CSV must have header: dpo_source,sft_source; "
                f"found {reader.fieldnames}"
            )
        aliases: Dict[str, str] = {}
        for row in reader:
            dpo_source = (row.get("dpo_source") or "").strip()
            sft_source = (row.get("sft_source") or "").strip()
            if not dpo_source or not sft_source:
                raise RuntimeError(
                    f"Invalid alias row; both fields required: {row}"
                )
            aliases[dpo_source] = sft_source
    return aliases


def load_dpo_dataset_groups(source_file: Path) -> Dict[str, Tuple[str, ...]]:
    """
    Load DPO dataset groups from the cleaned source values file.

    Args:
        source_file: Path to the cleaned DPO source values file.

    Returns:
        Mapping of dataset name to a tuple of source strings.

    Raises:
        RuntimeError: If the source file is missing or malformed.
    """
    if not source_file.exists():
        raise RuntimeError(f"DPO source file not found: {source_file}")

    groups: Dict[str, List[str]] = {
        dataset_name: [] for dataset_name in DPO_SECTION_TO_DATASET.values()
    }
    current_section: Optional[str] = None
    for raw_line in source_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            header = line[:-1].strip()
            if header not in DPO_SECTION_TO_DATASET:
                raise RuntimeError(f"Unknown DPO section header: {header!r}")
            current_section = header
            continue
        if line.startswith("- "):
            if current_section is None:
                raise RuntimeError(
                    "Encountered a source entry before any section header."
                )
            source = line[2:].strip()
            if source:
                dataset_name = DPO_SECTION_TO_DATASET[current_section]
                groups[dataset_name].append(source)
            continue
        raise RuntimeError(f"Unrecognized line in DPO source file: {raw_line!r}")

    dataset_groups: Dict[str, Tuple[str, ...]] = {}
    for dataset_name, sources in groups.items():
        unique_sources = list(dict.fromkeys(sources))
        if not unique_sources:
            raise RuntimeError(
                f"DPO dataset group {dataset_name!r} has no sources in {source_file}."
            )
        dataset_groups[dataset_name] = tuple(unique_sources)
    return dataset_groups


def build_message_text(messages: List[Dict[str, str]]) -> str:
    """
    Convert messages to a simple text format for token counting.

    Args:
        messages: List of message dicts with `role` and `content`.

    Returns:
        Flattened text representation.
    """
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def count_tokens_for_messages(
    messages: object, tokenizer: PreTrainedTokenizerBase
) -> int:
    """
    Count tokens for either chat messages or raw text.

    Args:
        messages: Message list, list of strings, or raw text.
        tokenizer: Tokenizer used for token counting.

    Returns:
        Token count for the serialized content.

    Raises:
        ValueError: If the message type is unsupported.
    """
    if messages is None:
        return 0
    if isinstance(messages, str):
        return len(tokenizer(messages).input_ids)
    if isinstance(messages, list):
        if not messages:
            return 0
        if isinstance(messages[0], dict) and "role" in messages[0]:
            if getattr(tokenizer, "chat_template", None):
                token_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
                return len(token_ids)
            text = build_message_text(messages)
            return len(tokenizer(text).input_ids)
        text = "\n".join(str(item) for item in messages)
        return len(tokenizer(text).input_ids)
    raise ValueError(f"Unsupported message type for token counting: {type(messages)}")


def extract_prompt_from_pair(chosen: object, rejected: object) -> str:
    """Extract the shared user prompt from a chosen/rejected pair.

    Args:
        chosen: Chosen response content (expected list of chat messages).
        rejected: Rejected response content (expected list of chat messages).

    Returns:
        The shared user prompt content.

    Raises:
        ValueError: If a shared user prompt cannot be found or types are unsupported.
    """
    if isinstance(chosen, list) and isinstance(rejected, list):
        def _user_contents(messages: List[Dict[str, str]]) -> List[str]:
            return [
                msg.get("content", "")
                for msg in messages
                if isinstance(msg, dict) and msg.get("role") == "user"
            ]

        chosen_users = _user_contents(chosen)
        rejected_users = set(_user_contents(rejected))
        for content in chosen_users:
            if content in rejected_users and content:
                return content
        raise ValueError("No shared user message found in chosen/rejected pair.")
    raise ValueError(
        "Unsupported chosen/rejected types for prompt extraction: "
        f"{type(chosen)} / {type(rejected)}"
    )


def add_prompt_column(dataset: Dataset) -> Dataset:
    """Add a prompt column derived from shared user messages.

    Args:
        dataset: Dataset to augment.

    Returns:
        Dataset with an added prompt column.
    """
    def _map(batch: Dict[str, List]) -> Dict[str, List[str]]:
        prompts: List[str] = []
        for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
            try:
                prompts.append(extract_prompt_from_pair(chosen, rejected))
            except ValueError as exc:
                raise RuntimeError(
                    "Failed to extract shared user prompt from chosen/rejected pair."
                ) from exc
        return {"prompt": prompts}

    return dataset.map(_map, batched=True, batch_size=64, num_proc=NUM_PROC)


def add_token_counts(
    dataset: Dataset, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    """Add token counts to a dataset.

    Args:
        dataset: Dataset to augment.
        tokenizer: Tokenizer used for token counting.

    Returns:
        Dataset with added token count columns.
    """
    def _map(batch: Dict[str, List]) -> Dict[str, List[int]]:
        global _TOKENIZER
        if _TOKENIZER is None:
            _TOKENIZER = tokenizer
        chosen_counts = [
            count_tokens_for_messages(chosen, _TOKENIZER)
            for chosen in batch["chosen"]
        ]
        rejected_counts = [
            count_tokens_for_messages(rejected, _TOKENIZER)
            for rejected in batch["rejected"]
        ]
        max_counts = [
            max(chosen, rejected)
            for chosen, rejected in zip(chosen_counts, rejected_counts)
        ]
        total_counts = [
            chosen + rejected
            for chosen, rejected in zip(chosen_counts, rejected_counts)
        ]
        return {
            "chosen_token_count": chosen_counts,
            "rejected_token_count": rejected_counts,
            "max_token_count": max_counts,
            "token_count": total_counts,
        }

    return dataset.map(_map, batched=True, batch_size=64, num_proc=NUM_PROC)


def normalize_columns(dataset: Dataset, source_column: str) -> Dataset:
    """Ensure the dataset has the expected column names.

    Args:
        dataset: Dataset to normalize.
        source_column: Column name used for the dataset source.

    Returns:
        Dataset with required columns only.
    """
    if source_column != "dataset_source":
        dataset = dataset.rename_column(source_column, "dataset_source")

    keep = {"chosen", "rejected", "dataset_source", "token_count", "prompt"}
    drop = [col for col in dataset.column_names if col not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)
    return dataset


def output_path_for_dataset(dataset_name: str) -> Path:
    """Create a filesystem path for a dataset name.

    Args:
        dataset_name: Hugging Face dataset name (owner/name).

    Returns:
        Filesystem path under the data/ directory.
    """
    safe_name = dataset_name.replace("/", "__")
    return OUTPUT_ROOT / safe_name


def allocate_token_budgets(
    dataset_names: Sequence[str],
    total_budget: int,
    available_tokens: Dict[str, int],
    strategy: str = "equal",
    manual_budgets: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Allocate token budgets across datasets.

    Args:
        dataset_names: Names of datasets to allocate budgets for.
        total_budget: Total token budget across all datasets.
        available_tokens: Available tokens per dataset after filtering.
        strategy: Allocation strategy: "equal" or "proportional".
        manual_budgets: Optional explicit per-dataset budgets.

    Returns:
        Mapping of dataset name to token budget.

    Raises:
        ValueError: If an unknown strategy is provided or manual budgets are invalid.
    """
    if manual_budgets is not None:
        expected = set(dataset_names)
        provided = set(manual_budgets)
        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            raise ValueError(
                "Manual token budgets must cover all datasets. "
                f"Missing={sorted(missing)}, extra={sorted(extra)}"
            )
        total_manual = sum(manual_budgets.values())
        if total_manual != total_budget:
            logger.warning(
                "Manual token budgets sum to {}, but total budget is {}.",
                total_manual,
                total_budget,
            )
        return dict(manual_budgets)

    if total_budget <= 0:
        raise ValueError("Total token budget must be positive.")

    if strategy == "equal":
        per_budget = total_budget // len(dataset_names)
        remainder = total_budget % len(dataset_names)
        budgets = {}
        for idx, name in enumerate(dataset_names):
            budgets[name] = per_budget + (1 if idx < remainder else 0)
        return budgets

    if strategy == "proportional":
        total_available = sum(available_tokens.values())
        if total_available == 0:
            return {name: 0 for name in dataset_names}
        raw = {
            name: (total_budget * available_tokens[name]) / total_available
            for name in dataset_names
        }
        budgets = {name: int(raw[name]) for name in dataset_names}
        remainder = total_budget - sum(budgets.values())
        if remainder > 0:
            ordered = sorted(
                dataset_names, key=lambda n: raw[n] - budgets[n], reverse=True
            )
            for name in ordered[:remainder]:
                budgets[name] += 1
        return budgets

    raise ValueError(f"Unknown token budget strategy: {strategy}")


def compute_reference_source_shares(
    dataset_name: str,
    source_column: str = "dataset_source",
    token_column: str = "token_count",
) -> Dict[str, float]:
    """Compute per-source token shares from a reference dataset on the Hub.

    Args:
        dataset_name: Hugging Face dataset name to load.
        source_column: Column containing dataset source values.
        token_column: Column containing token counts.

    Returns:
        Mapping of source name to token share (sums to 1.0).

    Raises:
        RuntimeError: If the dataset cannot be loaded or lacks required columns.
    """
    try:
        ref = load_dataset(dataset_name, split="train", streaming=False)
    except Exception as exc:
        logger.exception("Failed loading reference dataset {}", dataset_name)
        raise RuntimeError(
            f"Failed loading reference dataset {dataset_name}"
        ) from exc

    columns = set(getattr(ref, "column_names", []))
    missing = {source_column, token_column} - columns
    if missing:
        raise RuntimeError(
            f"Reference dataset {dataset_name} missing columns: {sorted(missing)}"
        )

    totals: Dict[str, int] = {}
    total_tokens = 0
    for source, tokens in zip(ref[source_column], ref[token_column]):
        token_count = int(tokens)
        totals[source] = totals.get(source, 0) + token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError(
            f"Reference dataset {dataset_name} has zero total tokens."
        )

    shares = {source: total / total_tokens for source, total in totals.items()}
    return shares


def allocate_source_budgets(
    source_shares: Dict[str, float],
    total_budget: int,
) -> Dict[str, int]:
    """Allocate token budgets per source based on reference shares."""
    raw = {src: share * total_budget for src, share in source_shares.items()}
    budgets = {src: int(value) for src, value in raw.items()}
    remainder = total_budget - sum(budgets.values())
    if remainder > 0:
        ordered = sorted(raw, key=lambda s: raw[s] - budgets[s], reverse=True)
        for src in ordered[:remainder]:
            budgets[src] += 1
    return budgets


def downsample_by_source_budgets(
    dataset: Dataset,
    source_budgets: Dict[str, int],
    seed: int,
    source_column: str = "dataset_source",
    strict: bool = True,
) -> tuple[Dataset, int]:
    """Downsample a dataset to match per-source token budgets."""
    if dataset.num_rows == 0 or not source_budgets:
        return dataset.select([]), 0

    dataset_sources = set(dataset[source_column])
    missing = set(source_budgets) - dataset_sources
    extra = dataset_sources - set(source_budgets)
    if missing:
        raise RuntimeError(f"Missing sources in dataset: {sorted(missing)}")
    if extra:
        raise RuntimeError(f"Unexpected sources in dataset: {sorted(extra)}")

    parts: List[Dataset] = []
    total_tokens = 0
    for idx, (source, budget) in enumerate(source_budgets.items()):
        subset = dataset.filter(
            lambda row: row[source_column] == source,
            num_proc=NUM_PROC,
        ).shuffle(seed=seed + idx)
        selected, kept = downsample_to_token_budget(
            subset, budget, seed=seed + idx, source_column=source_column
        )
        if kept < budget:
            message = (
                f"Source {source} has {kept} tokens (< budget {budget})."
            )
            if strict:
                raise RuntimeError(message)
            logger.warning(message + " Using all available tokens.")
        if selected.num_rows:
            parts.append(selected)
        total_tokens += kept

    combined = concatenate_datasets(parts) if parts else dataset.select([])
    if combined.num_rows:
        combined = combined.shuffle(seed=seed)
    return combined, total_tokens


def downsample_to_token_budget(
    dataset: Dataset,
    token_budget: int,
    seed: int,
    preferred_sources: Optional[Set[str]] = None,
    source_column: str = "dataset_source",
) -> tuple[Dataset, int]:
    """
    Downsample a dataset to fit within a token budget.

    Args:
        dataset: Dataset to downsample.
        token_budget: Maximum total tokens to keep.
        seed: Random seed for shuffling.
        preferred_sources: Optional set of preferred source names.
        source_column: Column containing dataset source values.

    Returns:
        Tuple of (downsampled dataset, total tokens kept).
    """
    if dataset.num_rows == 0 or token_budget <= 0:
        return dataset.select([]), 0

    def _select_rows(dataset_slice: Dataset, budget: int) -> tuple[Dataset, int]:
        if dataset_slice.num_rows == 0 or budget <= 0:
            return dataset_slice.select([]), 0
        counts = dataset_slice["token_count"]
        total = 0
        keep = 0
        for count in counts:
            if total + count > budget:
                break
            total += count
            keep += 1
        if keep < dataset_slice.num_rows:
            dataset_slice = dataset_slice.select(range(keep))
        return dataset_slice, total

    if not preferred_sources:
        shuffled = dataset.shuffle(seed=seed)
        return _select_rows(shuffled, token_budget)

    preferred = dataset.filter(
        lambda row: row[source_column] in preferred_sources,
        num_proc=NUM_PROC,
    ).shuffle(seed=seed)
    remaining = dataset.filter(
        lambda row: row[source_column] not in preferred_sources,
        num_proc=NUM_PROC,
    ).shuffle(seed=seed)

    preferred, preferred_tokens = _select_rows(preferred, token_budget)
    remaining_budget = token_budget - preferred_tokens
    remaining_tokens = 0
    if remaining_budget > 0:
        remaining, remaining_tokens = _select_rows(remaining, remaining_budget)

    if preferred.num_rows and remaining.num_rows:
        combined = concatenate_datasets([preferred, remaining])
    elif preferred.num_rows:
        combined = preferred
    else:
        combined = remaining

    if combined.num_rows:
        combined = combined.shuffle(seed=seed)
    return combined, preferred_tokens + remaining_tokens


def downsample_with_min_other_share(
    dataset: Dataset,
    token_budget: int,
    seed: int,
    primary_source: str,
    min_other_share: float,
    max_primary_share: Optional[float],
    policy: str,
    preferred_sources: Optional[Set[str]] = None,
    source_column: str = "dataset_source",
) -> tuple[Dataset, int]:
    """
    Downsample a dataset while enforcing a minimum token share for non-primary sources.

    Args:
        dataset: Dataset to downsample.
        token_budget: Maximum total tokens to keep.
        seed: Random seed for shuffling.
        primary_source: Source to treat as primary (dominant) source.
        min_other_share: Minimum share of tokens from non-primary sources.
        max_primary_share: Optional maximum share of tokens from primary source.
        policy: "strict" to raise on violations, "adaptive" to shrink budget.
        preferred_sources: Optional set of preferred source names.
        source_column: Column containing dataset source values.

    Returns:
        Tuple of (downsampled dataset, total tokens kept).
    """
    if dataset.num_rows == 0 or token_budget <= 0:
        return dataset.select([]), 0

    if min_other_share <= 0:
        return downsample_to_token_budget(
            dataset,
            token_budget,
            seed,
            preferred_sources=preferred_sources,
            source_column=source_column,
        )
    if min_other_share >= 1:
        raise ValueError("min_other_share must be in the (0, 1) interval.")
    if max_primary_share is not None:
        if not 0 < max_primary_share < 1:
            raise ValueError("max_primary_share must be in the (0, 1) interval.")
        min_other_share = max(min_other_share, 1.0 - max_primary_share)
    if policy not in {"strict", "adaptive"}:
        raise ValueError(f"Unknown diversity policy: {policy}")

    primary = dataset.filter(
        lambda row: row[source_column] == primary_source,
        num_proc=NUM_PROC,
    )
    other = dataset.filter(
        lambda row: row[source_column] != primary_source,
        num_proc=NUM_PROC,
    )
    primary_total = sum(primary["token_count"]) if primary.num_rows else 0
    other_total = sum(other["token_count"]) if other.num_rows else 0
    if other_total == 0:
        raise RuntimeError(
            "No non-primary sources available for min-share enforcement. "
            f"primary_source={primary_source}"
        )

    total_target = min(token_budget, primary_total + other_total)
    max_total_allowed = int(other_total / min_other_share)
    if total_target > max_total_allowed:
        message = (
            "Non-primary sources have {} tokens, which is below the minimum "
            "share requirement of {:.0%} for target budget {}."
        )
        if policy == "strict":
            raise RuntimeError(message.format(other_total, min_other_share, total_target))
        logger.warning(
            message + " Reducing token budget to {}.",
            other_total,
            min_other_share,
            total_target,
            max_total_allowed,
        )
        total_target = max_total_allowed
        if total_target <= 0:
            return dataset.select([]), 0

    required_other_min = int(math.ceil(total_target * min_other_share))
    required_other_min = min(required_other_min, other_total)
    required_other_to_fill = max(required_other_min, total_target - primary_total)
    other_budget = min(other_total, required_other_to_fill)
    primary_budget = max(total_target - other_budget, 0)

    other_selected, other_tokens = downsample_to_token_budget(
        other,
        other_budget,
        seed=seed,
        preferred_sources=preferred_sources,
        source_column=source_column,
    )
    primary_selected, primary_tokens = downsample_to_token_budget(
        primary,
        primary_budget,
        seed=seed,
        preferred_sources=preferred_sources,
        source_column=source_column,
    )
    total_tokens = other_tokens + primary_tokens
    if total_tokens > 0:
        actual_share = other_tokens / total_tokens
    else:
        actual_share = 0.0

    if actual_share < min_other_share:
        message = "Actual non-primary share {:.2%} is below required {:.2%}."
        if other_tokens == 0:
            raise RuntimeError(
                "No non-primary tokens available after sampling; cannot enforce share."
            )
        logger.warning(message + " Trimming primary tokens.", actual_share, min_other_share)
        max_primary_tokens = int(
            math.floor(other_tokens * (1.0 - min_other_share) / min_other_share)
        )
        primary_budget = max(max_primary_tokens, 0)
        primary_selected, primary_tokens = downsample_to_token_budget(
            primary,
            primary_budget,
            seed=seed,
            preferred_sources=preferred_sources,
            source_column=source_column,
        )
        total_tokens = other_tokens + primary_tokens
        actual_share = other_tokens / total_tokens if total_tokens else 0.0
        if actual_share < min_other_share and policy == "strict":
            raise RuntimeError(message.format(actual_share, min_other_share))

    if other_selected.num_rows and primary_selected.num_rows:
        combined = concatenate_datasets([other_selected, primary_selected])
    elif other_selected.num_rows:
        combined = other_selected
    else:
        combined = primary_selected

    if combined.num_rows:
        combined = combined.shuffle(seed=seed)
    return combined, total_tokens


def main() -> None:
    """
    Create grouped DPO datasets, save them, and push to the Hub.

    Raises:
        RuntimeError: If the source dataset cannot be loaded.
    """
    dataset_groups = load_dpo_dataset_groups(DPO_SOURCE_FILE)
    if set(dataset_groups) != {MATH_N_STEM_DATASET}:
        raise RuntimeError(
            "DPO dataset groups must contain only the math+stem dataset. "
            f"Found: {sorted(dataset_groups)}"
        )
    sft_sources = load_sft_source_values(SFT_SOURCE_FILE)
    dpo_sources = {source for sources in dataset_groups.values() for source in sources}
    alias_map = load_source_aliases(SIMILAR_SOURCES_FILE)
    direct_overlap = sft_sources & dpo_sources
    mapped_overlap = {
        dpo_source
        for dpo_source, sft_source in alias_map.items()
        if dpo_source in dpo_sources and sft_source in sft_sources
    }
    overlap_sources = direct_overlap | mapped_overlap
    logger.info("Loaded {} SFT sources from {}", len(sft_sources), SFT_SOURCE_FILE)
    logger.info("Loaded {} DPO sources from {}", len(dpo_sources), DPO_SOURCE_FILE)
    logger.info("Overlap sources (SFT and DPO): {}", len(overlap_sources))
    logger.info(
        "Overlap breakdown: direct={}, alias={}",
        len(direct_overlap),
        len(mapped_overlap),
    )
    logger.info("Using {} dataset groups from DATASET_GROUPS", len(dataset_groups))
    for dataset_name, sources in dataset_groups.items():
        logger.info("Group {} has {} sources", dataset_name, len(sources))

    try:
        dataset = load_dataset(DATASET_NAME, split="train", streaming=False)
    except Exception as exc:
        logger.exception("Failed loading dataset {}", DATASET_NAME)
        raise RuntimeError(f"Failed loading dataset {DATASET_NAME}") from exc

    source_column = detect_source_column(getattr(dataset, "column_names", []))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    logger.info("Using num_proc={}", NUM_PROC)
    available_tokens: Dict[str, int] = {}
    prepared: Dict[str, Dataset] = {}
    reference_shares: Optional[Dict[str, float]] = None
    if USE_REFERENCE_PROPORTIONS:
        reference_shares = compute_reference_source_shares(REFERENCE_DATASET_NAME)
        logger.info(
            "Using reference proportions from {} ({} sources).",
            REFERENCE_DATASET_NAME,
            len(reference_shares),
        )

    for target_name, sources in dataset_groups.items():
        source_set = set(sources)
        logger.info("Building {} from {} sources", target_name, len(source_set))

        filtered = dataset.filter(
            lambda row: row[source_column] in source_set,
            num_proc=NUM_PROC,
        )
        logger.info("{} examples selected for {}", filtered.num_rows, target_name)

        filtered = add_prompt_column(filtered)
        filtered = add_token_counts(filtered, tokenizer)
        filtered = filtered.filter(
            lambda row: row["chosen_token_count"] <= MAX_TOKENS
            and row["rejected_token_count"] <= MAX_TOKENS,
            num_proc=NUM_PROC,
        )
        logger.info(
            "{} examples remain after {} token cutoff",
            filtered.num_rows,
            MAX_TOKENS,
        )
        if MAX_ROWS is not None and filtered.num_rows > MAX_ROWS:
            filtered = filtered.shuffle(seed=SHUFFLE_SEED).select(range(MAX_ROWS))
            logger.info("Downsampled to {} examples for {}", MAX_ROWS, target_name)
        total_tokens = sum(filtered["token_count"])
        available_tokens[target_name] = total_tokens
        prepared[target_name] = filtered
        logger.info("Available tokens for {}: {}", target_name, total_tokens)

    total_budget = (
        sum(TOKEN_BUDGETS.values())
        if TOKEN_BUDGETS is not None
        else PER_DATASET_TOKEN_BUDGET * len(prepared)
    )
    token_budgets = allocate_token_budgets(
        list(prepared.keys()),
        total_budget,
        available_tokens,
        strategy=TOKEN_BUDGET_STRATEGY,
        manual_budgets=TOKEN_BUDGETS,
    )
    logger.info("Per-dataset token budget target: {}", PER_DATASET_TOKEN_BUDGET)
    logger.info("Total token budget: {}", total_budget)
    logger.info("Token budget strategy: {}", TOKEN_BUDGET_STRATEGY)
    logger.info(
        "Math+STEM diversity: primary_source={}, min_other_share={:.0%}, "
        "max_primary_share={:.0%}, policy={}",
        MATH_N_STEM_PRIMARY_SOURCE,
        MATH_N_STEM_MIN_OTHER_SHARE,
        MATH_N_STEM_MAX_PRIMARY_SHARE,
        MATH_N_STEM_DIVERSITY_POLICY,
    )
    for name, budget in token_budgets.items():
        logger.info("Token budget for {}: {}", name, budget)

    token_totals: Dict[str, int] = {}
    for target_name, filtered in prepared.items():
        budget = token_budgets[target_name]
        if target_name != MATH_N_STEM_DATASET:
            raise RuntimeError(
                "Only the math+stem dataset is supported for DPO creation."
            )

        if USE_REFERENCE_PROPORTIONS:
            assert reference_shares is not None
            target_budget = min(budget, available_tokens[target_name])
            source_budgets = allocate_source_budgets(reference_shares, target_budget)
            filtered, kept_tokens = downsample_by_source_budgets(
                filtered,
                source_budgets,
                seed=SHUFFLE_SEED,
                source_column=source_column,
                strict=REFERENCE_PROPORTIONS_STRICT,
            )
        else:
            filtered, kept_tokens = downsample_with_min_other_share(
                filtered,
                budget,
                seed=SHUFFLE_SEED,
                primary_source=MATH_N_STEM_PRIMARY_SOURCE,
                min_other_share=MATH_N_STEM_MIN_OTHER_SHARE,
                max_primary_share=MATH_N_STEM_MAX_PRIMARY_SHARE,
                policy=MATH_N_STEM_DIVERSITY_POLICY,
                preferred_sources=overlap_sources,
                source_column=source_column,
            )
        if kept_tokens < budget and available_tokens[target_name] < budget:
            logger.warning(
                "{} has {} available tokens (< budget {}); using all available tokens.",
                target_name,
                available_tokens[target_name],
                budget,
            )
        elif kept_tokens < budget:
            logger.warning(
                "{} selected {} tokens (< budget {}).",
                target_name,
                kept_tokens,
                budget,
            )
        token_totals[target_name] = kept_tokens
        logger.info("Selected tokens for {}: {}", target_name, kept_tokens)
        prepared[target_name] = filtered
        filtered = normalize_columns(filtered, source_column)

        output_path = output_path_for_dataset(target_name)
        output_path.mkdir(parents=True, exist_ok=True)
        filtered.save_to_disk(str(output_path))
        logger.info("Saved {} to {}", target_name, output_path)

        try:
            filtered.push_to_hub(target_name)
            logger.info("Pushed {} to the Hub.", target_name)
        except Exception as exc:
            logger.exception("Failed pushing {} to the Hub.", target_name)
            raise RuntimeError(f"Failed pushing {target_name} to the Hub.") from exc
    if token_totals:
        logger.info("Token totals by dataset:")
        for name, total in token_totals.items():
            logger.info("- {}: {}", name, total)
        logger.info("Total tokens across datasets: {}", sum(token_totals.values()))

    if prepared and len(prepared) > 1:
        raise RuntimeError(
            "Expected only the math+stem dataset to be built; "
            f"found {len(prepared)} datasets."
        )


if __name__ == "__main__":
    main()
