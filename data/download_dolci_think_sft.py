"""Download Dolci-Think-SFT-7B and write unique source values."""
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset
from loguru import logger

DATASET_NAME = "allenai/Dolci-Think-SFT-7B"
SPLIT_NAME = "train"
OUTPUT_PATH = Path("data/source_values.txt")


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


def write_output(values: List[str], output_path: Path) -> None:
    """
    Write unique values to a file, one per line.

    Args:
        values: Sorted list of unique source values.
        output_path: Destination path for the output file.

    Raises:
        RuntimeError: If the output file cannot be written.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(values) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.exception(f"Failed writing output to {output_path}")
        raise RuntimeError(f"Failed writing output to {output_path}") from exc


def main() -> None:
    """
    Download the dataset and emit unique source values to disk.

    Raises:
        RuntimeError: If the dataset cannot be loaded or processed.
    """
    try:
        dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME, streaming=False)
    except Exception as exc:
        logger.exception(f"Failed loading dataset {DATASET_NAME}")
        raise RuntimeError(f"Failed loading dataset {DATASET_NAME}") from exc

    column = detect_source_column(getattr(dataset, "column_names", []))
    logger.info(f"Using column: {column}")

    try:
        unique_values = sorted(dataset.unique(column))
    except Exception as exc:
        logger.exception(f"Failed computing unique values for {column}")
        raise RuntimeError(f"Failed computing unique values for {column}") from exc

    logger.info(f"Found {len(unique_values)} unique values.")
    write_output(unique_values, OUTPUT_PATH)
    logger.info(f"Wrote unique source values to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
