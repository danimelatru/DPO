"""
Data loading utilities for DPO training
"""

import logging
from typing import Dict, Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_dpo_dataset(
    data_path: str, split: str = "train", validation_split: Optional[float] = None
) -> Dataset:
    """
    Load DPO dataset from JSONL file with optional train/validation split.

    Args:
        data_path: Path to JSONL file
        split: Dataset split name
        validation_split: Fraction for validation (e.g., 0.1 for 10%)

    Returns:
        Dataset object ready for DPO training
    """
    logger.info(f"Loading dataset from {data_path}")

    dataset = load_dataset("json", data_files=data_path, split=split)

    if validation_split:
        logger.info(f"Splitting dataset with {validation_split*100}% for validation")
        split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
        return split_dataset

    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def format_dpo_example(example: Dict[str, str]) -> Dict[str, str]:
    """
    Validate and format a single DPO example.

    Args:
        example: Dict with 'prompt', 'chosen', 'rejected' keys

    Returns:
        Formatted example

    Raises:
        ValueError: If example is malformed
    """
    required_keys = ["prompt", "chosen", "rejected"]

    if not all(key in example for key in required_keys):
        raise ValueError(f"Example must contain keys: {required_keys}")

    # Ensure strings are non-empty
    for key in required_keys:
        if not isinstance(example[key], str) or not example[key].strip():
            raise ValueError(f"Key '{key}' must be a non-empty string")

    return {
        "prompt": example["prompt"].strip(),
        "chosen": example["chosen"].strip(),
        "rejected": example["rejected"].strip(),
    }


def validate_dataset(dataset: Dataset) -> bool:
    """
    Validate that all examples in dataset are properly formatted.

    Args:
        dataset: Dataset to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    logger.info("Validating dataset...")

    for idx, example in enumerate(dataset):
        try:
            format_dpo_example(example)
        except ValueError as e:
            logger.error(f"Invalid example at index {idx}: {e}")
            raise

    logger.info("Dataset validation passed")
    return True
