import pytest
from datasets import Dataset

from src.data.data_loader import format_dpo_example, validate_dataset


def test_format_dpo_example_valid():
    """Test formatting a valid example"""
    example = {"prompt": "User: help", "chosen": "Action: help", "rejected": "I cannot help"}
    formatted = format_dpo_example(example)
    assert formatted["prompt"] == "User: help"
    assert formatted["chosen"] == "Action: help"
    assert formatted["rejected"] == "I cannot help"


def test_format_dpo_example_missing_keys():
    """Test validation of missing keys"""
    example = {"prompt": "User: help", "chosen": "Action: help"}
    with pytest.raises(ValueError, match="Example must contain keys"):
        format_dpo_example(example)


def test_format_dpo_example_empty_strings():
    """Test validation of empty strings"""
    example = {"prompt": " ", "chosen": "Action: help", "rejected": "I cannot help"}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        format_dpo_example(example)


def test_validate_dataset_valid():
    """Test full dataset validation"""
    data = {"prompt": ["p1", "p2"], "chosen": ["c1", "c2"], "rejected": ["r1", "r2"]}
    dataset = Dataset.from_dict(data)
    assert validate_dataset(dataset) is True


def test_validate_dataset_invalid():
    """Test failure on invalid dataset"""
    data = {
        "prompt": ["p1", " "],  # second one is empty/invalid
        "chosen": ["c1", "c2"],
        "rejected": ["r1", "r2"],
    }
    dataset = Dataset.from_dict(data)
    with pytest.raises(ValueError):
        validate_dataset(dataset)
