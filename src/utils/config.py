"""
Configuration management utilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return config


def save_config(config: Dict, output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving config to {output_path}")

    with open(output_path, "w") as f:
        if output_path.suffix == ".yaml" or output_path.suffix == ".yml":
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif output_path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {output_path.suffix}")


def validate_config(config: Dict) -> bool:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["model", "lora", "training", "data"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model section
    if "base_model" not in config["model"]:
        raise ValueError("Missing 'base_model' in model config")

    # Validate data section
    if "path" not in config["data"]:
        raise ValueError("Missing 'path' in data config")

    logger.info("Configuration validation passed")
    return True
