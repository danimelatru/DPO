import os

import pytest

from src.utils.config import load_config


def test_config_loading():
    """Test that training configuration can be loaded"""
    config_path = "configs/train_config.yaml"
    if not os.path.exists(config_path):
        pytest.skip("Config file not found")

    config = load_config(config_path)
    assert "model" in config
    assert "training" in config
    assert "data" in config


def test_import_modules():
    """Test that all key modules can be imported without error"""
    try:
        from src.data import data_loader
        from src.training import train_dpo, trainer
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")
