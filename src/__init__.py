"""
DPO Agent Training Framework - Post-training alignment for agentic AI
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data.generate_data import generate_entry
from .data.generate_data import main as generate_data
from .inference.interactive_cli import InteractiveAgent
from .training.train_dpo import train

__all__ = ["generate_entry", "generate_data", "train", "InteractiveAgent"]
