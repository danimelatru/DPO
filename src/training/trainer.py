"""
Wrapper class for DPOTrainer with additional utilities
"""

import logging
from typing import Dict, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


class DPOTrainerWrapper:
    """
    High-level wrapper for DPO training with sensible defaults
    """

    def __init__(
        self, model_name: str, config: Dict, tokenizer: Optional[AutoTokenizer] = None
    ) -> None:
        """
        Initialize DPO trainer wrapper.

        Args:
            model_name: HuggingFace model identifier
            config: Training configuration dictionary
            tokenizer: Optional pre-loaded tokenizer
        """
        self.model_name = model_name
        self.config = config

        # Load tokenizer
        if tokenizer is None:
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Load model
        logger.info(f"Loading base model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Apply LoRA
        self._apply_lora()

        self.trainer = None

    def _apply_lora(self) -> None:
        """Apply LoRA configuration to model"""
        lora_config: Dict = self.config.get("lora", {})

        peft_config = LoraConfig(
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )

        logger.info(f"Applying LoRA with r={peft_config.r}")
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def setup_trainer(self, train_dataset: Dataset, output_dir: str) -> None:
        """
        Setup DPO trainer with dataset.

        Args:
            train_dataset: Training dataset
            output_dir: Directory for checkpoints
        """
        training_config = self.config.get("training", {})

        training_args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=training_config.get("batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 5e-6),
            logging_steps=training_config.get("logging_steps", 10),
            bf16=True,
            num_train_epochs=training_config.get("num_epochs", 3),
            warmup_ratio=training_config.get("warmup_ratio", 0.1),
            save_strategy="steps",
            save_steps=training_config.get("save_steps", 100),
            report_to=training_config.get("report_to", "none"),
            run_name=training_config.get("run_name", "agent-dpo"),
            remove_unused_columns=False,
            beta=training_config.get("beta", 0.1),
            max_prompt_length=training_config.get("max_prompt_length", 512),
            max_length=training_config.get("max_length", 1024),
        )

        logger.info("Initializing DPOTrainer")
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
        )

    def train(self) -> None:
        """Start training"""
        if self.trainer is None:
            raise RuntimeError("Trainer not setup. Call setup_trainer() first")

        logger.info("Starting DPO training...")
        self.trainer.train()
        logger.info("Training completed")

    def save_model(self, output_dir: str) -> None:
        """
        Save trained model.

        Args:
            output_dir: Directory to save model
        """
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
