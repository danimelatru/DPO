"""
DPO training script with configuration management
"""
import os
import argparse
import logging
from pathlib import Path

from src.utils.config import load_config, validate_config
from src.data.data_loader import load_dpo_dataset, validate_dataset
from src.training.trainer import DPOTrainerWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train DPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load and validate configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    validate_config(config)
    
    # Extract configuration
    model_name = config['model']['base_model']
    new_model_name = config['model']['new_model_name']
    data_path = config['data']['path']
    output_base = config['output']['dir']
    output_dir = os.path.join(output_base, new_model_name)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("="*60)
    logger.info("DPO Training Configuration")
    logger.info("="*60)
    logger.info(f"Base Model: {model_name}")
    logger.info(f"Output Model: {new_model_name}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*60)
    
    # Load dataset
    logger.info("Loading training dataset...")
    dataset = load_dpo_dataset(
        data_path=data_path,
        validation_split=config['data'].get('validation_split')
    )
    
    # Validate dataset
    if isinstance(dataset, dict):  # Has train/test split
        validate_dataset(dataset['train'])
        train_dataset = dataset['train']
    else:
        validate_dataset(dataset)
        train_dataset = dataset
    
    # Initialize trainer wrapper
    logger.info("Initializing DPO trainer...")
    trainer_wrapper = DPOTrainerWrapper(
        model_name=model_name,
        config=config
    )
    
    # Setup trainer
    trainer_wrapper.setup_trainer(
        train_dataset=train_dataset,
        output_dir=output_dir
    )
    
    # Train
    logger.info("Starting training...")
    trainer_wrapper.train()
    
    # Save model
    trainer_wrapper.save_model(output_dir)
    
    logger.info("="*60)
    logger.info("✅ Training completed successfully!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    train()
