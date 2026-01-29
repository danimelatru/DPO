#!/bin/bash
#SBATCH --job-name=agent_train_dpo
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Error handling
set -e
set -o pipefail

# Load environment
source ~/.bashrc
set -u

conda activate research

# Navigate to project directory
cd /gpfs/workdir/fernandeda/dpo || exit 1

# Log environment information
echo "========================================"
echo "Environment Information"
echo "========================================"
nvidia-smi
echo "----------------------------------------"
python --version
echo "----------------------------------------"
pip list | grep -E "(transformers|torch|trl|peft)"
echo "========================================"

# Create logs directory
mkdir -p logs

# Run training with config
echo "Starting DPO training..."
python -m src.training.train_dpo --config configs/train_config.yaml

# Notify completion
echo "========================================"
echo "Training completed at $(date)"
echo "========================================"