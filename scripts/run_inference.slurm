#!/bin/bash
#SBATCH --job-name=agent_infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Error handling
set -e
set -o pipefail

# Load environment
source ~/.bashrc
set -u

conda activate research

# Navigate to project
cd /gpfs/workdir/fernandeda/dpo || exit 1

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Running DPO Agent Inference"
echo "========================================"

# Choose which inference script to run
# For TinyLlama:
# python -m src.inference.inference --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter models/TinyLlama-Agent-DPO-v1

# For Llama 3:
python -m src.inference.inference_llama3 --base-model meta-llama/Meta-Llama-3-8B-Instruct --adapter models/Llama3-8B-Agent-DPO-v1

echo "========================================"
echo "Inference completed at $(date)"
echo "========================================"
