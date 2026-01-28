#!/bin/bash
#SBATCH --job-name=agent_interactive
#SBATCH --output=logs/interactive_%j.out
#SBATCH --error=logs/interactive_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Error handling
set -e
set -u
set -o pipefail

# Load environment
source ~/.bashrc
conda activate research

# Navigate to project
cd /gpfs/workdir/fernandeda/dpo || exit 1

echo "========================================"
echo "Starting Interactive DPO Agent"
echo "========================================"

# Run interactive CLI
python -m src.inference.interactive_cli \
    --base-model meta-llama/Meta-Llama-3-8B-Instruct \
    --adapter models/Llama3-8B-Agent-DPO-v1 \
    --temperature 0.1 \
    --max-tokens 256

echo "========================================"
echo "Session ended at $(date)"
echo "========================================"
