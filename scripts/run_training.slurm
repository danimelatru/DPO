#!/bin/bash
#SBATCH --job-name=agent_dpo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ruche-gpu02,ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
source ~/.bashrc

# Activate the specific environment for this project
conda activate meta-dpo

# Define the root of your project
# IMPORTANT: Change this path to where you actually cloned/created the folder
PROJECT_ROOT=/gpfs/workdir/fernandeda/dpo
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Library fix (Kept from your template, good for ensuring CUDA libs load)
if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# Create directories for logs and results if they don't exist
mkdir -p logs
mkdir -p results
mkdir -p models

cd "$PROJECT_ROOT"

echo "STARTING DPO TRAINING"
echo "Project Root: $PROJECT_ROOT"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run the training script (using -u for unbuffered output so logs update in real-time)
python -u src/train_dpo.py