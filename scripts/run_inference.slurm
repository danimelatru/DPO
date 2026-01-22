#!/bin/bash
#SBATCH --job-name=agent_infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate meta-dpo
cd /gpfs/workdir/fernandeda/dpo

python src/inference.py

# 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