# Post-Training Alignment for Agentic AI using DPO

A project focused on **post-training alignment** of Small Language Models (SLMs) for **agentic tool use**. This repository implements an end-to-end pipeline to steer models towards strict "Chain-of-Thought" reasoning and structured JSON actions using **Direct Preference Optimization (DPO)**.

> **Research Goal:** Validate the efficacy of DPO in enforcing rigid schema compliance (JSON) and reasoning traces in lightweight models (1.1B–8B parameters) without relying on massive-scale supervised fine-tuning.

---

## 🚀 Key Features

- **Synthetic Agentic Dataset:** Automated generation of "Thought → Action" trajectories simulating tool use (Calculator, Search, Calendar).
- **DPO Fine-Tuning:** Implementation of Direct Preference Optimization to penalize hallucinations and unstructured outputs.
- **HPC Optimized:** Full SLURM integration for training on A100 clusters using mixed-precision (`bf16`).
- **Parameter Efficient:** Uses LoRA (Low-Rank Adaptation) techniques for efficient fine-tuning.
- **Interactive CLI:** Real-time testing interface for conversational agent interaction.

---

## 📂 Project Structure

```text
dpo/
├── configs/
│   ├── train_config.yaml
│   └── train_config_tinyllama.yaml
├── data/
│   └── processed/
├── logs/
├── models/
├── scripts/
│   ├── run_training.slurm
│   ├── run_inference.slurm
│   └── run_interactive.slurm
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_data.py
│   │   └── data_loader.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_dpo.py
│   │   └── trainer.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── inference_llama3.py
│   │   └── interactive_cli.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── prompts.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### Requirements
- Python 3.10+
- CUDA 11.8+
- 16GB+ GPU memory (24GB+ recommended for Llama 3)

### Setup

```bash
git clone https://github.com/danimelatru/dpo.git
cd dpo

conda create -n dpo python=3.10 -y
conda activate dpo

pip install -r requirements.txt

mkdir -p logs models data/processed
```

---

## 🧪 Usage

### 1. Generate Synthetic Training Data

```bash
python -m src.data.generate_data --output data/processed/dpo_data.jsonl --num-examples 2500 --seed 42
```

### 2. Training

```bash
python -m src.training.train_dpo --config configs/train_config_tinyllama.yaml
```

or via SLURM:

```bash
sbatch scripts/run_training.slurm
```

---

## 📊 Results

| Metric | Base Model | After DPO |
|------|-----------|-----------|
| JSON Validity | 0% | 95%+ |
| Thought Prefix Rate | 10% | 98%+ |
| Full Compliance | 0% | 93%+ |

---

## 📜 License

This project is open-sourced under the MIT License.
