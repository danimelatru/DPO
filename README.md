# Post-Training Alignment for Agentic AI using DPO

A research project focused on **post-training alignment** of Small Language Models (SLMs) for **agentic tool use**. This repository implements an end-to-end pipeline to steer models towards strict "Chain-of-Thought" reasoning and structured JSON actions using **Direct Preference Optimization (DPO)**.

> **Research Goal:** Validate the efficacy of DPO in enforcing rigid schema compliance (JSON) and reasoning traces in lightweight models (1.1B parameters) without relying on massive-scale supervised fine-tuning.

## 🚀 Key Features

* **Synthetic Agentic Dataset:** Automated generation of "Thought → Action" trajectories simulating tool use (Calculator, Search, Calendar).
* **DPO Fine-Tuning:** Implementation of Direct Preference Optimization to penalize hallucinations and unstructured outputs.
* **HPC Optimized:** Full SLURM integration for training on A100 clusters using mixed-precision (`bf16`) and Flash Attention.
* **Parameter Efficient:** Uses LoRA (Low-Rank Adaptation) and QLoRA techniques for efficient fine-tuning.

## 📂 Project Structure

```text
meta-agentic-dpo/
├── configs/              # Configuration files for hyperparameters
├── data/                 # Data storage
│   └── processed/        # Generated synthetic JSONL datasets
├── models/               # Saved LoRA adapters and checkpoints
├── scripts/              # SLURM scripts for HPC execution
│   ├── run_training.slurm
│   └── run_inference.slurm
├── src/                  # Source code
│   ├── generate_data.py  # Synthetic data generation engine
│   ├── train_dpo.py      # Main DPO training loop (TRL based)
│   └── inference.py      # Inference script for qualitative evaluation
└── README.md
```

## 🛠️ Installation

This project is designed to run on Linux HPC environments.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_GITHUB_USER/meta-agentic-dpo.git
cd meta-agentic-dpo

# 2. Create the environment (Conda recommended for HPC)
conda create -n meta-dpo python=3.10 -y
conda activate meta-dpo

# 3. Install dependencies
# Note: Use specific versions to ensure reproducibility
pip install "transformers>=4.40.0" "datasets>=2.18.0" "peft>=0.10.0" "trl>=0.8.0" "bitsandbytes==0.42.0" "accelerate>=0.29.0"
pip install sympy networkx jinja2 scipy
```

## 🧪 Methodology & Usage

### 1. Data Generation

We generate synthetic preference pairs where:

* **Prompt:** User prompt requiring a tool.
* **Chosen (Winner):** Correct Chain-of-Thought + Valid JSON Action.
* **Rejected (Loser):** Hallucination, refusal, or unstructured text.

```bash
python src/generate_data.py
# Output: 2,500 examples in data/processed/dpo_data.jsonl
```

### 2. DPO Training

We fine-tune `TinyLlama-1.1B-Chat` (or `Llama-3-8B`) using the `DPOTrainer` from Hugging Face `trl`.

**For HPC Users (SLURM):**

```bash
sbatch scripts/run_training.slurm
```

**Training Highlights:**

* **Loss Reduction:** Achieved convergence from `~0.69` (random) to `~0.005` in 3 epochs.
* **Reward Margin:** Increased to `>5.0`, indicating strong discrimination between agentic and non-agentic formats.

### 3. Inference & Evaluation

Test the aligned agent's ability to follow instructions and call tools.

```bash
sbatch scripts/run_inference.slurm
# Or locally on a GPU node:
python src/inference.py
```

## 📊 Results

The model successfully learned to transition from unstructured generation to a strict agentic format.

**Before Alignment (Base Model):**

> *User:* Calculate 25 * 40.  
> *Model:* The answer is 1000. (Direct hallucination/answer, no tool usage)

**After DPO Alignment:**

> *User:* Calculate 25 * 40.  
> *Model:*  
> **Thought:** The user wants to calculate 25 * 40. This requires the calculator tool.  
> **Action:**  
> ```json
> {"tool": "calculator", "args": "25 * 40"}
> ```

## 📜 License

This project is open-sourced under the MIT License.

---

*Developed as a research prototype for agentic AI alignment.*