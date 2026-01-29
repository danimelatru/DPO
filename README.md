# Post-Training Alignment for Agentic AI using DPO

A complete implementation of **Direct Preference Optimization (DPO)** for training Small Language Models (SLMs) to perform **agentic tool use** with strict Chain-of-Thought reasoning and structured JSON outputs.

> **Research Goal:** Validate the efficacy of DPO in enforcing rigid schema compliance (JSON) and reasoning traces in lightweight models (1.1B–8B parameters) without relying on massive-scale supervised fine-tuning.

---

## 🚀 Key Features

- **Synthetic Agentic Dataset:** Automated generation of 2,500+ "Thought → Action" trajectories simulating tool use (Calculator, Search, Calendar)
- **DPO Fine-Tuning:** Implementation of Direct Preference Optimization to penalize hallucinations and unstructured outputs
- **HPC Optimized:** Full SLURM integration for training on A100 clusters using mixed-precision (`bf16`)
- **Parameter Efficient:** Uses LoRA (Low-Rank Adaptation) with rank 64 for efficient fine-tuning
- **Production Ready:** Includes comprehensive test suite and inference tools
- **Interactive CLI:** Real-time testing interface for conversational agent interaction

---

## 📂 Project Structure

```text
dpo/
├── configs/
│   ├── train_config.yaml              # Llama 3 8B training config
│   └── train_config_tinyllama.yaml    # TinyLlama 1.1B config
├── data/
│   └── processed/
│       └── dpo_data.jsonl             # 2,500 training examples
├── models/
│   └── Llama3-8B-Agent-DPO-v1/        # Trained model (LoRA adapters)
├── scripts/
│   ├── run_training.slurm             # SLURM training script
│   ├── run_inference.slurm            # SLURM inference script
│   └── run_interactive.slurm          # Interactive CLI launcher
├── src/
│   ├── data/
│   │   ├── generate_data.py           # Synthetic data generator
│   │   └── data_loader.py             # Dataset loading utilities
│   ├── training/
│   │   ├── train_dpo.py               # Main training script
│   │   └── trainer.py                 # DPO trainer wrapper
│   ├── inference/
│   │   ├── inference.py               # Batch inference
│   │   ├── inference_llama3.py        # Llama 3 specific inference
│   │   └── interactive_cli.py         # Interactive chat interface
│   ├── evaluation/
│   │   └── metrics.py                 # Evaluation metrics
│   └── utils/
│       ├── config.py                  # Configuration management
│       └── prompts.py                 # Prompt templates
├── tests/
│   ├── test_data_loader.py            # Data loading tests
│   ├── test_training.py               # Training pipeline tests
│   └── test_integration.py            # End-to-end tests
├── requirements.txt
├── pyproject.toml                     # Project configuration
├── Makefile                           # Common commands
└── README.md
```

---

## 🛠️ Installation

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU training/inference)
- 16GB+ GPU memory (24GB+ recommended for Llama 3 8B)

### Setup

```bash
git clone https://github.com/danimelatru/dpo.git
cd dpo

# Create virtual environment
conda create -n dpo python=3.10 -y
conda activate dpo

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs models data/processed
```

---

## 🧪 Usage

### 1. Generate Synthetic Training Data

```bash
python -m src.data.generate_data \
    --output data/processed/dpo_data.jsonl \
    --num-examples 2500 \
    --seed 42
```

**Output:** JSONL file with preference pairs (chosen vs rejected responses)

### 2. Training

#### Local Training

```bash
python -m src.training.train_dpo --config configs/train_config.yaml
```

#### HPC Training (SLURM)

```bash
sbatch scripts/run_training.slurm
```

**Training Configuration:**
- Base Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- LoRA Rank: 64
- LoRA Alpha: 16
- Batch Size: 4 (with gradient accumulation: 4)
- Learning Rate: 5e-6
- Epochs: 3
- DPO Beta: 0.1

### 3. Inference

#### Batch Inference

```bash
python -m src.inference.inference_llama3 \
    --model-path models/Llama3-8B-Agent-DPO-v1 \
    --test-file data/test_data.jsonl \
    --output results.jsonl
```

#### Interactive CLI

```bash
python -m src.inference.interactive_cli \
    --model-path models/Llama3-8B-Agent-DPO-v1
```

### 4. Testing

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/test_data_loader.py -v
pytest tests/test_training.py -v
pytest tests/test_integration.py -v

# Run with coverage
pytest --cov=src tests/
```

---

## 📊 Results

### Training Metrics

The model was trained on 2,500 synthetic examples with the following results:

| Metric | Base Model | After DPO |
|--------|-----------|-----------|
| JSON Validity | ~0% | **95%+** |
| Thought Prefix Rate | ~10% | **98%+** |
| Full Compliance | ~0% | **93%+** |

### Model Details

- **Model Name:** Llama3-8B-Agent-DPO-v1
- **Base Model:** meta-llama/Meta-Llama-3-8B-Instruct
- **Training Method:** Direct Preference Optimization (DPO)
- **Adapter Type:** LoRA (r=64, alpha=16, dropout=0.05)
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Time:** ~3 epochs on A100 GPU
- **Final Model Size:** ~640MB (adapters only)

---

## 🔧 Development

### Project Commands (Makefile)

```bash
make test          # Run all tests
make lint          # Run linters (black, isort, flake8)
make format        # Auto-format code
make clean         # Clean cache files
```

### Configuration

Training configurations are stored in `configs/`:
- `train_config.yaml` - Llama 3 8B configuration
- `train_config_tinyllama.yaml` - TinyLlama 1.1B configuration

Modify these files to adjust hyperparameters, model paths, or training settings.

---

## 📝 Dataset Format

The training data follows this structure:

```json
{
  "prompt": "User: Calculate 5 + 3\nYou are an agent with access to tools...",
  "chosen": "Thought: The user is asking about 'Calculate 5 + 3'. This requires using the calculator tool.\nAction: ```json\n{\"tool\": \"calculator\", \"args\": \"5 + 3\"}\n```",
  "rejected": "I cannot help with that request as I am an AI."
}
```

Each example contains:
- **prompt**: User request + system instructions
- **chosen**: Preferred response (structured, with Thought + Action)
- **rejected**: Dispreferred response (hallucination, refusal, or malformed)

---

## 🎯 Use Cases

This trained model can be used for:
- **Agentic AI Systems:** Building agents that use tools reliably
- **Structured Output Generation:** Ensuring JSON compliance in LLM outputs
- **Chain-of-Thought Reasoning:** Enforcing explicit reasoning traces
- **Research:** Studying preference optimization techniques

---

## 📜 License

This project is open-sourced under the MIT License.

---

## 🙏 Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- DPO implementation from [TRL](https://github.com/huggingface/trl)
- LoRA via [PEFT](https://github.com/huggingface/peft)
