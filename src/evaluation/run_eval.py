"""
Evaluate DPO model on test set
"""
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.evaluation.metrics import evaluate_model

# Load model
base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
adapter_path = "models/Llama3-8B-Agent-DPO-v1"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)

# Test prompts
test_prompts = [
    "Calculate 789 * 234",
    "Who is the CEO of Tesla?",
    "Weather in Barcelona today",
    "Schedule a meeting with Lisa for Friday at 3pm",
    "What is 15 percent of 340?",
]

# Evaluate
results = evaluate_model(model, tokenizer, test_prompts)

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
for metric, value in results['metrics'].items():
    print(f"{metric:25s}: {value:.2%}")
