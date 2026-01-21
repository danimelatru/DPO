import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

def train():
    # --- CONFIGURATION ---
    # We use the Instruct version as base. 
    # Make sure you have access to Llama-3 on Hugging Face or use a valid mirror.
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    new_model_name = "Llama-3-Agent-DPO-v1"
    
    # Paths
    data_path = "data/processed/dpo_data.jsonl"
    output_dir = f"models/{new_model_name}"

    # --- 1. LOAD MODEL & TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load in bfloat16 for A100 efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" 
    )

    # --- 2. LORA CONFIG ---
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    )

    # --- 3. LOAD DATA ---
    # Load jsonl dataset
    dataset = load_dataset("json", data_files=data_path, split="train")

    # --- 4. TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        learning_rate=5e-6,           # Low LR is standard for DPO
        logging_steps=10,
        bf16=True,                    # Enable bfloat16
        num_train_epochs=1,
        warmup_ratio=0.1,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",            # Will log to Weights & Biases
        run_name="agent-dpo-run-1",
        remove_unused_columns=False
    )

    # --- 5. INITIALIZE TRAINER ---
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,               # TRL handles the reference model internally
        args=training_args,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=1024,
    )

    # --- 6. TRAIN ---
    print("Starting training...")
    dpo_trainer.train()
    
    # --- 7. SAVE ---
    print(f"Saving model to {output_dir}")
    dpo_trainer.save_model(output_dir)

if __name__ == "__main__":
    train()