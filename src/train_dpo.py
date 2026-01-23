import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

def train():
    # --- CONFIGURATION ---
    # Using TinyLlama to avoid gated access issues and speed up testing
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    new_model_name = "TinyLlama-Agent-DPO-v1"
    
    # Paths
    data_path = "data/processed/dpo_data.jsonl"
    output_dir = f"models/{new_model_name}"

    # --- 1. LOAD MODEL & TOKENIZER ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model (TinyLlama)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
        # attn_implementation="flash_attention_2" # Commented out to avoid installation errors
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
    dataset = load_dataset("json", data_files=data_path, split="train")

    # --- 4. TRAINING ARGUMENTS (DPOConfig) ---
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        bf16=True,
        num_train_epochs=3,
        warmup_ratio=0.1,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",
        run_name="agent-dpo-run-1",
        remove_unused_columns=False,
        # DPO specific parameters must be here now:
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
    )

    # --- 5. INITIALIZE TRAINER ---
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        # BREAKING CHANGE FIX: 'tokenizer' was renamed to 'processing_class' in trl>=0.12.0
        processing_class=tokenizer, 
        peft_config=peft_config,
    )

    # --- 6. TRAIN ---
    print("Starting training...")
    dpo_trainer.train()
    
    # --- 7. SAVE ---
    print(f"Saving model to {output_dir}")
    dpo_trainer.save_model(output_dir)

if __name__ == "__main__":
    train()