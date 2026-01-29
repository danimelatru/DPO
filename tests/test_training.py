"""
Real training tests for the DPO project.
These tests validate the trained model and training pipeline.
"""
import pytest
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.data.data_loader import load_dpo_dataset, validate_dataset


def test_trained_model_exists():
    """Test that the trained model directory exists with required files"""
    model_path = Path("models/Llama3-8B-Agent-DPO-v1")
    
    assert model_path.exists(), f"Model directory not found: {model_path}"
    assert (model_path / "adapter_config.json").exists(), "adapter_config.json missing"
    assert (model_path / "adapter_model.safetensors").exists(), "adapter_model.safetensors missing"
    

def test_training_data_exists_and_valid():
    """Test that training data exists and is valid"""
    data_path = "data/processed/dpo_data.jsonl"
    
    assert os.path.exists(data_path), f"Training data not found: {data_path}"
    
    # Load and validate dataset
    dataset = load_dpo_dataset(data_path)
    assert len(dataset) > 0, "Dataset is empty"
    
    # Validate schema
    validate_dataset(dataset)
    
    # Check expected size (2500 examples)
    assert len(dataset) == 2500, f"Expected 2500 examples, got {len(dataset)}"


def test_adapter_config_matches_training():
    """Test that adapter configuration matches the training setup"""
    import json
    
    config_path = Path("models/Llama3-8B-Agent-DPO-v1/adapter_config.json")
    with open(config_path) as f:
        adapter_config = json.load(f)
    
    # Verify base model
    assert adapter_config["base_model_name_or_path"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Verify LoRA parameters
    assert adapter_config["r"] == 64
    assert adapter_config["lora_alpha"] == 16
    assert adapter_config["lora_dropout"] == 0.05
    
    # Verify target modules
    expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert set(adapter_config["target_modules"]) == set(expected_modules)


@pytest.mark.slow
def test_load_trained_model():
    """
    Test loading the trained model with adapters.
    This test requires the base model to be accessible (either locally or via HuggingFace).
    Mark as slow since it downloads/loads the full model.
    """
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "models/Llama3-8B-Agent-DPO-v1"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        load_in_8bit=True  # Use 8-bit to reduce memory
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Verify model is in eval mode
    model.eval()
    
    # Test a simple forward pass
    test_input = "User: Calculate 5 + 3\nYou are an agent with access to tools."
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with pytest.warns(None):  # Suppress warnings
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(response) > 0, "Model generated empty response"
