"""
Inference script for TinyLlama model
"""
import torch
import argparse
import logging
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.prompts import format_agent_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_response(prompt: str, model, tokenizer, max_tokens: int = 128):
    """
    Generate agent response for a prompt.
    
    Args:
        prompt: User query
        model: Loaded model
        tokenizer: Tokenizer
        max_tokens: Maximum generation length
    
    Returns:
        Generated response
    """
    full_prompt = format_agent_prompt(prompt, include_system=False)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            temperature=0.1,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract generated part
    try:
        return result.split('Thought:')[-1].strip()
    except:
        return result


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Run inference with DPO agent")
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="models/TinyLlama-Agent-DPO-v1",
        help="Path to DPO adapter"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    logger.info(f"Loading DPO adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    # Test cases
    test_questions = [
        "What is 25 * 40?",
        "Who is the CEO of Tesla?",
        "Schedule a meeting for Monday."
    ]

    logger.info("\n" + "="*60)
    logger.info("AGENT INFERENCE TEST")
    logger.info("="*60 + "\n")
    
    for q in test_questions:
        logger.info(f"Question: {q}")
        response = generate_response(q, model, tokenizer)
        logger.info(f"Agent Response:\n{response}")
        logger.info("-" * 60)


if __name__ == "__main__":
    main()
