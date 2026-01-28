"""
Inference script for Llama 3 model with chat template
"""
import torch
import argparse
import logging
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.prompts import SYSTEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_response(
    user_prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 300
) -> str:
    """
    Generate agent-style response using Llama 3 chat template.
    
    Args:
        user_prompt: User query
        model: Loaded model
        tokenizer: Tokenizer with chat template
        max_new_tokens: Maximum generation length
    
    Returns:
        Generated response starting with "Thought:"
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Apply Llama 3 chat template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prime the model to start with Thought:
    input_text = prompt_text + "Thought:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.01,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return "Thought:" + generated_text


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description="Run inference with Llama 3 DPO agent")
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="models/Llama3-8B-Agent-DPO-v1",
        help="Path to DPO adapter"
    )
    
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info(f"Loading DPO adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    # Test cases
    test_questions = [
        "What is 25 * 40?",
        "Who is the CEO of Tesla right now?",
        "Schedule a lunch with the Meta team for next Tuesday at 1pm.",
    ]

    logger.info("\n" + "="*60)
    logger.info("STARTING INFERENCE EVALUATION")
    logger.info("="*60 + "\n")

    for q in test_questions:
        logger.info(f"USER: {q}")
        try:
            response = generate_response(q, model, tokenizer)
            logger.info(f"AGENT:\n{response}")
        except Exception as e:
            logger.error(f"Generation error: {e}")
        logger.info("-" * 60)


if __name__ == "__main__":
    main()
