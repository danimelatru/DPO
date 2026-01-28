"""
Metrics for evaluating agentic behavior
"""
import json
import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_json_action(response: str) -> Tuple[bool, str]:
    """
    Check if response contains valid JSON action block.
    
    Args:
        response: Model response
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Look for JSON block in markdown code fence
        json_match = re.search(
            r'```json\s*(\{.*?\})\s*```',
            response,
            re.DOTALL
        )
        
        if not json_match:
            # Also try without code fence
            json_match = re.search(r'\{[^{}]*\}', response)
            if not json_match:
                return False, "No JSON block found"
        
        json_str = json_match.group(1) if json_match.lastindex else json_match.group(0)
        parsed = json.loads(json_str)
        
        # Validate structure
        if 'tool' not in parsed:
            return False, "Missing 'tool' key in JSON"
        
        if 'args' not in parsed:
            return False, "Missing 'args' key in JSON"
        
        return True, ""
        
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error parsing: {str(e)}"


def has_thought_prefix(response: str) -> bool:
    """
    Check if response starts with 'Thought:' prefix.
    
    Args:
        response: Model response
    
    Returns:
        True if response has thought prefix
    """
    return response.strip().lower().startswith("thought:")


def has_action_block(response: str) -> bool:
    """
    Check if response contains 'Action:' block.
    
    Args:
        response: Model response
    
    Returns:
        True if action block is present
    """
    return "action:" in response.lower()


def is_refusal(response: str) -> bool:
    """
    Check if response is a refusal to help.
    
    Args:
        response: Model response
    
    Returns:
        True if response is a refusal
    """
    refusal_patterns = [
        "cannot help",
        "can't help",
        "unable to assist",
        "i don't know",
        "i am not able"
    ]
    
    response_lower = response.lower()
    return any(pattern in response_lower for pattern in refusal_patterns)


def compute_agent_compliance(responses: List[str]) -> Dict[str, float]:
    """
    Calculate compliance metrics for agentic behavior.
    
    Args:
        responses: List of model responses
    
    Returns:
        Dictionary with metric scores
    """
    if not responses:
        return {}
    
    n = len(responses)
    
    json_valid_count = sum(validate_json_action(r)[0] for r in responses)
    thought_count = sum(has_thought_prefix(r) for r in responses)
    action_count = sum(has_action_block(r) for r in responses)
    refusal_count = sum(is_refusal(r) for r in responses)
    
    metrics = {
        "json_validity": json_valid_count / n,
        "thought_prefix_rate": thought_count / n,
        "action_block_rate": action_count / n,
        "refusal_rate": refusal_count / n,
        "full_compliance": sum(
            validate_json_action(r)[0] and has_thought_prefix(r)
            for r in responses
        ) / n
    }
    
    logger.info(f"Evaluated {n} responses")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.2%}")
    
    return metrics


def evaluate_model(
    model,
    tokenizer,
    test_prompts: List[str],
    max_tokens: int = 256
) -> Dict[str, any]:
    """
    Comprehensive model evaluation on test prompts.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        test_prompts: List of test prompts
        max_tokens: Maximum generation length
    
    Returns:
        Dictionary with evaluation results
    """
    import torch
    
    logger.info(f"Evaluating model on {len(test_prompts)} prompts")
    
    responses = []
    
    for prompt in test_prompts:
        full_prompt = (
            f"User: {prompt}\n"
            f"You are an agent with access to tools. Analyze the request "
            f"and call the correct tool if necessary.\n"
            f"Thought:"
        )
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Thought:")[-1].strip()
        responses.append(response)
    
    # Compute metrics
    metrics = compute_agent_compliance(responses)
    
    return {
        "metrics": metrics,
        "responses": responses,
        "prompts": test_prompts
    }