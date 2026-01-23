import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = """You are an expert AI agent capable of using tools to solve problems.
You have access to the following tools:

1. calculator: Use this for math expressions.
   - Arguments: {"expression": "string"}

2. search: Use this to find current information about people, companies, or events.
   - Arguments: {"query": "string"}

3. calendar: Use this to schedule meetings or check dates.
   - Arguments: {"action": "create" | "read", "details": "string"}

INSTRUCTIONS:
- You must output your reasoning starting with 'Thought:'.
- If a tool is needed, you must output a strict JSON block starting with 'Action:'.
- The JSON must contain the keys "tool" and "args".
- Do not answer directly if a tool can verify the result.

Example:
Thought: I need to calculate 25 times 40.
Action:
```json
{"tool": "calculator", "args": {"expression": "25 * 40"}}
```
"""


def generate_response(user_prompt: str, model, tokenizer, max_new_tokens: int = 300) -> str:
    """
    Generates an agent-style response that begins with 'Thought:' and may include
    an 'Action:' JSON block.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Llama 3 Instruct models support chat templates via tokenizer.apply_chat_template
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
            temperature=0.01,  # Low temperature helps schema/JSON compliance
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Ensure the returned text starts with Thought:
    return "Thought:" + generated_text


def main():
    # --- CONFIGURATION ---
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "models/Llama3-8B-Agent-DPO-v1"

    print(f"--> Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"--> Loading DPO adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # --- TEST CASES ---
    test_questions = [
        "What is 25 * 40?",
        "Who is the CEO of Tesla right now?",
        "Schedule a lunch with the Meta team for next Tuesday at 1pm.",
    ]

    print("\n=== STARTING INFERENCE EVALUATION ===\n")

    for q in test_questions:
        print(f"USER: {q}")
        try:
            response = generate_response(q, model, tokenizer)
            print(f"AGENT:\n{response}")
        except Exception as e:
            print(f"Generation error: {e}")
        print("-" * 60)


if __name__ == "__main__":
    main()