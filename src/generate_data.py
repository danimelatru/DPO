import json
import os
import random

# Define the output path
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dpo_data.jsonl")

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- TEMPLATES FOR AGENTIC BEHAVIOR ---

# Scenarios: The user asks a question that requires a specific tool.
scenarios = [
    {
        "intent": "calculator",
        "questions": ["Calculate 345 * 12.", "What is the square root of 144?", "Multiply 50 by 10."],
        "tool_name": "calculator",
        "correct_tool_call": lambda q: f'{{"tool": "calculator", "input": "{q}"}}'
    },
    {
        "intent": "search",
        "questions": ["Who is the CEO of Meta?", "What is the population of London?", "Current stock price of NVIDIA."],
        "tool_name": "web_search",
        "correct_tool_call": lambda q: f'{{"tool": "web_search", "query": "{q}"}}'
    },
    {
        "intent": "calendar",
        "questions": ["Schedule a meeting with Mark for tomorrow.", "Check my availability for Friday.", "Add a reminder."],
        "tool_name": "calendar_api",
        "correct_tool_call": lambda q: f'{{"tool": "calendar_api", "action": "parse_intent", "text": "{q}"}}'
    }
]

def generate_entry(scenario, question):
    """
    Generates a single DPO triplet (prompt, chosen, rejected).
    """
    
    # 1. The Prompt
    prompt = f"User: {question}\nYou are an agent with access to tools. Analyze the request and call the correct tool if necessary."

    # 2. The CHOSEN Response (The "Winner")
    # Features: Reasoning trace + Strict JSON tool call
    chosen = (
        f"Thought: The user is asking a question that requires external information or calculation. "
        f"I should use the {scenario['tool_name']} tool to handle this.\n"
        f"Action: ```json\n{scenario['correct_tool_call'](question)}\n```"
    )

    # 3. The REJECTED Response (The "Loser")
    # Features: Hallucination (guessing) OR Wrong format OR Lazy answering
    failure_type = random.choice(["hallucination", "wrong_format", "lazy"])
    
    if failure_type == "hallucination":
        rejected = f"The answer is definitively 42. I am sure about this."
    elif failure_type == "wrong_format":
        # Missing the JSON block or Thought process
        rejected = f"Call {scenario['tool_name']} with input {question}"
    else:
        rejected = "I cannot help with that."

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    print(f"Generating synthetic agentic data at {OUTPUT_FILE}...")
    
    dataset = []
    
    # Generate 500 synthetic examples (For a real project, we'd want 1k-5k, but 500 is enough to test pipeline)
    for _ in range(500):
        scenario = random.choice(scenarios)
        question = random.choice(scenario["questions"])
        
        # Add some noise/variation to the question if you want (optional)
        entry = generate_entry(scenario, question)
        dataset.append(entry)

    # Save to JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in dataset:
            json.dump(entry, f)
            f.write('\n')

    print(f"Success! Generated {len(dataset)} examples.")
    print("Example entry:")
    print(json.dumps(dataset[0], indent=2))

if __name__ == "__main__":
    main()