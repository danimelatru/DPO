import json
import os
import random

# CONFIGURATION
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dpo_data.jsonl")
NUM_EXAMPLES = 2500  # Increased from 500 to 2500 for better learning

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SCENARIOS ---
# Expanded scenarios to give the model more context
tools = [
    {
        "name": "calculator",
        "desc": "Calculate math expressions",
        "queries": [
            ("Calculate 345 * 12", "345 * 12"),
            ("Square root of 144", "sqrt(144)"),
            ("What is 50 plus 10 divided by 2?", "(50 + 10) / 2"),
            ("25 * 40", "25 * 40"),
            ("100 minus 45", "100 - 45")
        ],
        "format": lambda q: f'{{"tool": "calculator", "args": "{q}"}}'
    },
    {
        "name": "web_search",
        "desc": "Search the internet for current facts",
        "queries": [
            ("Who is the CEO of Meta?", "CEO of Meta"),
            ("Current stock price of NVIDIA", "NVIDIA stock price"),
            ("Population of London 2024", "London population 2024"),
            ("Who won the last Super Bowl?", "Super Bowl winner"),
            ("Weather in Paris today", "weather Paris")
        ],
        "format": lambda q: f'{{"tool": "web_search", "query": "{q}"}}'
    },
    {
        "name": "calendar",
        "desc": "Schedule meetings and check availability",
        "queries": [
            ("Schedule a meeting with Mark for Monday", "Schedule meeting with Mark Monday"),
            ("Check my availability for Friday", "Check availability Friday"),
            ("Set a reminder for 5 PM", "Set reminder 5 PM"),
            ("Book a room for the team sync", "Book room team sync")
        ],
        "format": lambda q: f'{{"tool": "calendar", "action": "{q}"}}'
    }
]

def generate_entry():
    # Pick a random tool scenario
    tool = random.choice(tools)
    user_query, tool_input = random.choice(tool["queries"])
    
    # 1. The Prompt
    # We make the system prompt stricter
    prompt = (
        f"User: {user_query}\n"
        f"You are an agent with access to tools. Analyze the request. "
        f"If the user asks for something that requires calculation, external knowledge, or actions, "
        f"you MUST output a 'Thought' followed by an 'Action' in JSON format.\n"
    )

    # 2. The CHOSEN Response (Correct Agent Behavior)
    chosen = (
        f"Thought: The user is asking about '{user_query}'. This requires using the {tool['name']} tool.\n"
        f"Action: ```json\n{tool['format'](tool_input)}\n```"
    )

    # 3. The REJECTED Response (Common failures we want to penalize)
    fail_type = random.choice(["direct_answer", "refusal", "bad_format"])
    
    if fail_type == "direct_answer":
        # The model answers directly without using the tool (hallucination risk)
        rejected = f"Thought: I know the answer.\nThe answer is result."
    elif fail_type == "refusal":
        # The model is lazy or "safe"
        rejected = "I cannot help with that request as I am an AI."
    else:
        # The model forgets JSON
        rejected = f"Action: {tool['name']} with {tool_input}"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    print(f"Generating {NUM_EXAMPLES} synthetic agentic examples...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for _ in range(NUM_EXAMPLES):
            entry = generate_entry()
            json.dump(entry, f)
            f.write('\n')

    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()