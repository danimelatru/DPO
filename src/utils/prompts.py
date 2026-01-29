"""
Centralized prompt templates for the agent
"""

SYSTEM_PROMPT = """You are an expert AI agent capable of using tools to solve problems.
You have access to the following tools:

1. calculator: Use this for math expressions.
   - Arguments: {"expression": "string"}

2. web_search: Use this to find current information about people, companies, or events.
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
"""

TOOL_DESCRIPTIONS = {
    "calculator": {
        "name": "calculator",
        "description": "Calculate mathematical expressions",
        "args_schema": {"expression": "string"},
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the internet for current facts",
        "args_schema": {"query": "string"},
    },
    "calendar": {
        "name": "calendar",
        "description": "Schedule meetings and check availability",
        "args_schema": {"action": "create | read", "details": "string"},
    },
}


def format_agent_prompt(user_query: str, include_system: bool = True) -> str:
    """
    Format a user query into an agent prompt.

    Args:
        user_query: User's question or request
        include_system: Whether to include system prompt

    Returns:
        Formatted prompt string
    """
    prompt = ""

    if include_system:
        prompt += SYSTEM_PROMPT + "\n\n"

    prompt += (
        f"User: {user_query}\n"
        f"You are an agent with access to tools. Analyze the request "
        f"and call the correct tool if necessary.\n"
        f"Thought:"
    )

    return prompt


def format_llama3_chat(user_query: str, tokenizer) -> str:
    """
    Format prompt using Llama 3 chat template.
    Args:
        user_query: User's question
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted prompt
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt + "Thought:"
