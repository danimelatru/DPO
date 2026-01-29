"""
Synthetic data generation for DPO training with MAXIMUM diversity
"""

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_calculator_query():
    """Generate diverse calculator queries"""
    operations = [
        ("Calculate {a} * {b}", "{a} * {b}"),
        ("What is {a} plus {b}?", "{a} + {b}"),
        ("Subtract {b} from {a}", "{a} - {b}"),
        ("Divide {a} by {b}", "{a} / {b}"),
        ("What's {a} to the power of {b}?", "{a} ** {b}"),
        ("Calculate the square root of {a}", "sqrt({a})"),
        ("What is {a} percent of {b}?", "({a}/100) * {b}"),
        ("{a} times {b}", "{a} * {b}"),
        ("{a} divided by {b}", "{a} / {b}"),
        ("({a} + {b}) multiplied by {c}", "({a} + {b}) * {c}"),
        ("What's {a} minus {b}?", "{a} - {b}"),
        ("{a} plus {b} equals what?", "{a} + {b}"),
        ("Calculate {a} mod {b}", "{a} % {b}"),
        ("What is {a} squared?", "{a} ** 2"),
    ]

    q_template, a_template = random.choice(operations)

    # Generate random numbers with WIDE range for uniqueness
    a = random.randint(2, 999)
    b = random.randint(2, 250)
    c = random.randint(2, 99)

    user_query = q_template.format(a=a, b=b, c=c)
    tool_input = a_template.format(a=a, b=b, c=c)

    return user_query, tool_input


def generate_search_query():
    """Generate diverse web search queries with MANY parameters"""
    # Expanded lists for maximum variety
    companies = [
        "Tesla",
        "Apple",
        "Google",
        "Meta",
        "Microsoft",
        "Amazon",
        "NVIDIA",
        "OpenAI",
        "SpaceX",
        "Netflix",
        "Samsung",
        "Intel",
        "IBM",
        "Oracle",
        "Salesforce",
        "Adobe",
        "Shopify",
        "Twitter",
        "Uber",
        "Airbnb",
        "PayPal",
        "Stripe",
        "Zoom",
        "Spotify",
        "Pinterest",
        "Reddit",
        "TikTok",
        "LinkedIn",
        "Dropbox",
        "Slack",
        "Discord",
        "GitHub",
    ]

    cities = [
        "London",
        "Paris",
        "Tokyo",
        "New York",
        "Berlin",
        "Madrid",
        "Rome",
        "Sydney",
        "Toronto",
        "Dubai",
        "Singapore",
        "Barcelona",
        "Amsterdam",
        "Vienna",
        "Stockholm",
        "Copenhagen",
        "Zurich",
        "Prague",
        "Budapest",
        "Warsaw",
        "Athens",
        "Lisbon",
        "Dublin",
        "Edinburgh",
        "Brussels",
        "Oslo",
        "Helsinki",
        "Munich",
        "Milan",
        "Venice",
        "Florence",
        "Chicago",
        "Los Angeles",
        "San Francisco",
    ]

    topics = [
        "AI",
        "climate change",
        "electric vehicles",
        "space exploration",
        "quantum computing",
        "renewable energy",
        "blockchain",
        "5G networks",
        "cryptocurrency",
        "machine learning",
        "robotics",
        "biotechnology",
        "nanotechnology",
        "virtual reality",
        "augmented reality",
        "IoT",
        "cybersecurity",
        "cloud computing",
        "edge computing",
        "metaverse",
    ]

    people = [
        "Einstein",
        "Marie Curie",
        "Steve Jobs",
        "Elon Musk",
        "Ada Lovelace",
        "Tim Berners-Lee",
        "Grace Hopper",
        "Alan Turing",
        "Isaac Newton",
        "Galileo",
        "Darwin",
        "Tesla",
        "Edison",
        "Bill Gates",
        "Mark Zuckerberg",
        "Jeff Bezos",
        "Warren Buffett",
        "Oprah Winfrey",
        "Nelson Mandela",
        "Martin Luther King",
        "Gandhi",
        "Shakespeare",
        "Leonardo da Vinci",
    ]

    years = ["2020", "2021", "2022", "2023", "2024", "2025"]

    # More diverse templates
    templates = [
        ("Who is the CEO of {company}?", "CEO of {company}"),
        ("Current stock price of {company}", "{company} stock price"),
        ("Latest news about {topic}", "{topic} news latest"),
        ("Population of {city} in {year}", "{city} population {year}"),
        ("Weather in {city} today", "weather {city}"),
        ("What is the GDP of {city}?", "{city} GDP"),
        ("History of {company}", "{company} history"),
        ("{person} biography", "{person} bio"),
        ("When did {person} die?", "{person} death date"),
        ("Founder of {company}", "{company} founder"),
        ("Where was {person} born?", "{person} birthplace"),
        ("Average temperature in {city}", "{city} average temperature"),
        ("{company} market cap", "{company} market capitalization"),
        ("Tourist attractions in {city}", "{city} tourist attractions"),
        ("What did {person} invent?", "{person} inventions"),
        ("Revenue of {company} in {year}", "{company} revenue {year}"),
        ("Cost of living in {city}", "{city} cost of living"),
        ("Research papers on {topic}", "{topic} research papers"),
    ]

    q_template, a_template = random.choice(templates)

    params = {
        "company": random.choice(companies),
        "city": random.choice(cities),
        "topic": random.choice(topics),
        "person": random.choice(people),
        "year": random.choice(years),
    }

    user_query = q_template.format(**params)
    tool_input = a_template.format(**params)

    return user_query, tool_input


def generate_calendar_query():
    """Generate diverse calendar queries with MANY parameters"""
    # Expanded lists
    people = [
        "Mark",
        "Sarah",
        "John",
        "Alice",
        "David",
        "Emma",
        "Michael",
        "Lisa",
        "Peter",
        "Jennifer",
        "Robert",
        "Maria",
        "James",
        "Anna",
        "Thomas",
        "Laura",
        "Daniel",
        "Sophie",
        "Chris",
        "Emily",
        "Andrew",
        "Kate",
        "Brian",
        "Rachel",
        "Kevin",
        "Michelle",
        "Steven",
        "Jessica",
        "Paul",
        "Amanda",
        "Eric",
        "Nicole",
    ]

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "next week",
        "tomorrow",
        "next Monday",
        "this Friday",
        "next Tuesday",
        "this Wednesday",
        "next Thursday",
        "Saturday",
        "Sunday",
        "next month",
        "today",
    ]

    times = [
        "9am",
        "2pm",
        "5pm",
        "noon",
        "3:30pm",
        "10am",
        "4pm",
        "11am",
        "1pm",
        "8am",
        "6pm",
        "7pm",
        "8:30am",
        "3pm",
        "9:30am",
        "10:30am",
        "2:30pm",
        "4:30pm",
        "midday",
    ]

    meetings = [
        "team sync",
        "client call",
        "standup",
        "review meeting",
        "presentation",
        "workshop",
        "brainstorm session",
        "sprint planning",
        "one-on-one",
        "board meeting",
        "product demo",
        "training session",
        "interview",
        "performance review",
        "kickoff meeting",
        "retrospective",
        "strategy session",
        "town hall",
        "coffee chat",
        "lunch meeting",
    ]

    templates = [
        ("Schedule a meeting with {person} for {day}", "Schedule meeting with {person} {day}"),
        ("Check my availability for {day}", "Check availability {day}"),
        ("Set a reminder for {time}", "Set reminder {time}"),
        ("Book a room for {meeting}", "Book room {meeting}"),
        ("Create a calendar event for {day} at {time}", "Create event {day} {time}"),
        ("Cancel my meeting with {person}", "Cancel meeting {person}"),
        ("Reschedule {meeting} to {day}", "Reschedule {meeting} {day}"),
        ("What meetings do I have on {day}?", "List meetings {day}"),
        ("Schedule {meeting} with {person} at {time}", "Schedule {meeting} {person} {time}"),
        ("Block {time} on {day} for {meeting}", "Block {time} {day} {meeting}"),
        ("Move my {meeting} to {day}", "Move {meeting} {day}"),
        ("Add {person} to {meeting}", "Add {person} {meeting}"),
    ]

    q_template, a_template = random.choice(templates)

    params = {
        "person": random.choice(people),
        "day": random.choice(days),
        "time": random.choice(times),
        "meeting": random.choice(meetings),
    }

    user_query = q_template.format(**params)
    tool_input = a_template.format(**params)

    return user_query, tool_input


def generate_entry():
    """
    Generate a single DPO training example with maximum diversity.

    Returns:
        Dictionary with prompt, chosen, and rejected responses
    """
    # Pick tool type
    tool_type = random.choice(["calculator", "web_search", "calendar"])

    if tool_type == "calculator":
        user_query, tool_input = generate_calculator_query()
        tool_name = "calculator"
    elif tool_type == "web_search":
        user_query, tool_input = generate_search_query()
        tool_name = "web_search"
    else:
        user_query, tool_input = generate_calendar_query()
        tool_name = "calendar"

    # Build prompt
    prompt = (
        f"User: {user_query}\n"
        f"You are an agent with access to tools. Analyze the request. "
        f"If the user asks for something that requires calculation, external knowledge, or actions, "
        f"you MUST output a 'Thought' followed by an 'Action' in JSON format.\n"
    )

    # Chosen response (correct behavior)
    chosen = (
        f"Thought: The user is asking about '{user_query}'. This requires using the {tool_name} tool.\n"
        f'Action: ```json\n{{"tool": "{tool_name}", "args": "{tool_input}"}}\n```'
    )

    # Rejected response (random failure mode)
    fail_type = random.choice(["direct_answer", "refusal", "bad_format"])

    if fail_type == "direct_answer":
        rejected = f"Thought: I know the answer.\nThe answer is result."
    elif fail_type == "refusal":
        rejected = "I cannot help with that request as I am an AI."
    else:
        rejected = f"Action: {tool_name} with {tool_input}"

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    """Main entry point for data generation"""
    parser = argparse.ArgumentParser(description="Generate diverse synthetic DPO training data")
    parser.add_argument(
        "--output", type=str, default="data/processed/dpo_data.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--num-examples", type=int, default=2500, help="Number of examples to generate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.num_examples} diverse synthetic examples...")
    logger.info(f"Random seed: {args.seed}")

    # Track unique prompts
    unique_prompts = set()

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            entry = generate_entry()
            unique_prompts.add(entry["prompt"])
            json.dump(entry, f)
            f.write("\n")

            if (i + 1) % 500 == 0:
                logger.info(f"Generated {i + 1}/{args.num_examples} examples")
                logger.info(f"  Unique prompts so far: {len(unique_prompts)}")

    logger.info(f"✅ Done! Saved to {output_path}")
    logger.info(f"📊 Statistics:")
    logger.info(f"   Total examples: {args.num_examples}")
    logger.info(f"   Unique prompts: {len(unique_prompts)}")
    logger.info(f"   Diversity rate: {len(unique_prompts)/args.num_examples*100:.1f}%")


if __name__ == "__main__":
    main()
