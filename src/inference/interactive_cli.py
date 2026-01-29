"""
Interactive CLI for DPO-aligned agent
"""

import argparse
import logging
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class InteractiveAgent:
    """Interactive command-line interface for the agent"""

    def __init__(
        self, base_model: str, adapter_path: str, temperature: float = 0.1, max_tokens: int = 256
    ):
        """
        Initialize interactive agent.

        Args:
            base_model: Base model identifier
            adapter_path: Path to trained adapter
            temperature: Sampling temperature
            max_tokens: Maximum generation length
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Loading model: {base_model}")
        print(f"🔄 Loading model: {base_model}...")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )

        logger.info(f"Loading adapter: {adapter_path}")
        print(f"🔄 Loading adapter: {adapter_path}...")

        self.model = PeftModel.from_pretrained(base_model_obj, adapter_path)
        print("✅ Model loaded successfully!\n")

    def generate(self, prompt: str) -> str:
        """
        Generate agent response for a user prompt.

        Args:
            prompt: User input

        Returns:
            Agent response
        """
        full_prompt = (
            f"User: {prompt}\n"
            f"You are an agent with access to tools. Analyze the request "
            f"and call the correct tool if necessary.\n"
            f"Thought:"
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only generated part after "Thought:"
        try:
            return result.split("Thought:")[-1].strip()
        except:
            return result

    def run(self):
        """Start interactive session"""
        print("=" * 70)
        print("🤖 DPO Agent - Interactive Mode")
        print("=" * 70)
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • 'exit' or 'quit' - Exit the program")
        print("  • 'clear' - Clear screen")
        print("  • 'help' - Show this help message")
        print("\n" + "=" * 70 + "\n")

        while True:
            try:
                user_input = input("👤 You: ").strip()

                # Handle commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "clear":
                    print("\n" * 50)
                    continue

                if user_input.lower() == "help":
                    print("\n📖 Available commands:")
                    print("  • exit/quit - Exit the program")
                    print("  • clear - Clear screen")
                    print("  • help - Show this message\n")
                    continue

                if not user_input:
                    continue

                # Generate response
                print("\n🤖 Agent:")
                response = self.generate(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                print(f"❌ Error: {e}\n")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive DPO Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name or path",
    )

    parser.add_argument(
        "--adapter", type=str, default="models/TinyLlama-Agent-DPO-v1", help="Path to DPO adapter"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)"
    )

    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum generation length (default: 256)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run agent
    agent = InteractiveAgent(
        base_model=args.base_model,
        adapter_path=args.adapter,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    agent.run()


if __name__ == "__main__":
    main()
