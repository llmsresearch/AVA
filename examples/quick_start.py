"""Quick start example for AVA.

Demonstrates solving a math problem with AVA vs. self-consistency baseline.
Requires either an Ollama model running locally or Azure OpenAI credentials in .env.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.agents.ava_agent import AVAAgent
from ava.baselines.self_consistency import self_consistency
from ava.core.interfaces import Budget


def main():
    # Choose model backend: "ollama" or "azure"
    backend = "ollama"

    if backend == "ollama":
        from ava.models.ollama_model import OllamaModel
        model = OllamaModel(model_name="qwen2.5:7b")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        from ava.models.azure_openai import AzureOpenAIModel
        model = AzureOpenAIModel()

    prompt = "Question: A store sells apples for $2 each and oranges for $3 each. If Sarah buys 4 apples and 5 oranges, how much does she spend in total?\nAnswer:"

    # --- AVA ---
    budget_ava = Budget(token_limit=800, tool_calls_limit=20, verify_calls_limit=20)
    agent = AVAAgent(model, target_reliability=0.9)
    result = agent.solve(prompt, budget_ava)
    print(f"AVA answer: {result.text[:200]}")
    print(f"AVA tokens used: {budget_ava.tokens_used}")

    # --- Self-Consistency baseline ---
    budget_sc = Budget(token_limit=800, tool_calls_limit=20, verify_calls_limit=20)
    answer, votes = self_consistency(prompt, model, budget_sc, k=5)
    print(f"\nSelf-Consistency answer: {answer[:200]}")
    print(f"Self-Consistency tokens used: {budget_sc.tokens_used}")


if __name__ == "__main__":
    main()
