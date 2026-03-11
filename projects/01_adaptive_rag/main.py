#!/usr/bin/env python3
"""Run the adaptive RAG graph against a set of demo questions.

Each question is designed to trigger a different routing path:
  1. Vectorstore retrieval  (domain-specific AI question)
  2. Web search             (current events / recent info)
  3. Direct LLM answer      (simple general knowledge)
"""

from dotenv import load_dotenv

load_dotenv()

from adaptive_rag.graph import build_graph  # noqa: E402

DEMO_QUESTIONS = [
    "What are the different types of memory in LLM-based agents?",
    "Who won the most recent Super Bowl and what was the score?",
    "What is the capital of France?",
]


def main() -> None:
    graph = build_graph()

    for question in DEMO_QUESTIONS:
        print(f"\n{'=' * 72}")
        print(f"QUESTION: {question}")
        print("=" * 72)

        result = graph.invoke({"question": question})

        print(f"\nANSWER:\n{result['generation']}")
        print("=" * 72)


if __name__ == "__main__":
    main()
