#!/usr/bin/env python3
"""Research Multi-Agent System entry point."""
import sys
from orchestrator import ResearchOrchestrator


def main():
    orchestrator = ResearchOrchestrator()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Research Multi-Agent System")
        print("Agents: Search → Analyst → Coder → Writer")
        print("-" * 40)
        question = input("Research question: ").strip()
        if not question:
            print("No question provided.")
            sys.exit(1)

    result = orchestrator.research(question)
    print(f"\n{'=' * 60}\n{result}")


if __name__ == "__main__":
    main()
