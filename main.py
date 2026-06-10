#!/usr/bin/env python3
"""
CLI entry point for the FAQ tool.

Usage:
    # Interactive REPL
    python main.py

    # Single question
    python main.py --question "Can I work during IVF?"

    # Show retrieved sources
    python main.py --question "Is IVF painful?" --show-sources
"""

import argparse
import os
import sys
from pathlib import Path

# Make sure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import FAQBot


DATA_PATH = Path(__file__).parent / "data" / "faqs.json"


def check_api_key() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY not set")
        return


def print_result(result: dict, show_sources: bool) -> None:
    print("\n" + "─" * 60)
    print("Answer:")
    print(result["answer"])
    if show_sources:
        print("\nTop sources used:")
        for s in result["sources"]:
            bar = "█" * int(s["score"] * 20)
            print(f"  [{bar:<20}] {s['score']:.3f}  {s['question']}")
    print("─" * 60 + "\n")


def run_repl(bot: FAQBot, show_sources: bool) -> None:
    print("FAQ Bot ready. Type your question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        result = bot.ask(question)
        print_result(result, show_sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ answer tool using RAG + Claude")
    parser.add_argument("--question", "-q", help="Ask a single question and exit")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the FAQ sources used to generate the answer",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help="Path to faqs.json (default: data/faqs.json)",
    )
    args = parser.parse_args()

    check_api_key()

    print("Loading FAQ index…")
    bot = FAQBot(args.data)

    if args.question:
        result = bot.ask(args.question)
        print_result(result, args.show_sources)
    else:
        run_repl(bot, args.show_sources)


if __name__ == "__main__":
    main()
