from __future__ import annotations

import argparse

from app.config import settings
from app.graph import invoke_agent
from app.logging_utils import format_reasoning_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bitext Customer Service Data Analyst Agent"
    )
    parser.add_argument(
        "--session",
        default=settings.default_session_id,
        help="Conversation session ID.",
    )
    parser.add_argument(
        "--user",
        default=settings.default_user_id,
        help="Persistent user ID.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.max_iterations,
        help="Maximum reasoning/tool-use iterations.",
    )
    return parser.parse_args()


def print_welcome(session_id: str, user_id: str, max_iterations: int) -> None:
    print("Bitext Customer Service Data Analyst Agent")
    print(f"Session: {session_id}")
    print(f"User: {user_id}")
    print(f"Max iterations: {max_iterations}")
    print("Type 'exit' or 'quit' to stop.")
    print()


def run_cli() -> None:

    args = parse_args()
    max_iterations = settings.normalize_max_iterations(args.max_iterations)

    print_welcome(
        session_id=args.session,
        user_id=args.user,
        max_iterations=max_iterations,
    )

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not query:
            continue

        try:
            result = invoke_agent(
                query=query,
                session_id=args.session,
                user_id=args.user,
                max_iterations=max_iterations,
            )
        except RuntimeError as exc:
            print(f"\nError: {exc}\n")
            continue

        trace = format_reasoning_trace(
            route=result.get("route"),
            route_reason=result.get("route_reason"),
            tool_trace=result.get("tool_trace", []),
        )

        if trace:
            print()
            print(trace)
            print()

        print(f"Agent: {result.get('final_answer')}")
        print()


if __name__ == "__main__":
    run_cli()