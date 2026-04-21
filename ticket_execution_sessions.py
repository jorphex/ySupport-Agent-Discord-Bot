from __future__ import annotations

import argparse
import json

import config
from codex_support_sessions import CodexSupportSessionManager


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and manage stored Codex support sessions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Show active session summary.")

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a single session record by conversation key.",
    )
    inspect_parser.add_argument("conversation_key")

    reset_parser = subparsers.add_parser(
        "reset",
        help="Delete a single session record by conversation key.",
    )
    reset_parser.add_argument("conversation_key")

    subparsers.add_parser(
        "prune",
        help="Delete expired session records and report how many were removed.",
    )
    return parser


def build_session_manager() -> CodexSupportSessionManager:
    return CodexSupportSessionManager(
        config.TICKET_EXECUTION_CODEX_SESSION_DIR,
        max_age_hours=config.TICKET_EXECUTION_CODEX_SESSION_MAX_AGE_HOURS,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manager = build_session_manager()

    if args.command == "summary":
        print(json.dumps(manager.summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "inspect":
        print(json.dumps(manager.inspect(args.conversation_key), indent=2, sort_keys=True))
        return 0
    if args.command == "reset":
        manager.reset(args.conversation_key)
        print(
            json.dumps(
                {
                    "conversation_key": args.conversation_key,
                    "reset": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prune":
        print(
            json.dumps(
                {
                    "removed": manager.prune_expired(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
