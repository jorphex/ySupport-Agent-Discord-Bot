from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Iterator

import config
from codex_support_sessions import CodexSupportSessionManager
from ticket_execution.runtime_factory import build_local_ticket_investigation_executor
from ticket_investigation.json_endpoint import build_ticket_execution_json_endpoint
from ticket_investigation.transport import TicketExecutionTransportRequest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a ticket execution transport request against a chosen backend.",
    )
    parser.add_argument(
        "request_path",
        nargs="?",
        help="Optional path to a transport request JSON file. Reads stdin when omitted or '-'.",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "subprocess", "codex_support_exec"),
        help="Override the primary ticket execution backend for this replay run.",
    )
    parser.add_argument(
        "--fallback-mode",
        choices=("local", "subprocess", "codex_support_exec"),
        help="Override the fallback backend for this replay run.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable the configured fallback backend for this replay run.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the result JSON.",
    )
    parser.add_argument(
        "--session-dir",
        help=(
            "Optional isolated Codex session directory for replay continuity. "
            "Replays are ephemeral by default."
        ),
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Reset this request in the explicit --session-dir before replaying it.",
    )
    return parser


def load_request_json(request_path: str | None) -> str:
    if request_path and request_path != "-":
        with open(request_path, encoding="utf-8") as handle:
            return handle.read()
    return sys.stdin.read()


@contextmanager
def temporary_ticket_execution_modes(
    mode: str | None,
    fallback_mode: str | None,
    session_dir: str | None,
) -> Iterator[None]:
    original_mode = config.TICKET_EXECUTION_ENDPOINT
    original_fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
    original_session_dir = config.TICKET_EXECUTION_CODEX_SESSION_DIR
    try:
        if mode is not None:
            config.TICKET_EXECUTION_ENDPOINT = mode
        if fallback_mode is not None:
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT = fallback_mode
        config.TICKET_EXECUTION_CODEX_SESSION_DIR = session_dir
        yield
    finally:
        config.TICKET_EXECUTION_ENDPOINT = original_mode
        config.TICKET_EXECUTION_FALLBACK_ENDPOINT = original_fallback_mode
        config.TICKET_EXECUTION_CODEX_SESSION_DIR = original_session_dir


async def execute_request_json(
    request_json: str,
    *,
    mode: str | None = None,
    fallback_mode: str | None = None,
    fresh_session: bool = False,
    session_dir: str | None = None,
) -> str:
    _validate_replay_session_dir(session_dir)
    if fresh_session:
        if not session_dir:
            raise ValueError("fresh_session requires an explicit replay session_dir.")
        _reset_replay_session(request_json, session_dir=session_dir)
    with temporary_ticket_execution_modes(mode, fallback_mode, session_dir):
        delegate = _build_replay_delegate(
            primary_mode=config.TICKET_EXECUTION_ENDPOINT,
            fallback_mode=config.TICKET_EXECUTION_FALLBACK_ENDPOINT,
        )
        endpoint = build_ticket_execution_json_endpoint(delegate)
        return await endpoint.execute_json_turn(request_json)


def _validate_replay_session_dir(session_dir: str | None) -> None:
    production_session_dir = config.TICKET_EXECUTION_CODEX_SESSION_DIR
    if not session_dir or not production_session_dir:
        return
    if Path(session_dir).expanduser().resolve() == Path(
        production_session_dir
    ).expanduser().resolve():
        raise ValueError("Replay session_dir must not be the configured live session directory.")


def _build_replay_delegate(*, primary_mode: str, fallback_mode: str) -> object:
    if primary_mode == "local" or fallback_mode == "local":
        return build_local_ticket_investigation_executor()
    return _NoopExecutor()


class _NoopExecutor:
    async def execute_turn(self, request, hooks=None):
        raise AssertionError("Replay delegate should not execute for non-local backends.")


def _reset_replay_session(request_json: str, *, session_dir: str) -> None:
    request = TicketExecutionTransportRequest.from_json(request_json)
    manager = CodexSupportSessionManager(
        session_dir,
        max_age_hours=config.TICKET_EXECUTION_CODEX_SESSION_MAX_AGE_HOURS,
    )
    conversation_key = manager.conversation_key_for_request(request)
    if conversation_key is not None:
        manager.reset(conversation_key)


async def _main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.fresh_session and not args.session_dir:
        parser.error("--fresh-session requires --session-dir.")
    try:
        _validate_replay_session_dir(args.session_dir)
    except ValueError as exc:
        parser.error(str(exc))
    effective_mode = args.mode or config.TICKET_EXECUTION_ENDPOINT
    effective_fallback_mode = "" if args.no_fallback else (
        args.fallback_mode
        if args.fallback_mode is not None
        else config.TICKET_EXECUTION_FALLBACK_ENDPOINT
    )
    if effective_mode == "local" or effective_fallback_mode == "local":
        from agents import set_default_openai_key

        set_default_openai_key(config.OPENAI_API_KEY)

    request_json = load_request_json(args.request_path)
    if not request_json.strip():
        print("Expected ticket execution JSON from stdin or request_path.", file=sys.stderr)
        return 1

    response_json = await execute_request_json(
        request_json,
        mode=args.mode,
        fallback_mode=effective_fallback_mode if args.no_fallback or args.fallback_mode is not None else None,
        fresh_session=args.fresh_session,
        session_dir=args.session_dir,
    )
    if args.pretty:
        sys.stdout.write(json.dumps(json.loads(response_json), indent=2, sort_keys=True))
    else:
        sys.stdout.write(response_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
