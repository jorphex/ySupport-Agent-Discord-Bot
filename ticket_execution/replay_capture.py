from __future__ import annotations

import argparse
import sys

from ticket_investigation.transport import TicketExecutionTransportRequest
from ticket_transcript_fetch import (
    TranscriptMessage,
    extract_channel_id,
    fetch_channel_messages,
    fetch_channel_metadata,
    normalize_message,
)


_DEPOSITS_WITHDRAWALS_PROMPT = (
    "Got it. If you want deposit or withdrawal help, send your wallet address (0x...). "
    "If you are trying to find a vault or check vault info first, send the token, vault name, "
    "or vault address instead."
)
_GENERAL_INFO_PROMPT_PREFIX = (
    "Great! What specific information or product are you looking for, or what how-to question "
    "do you have about "
)
_INVESTIGATE_ISSUE_PROMPT = (
    "Describe the issue in as much detail as you have. You can send multiple messages now and I "
    "will review them together. Include the product or vault involved, the exact page or URL if "
    "relevant, what you expected to happen, what actually happened, any error text, any tx hash, "
    "and your browser/device or wallet if that matters."
)
_OTHER_ISSUE_PROMPT = (
    "Okay, please describe your issue or question in detail below. I'll do my best to assist or "
    "find the right help for you."
)


def infer_initial_button_intent(prompt_text: str) -> str | None:
    normalized = prompt_text.strip()
    if normalized == _DEPOSITS_WITHDRAWALS_PROMPT:
        return "data_deposits_withdrawals_start"
    if normalized.startswith(_GENERAL_INFO_PROMPT_PREFIX):
        return "docs_qa"
    if normalized == _INVESTIGATE_ISSUE_PROMPT:
        return "investigate_issue"
    if normalized == _OTHER_ISSUE_PROMPT:
        return "other_free_form"
    return None


def build_first_turn_transport_request(
    *,
    channel_id: str,
    messages: list[TranscriptMessage],
    target_message_id: str | None = None,
    project_context: str = "yearn",
) -> TicketExecutionTransportRequest:
    if not messages:
        raise ValueError("Transcript is empty.")

    target_index = _resolve_target_index(messages, target_message_id)
    start_index, end_index = _expand_ticket_user_block(messages, target_index)
    if any(message.author_name == "ySupport" and message.content == "Got it — updating based on your latest message." for message in messages[:start_index]):
        raise ValueError(
            "Target message is not a first-turn user block; later turns need captured transport state."
        )
    if any(message.author_name != "Ticket Tool" and _message_role(message) == "user" for message in messages[:start_index]):
        raise ValueError(
            "Target message is not the first user turn in the ticket; later-turn replay is not supported here."
        )

    initial_intent = _find_initial_intent(messages[:start_index])
    aggregated_text = "\n".join(
        message.content for message in messages[start_index : end_index + 1]
    ).strip()
    requested_intent = initial_intent
    mode = "collecting" if initial_intent == "investigate_issue" else "waiting_for_user"
    if requested_intent is None:
        mode = "idle"

    return TicketExecutionTransportRequest(
        aggregated_text=aggregated_text,
        input_list=[{"role": "user", "content": aggregated_text}],
        current_history=[],
        run_context={
            "channel_id": int(channel_id),
            "category_id": None,
            "is_public_trigger": False,
            "project_context": project_context,
            "initial_button_intent": initial_intent,
            "repo_search_calls": 0,
            "repo_fetch_calls": 0,
            "repo_searches_without_fetch": 0,
            "repo_last_search_query": None,
            "repo_last_search_artifact_refs": [],
        },
        investigation_job={
            "channel_id": int(channel_id),
            "requested_intent": requested_intent,
            "mode": mode,
            "current_specialty": None,
            "last_specialty": None,
            "evidence": {
                "wallet": None,
                "chain": None,
                "tx_hashes": [],
                "withdrawal_target_chain": None,
                "withdrawal_target_vault": None,
            },
        },
        workflow_name=f"ticket_replay.first_turn.{channel_id}",
        wants_bug_review_status=False,
    )


def _resolve_target_index(
    messages: list[TranscriptMessage],
    target_message_id: str | None,
) -> int:
    if target_message_id is None:
        for index, message in enumerate(messages):
            if _message_role(message) == "user":
                return index
        raise ValueError("No ticket user messages found in transcript.")
    for index, message in enumerate(messages):
        if message.id == target_message_id:
            if _message_role(message) != "user":
                raise ValueError("target_message_id must point to a ticket user message.")
            return index
    raise ValueError(f"target_message_id {target_message_id} was not found in transcript.")


def _expand_ticket_user_block(
    messages: list[TranscriptMessage],
    target_index: int,
) -> tuple[int, int]:
    target_message = messages[target_index]
    target_author_id = target_message.author_id
    start_index = target_index
    while start_index > 0:
        candidate = messages[start_index - 1]
        if _message_role(candidate) != "user" or candidate.author_id != target_author_id:
            break
        start_index -= 1

    end_index = target_index
    while end_index + 1 < len(messages):
        candidate = messages[end_index + 1]
        if _message_role(candidate) != "user" or candidate.author_id != target_author_id:
            break
        end_index += 1
    return start_index, end_index


def _find_initial_intent(messages: list[TranscriptMessage]) -> str | None:
    for message in reversed(messages):
        if message.author_is_bot and message.author_name == "ySupport":
            intent = infer_initial_button_intent(message.content)
            if intent is not None:
                return intent
    return None


def _message_role(message: TranscriptMessage) -> str:
    if message.author_is_bot:
        return "assistant"
    return "user"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a first-turn Discord ticket replay request as TicketExecutionTransportRequest JSON.",
    )
    parser.add_argument("channel", help="Discord ticket channel id or discord.com/channels/... link")
    parser.add_argument(
        "--target-message-id",
        help="Optional ticket-user message id inside the first-turn block to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum number of recent messages to fetch (default: 80).",
    )
    parser.add_argument(
        "--project-context",
        default="yearn",
        help="Project context to stamp into the replay request (default: yearn).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        channel_id = extract_channel_id(args.channel)
        fetch_channel_metadata(channel_id)
        messages = [
            normalize_message(message)
            for message in fetch_channel_messages(channel_id, limit=args.limit)
        ]
        request = build_first_turn_transport_request(
            channel_id=channel_id,
            messages=messages,
            target_message_id=args.target_message_id,
            project_context=args.project_context,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(request.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
