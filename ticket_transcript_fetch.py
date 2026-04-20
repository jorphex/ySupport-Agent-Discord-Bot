import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from urllib import parse
from discord_api import DISCORD_API_BASE, discord_get_json


@dataclass(frozen=True)
class TranscriptMessage:
    id: str
    timestamp: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content: str
    attachments: tuple[str, ...]


@dataclass(frozen=True)
class ChannelMetadata:
    id: str
    name: str


SUPPORT_BOT_NAMES = {"ysupport"}


def extract_channel_id(channel_or_link: str) -> str:
    candidate = channel_or_link.strip()
    if candidate.isdigit():
        return candidate
    parsed = parse.urlparse(candidate)
    path_parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc == "discord.com" and len(path_parts) >= 3 and path_parts[0] == "channels":
        channel_id = path_parts[2]
        if channel_id.isdigit():
            return channel_id
    raise ValueError("Expected a Discord channel id or discord.com/channels/... link.")


def fetch_channel_messages(channel_id: str, limit: int = 100) -> list[dict[str, Any]]:
    remaining = max(limit, 0)
    before_message_id: str | None = None
    messages: list[dict[str, Any]] = []
    while remaining > 0:
        batch_size = min(remaining, 100)
        query = {"limit": str(batch_size)}
        if before_message_id:
            query["before"] = before_message_id
        url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages?{parse.urlencode(query)}"
        batch = discord_get_json(url)
        if not isinstance(batch, list) or not batch:
            break
        messages.extend(batch)
        before_message_id = batch[-1]["id"]
        remaining -= len(batch)
        if len(batch) < batch_size:
            break
    return list(reversed(messages))


def fetch_channel_metadata(channel_id: str) -> ChannelMetadata:
    raw_channel = discord_get_json(f"{DISCORD_API_BASE}/channels/{channel_id}")
    channel_name = str(raw_channel.get("name") or channel_id)
    return ChannelMetadata(id=channel_id, name=channel_name)


def normalize_message(raw_message: dict[str, Any]) -> TranscriptMessage:
    author = raw_message.get("author", {})
    attachments = tuple(
        attachment.get("url", "")
        for attachment in raw_message.get("attachments", [])
        if attachment.get("url")
    )
    content = (raw_message.get("content") or "").strip()
    if not content and attachments:
        content = "(attachment only)"
    return TranscriptMessage(
        id=str(raw_message["id"]),
        timestamp=str(raw_message["timestamp"]),
        author_id=str(author.get("id") or ""),
        author_name=str(author.get("global_name") or author.get("username") or "unknown"),
        author_is_bot=bool(author.get("bot")),
        content=content,
        attachments=attachments,
    )


def _normalize_speaker_key(message: TranscriptMessage) -> str:
    author_id = message.author_id.strip()
    if author_id:
        return author_id
    return message.author_name.strip().lower()


def _speaker_role(message: TranscriptMessage, *, ticket_user_key: str | None) -> str:
    if message.author_is_bot:
        if message.author_name.strip().lower() in SUPPORT_BOT_NAMES:
            return "support_bot"
        return "other_bot"
    if ticket_user_key and _normalize_speaker_key(message) == ticket_user_key:
        return "ticket_user"
    return "human_contributor"


def _collect_speaker_summary(messages: list[TranscriptMessage], *, ticket_user_key: str | None) -> list[str]:
    speakers_by_role: dict[str, list[str]] = {
        "ticket_user": [],
        "support_bot": [],
        "human_contributor": [],
        "other_bot": [],
    }
    seen_names_by_role: dict[str, set[str]] = {role: set() for role in speakers_by_role}

    for message in messages:
        role = _speaker_role(message, ticket_user_key=ticket_user_key)
        speaker_name = message.author_name.strip() or "unknown"
        normalized_name = speaker_name.lower()
        if normalized_name in seen_names_by_role[role]:
            continue
        seen_names_by_role[role].add(normalized_name)
        speakers_by_role[role].append(speaker_name)

    lines = ["Speakers:"]
    for role in ("ticket_user", "support_bot", "human_contributor", "other_bot"):
        names = speakers_by_role[role]
        if not names:
            continue
        lines.append(f"- {role}: {', '.join(names)}")
    return lines


def render_transcript(messages: Iterable[TranscriptMessage]) -> str:
    message_list = list(messages)
    ticket_user_key = next(
        (_normalize_speaker_key(message) for message in message_list if not message.author_is_bot),
        None,
    )
    rendered_lines: list[str] = []
    if message_list:
        rendered_lines.extend(_collect_speaker_summary(message_list, ticket_user_key=ticket_user_key))
        rendered_lines.append("")
    for message in message_list:
        try:
            timestamp = datetime.fromisoformat(message.timestamp.replace("Z", "+00:00")).isoformat()
        except ValueError:
            timestamp = message.timestamp
        speaker_role = _speaker_role(message, ticket_user_key=ticket_user_key)
        speaker = f"{speaker_role}({message.author_name})"
        rendered_lines.append(f"[{timestamp}] {speaker}: {message.content}")
        for attachment_url in message.attachments:
            rendered_lines.append(f"  attachment: {attachment_url}")
    return "\n".join(rendered_lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a Discord ticket transcript read-only and normalize it for regression review.",
    )
    parser.add_argument("channel", help="Discord ticket channel id or discord.com/channels/... link")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of recent messages to fetch (default: 50, max per request is paginated automatically).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit normalized JSON instead of plain text transcript output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        channel_id = extract_channel_id(args.channel)
        channel_metadata = fetch_channel_metadata(channel_id)
        raw_messages = fetch_channel_messages(channel_id, limit=args.limit)
        normalized_messages = [normalize_message(message) for message in raw_messages]
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.as_json:
        payload = {
            "channel_id": channel_id,
            "channel_name": channel_metadata.name,
            "message_count": len(normalized_messages),
            "messages": [
                {
                    "id": message.id,
                    "timestamp": message.timestamp,
                    "author_id": message.author_id,
                    "author_name": message.author_name,
                    "author_is_bot": message.author_is_bot,
                    "speaker_role": _speaker_role(
                        message,
                        ticket_user_key=next(
                            (
                                _normalize_speaker_key(candidate)
                                for candidate in normalized_messages
                                if not candidate.author_is_bot
                            ),
                            None,
                        ),
                    ),
                    "content": message.content,
                    "attachments": list(message.attachments),
                }
                for message in normalized_messages
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(render_transcript(normalized_messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
