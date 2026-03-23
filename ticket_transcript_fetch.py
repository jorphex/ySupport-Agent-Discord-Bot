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
    author_name: str
    author_is_bot: bool
    content: str
    attachments: tuple[str, ...]


@dataclass(frozen=True)
class ChannelMetadata:
    id: str
    name: str


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
        author_name=str(raw_message.get("author", {}).get("global_name") or raw_message.get("author", {}).get("username") or "unknown"),
        author_is_bot=bool(raw_message.get("author", {}).get("bot")),
        content=content,
        attachments=attachments,
    )


def render_transcript(messages: Iterable[TranscriptMessage]) -> str:
    rendered_lines: list[str] = []
    for message in messages:
        try:
            timestamp = datetime.fromisoformat(message.timestamp.replace("Z", "+00:00")).isoformat()
        except ValueError:
            timestamp = message.timestamp
        speaker = f"{message.author_name}{' [bot]' if message.author_is_bot else ''}"
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
                    "author_name": message.author_name,
                    "author_is_bot": message.author_is_bot,
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
