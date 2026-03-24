import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config
from discord_api import DISCORD_API_BASE, discord_get_json, discord_post_json
from knowledge_gap_reporting import (
    KnowledgeGapCandidate,
    KnowledgeGapReport,
    PreparedTicketTranscript,
    _build_repo_grounding_query,
    _should_fetch_repo_grounding,
    analyze_transcript_for_knowledge_gap,
    finalize_knowledge_gap_report,
    format_knowledge_gap_report,
)
from ticket_transcript_fetch import (
    extract_channel_id,
    fetch_channel_metadata,
    fetch_channel_messages,
    normalize_message,
    render_transcript,
)
from utils import split_long_message

__all__ = [
    "KnowledgeGapCandidate",
    "KnowledgeGapReport",
    "PreparedTicketTranscript",
    "_build_repo_grounding_query",
    "_is_closed_ticket",
    "_build_report_signature",
    "_should_fetch_repo_grounding",
    "_snowflake_sort_key",
    "discover_recent_closed_ticket_channels",
    "preview_recent_closed_ticket_channels",
    "finalize_knowledge_gap_report",
    "prepare_ticket_transcript",
    "analyze_transcript_for_knowledge_gap",
    "format_knowledge_gap_report",
    "post_report_message",
    "process_ticket",
    "process_tickets",
    "_main_async",
    "main",
]


@dataclass
class PendingReportGroup:
    report: KnowledgeGapReport
    transcripts: list[PreparedTicketTranscript]
    result_indexes: list[int]


def prepare_ticket_transcript(channel_or_link: str, limit: int = 80) -> PreparedTicketTranscript:
    channel_id = extract_channel_id(channel_or_link)
    channel_metadata = fetch_channel_metadata(channel_id)
    raw_messages = fetch_channel_messages(channel_id, limit=limit)
    normalized_messages = [normalize_message(message) for message in raw_messages]
    return PreparedTicketTranscript(
        channel_id=channel_id,
        channel_name=channel_metadata.name,
        message_count=len(normalized_messages),
        transcript_text=render_transcript(normalized_messages),
    )


def _is_closed_ticket(prepared_transcript: PreparedTicketTranscript) -> bool:
    return prepared_transcript.channel_name.lower().startswith("closed-")


def _build_report_signature(report: KnowledgeGapReport) -> str:
    signature_payload = {
        "category": report.category,
        "topic": report.topic.strip().lower(),
        "product": (report.product or "").strip().lower(),
        "chain": (report.chain or "").strip().lower(),
    }
    serialized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_reported_signatures(state_path: str | None) -> set[str]:
    if not state_path:
        return set()
    path = Path(state_path)
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    signatures = payload.get("reported_signatures")
    if not isinstance(signatures, list):
        return set()
    return {str(signature) for signature in signatures if str(signature).strip()}


def _save_reported_signatures(state_path: str | None, signatures: set[str]) -> None:
    if not state_path:
        return
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reported_signatures": sorted(signatures)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _snowflake_sort_key(snowflake: str | None) -> int:
    if not snowflake or not str(snowflake).isdigit():
        return 0
    return int(snowflake)


def discover_recent_closed_ticket_channels(limit: int) -> list[str]:
    if config.YEARN_TICKET_CATEGORY_ID is None:
        raise ValueError("YEARN_TICKET_CATEGORY_ID is required for recent closed-ticket discovery.")

    category_payload = discord_get_json(
        f"{DISCORD_API_BASE}/channels/{config.YEARN_TICKET_CATEGORY_ID}"
    )
    guild_id = category_payload.get("guild_id")
    if not guild_id:
        raise ValueError("Could not determine guild_id from YEARN_TICKET_CATEGORY_ID.")

    guild_channels = discord_get_json(f"{DISCORD_API_BASE}/guilds/{guild_id}/channels")
    if not isinstance(guild_channels, list):
        raise ValueError("Unexpected Discord response while listing guild channels.")

    closed_channels: list[dict] = []
    same_parent_closed_channels: list[dict] = []
    for raw_channel in guild_channels:
        if raw_channel.get("type") != 0:
            continue
        channel_name = str(raw_channel.get("name") or "")
        if not channel_name.startswith("closed-"):
            continue
        closed_channels.append(raw_channel)
        if str(raw_channel.get("parent_id") or "") == str(config.YEARN_TICKET_CATEGORY_ID):
            same_parent_closed_channels.append(raw_channel)

    preferred_channels = same_parent_closed_channels or closed_channels
    ranked_channels = sorted(
        preferred_channels,
        key=lambda raw_channel: _snowflake_sort_key(
            str(raw_channel.get("last_message_id") or raw_channel.get("id") or "")
        ),
        reverse=True,
    )
    return [
        str(raw_channel["id"])
        for raw_channel in ranked_channels[: max(limit, 0)]
        if raw_channel.get("id")
    ]


def preview_recent_closed_ticket_channels(limit: int) -> list[dict[str, str]]:
    channel_ids = discover_recent_closed_ticket_channels(limit)
    previews: list[dict[str, str]] = []
    for channel_id in channel_ids:
        metadata = fetch_channel_metadata(channel_id)
        previews.append(
            {
                "channel_id": metadata.id,
                "channel_name": metadata.name,
            }
        )
    return previews


def post_report_message(channel_id: int, content: str) -> None:
    for chunk in split_long_message(content):
        discord_post_json(
            f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
            {"content": chunk, "flags": 4},
        )


async def process_ticket(
    channel_or_link: str,
    *,
    limit: int,
    report_channel_id: int,
    dry_run: bool,
    closed_only: bool = False,
    known_signatures: Optional[set[str]] = None,
    max_posts: Optional[int] = None,
    posted_count: int = 0,
) -> dict[str, object]:
    results = await process_tickets(
        [channel_or_link],
        limit=limit,
        report_channel_id=report_channel_id,
        dry_run=dry_run,
        closed_only=closed_only,
        state_path=None,
        max_posts=max_posts,
        known_signatures_override=known_signatures if known_signatures is not None else set(),
        posted_groups_so_far=posted_count,
    )
    return results[0]


async def process_tickets(
    channels: list[str],
    *,
    limit: int,
    report_channel_id: int,
    dry_run: bool,
    closed_only: bool,
    state_path: str | None,
    max_posts: Optional[int],
    known_signatures_override: Optional[set[str]] = None,
    posted_groups_so_far: int = 0,
) -> list[dict[str, object]]:
    known_signatures = (
        known_signatures_override
        if known_signatures_override is not None
        else _load_reported_signatures(state_path)
    )
    results: list[dict[str, object]] = []
    pending_groups: dict[str, PendingReportGroup] = {}

    for channel in channels:
        prepared = await asyncio.to_thread(prepare_ticket_transcript, channel, limit)
        if closed_only and not _is_closed_ticket(prepared):
            results.append(
                {
                    "channel_id": prepared.channel_id,
                    "channel_name": prepared.channel_name,
                    "message_count": prepared.message_count,
                    "report_posted": False,
                    "report": None,
                    "skipped_reason": "open_ticket",
                }
            )
            continue

        report = await analyze_transcript_for_knowledge_gap(prepared)
        if report is None:
            results.append(
                {
                    "channel_id": prepared.channel_id,
                    "channel_name": prepared.channel_name,
                    "message_count": prepared.message_count,
                    "report_posted": False,
                    "report": None,
                }
            )
            continue

        report_signature = _build_report_signature(report)
        if report_signature in known_signatures:
            results.append(
                {
                    "channel_id": prepared.channel_id,
                    "channel_name": prepared.channel_name,
                    "message_count": prepared.message_count,
                    "report_posted": False,
                    "report": report.model_dump(),
                    "report_signature": report_signature,
                    "skipped_reason": "duplicate_report",
                }
            )
            continue

        result_index = len(results)
        results.append(
            {
                "channel_id": prepared.channel_id,
                "channel_name": prepared.channel_name,
                "message_count": prepared.message_count,
                "report_posted": False,
                "report": report.model_dump(),
                "report_signature": report_signature,
            }
        )
        if report_signature not in pending_groups:
            pending_groups[report_signature] = PendingReportGroup(
                report=report,
                transcripts=[prepared],
                result_indexes=[result_index],
            )
        else:
            pending_groups[report_signature].transcripts.append(prepared)
            pending_groups[report_signature].result_indexes.append(result_index)

    posted_group_count = posted_groups_so_far
    for report_signature, group in pending_groups.items():
        formatted = format_knowledge_gap_report(group.report, affected_channels=group.transcripts)
        within_post_budget = max_posts is None or posted_group_count < max_posts
        if within_post_budget and not dry_run:
            await asyncio.to_thread(post_report_message, report_channel_id, formatted)
            known_signatures.add(report_signature)
            posted_group_count += 1
        elif within_post_budget:
            posted_group_count += 1

        for result_index in group.result_indexes:
            results[result_index]["formatted_report"] = formatted
            results[result_index]["group_size"] = len(group.transcripts)
            results[result_index]["group_channel_ids"] = [ticket.channel_id for ticket in group.transcripts]
            if within_post_budget:
                results[result_index]["report_posted"] = not dry_run
            else:
                results[result_index]["skipped_reason"] = "max_posts_reached"

    if not dry_run and known_signatures_override is None:
        _save_reported_signatures(state_path, known_signatures)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Discord support tickets offline for docs gaps, FAQ candidates, "
            "and other private internal knowledge-gap reports."
        ),
    )
    parser.add_argument(
        "channels",
        nargs="*",
        help="Discord ticket channel ids or discord.com/channels/... links.",
    )
    parser.add_argument(
        "--recent-closed",
        type=int,
        default=0,
        help="Discover and analyze the N most recent closed-* ticket channels from the Yearn ticket guild.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum number of recent messages to fetch per ticket.",
    )
    parser.add_argument(
        "--report-channel-id",
        type=int,
        default=config.KNOWLEDGE_GAP_REPORT_CHANNEL_ID,
        help="Private Discord channel id that receives formatted reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print JSON results without posting to Discord.",
    )
    parser.add_argument(
        "--closed-only",
        action="store_true",
        help="Only analyze ticket channels whose names start with 'closed-'.",
    )
    parser.add_argument(
        "--state-path",
        default=None,
        help="Optional JSON file used to persist already-posted report signatures for dedupe.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Optional maximum number of reports to post in a single run after dedupe/filtering.",
    )
    parser.add_argument(
        "--preview-discovery",
        action="store_true",
        help="Print the recent closed-ticket discovery result and exit before analysis/posting.",
    )
    return parser


async def _main_async(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    channels = list(args.channels)
    if args.recent_closed > 0:
        discovered_channels = await asyncio.to_thread(
            discover_recent_closed_ticket_channels,
            args.recent_closed,
        )
        channels.extend(
            channel_id
            for channel_id in discovered_channels
            if channel_id not in channels
        )
    if args.preview_discovery:
        discovered_preview = []
        if args.recent_closed > 0:
            discovered_preview = await asyncio.to_thread(
                preview_recent_closed_ticket_channels,
                args.recent_closed,
            )
        print(
            json.dumps(
                {
                    "preview_only": True,
                    "selected_channels": channels,
                    "recent_closed_preview": discovered_preview,
                },
                indent=2,
            )
        )
        return 0
    if not channels:
        parser.error("Provide at least one channel or use --recent-closed N.")

    results = await process_tickets(
        channels,
        limit=args.limit,
        report_channel_id=args.report_channel_id,
        dry_run=args.dry_run,
        closed_only=args.closed_only,
        state_path=args.state_path,
        max_posts=args.max_posts,
    )
    print(json.dumps({"results": results}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
