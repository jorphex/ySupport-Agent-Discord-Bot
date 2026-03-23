import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from agents import Agent, Runner, RunConfig, ModelSettings
from agents.model_settings import Reasoning

import config
import tools_lib
from discord_api import DISCORD_API_BASE, discord_get_json, discord_post_json
from ticket_transcript_fetch import (
    extract_channel_id,
    fetch_channel_metadata,
    fetch_channel_messages,
    normalize_message,
    render_transcript,
)
from utils import split_long_message


class KnowledgeGapCandidate(BaseModel):
    reportable: bool = Field(..., description="Whether this ticket is worth private internal reporting.")
    category: Literal[
        "docs_gap",
        "faq_candidate",
        "bot_behavior_gap",
        "product_confusion",
        "issue_draft_candidate",
        "no_action",
    ] = Field(..., description="Best-fit internal report category.")
    title: str = Field(..., description="Short internal title.")
    topic: str = Field(..., description="Main topic or user confusion area.")
    product: Optional[str] = Field(default=None, description="Relevant product or system if clear.")
    chain: Optional[str] = Field(default=None, description="Relevant chain if clear.")
    grounding_query: str = Field(
        ...,
        description="A concise query for official docs/repo grounding, based on the real issue in the transcript.",
    )
    evidence_summary: str = Field(..., description="Short neutral summary of what happened in the ticket.")
    suggested_action: str = Field(..., description="Best next internal action.")
    needs_repo_context: bool = Field(
        ...,
        description="Whether repo context is likely needed in addition to docs to assess the issue properly.",
    )


class KnowledgeGapReport(BaseModel):
    should_post: bool = Field(..., description="Whether the final grounded result should be posted to the private channel.")
    category: Literal[
        "docs_gap",
        "faq_candidate",
        "bot_behavior_gap",
        "product_confusion",
        "issue_draft_candidate",
        "no_action",
    ] = Field(..., description="Best-fit internal report category.")
    title: str = Field(..., description="Short report title.")
    topic: str = Field(..., description="Main topic or problem area.")
    product: Optional[str] = Field(default=None, description="Relevant product if clear.")
    chain: Optional[str] = Field(default=None, description="Relevant chain if clear.")
    evidence_summary: str = Field(..., description="Short evidence-backed summary of the ticket issue.")
    current_official_grounding: str = Field(
        ...,
        description="What official docs/repo context currently supports, or that it appears missing.",
    )
    assessment: str = Field(..., description="Why this should or should not become an internal report.")
    suggested_action: str = Field(..., description="Recommended internal action.")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Confidence in the assessment.")


def _knowledge_gap_model_settings() -> ModelSettings:
    return ModelSettings(
        reasoning=Reasoning(effort=config.LLM_KNOWLEDGE_GAP_REASONING_EFFORT),
        verbosity=config.LLM_KNOWLEDGE_GAP_VERBOSITY,
    )


KNOWLEDGE_GAP_CANDIDATE_INSTRUCTIONS = """
You are an internal Yearn support-quality analyst.

Your job is to decide whether a Discord support ticket should become a private internal
knowledge-gap report.

Report only if the transcript reveals one of these:
- missing official docs or missing official source coverage
- repeated or likely recurring confusion that suggests an FAQ
- a bot behavior gap where the official source likely exists but was not surfaced well
- product/UI confusion that should be surfaced internally
- a likely engineering/product issue that may deserve an internal issue draft

Do NOT report:
- one-off resolved account-specific questions with no broader lesson
- normal successful support turns
- cases where the transcript is too thin to justify any internal follow-up

Prefer a narrow, durable framing over ticket-specific details.
The grounding_query must be the real question that official docs or repo sources should answer.
If no report is warranted, set category to `no_action`, reportable to false, and keep the rest concise.
"""


KNOWLEDGE_GAP_REPORT_INSTRUCTIONS = """
You are finalizing a private internal Yearn support-quality report.

You will receive:
- the ticket transcript
- an initial candidate assessment
- official docs grounding
- optional repo grounding

Your job:
- decide whether the issue should actually be posted to the private internal channel
- stay grounded in the provided official source context
- if official grounding is missing, say that plainly instead of inventing support
- if official grounding exists but the bot or ticket flow failed to surface it, classify that as bot_behavior_gap or faq_candidate
- if the issue is mostly product/UI confusion rather than missing docs, say that directly
- if it is not worth surfacing, set should_post to false and category to no_action

Keep the output concise and internal-facing.
Do not write as if you are replying to the end user.
"""


knowledge_gap_candidate_agent = Agent(
    name="Knowledge Gap Candidate Agent",
    instructions=KNOWLEDGE_GAP_CANDIDATE_INSTRUCTIONS,
    output_type=KnowledgeGapCandidate,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


knowledge_gap_report_agent = Agent(
    name="Knowledge Gap Report Agent",
    instructions=KNOWLEDGE_GAP_REPORT_INSTRUCTIONS,
    output_type=KnowledgeGapReport,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


@dataclass(frozen=True)
class PreparedTicketTranscript:
    channel_id: str
    channel_name: str
    message_count: int
    transcript_text: str


@dataclass
class PendingReportGroup:
    report: KnowledgeGapReport
    transcripts: list[PreparedTicketTranscript]
    result_indexes: list[int]


@lru_cache(maxsize=1)
def _supported_chain_id_map() -> dict[int, str]:
    chain_ids: dict[int, str] = {}
    for chain_name, web3_instance in tools_lib.WEB3_INSTANCES.items():
        try:
            chain_ids[int(web3_instance.eth.chain_id)] = chain_name
        except Exception:
            continue
    return chain_ids


def _supported_chain_names() -> set[str]:
    return set(config.RPC_URLS)


def _extract_transcript_chain_hint(prepared_transcript: PreparedTicketTranscript) -> Optional[str]:
    transcript_text = prepared_transcript.transcript_text.lower()
    chain_id_matches = {
        int(match.group(1))
        for match in re.finditer(r"\bchain(?:id)?\s*[:=]?\s*(\d+)\b", transcript_text)
    }
    supported_chain_ids = _supported_chain_id_map()
    matched_supported_names = {
        supported_chain_ids[chain_id]
        for chain_id in chain_id_matches
        if chain_id in supported_chain_ids
    }
    if len(matched_supported_names) == 1:
        return next(iter(matched_supported_names))
    if len(matched_supported_names) > 1:
        return None

    explicit_name_matches = {
        chain_name
        for chain_name in _supported_chain_names()
        if re.search(rf"\b{re.escape(chain_name)}\b", transcript_text)
    }
    if len(explicit_name_matches) == 1:
        return next(iter(explicit_name_matches))
    return None


def _normalize_report_chain(chain_value: Optional[str], *, transcript_chain_hint: Optional[str]) -> Optional[str]:
    if transcript_chain_hint:
        return transcript_chain_hint
    if not chain_value:
        return None

    normalized_value = chain_value.strip().lower()
    if not normalized_value:
        return None

    for chain_name in _supported_chain_names():
        if re.search(rf"\b{re.escape(chain_name)}\b", normalized_value):
            return chain_name

    for chain_id, chain_name in _supported_chain_id_map().items():
        if re.search(rf"\b{chain_id}\b", normalized_value):
            return chain_name

    return None


def finalize_knowledge_gap_report(
    report: KnowledgeGapReport,
    prepared_transcript: PreparedTicketTranscript,
) -> KnowledgeGapReport:
    transcript_chain_hint = _extract_transcript_chain_hint(prepared_transcript)
    normalized_payload = report.model_dump()
    normalized_payload["chain"] = _normalize_report_chain(
        report.chain,
        transcript_chain_hint=transcript_chain_hint,
    )
    return KnowledgeGapReport(**normalized_payload)


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
        "title": report.title.strip().lower(),
        "topic": report.topic.strip().lower(),
        "product": (report.product or "").strip().lower(),
        "chain": (report.chain or "").strip().lower(),
        "suggested_action": report.suggested_action.strip().lower(),
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


async def _run_structured_agent(agent: Agent, input_text: str, output_type: type[BaseModel], workflow_name: str):
    runner = Runner()
    result = await runner.run(
        starting_agent=agent,
        input=input_text,
        run_config=RunConfig(workflow_name=workflow_name, tracing_disabled=True),
    )
    return result.final_output_as(output_type)


async def analyze_transcript_for_knowledge_gap(
    prepared_transcript: PreparedTicketTranscript,
) -> Optional[KnowledgeGapReport]:
    candidate = await _run_structured_agent(
        knowledge_gap_candidate_agent,
        prepared_transcript.transcript_text,
        KnowledgeGapCandidate,
        "Knowledge Gap Candidate Analysis",
    )
    if not candidate.reportable or candidate.category == "no_action":
        return None

    docs_grounding = await tools_lib.core_answer_from_docs(candidate.grounding_query)
    repo_grounding = ""
    if candidate.needs_repo_context:
        repo_grounding = await tools_lib.core_pretriage_repo_claim(
            candidate.grounding_query,
            include_docs=False,
        )

    report_input = (
        f"Channel ID: {prepared_transcript.channel_id}\n"
        f"Message count: {prepared_transcript.message_count}\n\n"
        f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
        f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
        f"Docs grounding:\n{docs_grounding}\n\n"
        f"Repo grounding:\n{repo_grounding or 'none'}"
    )
    report = await _run_structured_agent(
        knowledge_gap_report_agent,
        report_input,
        KnowledgeGapReport,
        "Knowledge Gap Final Report",
    )
    if not report.should_post or report.category == "no_action":
        return None
    return finalize_knowledge_gap_report(report, prepared_transcript)


def format_knowledge_gap_report(
    report: KnowledgeGapReport,
    *,
    affected_channels: list[PreparedTicketTranscript],
) -> str:
    affected_ticket_refs = ", ".join(
        f"<#{ticket.channel_id}> ({ticket.channel_id})" for ticket in affected_channels
    )
    lines = ["**Knowledge-Gap Report**", ""]
    lines.extend(
        [
            f"**Title:** {report.title}",
            f"**Category:** {report.category}",
            f"**Topic:** {report.topic}",
        ]
    )
    if report.product:
        lines.append(f"**Product:** {report.product}")
    if report.chain:
        lines.append(f"**Chain:** {report.chain}")
    lines.extend(
        [
            f"**Affected tickets:** {affected_ticket_refs}",
            f"**Confidence:** {report.confidence}",
            "",
            "**Evidence**",
            report.evidence_summary,
            "",
            "**Current official grounding**",
            report.current_official_grounding,
            "",
            "**Assessment**",
            report.assessment,
            "",
            "**Suggested action**",
            report.suggested_action,
        ]
    )
    return "\n".join(lines).strip()


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
    prepared = await asyncio.to_thread(prepare_ticket_transcript, channel_or_link, limit)
    if closed_only and not _is_closed_ticket(prepared):
        return {
            "channel_id": prepared.channel_id,
            "channel_name": prepared.channel_name,
            "message_count": prepared.message_count,
            "report_posted": False,
            "report": None,
            "skipped_reason": "open_ticket",
        }
    report = await analyze_transcript_for_knowledge_gap(prepared)
    if report is None:
        return {
            "channel_id": prepared.channel_id,
            "channel_name": prepared.channel_name,
            "message_count": prepared.message_count,
            "report_posted": False,
            "report": None,
        }

    report_signature = _build_report_signature(report)
    if known_signatures is not None and report_signature in known_signatures:
        return {
            "channel_id": prepared.channel_id,
            "channel_name": prepared.channel_name,
            "message_count": prepared.message_count,
            "report_posted": False,
            "report": report.model_dump(),
            "report_signature": report_signature,
            "skipped_reason": "duplicate_report",
        }

    if max_posts is not None and posted_count >= max_posts:
        return {
            "channel_id": prepared.channel_id,
            "channel_name": prepared.channel_name,
            "message_count": prepared.message_count,
            "report_posted": False,
            "report": report.model_dump(),
            "report_signature": report_signature,
            "skipped_reason": "max_posts_reached",
        }

    formatted = format_knowledge_gap_report(report, affected_channels=[prepared])
    if not dry_run:
        await asyncio.to_thread(post_report_message, report_channel_id, formatted)
    return {
        "channel_id": prepared.channel_id,
        "channel_name": prepared.channel_name,
        "message_count": prepared.message_count,
        "report_posted": not dry_run,
        "report": report.model_dump(),
        "report_signature": report_signature,
        "formatted_report": formatted,
    }


async def process_tickets(
    channels: list[str],
    *,
    limit: int,
    report_channel_id: int,
    dry_run: bool,
    closed_only: bool,
    state_path: str | None,
    max_posts: Optional[int],
) -> list[dict[str, object]]:
    known_signatures = _load_reported_signatures(state_path)
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

    posted_group_count = 0
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

    if not dry_run:
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
