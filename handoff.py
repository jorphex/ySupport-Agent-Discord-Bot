from __future__ import annotations

import asyncio
from dataclasses import dataclass
import html
import json
import logging
import os
import re
import sys
from typing import Any, Literal
from urllib import request

import config


HandoffTarget = Literal[
    "support_manual",
    "strategist_ops",
    "web_team",
    "security_team",
]


@dataclass(frozen=True)
class HandoffRoute:
    target: HandoffTarget
    team_label: str
    reason: str


@dataclass(frozen=True)
class TelegramSentMessage:
    chat_id: str
    message_id: int
    message_text: str = ""


_STRATEGIST_OPS_TOKENS = (
    "harvest",
    "report",
    "keeper",
    "rewards",
    "reward",
    "dump",
    "compound",
    "compounding",
    "profit unlock",
    "pps",
    "queued",
    "pending signatures",
    "multisig",
    "swap",
)

_WEB_TEAM_TOKENS = (
    "frontend",
    "front-end",
    "website",
    "site",
    "page",
    "button",
    "ui ",
    " ui",
    "display",
    "chart",
    "loading",
    "load forever",
    "spinner",
    "wallet connect",
    "connect wallet",
)

_SECURITY_TEAM_TOKENS = (
    "immunefi",
    "sherlock",
    "bug bounty",
    "security report",
    "kyc",
    "jurisdiction",
    "official security process",
)

_VAGUE_TEAM_REPLIES = {
    "ok",
    "okay",
    "noted",
    "noted thanks",
    "thanks",
    "thank you",
    "seen",
    "ack",
    "acknowledged",
    "check later",
    "later",
    "we saw it",
    "got it",
    "done",
}


def strip_handoff_placeholder(text: str | None) -> str:
    if not text:
        return ""
    stripped = text.replace(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, " ").strip()
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    return stripped.strip()


def infer_handoff_route(
    *texts: str | None,
    explicit_target: HandoffTarget | None = None,
    explicit_reason: str | None = None,
) -> HandoffRoute:
    target = explicit_target or _infer_target_from_texts(*texts)
    return HandoffRoute(
        target=target,
        team_label=_team_label_for_target(target),
        reason=(explicit_reason or _default_reason_for_target(target)).strip(),
    )


def build_user_handoff_reply(
    base_reply: str | None,
    route: HandoffRoute,
    *,
    location: Literal["ticket", "here"] = "ticket",
) -> str:
    cleaned = strip_handoff_placeholder(base_reply)
    follow_up = (
        f"I've notified the {route.team_label} about this. "
        f"They'll follow up {'in this ticket' if location == 'ticket' else 'here'}."
    )
    if not cleaned:
        return follow_up
    if cleaned.endswith((".", "!", "?")):
        return f"{cleaned} {follow_up}"
    return f"{cleaned}. {follow_up}"


def build_handoff_notice(
    route: HandoffRoute,
    *,
    summary: str,
    channel_id: int,
    guild_id: int | None,
    ticket_label: str | None = None,
) -> str:
    label = ticket_label or _default_ticket_label(channel_id)
    link_line = (
        f'<a href="https://discord.com/channels/{guild_id}/{channel_id}">{html.escape(label)}</a>: '
        f"<code>{html.escape(route.target)}</code>"
        if guild_id is not None
        else f"<b>{html.escape(label)}</b>: <code>{html.escape(route.target)}</code>"
    )
    lines = [link_line]
    if guild_id is not None:
        lines.append("")
    lines.append(f"<b>Reason</b>: {html.escape(route.reason)}")
    lines.append(f"<b>Summary</b>: {html.escape(_summarize_text(summary))}")
    lines.append("")
    lines.append(
        "Reply to this message with what I should tell the user or do next. "
        "Your reply will be used for the next ticket update. "
        "Only one reply is accepted."
    )
    return "\n".join(lines)


def build_closed_handoff_notice(original_notice: str) -> str:
    lines = original_notice.splitlines()
    footer = (
        "Reply to this message with what I should tell the user or do next. "
        "Your reply will be used for the next ticket update. "
        "Only one reply is accepted."
    )
    closed_footer = "<i>Reply received. Replies closed.</i>"
    if lines and lines[-1].strip() == footer:
        lines[-1] = closed_footer
        return "\n".join(lines)
    return f"{original_notice}\n\n{closed_footer}"


async def send_handoff_notice(message_text: str) -> TelegramSentMessage | None:
    return await send_telegram_message(
        chat_id=config.TELEGRAM_YSUPPORT_CHAT,
        message_text=message_text,
    )


async def send_telegram_message(
    *,
    chat_id: str | None,
    message_text: str,
    reply_to_message_id: int | None = None,
) -> TelegramSentMessage | None:
    if "unittest" in sys.modules and os.getenv("YSUPPORT_ALLOW_TEST_TELEGRAM") != "1":
        return None
    if not config.TELEGRAM_BOT_TOKEN or not chat_id:
        return None

    payload = {
        "chat_id": chat_id,
        "text": message_text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    if reply_to_message_id is not None:
        payload["reply_to_message_id"] = reply_to_message_id
    response_payload = await _telegram_api_call("sendMessage", payload)
    if not response_payload:
        return None
    result = response_payload.get("result") or {}
    message_id = result.get("message_id")
    chat = result.get("chat") or {}
    chat_id = chat.get("id")
    if message_id is None or chat_id is None:
        return None
    return TelegramSentMessage(
        chat_id=str(chat_id),
        message_id=int(message_id),
        message_text=message_text,
    )


async def edit_handoff_notice(
    *,
    chat_id: str,
    message_id: int,
    message_text: str,
) -> bool:
    if "unittest" in sys.modules and os.getenv("YSUPPORT_ALLOW_TEST_TELEGRAM") != "1":
        return False
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": message_text,
        "disable_web_page_preview": True,
        "parse_mode": "HTML",
    }
    response_payload = await _telegram_api_call("editMessageText", payload)
    return bool(response_payload)


async def fetch_telegram_updates(offset: int | None = None) -> list[dict[str, Any]]:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_YSUPPORT_CHAT:
        return []
    payload: dict[str, Any] = {
        "timeout": 25,
        "allowed_updates": ["message"],
    }
    if offset is not None:
        payload["offset"] = offset
    response_payload = await _telegram_api_call("getUpdates", payload)
    if not response_payload:
        return []
    result = response_payload.get("result")
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


async def _telegram_api_call(
    method: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not config.TELEGRAM_BOT_TOKEN:
        return None
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"

    body = json.dumps(payload).encode("utf-8")

    def _send() -> dict[str, Any] | None:
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=35) as response:
            parsed = json.loads(response.read().decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
            return None

    try:
        result = await asyncio.to_thread(_send)
        if not result or result.get("ok") is not True:
            logging.error("Telegram API call %s failed: %s", method, result)
            return None
        return result
    except Exception as exc:
        logging.error("Telegram API call %s failed: %s", method, exc, exc_info=True)
        return None


def _default_ticket_label(channel_id: int) -> str:
    suffix = str(channel_id)[-4:]
    return f"Ticket-{suffix}"


def is_substantive_team_reply(text: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    normalized = normalized.strip(" .,!?:;")
    if not normalized:
        return False
    if normalized in _VAGUE_TEAM_REPLIES:
        return False
    words = normalized.split()
    if len(words) <= 2 and all(word in {"ok", "okay", "noted", "thanks", "seen"} for word in words):
        return False
    return True


def build_vague_team_reply_feedback() -> str:
    return (
        "That reply is too vague to send to the user. "
        "Reply with the exact update I should give them. "
        "Only one clear reply will be used."
    )


def _infer_target_from_texts(*texts: str | None) -> HandoffTarget:
    lowered = " ".join((text or "").lower() for text in texts)
    if any(token in lowered for token in _SECURITY_TEAM_TOKENS):
        return "security_team"
    if any(token in lowered for token in _WEB_TEAM_TOKENS):
        return "web_team"
    if any(token in lowered for token in _STRATEGIST_OPS_TOKENS):
        return "strategist_ops"
    return "support_manual"


def _team_label_for_target(target: HandoffTarget) -> str:
    if target == "strategist_ops":
        return "strategy team"
    if target == "web_team":
        return "web team"
    if target == "security_team":
        return "security team"
    return "support team"


def _default_reason_for_target(target: HandoffTarget) -> str:
    if target == "strategist_ops":
        return "likely manual strategist or ops action"
    if target == "web_team":
        return "likely frontend or website issue"
    if target == "security_team":
        return "security-process or disclosure follow-up"
    return "manual follow-up needed"


def _summarize_text(text: str, limit: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
