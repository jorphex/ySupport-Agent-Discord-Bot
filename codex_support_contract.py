from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from ticket_investigation_transport import TicketExecutionTransportRequest, TicketExecutionTransportResult

_MAX_RECENT_TRANSCRIPT_ITEMS = 12
_FORBIDDEN_DISCORD_REDIRECT_PATTERNS = (
    "discord.gg/",
    "discord.com/invite",
    "join discord",
    "go to discord",
    "open a discord ticket",
    "open a ticket in discord",
)
_EXPLICIT_HANDOFF_REQUEST_PATTERNS = (
    "need a human",
    "want a human",
    "can a human",
    "could a human",
    "need a person",
    "want a person",
    "can someone review",
    "could someone review",
    "need a moderator",
    "need an admin",
    "manual review",
    "strategist review",
)
_HUMAN_ONLY_HANDOFF_REASON_PATTERNS = (
    "moderator",
    "admin access",
    "server access",
    "verification access",
    "manual recovery",
    "recovery",
    "refund",
    "wrong address",
    "treasury",
    "private internal",
    "private context",
    "account-specific",
    "personal balance",
    "eligibility",
    "security process",
    "sensitive report",
    "receipt confirmation",
)
_OPTIONAL_HANDOFF_SENTENCE_PATTERNS = (
    "hand this off",
    "hand it off",
    "human review",
    "strategist review",
    "team review",
    "manual review",
    "someone can review",
)


CODEX_SUPPORT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "answer",
        "requires_human_handoff",
        "handoff_reason",
        "evidence_summary",
        "used_tools",
    ],
    "properties": {
        "answer": {"type": "string"},
        "requires_human_handoff": {"type": "boolean"},
        "handoff_reason": {"type": ["string", "null"]},
        "evidence_summary": {"type": "string"},
        "used_tools": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


@dataclass
class SupportTurnRequest:
    current_user_message: str
    recent_transcript: list[dict[str, Any]]
    channel_type: str
    channel_id: int | None
    project_context: str
    workflow_name: str
    initial_button_intent: str | None
    requested_intent: str | None
    evidence: dict[str, Any]
    support_state: dict[str, Any]
    constraints: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "current_user_message": self.current_user_message,
            "recent_transcript": list(self.recent_transcript),
            "channel_type": self.channel_type,
            "channel_id": self.channel_id,
            "project_context": self.project_context,
            "workflow_name": self.workflow_name,
            "initial_button_intent": self.initial_button_intent,
            "requested_intent": self.requested_intent,
            "evidence": dict(self.evidence),
            "support_state": dict(self.support_state),
            "constraints": dict(self.constraints),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload())

    @classmethod
    def from_ticket_execution_request(
        cls,
        request: TicketExecutionTransportRequest,
        *,
        ysupport_mcp_enabled: bool = True,
    ) -> "SupportTurnRequest":
        recent_history = [
            item
            for item in list(request.current_history)[-_MAX_RECENT_TRANSCRIPT_ITEMS:]
            if isinstance(item, dict)
        ]
        channel_type = "public" if request.run_context.get("is_public_trigger") else "ticket"
        allowed_tools = ["shell", "web_search"]
        if ysupport_mcp_enabled:
            allowed_tools.append("ysupport_mcp")
        investigation_job = request.investigation_job
        run_context = request.run_context
        evidence = dict(investigation_job.get("evidence", {}))
        initial_button_intent = run_context.get("initial_button_intent")
        requested_intent = investigation_job.get("requested_intent")
        withdrawal_target = None
        if evidence.get("withdrawal_target_chain") or evidence.get(
            "withdrawal_target_vault"
        ):
            withdrawal_target = {
                "chain": evidence.get("withdrawal_target_chain"),
                "vault": evidence.get("withdrawal_target_vault"),
            }
        return cls(
            current_user_message=request.aggregated_text,
            recent_transcript=recent_history,
            channel_type=channel_type,
            channel_id=run_context.get("channel_id"),
            project_context=run_context.get("project_context", "unknown"),
            workflow_name=request.workflow_name,
            initial_button_intent=initial_button_intent,
            requested_intent=requested_intent,
            evidence=evidence,
            support_state={
                "investigation_mode": investigation_job.get("mode"),
                "human_handoff_active": investigation_job.get("mode")
                == "escalated_to_human",
                "current_specialty": investigation_job.get("current_specialty"),
                "last_specialty": investigation_job.get("last_specialty"),
                "known_targets": {
                    "wallet": evidence.get("wallet"),
                    "chain": evidence.get("chain"),
                    "tx_hashes": list(evidence.get("tx_hashes", [])),
                    "withdrawal_target": withdrawal_target,
                },
                "repo_context": {
                    "last_search_query": run_context.get("repo_last_search_query"),
                    "last_search_artifact_refs": list(
                        run_context.get("repo_last_search_artifact_refs", [])
                    ),
                },
                "workflow_context": {
                    "surface": channel_type,
                    "button_context_known": bool(initial_button_intent),
                    "initial_button_intent": initial_button_intent,
                    "requested_intent": requested_intent,
                    "guardrail_profile": _derive_guardrail_profile(
                        channel_type=channel_type,
                        initial_button_intent=initial_button_intent,
                        requested_intent=requested_intent,
                    ),
                    "expected_first_actions": _derive_expected_first_actions(
                        channel_type=channel_type,
                        initial_button_intent=initial_button_intent,
                        requested_intent=requested_intent,
                    ),
                    "non_support_boundaries": [
                        "listing",
                        "partnership",
                        "marketing",
                        "vendor_security",
                        "job_inquiry",
                    ],
                },
            },
            constraints={
                "no_discord_redirects": True,
                "no_file_writes": True,
                "allowed_tools": allowed_tools,
            },
        )


@dataclass
class SupportTurnResult:
    answer: str
    requires_human_handoff: bool
    handoff_reason: str | None
    evidence_summary: str
    used_tools: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "requires_human_handoff": self.requires_human_handoff,
            "handoff_reason": self.handoff_reason,
            "evidence_summary": self.evidence_summary,
            "used_tools": list(self.used_tools),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload())

    @classmethod
    def from_json(cls, raw_json: str) -> "SupportTurnResult":
        payload = json.loads(raw_json)
        return cls(
            answer=payload["answer"].strip(),
            requires_human_handoff=payload["requires_human_handoff"],
            handoff_reason=_normalize_optional_text(payload.get("handoff_reason")),
            evidence_summary=payload["evidence_summary"].strip(),
            used_tools=_normalize_used_tools(payload.get("used_tools", [])),
        )


def verify_support_turn_result(
    result: SupportTurnResult,
    request: SupportTurnRequest,
) -> SupportTurnResult:
    if not result.answer:
        raise ValueError("Support result answer cannot be empty.")
    if not result.evidence_summary:
        raise ValueError("Support result evidence summary cannot be empty.")
    if result.requires_human_handoff and not result.handoff_reason:
        raise ValueError(
            "Support result requires a handoff reason when requires_human_handoff is true."
        )

    lowered_answer = result.answer.lower()
    forbidden_pattern = next(
        (
            pattern
            for pattern in _FORBIDDEN_DISCORD_REDIRECT_PATTERNS
            if pattern in lowered_answer
        ),
        None,
    )
    if forbidden_pattern is not None:
        raise ValueError(
            f"Support result contains a forbidden Discord redirect pattern: {forbidden_pattern}"
        )

    allowed_tools = set(request.constraints.get("allowed_tools", []))
    unexpected_tools = [
        tool for tool in result.used_tools if not _is_allowed_reported_tool(tool, allowed_tools)
    ]
    if unexpected_tools:
        raise ValueError(
            "Support result reported tools that were not allowed for this run: "
            + ", ".join(unexpected_tools)
        )

    normalized_result = SupportTurnResult(
        answer=result.answer,
        requires_human_handoff=result.requires_human_handoff,
        handoff_reason=result.handoff_reason,
        evidence_summary=result.evidence_summary,
        used_tools=result.used_tools,
    )
    if normalized_result.requires_human_handoff and not _handoff_is_allowed(
        normalized_result,
        request,
    ):
        normalized_result = SupportTurnResult(
            answer=_strip_optional_handoff_language(normalized_result.answer),
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary=normalized_result.evidence_summary,
            used_tools=normalized_result.used_tools,
        )

    return normalized_result


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_used_tools(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tool = str(value).strip()
        if not tool or tool in seen:
            continue
        seen.add(tool)
        normalized.append(tool)
    return normalized


def _is_allowed_reported_tool(tool: str, allowed_tools: set[str]) -> bool:
    if tool in allowed_tools:
        return True
    prefixes = {
        "ysupport_mcp": ("ysupport_mcp.",),
        "web_search": ("web_search", "browser", "web."),
        "shell": ("shell", "bash", "exec", "command"),
    }
    for allowed in allowed_tools:
        for prefix in prefixes.get(allowed, ()):
            if tool == prefix or tool.startswith(prefix):
                return True
    return False


def _handoff_is_allowed(
    result: SupportTurnResult,
    request: SupportTurnRequest,
) -> bool:
    if request.support_state.get("human_handoff_active"):
        return True
    current_message = request.current_user_message.lower()
    if any(pattern in current_message for pattern in _EXPLICIT_HANDOFF_REQUEST_PATTERNS):
        return True
    handoff_reason = (result.handoff_reason or "").lower()
    return any(
        pattern in handoff_reason for pattern in _HUMAN_ONLY_HANDOFF_REASON_PATTERNS
    )


def _strip_optional_handoff_language(answer: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]
    filtered = [
        sentence
        for sentence in sentences
        if not any(
            pattern in sentence.lower()
            for pattern in _OPTIONAL_HANDOFF_SENTENCE_PATTERNS
        )
    ]
    if not filtered:
        return answer.strip()
    return " ".join(filtered).strip()


def _derive_guardrail_profile(
    *,
    channel_type: str,
    initial_button_intent: str | None,
    requested_intent: str | None,
) -> str:
    if channel_type == "public":
        return "public_support"
    intent = requested_intent or initial_button_intent
    if intent == "data_deposits_withdrawals_start":
        return "ticket_deposits_withdrawals"
    if intent == "docs_qa":
        return "ticket_docs_qa"
    if intent == "investigate_issue":
        return "ticket_issue_investigation"
    if intent == "other_free_form":
        return "ticket_free_form_support"
    return "ticket_general_support"


def _derive_expected_first_actions(
    *,
    channel_type: str,
    initial_button_intent: str | None,
    requested_intent: str | None,
) -> list[str]:
    intent = requested_intent or initial_button_intent
    if channel_type == "public":
        return [
            "Answer directly in-channel and keep public-channel replies concise.",
        ]
    if intent == "data_deposits_withdrawals_start":
        return [
            "If the user provides a wallet address, start with wallet position lookup before asking for more detail.",
            "If the user provides a vault address, vault name, or token, start with vault lookup.",
            "If the user provides a tx hash, investigate the transaction directly.",
        ]
    if intent == "docs_qa":
        return [
            "Treat this as a direct product/docs question before escalating.",
        ]
    if intent == "investigate_issue":
        return [
            "Treat linked artifacts, tx hashes, and concrete product targets as investigation inputs, not generic support chatter.",
        ]
    if intent == "other_free_form":
        return [
            "Treat this as a free-form support request and infer the best Yearn support path from the user message and known context.",
        ]
    return []


def support_result_to_transport_result(
    result: SupportTurnResult,
    request: TicketExecutionTransportRequest,
) -> TicketExecutionTransportResult:
    updated_job = dict(request.investigation_job)
    updated_job["mode"] = (
        "escalated_to_human" if result.requires_human_handoff else "waiting_for_user"
    )
    updated_job["current_specialty"] = None
    conversation_history = list(request.current_history) + [
        {"role": "user", "content": request.aggregated_text},
        {"role": "assistant", "content": result.answer},
    ]
    return TicketExecutionTransportResult(
        flow_outcome={
            "raw_final_reply": result.answer,
            "conversation_history": conversation_history,
            "completed_agent_key": None,
            "requires_human_handoff": result.requires_human_handoff,
        },
        updated_job=updated_job,
    )
