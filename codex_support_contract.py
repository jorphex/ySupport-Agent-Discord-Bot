from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from ticket_investigation.transport import TicketExecutionTransportRequest, TicketExecutionTransportResult
from ticket_investigation.executor import TicketExecutionNonFallbackError

_MAX_RECENT_TRANSCRIPT_ITEMS = 12
_FORBIDDEN_DISCORD_REDIRECT_PATTERNS = (
    "discord.gg/",
    "discord.com/invite",
    "join discord",
    "go to discord",
    "open a discord ticket",
    "open a ticket in discord",
)
_HANDOFF_KINDS = (
    "access_or_permission_action",
    "fund_or_account_recovery",
    "security_process",
    "manual_strategy_action",
    "private_internal_fact",
    "human_decision",
)
_OPTIONAL_HANDOFF_SENTENCE_PATTERNS = (
    "hand this off",
    "hand it off",
    "human review",
    "human ops review",
    "strategist review",
    "team review",
    "manual review",
    "ops review",
    "operator review",
    "should get a human",
    "someone can review",
    "human can review",
    "moderator can review",
    "admin can review",
    "strategist can review",
    "team can review",
)
_OPTIONAL_HANDOFF_CLAUSE_PATTERNS = (
    r",?\s*so this should get a human ops review\.?$",
    r",?\s*so this should get an ops review\.?$",
    r",?\s*so this should get a human review\.?$",
    r",?\s*so this should get operator review\.?$",
    r",?\s*so this should get a human operator review\.?$",
)
# EIP-2718 transaction types occupy 0x01–0x7f, while signed legacy and current
# typed transactions use the 0xf8–0xff long-list RLP envelope. The 80-byte hex
# tail floor includes compact signed transfers while excluding hashes and addresses.
_TRANSACTION_SIZED_HEX_PAYLOAD_RE = re.compile(
    r"(?<![0-9a-f])0x(?:0[1-9a-f]|[1-7][0-9a-f])?f[89a-f]"
    r"[0-9a-f]{160,}(?![0-9a-f])",
    re.IGNORECASE,
)
CODEX_SUPPORT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "answer",
        "requires_human_handoff",
        "handoff_reason",
        "handoff_kind",
        "evidence_summary",
        "used_tools",
    ],
    "properties": {
        "answer": {"type": "string"},
        "requires_human_handoff": {"type": "boolean"},
        "handoff_reason": {"type": ["string", "null"]},
        "handoff_kind": {
            "type": ["string", "null"],
            "enum": [*_HANDOFF_KINDS, None],
        },
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
    current_turn_source: str = "user"
    current_turn_instruction: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "current_user_message": self.current_user_message,
            "current_turn_source": self.current_turn_source,
            "current_turn_instruction": self.current_turn_instruction,
            "recent_transcript": list(self.recent_transcript),
            "attachments": list(self.attachments),
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
        attachments = list(request.attachments)
        investigation_job = request.investigation_job
        run_context = request.run_context
        evidence = dict(investigation_job.get("evidence", {}))
        current_turn_context = _current_turn_system_context(request)
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
            current_turn_source=str(request.turn_source or "user"),
            current_turn_instruction=_normalize_optional_text(request.turn_instruction),
            recent_transcript=recent_history,
            attachments=attachments,
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
                "current_turn_context": current_turn_context,
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


def _current_turn_system_context(
    request: TicketExecutionTransportRequest,
) -> list[str]:
    current_turn_items = request.input_list[len(request.current_history) :]
    turn_instruction = _normalize_optional_text(request.turn_instruction)
    return [
        content.strip()
        for item in current_turn_items
        if isinstance(item, dict)
        and item.get("role") == "system"
        and isinstance((content := item.get("content")), str)
        and content.strip()
        and content.strip() != turn_instruction
    ]


@dataclass
class SupportTurnResult:
    answer: str
    requires_human_handoff: bool
    handoff_reason: str | None
    evidence_summary: str
    used_tools: list[str]
    handoff_kind: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "requires_human_handoff": self.requires_human_handoff,
            "handoff_reason": self.handoff_reason,
            "handoff_kind": self.handoff_kind,
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
            handoff_kind=_normalize_optional_text(payload.get("handoff_kind")),
        )


class SupportTurnPolicyViolation(TicketExecutionNonFallbackError):
    """Raised when a parsed support result violates an enforced policy."""


class SignedTransactionSafetyViolation(SupportTurnPolicyViolation):
    """Raised when a support response exposes a transaction-sized hex payload."""


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
    visible_text = "\n".join(
        value
        for value in (
            result.answer,
            result.evidence_summary,
            result.handoff_reason,
        )
        if value
    )
    if _TRANSACTION_SIZED_HEX_PAYLOAD_RE.search(visible_text):
        raise SignedTransactionSafetyViolation(
            "Support result contains a transaction-sized serialized hex payload."
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
        raise SupportTurnPolicyViolation(
            f"Support result contains a forbidden Discord redirect pattern: {forbidden_pattern}"
        )

    allowed_tools = _effective_allowed_tools(request)
    unexpected_tools = [
        tool for tool in result.used_tools if not _is_allowed_reported_tool(tool, allowed_tools)
    ]
    if unexpected_tools:
        raise SupportTurnPolicyViolation(
            "Support result reported tools that were not allowed for this run: "
            + ", ".join(unexpected_tools)
        )

    normalized_result = SupportTurnResult(
        answer=result.answer,
        requires_human_handoff=result.requires_human_handoff,
        handoff_reason=result.handoff_reason,
        evidence_summary=result.evidence_summary,
        used_tools=result.used_tools,
        handoff_kind=result.handoff_kind,
    )
    if not normalized_result.requires_human_handoff and (
        normalized_result.handoff_reason or normalized_result.handoff_kind
    ):
        normalized_result = SupportTurnResult(
            answer=normalized_result.answer,
            requires_human_handoff=False,
            handoff_reason=None,
            evidence_summary=normalized_result.evidence_summary,
            used_tools=normalized_result.used_tools,
            handoff_kind=None,
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
            handoff_kind=None,
        )

    return normalized_result


def _effective_allowed_tools(request: SupportTurnRequest) -> set[str]:
    allowed_tools = set(request.constraints.get("allowed_tools", []))
    if any(
        isinstance(attachment, dict) and attachment.get("is_image")
        for attachment in request.attachments
    ):
        allowed_tools.add("view_image")
    return allowed_tools


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
        if tool.startswith("functions."):
            tool = tool[len("functions.") :]
        if not tool or tool in seen:
            continue
        seen.add(tool)
        normalized.append(tool)
    return normalized


def _is_allowed_reported_tool(tool: str, allowed_tools: set[str]) -> bool:
    if tool in allowed_tools:
        return True
    prefixes = {
        "ysupport_mcp": ("ysupport_mcp.", "mcp__ysupport__", "mcp__ysupport."),
        "web_search": ("web_search", "browser", "web."),
        "shell": ("shell", "bash", "exec", "command"),
        "view_image": ("view_image",),
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
    if request.current_turn_source != "user":
        return False
    return result.handoff_kind in _HANDOFF_KINDS


def _strip_optional_handoff_language(answer: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]
    filtered: list[str] = []
    for sentence in sentences:
        stripped_sentence = sentence
        for pattern in _OPTIONAL_HANDOFF_CLAUSE_PATTERNS:
            stripped_sentence = re.sub(
                pattern,
                ".",
                stripped_sentence,
                flags=re.IGNORECASE,
            ).strip()
        if not stripped_sentence:
            continue
        if any(
            pattern in stripped_sentence.lower()
            for pattern in _OPTIONAL_HANDOFF_SENTENCE_PATTERNS
        ):
            continue
        filtered.append(stripped_sentence)
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
    conversation_history = list(request.current_history)
    if request.turn_source == "user":
        conversation_history.append({"role": "user", "content": request.aggregated_text})
    conversation_history.append({"role": "assistant", "content": result.answer})
    return TicketExecutionTransportResult(
        flow_outcome={
            "raw_final_reply": result.answer,
            "conversation_history": conversation_history,
            "completed_agent_key": None,
            "requires_human_handoff": result.requires_human_handoff,
            "handoff_reason": result.handoff_reason,
        },
        updated_job=updated_job,
    )
