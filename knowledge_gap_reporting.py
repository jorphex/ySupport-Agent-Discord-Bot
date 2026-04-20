import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.model_settings import Reasoning
from pydantic import BaseModel, Field

import config
import tools_lib


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
    user_problem: Optional[str] = Field(
        default=None,
        description="For bot behavior gaps: what the user was actually trying to resolve.",
    )
    bot_failure: Optional[str] = Field(
        default=None,
        description="For bot behavior gaps: the concrete support or routing failure from the bot.",
    )
    human_follow_up: Optional[str] = Field(
        default=None,
        description="For bot behavior gaps: what later human contributors clarified or fixed.",
    )
    unresolved_risk: Optional[str] = Field(
        default=None,
        description="For bot behavior gaps: what remains unresolved or risky after the ticket.",
    )
    recommended_owner: Optional[
        Literal["bot", "docs", "support_ops", "product", "engineering", "mixed"]
    ] = Field(
        default=None,
        description="Primary internal owner for the next action when clear.",
    )
    missing_or_unclear_docs: Optional[str] = Field(
        default=None,
        description="For docs gaps: what official documentation is missing, unclear, or too hard to find.",
    )
    current_workaround: Optional[str] = Field(
        default=None,
        description="For docs gaps: any current answer, workaround, or contributor-provided path that partly resolves the issue.",
    )
    reported_issue: Optional[str] = Field(
        default=None,
        description="For issue-draft candidates: the concrete issue or claim being reported.",
    )
    plausibility_basis: Optional[str] = Field(
        default=None,
        description="For issue-draft candidates: why the claim seems plausible based on provided grounding or evidence.",
    )
    blocking_unknown: Optional[str] = Field(
        default=None,
        description="For issue-draft candidates: the most important unresolved unknown blocking confident triage.",
    )
    immediate_triage_need: Optional[str] = Field(
        default=None,
        description="For issue-draft candidates: what kind of internal review or next step is immediately needed.",
    )
    confusing_behavior: Optional[str] = Field(
        default=None,
        description="For product confusion reports: the specific product or UI behavior that confused the user.",
    )
    faq_answer: Optional[str] = Field(
        default=None,
        description="For FAQ candidates: the concise reusable answer that support should give.",
    )
    recurrence_signal: Optional[str] = Field(
        default=None,
        description="For FAQ candidates: why this question seems likely to recur or deserves FAQ treatment.",
    )


class BotBehaviorGapExtraction(BaseModel):
    user_problem: str = Field(..., description="What the user was actually trying to figure out or resolve.")
    bot_failure: str = Field(..., description="The specific way ySupport failed in the conversation.")
    human_follow_up: str = Field(
        ...,
        description="What later human contributor replies added, corrected, or escalated.",
    )
    unresolved_risk: str = Field(
        ...,
        description="What uncertainty, risk, or support-quality issue remains after the ticket.",
    )
    docs_gap_present: bool = Field(
        ...,
        description="Whether the transcript suggests a true docs/source gap in addition to the bot failure.",
    )
    recommended_owner: Literal["bot", "docs", "support_ops", "product", "engineering", "mixed"] = Field(
        ...,
        description="Who should primarily own the next internal action.",
    )


class DocsGapExtraction(BaseModel):
    user_problem: str = Field(..., description="What the user was trying to learn or do.")
    missing_or_unclear_docs: str = Field(
        ...,
        description="What official documentation is missing, unclear, too hidden, or too indirect.",
    )
    current_workaround: str = Field(
        ...,
        description="What current workaround, human reply, or partial answer exists today.",
    )
    unresolved_risk: str = Field(
        ...,
        description="What confusion or support burden remains until docs improve.",
    )
    recommended_owner: Literal["bot", "docs", "support_ops", "product", "engineering", "mixed"] = Field(
        ...,
        description="Who should primarily own the next internal action.",
    )


class IssueDraftExtraction(BaseModel):
    reported_issue: str = Field(..., description="The concrete protocol/product/security issue being reported.")
    plausibility_basis: str = Field(
        ...,
        description="Why the report seems plausible enough to justify internal triage.",
    )
    blocking_unknown: str = Field(
        ...,
        description="The most important unresolved unknown blocking confident assessment.",
    )
    immediate_triage_need: str = Field(
        ...,
        description="What internal review or next step should happen immediately.",
    )
    recommended_owner: Literal["bot", "docs", "support_ops", "product", "engineering", "mixed"] = Field(
        ...,
        description="Who should primarily own the next internal action.",
    )


class ProductConfusionExtraction(BaseModel):
    user_problem: str = Field(..., description="What the user was trying to understand or accomplish.")
    confusing_behavior: str = Field(
        ...,
        description="The specific product or UI behavior that made the experience confusing.",
    )
    current_workaround: str = Field(
        ...,
        description="Any current workaround, contributor explanation, or support path that partly resolves it.",
    )
    unresolved_risk: str = Field(
        ...,
        description="What user-facing confusion or product risk remains unresolved.",
    )
    recommended_owner: Literal["bot", "docs", "support_ops", "product", "engineering", "mixed"] = Field(
        ...,
        description="Who should primarily own the next internal action.",
    )


class FaqCandidateExtraction(BaseModel):
    user_problem: str = Field(..., description="The recurring user question or request.")
    faq_answer: str = Field(
        ...,
        description="The concise reusable answer that support should give.",
    )
    recurrence_signal: str = Field(
        ...,
        description="Why this question seems likely to recur or deserves FAQ treatment.",
    )
    current_workaround: str = Field(
        ...,
        description="Any current workaround, support macro, or human explanation used today.",
    )
    recommended_owner: Literal["bot", "docs", "support_ops", "product", "engineering", "mixed"] = Field(
        ...,
        description="Who should primarily own the next internal action.",
    )


@dataclass(frozen=True)
class PreparedTicketTranscript:
    channel_id: str
    channel_name: str
    message_count: int
    transcript_text: str


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

Transcript speaker labels may include:
- `ticket_user(...)` for the original requester in the ticket
- `support_bot(...)` for ySupport
- `human_contributor(...)` for later human follow-up from contributors/mods/community helpers
- `other_bot(...)` for non-support automation messages

Use those roles explicitly when deciding whether the broader lesson is about the user, the bot, or contributor follow-up.
Do not confuse a human contributor's later explanation with the original user request.

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

Transcript speaker labels may include:
- `ticket_user(...)`
- `support_bot(...)`
- `human_contributor(...)`
- `other_bot(...)`

Treat `human_contributor(...)` replies as follow-up context, not as the original support failure source unless the contributor itself caused confusion.

Keep the output concise and internal-facing.
Do not write as if you are replying to the end user.

If a `bot_behavior_gap extraction` is provided:
- preserve its separation between the user's problem, the bot failure, and the later human follow-up
- populate the structured `user_problem`, `bot_failure`, `human_follow_up`, `unresolved_risk`, and `recommended_owner` fields
- make the `suggested_action` prioritize the highest-leverage internal fix first
- do not over-weight docs additions when the primary failure was bot behavior or escalation handling

If a `docs_gap extraction` is provided:
- preserve its separation between the user's problem, the missing/unclear docs, and any current workaround
- populate the structured `user_problem`, `missing_or_unclear_docs`, `current_workaround`, `unresolved_risk`, and `recommended_owner` fields
- keep the suggested action focused on doc discoverability/clarity, not generic support process changes

If an `issue_draft extraction` is provided:
- preserve the distinction between the reported issue, why it seems plausible, and what is still unknown
- populate the structured `reported_issue`, `plausibility_basis`, `blocking_unknown`, `immediate_triage_need`, and `recommended_owner` fields
- keep the suggested action focused on the next internal triage step instead of generic docs advice

If a `product_confusion extraction` is provided:
- preserve the distinction between the user's confusion, the confusing product/UI behavior, and any current workaround
- populate the structured `user_problem`, `confusing_behavior`, `current_workaround`, `unresolved_risk`, and `recommended_owner` fields
- keep the suggested action focused on the product/UI or visibility problem rather than defaulting to docs or bot fixes unless the extraction clearly points there

If a `faq_candidate extraction` is provided:
- preserve the recurring user question, the concise reusable answer, and why it recurs
- populate the structured `user_problem`, `faq_answer`, `recurrence_signal`, `current_workaround`, and `recommended_owner` fields
- keep the suggested action focused on making the answer reusable (FAQ/support macro/bot answer), not on generic product triage
"""


BOT_BEHAVIOR_GAP_EXTRACTION_INSTRUCTIONS = """
You are extracting structured internal findings from a Yearn support ticket that appears to be a bot behavior gap.

Focus on five things only:
- what the user was trying to resolve
- how ySupport failed
- what later human contributors added or corrected
- what remains unresolved or risky
- who should own the next internal fix

Be concrete. Prefer the real support failure over generic commentary.
Do not write the final report. Extract the minimum durable findings the final report should preserve.
"""


DOCS_GAP_EXTRACTION_INSTRUCTIONS = """
You are extracting structured internal findings from a Yearn support ticket that appears to reveal a docs gap.

Focus on five things only:
- what the user was trying to learn or do
- what official docs are missing, unclear, or too hard to find
- what current workaround or human answer exists today
- what confusion/support burden remains unresolved
- who should own the next internal fix

Be concrete. Prefer the actual missing explanation or navigation gap over generic commentary.
Do not write the final report. Extract the minimum durable findings the final report should preserve.
"""


ISSUE_DRAFT_EXTRACTION_INSTRUCTIONS = """
You are extracting structured internal findings from a Yearn support ticket that may deserve an internal issue draft.

Focus on five things only:
- what issue or claim is being reported
- why it seems plausible enough to warrant internal review
- what key unknown still blocks confident assessment
- what immediate internal triage should happen next
- who should own that next step

Be concrete and grounded. Do not decide the final report category here.
Do not write the final report. Extract the minimum durable findings the final report should preserve.
"""


PRODUCT_CONFUSION_EXTRACTION_INSTRUCTIONS = """
You are extracting structured internal findings from a Yearn support ticket that appears to reveal product or UI confusion.

Focus on five things only:
- what the user was trying to understand or accomplish
- what specific product or UI behavior was confusing
- what current workaround or human explanation exists
- what user-facing confusion or product risk remains unresolved
- who should own the next internal fix

Be concrete. Prefer the actual confusing behavior over generic UX commentary.
Do not write the final report. Extract the minimum durable findings the final report should preserve.
"""


FAQ_CANDIDATE_EXTRACTION_INSTRUCTIONS = """
You are extracting structured internal findings from a Yearn support ticket that appears to be a good FAQ candidate.

Focus on five things only:
- the recurring user question
- the concise reusable answer support should give
- why this question seems likely to recur
- what current workaround, support macro, or human explanation exists today
- who should own the next internal fix

Be concrete. Prefer the shortest correct reusable answer over a long explanation.
Do not write the final report. Extract the minimum durable findings the final report should preserve.
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


bot_behavior_gap_extraction_agent = Agent(
    name="Bot Behavior Gap Extraction Agent",
    instructions=BOT_BEHAVIOR_GAP_EXTRACTION_INSTRUCTIONS,
    output_type=BotBehaviorGapExtraction,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


docs_gap_extraction_agent = Agent(
    name="Docs Gap Extraction Agent",
    instructions=DOCS_GAP_EXTRACTION_INSTRUCTIONS,
    output_type=DocsGapExtraction,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


issue_draft_extraction_agent = Agent(
    name="Issue Draft Extraction Agent",
    instructions=ISSUE_DRAFT_EXTRACTION_INSTRUCTIONS,
    output_type=IssueDraftExtraction,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


product_confusion_extraction_agent = Agent(
    name="Product Confusion Extraction Agent",
    instructions=PRODUCT_CONFUSION_EXTRACTION_INSTRUCTIONS,
    output_type=ProductConfusionExtraction,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


faq_candidate_extraction_agent = Agent(
    name="FAQ Candidate Extraction Agent",
    instructions=FAQ_CANDIDATE_EXTRACTION_INSTRUCTIONS,
    output_type=FaqCandidateExtraction,
    model=config.LLM_KNOWLEDGE_GAP_MODEL,
    model_settings=_knowledge_gap_model_settings(),
)


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


def _build_repo_grounding_query(candidate: KnowledgeGapCandidate) -> str:
    parts = [
        candidate.grounding_query.strip(),
        candidate.topic.strip(),
        candidate.evidence_summary.strip(),
    ]
    seen: set[str] = set()
    ordered_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        normalized = part.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_parts.append(part)
    return "\n\n".join(ordered_parts)


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


def _merge_bot_behavior_gap_details(
    report: KnowledgeGapReport,
    extraction: BotBehaviorGapExtraction | None,
) -> KnowledgeGapReport:
    if extraction is None:
        return report

    normalized_payload = report.model_dump()
    for field_name in (
        "user_problem",
        "bot_failure",
        "human_follow_up",
        "unresolved_risk",
        "recommended_owner",
    ):
        if normalized_payload.get(field_name):
            continue
        normalized_payload[field_name] = getattr(extraction, field_name)
    return KnowledgeGapReport(**normalized_payload)


def _merge_docs_gap_details(
    report: KnowledgeGapReport,
    extraction: DocsGapExtraction | None,
) -> KnowledgeGapReport:
    if extraction is None:
        return report

    normalized_payload = report.model_dump()
    for field_name in (
        "user_problem",
        "missing_or_unclear_docs",
        "current_workaround",
        "unresolved_risk",
        "recommended_owner",
    ):
        if normalized_payload.get(field_name):
            continue
        normalized_payload[field_name] = getattr(extraction, field_name)
    return KnowledgeGapReport(**normalized_payload)


def _merge_issue_draft_details(
    report: KnowledgeGapReport,
    extraction: IssueDraftExtraction | None,
) -> KnowledgeGapReport:
    if extraction is None:
        return report

    normalized_payload = report.model_dump()
    for field_name in (
        "reported_issue",
        "plausibility_basis",
        "blocking_unknown",
        "immediate_triage_need",
        "recommended_owner",
    ):
        if normalized_payload.get(field_name):
            continue
        normalized_payload[field_name] = getattr(extraction, field_name)
    return KnowledgeGapReport(**normalized_payload)


def _merge_product_confusion_details(
    report: KnowledgeGapReport,
    extraction: ProductConfusionExtraction | None,
) -> KnowledgeGapReport:
    if extraction is None:
        return report

    normalized_payload = report.model_dump()
    for field_name in (
        "user_problem",
        "confusing_behavior",
        "current_workaround",
        "unresolved_risk",
        "recommended_owner",
    ):
        if normalized_payload.get(field_name):
            continue
        normalized_payload[field_name] = getattr(extraction, field_name)
    return KnowledgeGapReport(**normalized_payload)


def _merge_faq_candidate_details(
    report: KnowledgeGapReport,
    extraction: FaqCandidateExtraction | None,
) -> KnowledgeGapReport:
    if extraction is None:
        return report

    normalized_payload = report.model_dump()
    for field_name in (
        "user_problem",
        "faq_answer",
        "recurrence_signal",
        "current_workaround",
        "recommended_owner",
    ):
        if normalized_payload.get(field_name):
            continue
        normalized_payload[field_name] = getattr(extraction, field_name)
    return KnowledgeGapReport(**normalized_payload)


def _placeholder_like_text(text: Optional[str]) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    return normalized in {
        "evidence summary",
        "grounding",
        "docs grounding",
        "assessment",
        "suggested action",
        "current grounding",
    }


def _report_quality_issues(report: KnowledgeGapReport) -> list[str]:
    issues: list[str] = []
    core_fields = {
        "title": report.title,
        "topic": report.topic,
        "evidence_summary": report.evidence_summary,
        "current_official_grounding": report.current_official_grounding,
        "assessment": report.assessment,
        "suggested_action": report.suggested_action,
    }
    for field_name, value in core_fields.items():
        if _placeholder_like_text(value):
            issues.append(f"{field_name}_missing_or_placeholder")

    required_fields_by_category: dict[str, tuple[str, ...]] = {
        "bot_behavior_gap": (
            "user_problem",
            "bot_failure",
            "human_follow_up",
            "unresolved_risk",
            "recommended_owner",
        ),
        "docs_gap": (
            "user_problem",
            "missing_or_unclear_docs",
            "current_workaround",
            "unresolved_risk",
            "recommended_owner",
        ),
        "issue_draft_candidate": (
            "reported_issue",
            "plausibility_basis",
            "blocking_unknown",
            "immediate_triage_need",
            "recommended_owner",
        ),
        "product_confusion": (
            "user_problem",
            "confusing_behavior",
            "current_workaround",
            "unresolved_risk",
            "recommended_owner",
        ),
        "faq_candidate": (
            "user_problem",
            "faq_answer",
            "recurrence_signal",
            "current_workaround",
            "recommended_owner",
        ),
    }
    for field_name in required_fields_by_category.get(report.category, ()):
        value = getattr(report, field_name)
        if isinstance(value, str):
            if _placeholder_like_text(value):
                issues.append(f"{field_name}_missing_or_placeholder")
        elif value is None:
            issues.append(f"{field_name}_missing")

    return issues


def _should_fetch_repo_grounding(candidate: KnowledgeGapCandidate) -> bool:
    return candidate.needs_repo_context or candidate.category == "issue_draft_candidate"


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
    if _should_fetch_repo_grounding(candidate):
        repo_grounding = await tools_lib.core_pretriage_repo_claim(
            _build_repo_grounding_query(candidate),
            include_docs=False,
        )
    bot_behavior_gap_extraction = None
    docs_gap_extraction = None
    issue_draft_extraction = None
    product_confusion_extraction = None
    faq_candidate_extraction = None
    if candidate.category == "bot_behavior_gap":
        bot_behavior_gap_extraction = await _run_structured_agent(
            bot_behavior_gap_extraction_agent,
            (
                f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
                f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Docs grounding:\n{docs_grounding}\n\n"
                f"Repo grounding:\n{repo_grounding or 'none'}"
            ),
            BotBehaviorGapExtraction,
            "Knowledge Gap Bot Behavior Extraction",
        )
    if candidate.category == "docs_gap":
        docs_gap_extraction = await _run_structured_agent(
            docs_gap_extraction_agent,
            (
                f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
                f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Docs grounding:\n{docs_grounding}\n\n"
                f"Repo grounding:\n{repo_grounding or 'none'}"
            ),
            DocsGapExtraction,
            "Knowledge Gap Docs Extraction",
        )
    if candidate.category == "issue_draft_candidate":
        issue_draft_extraction = await _run_structured_agent(
            issue_draft_extraction_agent,
            (
                f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
                f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Docs grounding:\n{docs_grounding}\n\n"
                f"Repo grounding:\n{repo_grounding or 'none'}"
            ),
            IssueDraftExtraction,
            "Knowledge Gap Issue Draft Extraction",
        )
    if candidate.category == "product_confusion":
        product_confusion_extraction = await _run_structured_agent(
            product_confusion_extraction_agent,
            (
                f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
                f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Docs grounding:\n{docs_grounding}\n\n"
                f"Repo grounding:\n{repo_grounding or 'none'}"
            ),
            ProductConfusionExtraction,
            "Knowledge Gap Product Confusion Extraction",
        )
    if candidate.category == "faq_candidate":
        faq_candidate_extraction = await _run_structured_agent(
            faq_candidate_extraction_agent,
            (
                f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
                f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
                f"Docs grounding:\n{docs_grounding}\n\n"
                f"Repo grounding:\n{repo_grounding or 'none'}"
            ),
            FaqCandidateExtraction,
            "Knowledge Gap FAQ Extraction",
        )

    report_input = (
        f"Channel ID: {prepared_transcript.channel_id}\n"
        f"Message count: {prepared_transcript.message_count}\n\n"
        f"Transcript:\n{prepared_transcript.transcript_text}\n\n"
        f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
        f"Docs grounding:\n{docs_grounding}\n\n"
        f"Repo grounding:\n{repo_grounding or 'none'}\n\n"
        "bot_behavior_gap extraction:\n"
        f"{bot_behavior_gap_extraction.model_dump_json(indent=2) if bot_behavior_gap_extraction else 'none'}\n\n"
        "docs_gap extraction:\n"
        f"{docs_gap_extraction.model_dump_json(indent=2) if docs_gap_extraction else 'none'}\n\n"
        "issue_draft extraction:\n"
        f"{issue_draft_extraction.model_dump_json(indent=2) if issue_draft_extraction else 'none'}\n\n"
        "product_confusion extraction:\n"
        f"{product_confusion_extraction.model_dump_json(indent=2) if product_confusion_extraction else 'none'}\n\n"
        "faq_candidate extraction:\n"
        f"{faq_candidate_extraction.model_dump_json(indent=2) if faq_candidate_extraction else 'none'}"
    )
    report = await _run_structured_agent(
        knowledge_gap_report_agent,
        report_input,
        KnowledgeGapReport,
        "Knowledge Gap Final Report",
    )
    report = _merge_bot_behavior_gap_details(report, bot_behavior_gap_extraction)
    report = _merge_docs_gap_details(report, docs_gap_extraction)
    report = _merge_issue_draft_details(report, issue_draft_extraction)
    report = _merge_product_confusion_details(report, product_confusion_extraction)
    report = _merge_faq_candidate_details(report, faq_candidate_extraction)
    if not report.should_post or report.category == "no_action":
        return None
    if _report_quality_issues(report):
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
        ]
    )
    if report.category == "issue_draft_candidate":
        if report.reported_issue:
            lines.extend(["**Reported issue**", report.reported_issue, ""])
        else:
            lines.extend(["**Reported issue**", report.evidence_summary, ""])
        if report.plausibility_basis:
            lines.extend(["**Why plausible**", report.plausibility_basis, ""])
        lines.extend(["**Current grounding**", report.current_official_grounding, ""])
        if report.blocking_unknown:
            lines.extend(["**Blocking unknown**", report.blocking_unknown, ""])
        lines.extend(["**Triage assessment**", report.assessment, ""])
        if report.immediate_triage_need:
            lines.extend(["**Immediate triage need**", report.immediate_triage_need, ""])
        if report.recommended_owner:
            lines.extend(["**Recommended owner**", report.recommended_owner, ""])
        lines.extend(["**Recommended next step**", report.suggested_action])
    elif (
        report.user_problem
        or report.bot_failure
        or report.human_follow_up
        or report.unresolved_risk
        or report.recommended_owner
    ) and report.category == "bot_behavior_gap":
        if report.user_problem:
            lines.extend(["**User issue**", report.user_problem, ""])
        if report.bot_failure:
            lines.extend(["**Bot failure**", report.bot_failure, ""])
        if report.human_follow_up:
            lines.extend(["**Human follow-up**", report.human_follow_up, ""])
        lines.extend(["**Current official grounding**", report.current_official_grounding, ""])
        if report.unresolved_risk:
            lines.extend(["**Unresolved risk**", report.unresolved_risk, ""])
        lines.extend(["**Assessment**", report.assessment, ""])
        if report.recommended_owner:
            lines.extend(["**Recommended owner**", report.recommended_owner, ""])
        lines.extend(["**Recommended next step**", report.suggested_action])
    elif report.category == "docs_gap" and (
        report.user_problem
        or report.missing_or_unclear_docs
        or report.current_workaround
        or report.unresolved_risk
        or report.recommended_owner
    ):
        if report.user_problem:
            lines.extend(["**User issue**", report.user_problem, ""])
        if report.missing_or_unclear_docs:
            lines.extend(["**Missing or unclear docs**", report.missing_or_unclear_docs, ""])
        if report.current_workaround:
            lines.extend(["**Current workaround**", report.current_workaround, ""])
        lines.extend(["**Current official grounding**", report.current_official_grounding, ""])
        if report.unresolved_risk:
            lines.extend(["**Unresolved risk**", report.unresolved_risk, ""])
        lines.extend(["**Assessment**", report.assessment, ""])
        if report.recommended_owner:
            lines.extend(["**Recommended owner**", report.recommended_owner, ""])
        lines.extend(["**Recommended next step**", report.suggested_action])
    elif report.category == "product_confusion" and (
        report.user_problem
        or report.confusing_behavior
        or report.current_workaround
        or report.unresolved_risk
        or report.recommended_owner
    ):
        if report.user_problem:
            lines.extend(["**User confusion**", report.user_problem, ""])
        if report.confusing_behavior:
            lines.extend(["**Confusing product behavior**", report.confusing_behavior, ""])
        if report.current_workaround:
            lines.extend(["**Current workaround**", report.current_workaround, ""])
        lines.extend(["**Current official grounding**", report.current_official_grounding, ""])
        if report.unresolved_risk:
            lines.extend(["**Residual product risk**", report.unresolved_risk, ""])
        lines.extend(["**Assessment**", report.assessment, ""])
        if report.recommended_owner:
            lines.extend(["**Recommended owner**", report.recommended_owner, ""])
        lines.extend(["**Recommended next step**", report.suggested_action])
    elif report.category == "faq_candidate" and (
        report.user_problem
        or report.faq_answer
        or report.recurrence_signal
        or report.current_workaround
        or report.recommended_owner
    ):
        if report.user_problem:
            lines.extend(["**Recurring question**", report.user_problem, ""])
        if report.faq_answer:
            lines.extend(["**Reusable answer**", report.faq_answer, ""])
        if report.recurrence_signal:
            lines.extend(["**Why it recurs**", report.recurrence_signal, ""])
        if report.current_workaround:
            lines.extend(["**Current workaround**", report.current_workaround, ""])
        lines.extend(["**Current official grounding**", report.current_official_grounding, ""])
        lines.extend(["**Assessment**", report.assessment, ""])
        if report.recommended_owner:
            lines.extend(["**Recommended owner**", report.recommended_owner, ""])
        lines.extend(["**Recommended next step**", report.suggested_action])
    else:
        lines.extend(
            [
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
