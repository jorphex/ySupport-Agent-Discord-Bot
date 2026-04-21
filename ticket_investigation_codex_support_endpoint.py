from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from codex_support_home import prepare_codex_support_home
from codex_support_contract import (
    CODEX_SUPPORT_RESULT_SCHEMA,
    SupportTurnRequest,
    SupportTurnResult,
    support_result_to_transport_result,
    verify_support_turn_result,
)
from codex_support_sessions import CodexSupportSessionManager
from ticket_execution_subprocess_utils import (
    build_effective_execution_env,
    run_bounded_subprocess,
    safe_export_workspace_copy,
    validate_allowed_command_prefix,
)
from ticket_execution_workspace import TicketExecutionWorkspace
from ticket_investigation_codex_bundle import DEFAULT_CODEX_EXEC_COMMAND
from ticket_investigation_executor import TicketExecutionHooks
from ticket_investigation_transport import (
    build_smoke_transport_result,
    TicketExecutionTransportRequest,
)


@dataclass
class CodexSupportExecutionBundle:
    command: list[str]
    prompt_text: str
    prompt_path: Path
    support_request_path: Path
    response_schema_path: Path
    resumed_session_id: str | None


class _ConversationExecutionLocks:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, conversation_key: str | None):
        if not conversation_key:
            yield
            return
        async with self._guard:
            lock = self._locks.get(conversation_key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[conversation_key] = lock
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()


class CodexSupportTicketExecutionJsonEndpoint:
    def __init__(
        self,
        *,
        codex_command: Sequence[str] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        repo_root: str | Path | None = None,
        codex_home: str | Path | None = None,
        codex_auth_source: str | Path | None = None,
        session_dir: str | Path | None = None,
        session_max_age_hours: int | None = None,
        ysupport_mcp_url: str | None = None,
        ysupport_mcp_container: str | None = None,
        mcp_server_api_key: str | None = None,
        web_search_mode: str = "live",
        allowed_command_prefixes: Sequence[Sequence[str]] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_parent_env: bool = False,
        artifact_dir: str | None = None,
        run_dir_root: str | None = None,
        timeout_seconds: float = 300.0,
        max_output_chars: int = 200000,
        max_error_chars: int = 4000,
    ) -> None:
        command = list(codex_command or DEFAULT_CODEX_EXEC_COMMAND)
        if not command:
            raise ValueError("Codex support execution command cannot be empty.")
        self.codex_command = command
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.repo_root = Path(repo_root) if repo_root else None
        self.codex_home = Path(codex_home) if codex_home else None
        self.codex_auth_source = Path(codex_auth_source) if codex_auth_source else None
        self.session_manager = (
            CodexSupportSessionManager(
                session_dir,
                max_age_hours=session_max_age_hours,
            )
            if session_dir
            else None
        )
        self.ysupport_mcp_url = ysupport_mcp_url
        self.ysupport_mcp_container = ysupport_mcp_container
        self.mcp_server_api_key = mcp_server_api_key
        self.web_search_mode = web_search_mode
        self.allowed_command_prefixes = [
            list(prefix) for prefix in (allowed_command_prefixes or [])
        ]
        validate_allowed_command_prefix(
            self.codex_command,
            self.allowed_command_prefixes,
            error_message="Codex support execution command is not in the allowed prefix list.",
        )
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self.inherit_parent_env = inherit_parent_env
        self.artifact_dir = artifact_dir
        self.run_dir_root = run_dir_root
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_error_chars = max_error_chars
        self.execution_locks = _ConversationExecutionLocks()

    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        request = TicketExecutionTransportRequest.from_json(request_json)
        if request.wants_bug_review_status and hooks is not None:
            if hooks.send_bug_review_status is not None:
                await hooks.send_bug_review_status()
        if request.smoke_mode:
            return build_smoke_transport_result(
                request,
                endpoint_mode="codex_support_exec",
            ).to_json()
        conversation_key = _conversation_key_for_request(request)
        async with self.execution_locks.acquire(conversation_key):
            ysupport_mcp_enabled = False
            if self.codex_home and self.ysupport_mcp_url and self.mcp_server_api_key:
                support_home = prepare_codex_support_home(
                    codex_home=self.codex_home,
                    repo_root=self.repo_root,
                    mcp_container_name=self.ysupport_mcp_container,
                    auth_source=self.codex_auth_source,
                    ysupport_mcp_url=self.ysupport_mcp_url,
                    mcp_server_api_key=self.mcp_server_api_key,
                    web_search_mode=self.web_search_mode,
                )
                ysupport_mcp_enabled = support_home.ysupport_mcp_enabled

            workspace = TicketExecutionWorkspace(
                artifact_dir=self.artifact_dir or None,
                run_dir_root=self.run_dir_root or None,
                prefix="ticket-codex-support-run-",
            )
            with workspace as run_dir:
                support_request = SupportTurnRequest.from_ticket_execution_request(
                    request,
                    ysupport_mcp_enabled=ysupport_mcp_enabled,
                )
                workflow_context = support_request.support_state.get("workflow_context", {})
                session_record = (
                    self.session_manager.load_for_turn(
                        conversation_key=conversation_key,
                        requested_intent=support_request.requested_intent,
                        guardrail_profile=workflow_context.get("guardrail_profile"),
                        human_handoff_active=bool(
                            support_request.support_state.get("human_handoff_active")
                        ),
                    )
                    if self.session_manager is not None and conversation_key is not None
                    else None
                )
                bundle = _build_codex_support_execution_bundle(
                    support_request=support_request,
                    run_dir=run_dir,
                    codex_command=self.codex_command,
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    resume_session_id=session_record.session_id if session_record else None,
                )
                exported_run_dir: Path | None = None
                try:
                    response_text = await run_bounded_subprocess(
                        command=bundle.command,
                        stdin_text=bundle.prompt_text,
                        cwd=self.cwd or str(run_dir),
                        env=build_effective_execution_env(
                            env=self.env,
                            inherit_parent_env=self.inherit_parent_env,
                            run_dir=run_dir,
                        ),
                        timeout_seconds=self.timeout_seconds,
                        max_output_chars=self.max_output_chars,
                        max_error_chars=self.max_error_chars,
                        timeout_message=(
                            f"Codex support execution timed out after {self.timeout_seconds} seconds."
                        ),
                        empty_stdout_message="Codex support execution returned empty stdout.",
                        oversized_stdout_message="Codex support execution returned too much stdout.",
                        metadata={
                            "base_command": self.codex_command,
                            "command": bundle.command,
                            "cwd": self.cwd or str(run_dir),
                        },
                        artifact_run_dir=run_dir,
                    )
                except Exception as exc:
                    if (
                        self.session_manager is not None
                        and conversation_key is not None
                        and session_record is not None
                    ):
                        self.session_manager.record_failure(
                            conversation_key=conversation_key,
                            error_text=str(exc),
                        )
                    raise
                finally:
                    exported_run_dir = safe_export_workspace_copy(
                        workspace,
                        logger_name=__name__,
                        context="codex support execution",
                    )

                if self.session_manager is not None and conversation_key is not None:
                    session_id = self._extract_session_id_from_run_dir(run_dir)
                    if session_id is not None:
                        self.session_manager.record_success(
                            conversation_key=conversation_key,
                            session_id=session_id,
                            artifact_dir=str(exported_run_dir) if exported_run_dir else None,
                            requested_intent=support_request.requested_intent,
                            guardrail_profile=workflow_context.get("guardrail_profile"),
                            human_handoff_active=bool(
                                support_request.support_state.get("human_handoff_active")
                            ),
                        )
                support_result = verify_support_turn_result(
                    SupportTurnResult.from_json(response_text),
                    support_request,
                )
                return support_result_to_transport_result(support_result, request).to_json()

    def _extract_session_id_from_run_dir(self, run_dir: Path) -> str | None:
        stderr_path = run_dir / "stderr.txt"
        if not stderr_path.exists():
            return None
        try:
            stderr_text = stderr_path.read_text(encoding="utf-8")
        except OSError:
            return None
        return (
            self.session_manager.extract_session_id(stderr_text)
            if self.session_manager is not None
            else None
        )


def _build_codex_support_execution_bundle(
    *,
    support_request: SupportTurnRequest,
    run_dir: str | Path,
    codex_command: Sequence[str],
    model: str | None,
    reasoning_effort: str | None,
    resume_session_id: str | None = None,
) -> CodexSupportExecutionBundle:
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    support_request_path = run_dir_path / "support_request.json"
    response_schema_path = run_dir_path / "support_response_schema.json"
    prompt_path = run_dir_path / "codex_support_prompt.txt"

    support_request_path.write_text(support_request.to_json(), encoding="utf-8")
    response_schema_path.write_text(
        json.dumps(CODEX_SUPPORT_RESULT_SCHEMA, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    prompt_text = _codex_support_prompt(
        support_request_path=support_request_path,
        response_schema_path=response_schema_path,
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")

    command = _build_codex_support_command(
        codex_command=codex_command,
        model=model,
        reasoning_effort=reasoning_effort,
        response_schema_path=response_schema_path,
        run_dir_path=run_dir_path,
        resume_session_id=resume_session_id,
    )
    return CodexSupportExecutionBundle(
        command=command,
        prompt_text=prompt_text,
        prompt_path=prompt_path,
        support_request_path=support_request_path,
        response_schema_path=response_schema_path,
        resumed_session_id=resume_session_id,
    )


def _conversation_key_for_request(
    request: TicketExecutionTransportRequest,
) -> str | None:
    channel_id = request.run_context.get("channel_id")
    if channel_id is None:
        return None
    channel_type = "public" if request.run_context.get("is_public_trigger") else "ticket"
    return f"{channel_type}:{channel_id}"


def _build_codex_support_command(
    *,
    codex_command: Sequence[str],
    model: str | None,
    reasoning_effort: str | None,
    response_schema_path: Path,
    run_dir_path: Path,
    resume_session_id: str | None,
) -> list[str]:
    if resume_session_id:
        command = _build_codex_resume_command(
            codex_command=codex_command,
            session_id=resume_session_id,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        command.append("-")
        return command

    command = [
        arg for arg in codex_command
        if arg != "--ephemeral"
    ]
    if model:
        command.extend(["-m", model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    command.extend(
        [
            "--output-schema",
            str(response_schema_path),
            "-C",
            str(run_dir_path),
            "-",
        ]
    )
    return command


def _build_codex_resume_command(
    *,
    codex_command: Sequence[str],
    session_id: str,
    model: str | None,
    reasoning_effort: str | None,
) -> list[str]:
    exec_index = next(
        (index for index, token in enumerate(codex_command) if token == "exec"),
        None,
    )
    if exec_index is None:
        raise ValueError("Codex support command must include an exec subcommand.")
    command = list(codex_command[:exec_index]) + ["exec", "resume", session_id]
    if "--skip-git-repo-check" in codex_command[exec_index + 1 :]:
        command.append("--skip-git-repo-check")
    if model:
        command.extend(["-m", model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    return command


def _codex_support_prompt(
    *,
    support_request_path: Path,
    response_schema_path: Path,
) -> str:
    return (
        "You are the Yearn Discord support bot running in a constrained execution backend.\n\n"
        f"Read the support turn request from {support_request_path.name}.\n"
        f"Return only JSON matching {response_schema_path.name}.\n"
        "Treat `channel_type`, `channel_id`, `initial_button_intent`, and `requested_intent` in the request "
        "as support workflow context.\n"
        "Treat `support_state` in the request as explicit bot-known runtime state. Prefer it over re-inferring the "
        "same facts from transcript alone when they are consistent.\n"
        "Treat `support_state.workflow_context` as outer Discord/UI context coming from the live bot. If it gives "
        "expected first actions for the current support flow, follow them before falling back to generic "
        "clarification.\n"
        "Treat `support_state.workflow_context.non_support_boundaries` as hard outer-bot guardrails for topics "
        "that should not be handled like normal support.\n"
        "Use only the tools listed in `constraints.allowed_tools`.\n"
        "If `ysupport_mcp` is listed, prefer it for Yearn-specific docs, repo context, vault context, and support "
        "facts before relying only on generic web or shell results.\n"
        "Do not write or modify files as part of your investigation.\n"
        "Do not tell the user to go to Discord or open a Discord ticket.\n"
        "Before setting requires_human_handoff=true, do the strongest direct support work you can from the request "
        "and the available tools.\n"
        "If the user links external artifacts such as gists or reports, inspect them and give a preliminary "
        "technical assessment before handoff.\n"
        "If the user asks for a human but also provides a substantive target or question, acknowledge that a human "
        "can review it while still answering what you can right now.\n"
        "Answer directly when possible. If human review is needed, set requires_human_handoff=true and explain "
        "the reason briefly.\n"
        "Keep the answer concise and support-oriented.\n"
    )
