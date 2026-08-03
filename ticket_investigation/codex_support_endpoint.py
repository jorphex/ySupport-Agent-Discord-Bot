from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Sequence
from codex_support_home import (
    prepare_codex_auth_state,
    prepare_codex_support_home,
)
from codex_support_contract import (
    CODEX_SUPPORT_RESULT_SCHEMA,
    SignedTransactionSafetyViolation,
    SupportTurnRequest,
    SupportTurnResult,
    support_result_to_transport_result,
    verify_support_turn_result,
)
from codex_support_sessions import CodexSupportSessionManager
from bot_behavior import SECURITY_PROCESS_URL
from ticket_investigation.codex_support_attachments import (
    image_attachment_paths,
    prepare_support_request_attachments,
)
from ticket_investigation.codex_support_subprocess import (
    CodexSupportExecutionOutput,
    run_codex_support_json_subprocess,
)
from ticket_execution.subprocess_utils import (
    build_effective_execution_env,
    run_bounded_subprocess,
    safe_export_workspace_copy,
    validate_allowed_command_prefix,
)
from ticket_execution.workspace import TicketExecutionWorkspace
from ticket_investigation.executor import TicketExecutionHooks
from ticket_investigation.transport import (
    build_smoke_transport_result,
    TicketExecutionTransportRequest,
)

DEFAULT_CODEX_EXEC_COMMAND = [
    "codex",
    "exec",
    "--skip-git-repo-check",
    "--color",
    "never",
]
_SESSION_UUID_RE = re.compile(r"[0-9a-fA-F-]{36}")
_ROLLOUT_SESSION_ID_RE = re.compile(
    r"(?P<session_id>[0-9a-fA-F-]{36})\.jsonl$"
)
_CODEX_SESSION_DELETE_TIMEOUT_SECONDS = 30.0
_TRANSACTION_SAFETY_INSTRUCTION = (
    "For transaction troubleshooting, remain read-only. You may use transaction "
    "hashes, decoded fields, statuses, non-mutating calls or simulations, and "
    "official wallet or Yearn UI recovery flows. Never ask for, retrieve, retain, "
    "reconstruct, quote, display, submit, broadcast, or recommend manually "
    "broadcasting a raw signed transaction. Do not direct the user to a generic "
    "third-party transaction broadcaster. Reaching this safety boundary does not "
    "by itself justify human handoff."
)
_GAS_SUFFICIENCY_INSTRUCTION = (
    "For any gas-sufficiency conclusion, compare the spendable native-token balance "
    "with the transaction's native-token value plus its maximum gas cost: gas limit "
    "multiplied by maximum fee per gas, or by legacy gas price. Retain a conservative "
    "buffer and also account for the gas and native-token value committed by pending "
    "or wallet-queued transactions. Never claim the wallet definitely has enough gas "
    "from its current balance alone. If any required transaction-value, fee, or queue "
    "evidence is unknown, state that sufficiency is conditional and name the missing "
    "check."
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
        codex_auth_sync_source: str | Path | None = None,
        codex_auth_link_source: str | Path | None = None,
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
        self.codex_auth_sync_source = (
            Path(codex_auth_sync_source) if codex_auth_sync_source else None
        )
        self.codex_auth_link_source = (
            Path(codex_auth_link_source) if codex_auth_link_source else None
        )
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
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_error_chars = max_error_chars
        self.execution_locks = _ConversationExecutionLocks()
        self.codex_runtime_lock = asyncio.Lock()
        self._prepare_support_home()

    async def prune_expired_sessions(self) -> int:
        if (
            self.session_manager is None
            or self.session_manager.max_age is None
            or self.codex_home is None
        ):
            return 0
        async with self.codex_runtime_lock:
            active_session_ids = {
                record.session_id for record in self.session_manager.list_records()
            }
            cutoff = datetime.now(timezone.utc) - self.session_manager.max_age
            session_ids = _expired_unreferenced_rollout_ids(
                self.codex_home,
                active_session_ids=active_session_ids,
                cutoff=cutoff,
            )
            removed = 0
            for session_id in session_ids:
                if await self._delete_codex_session(session_id):
                    removed += 1
            return removed

    async def _delete_codex_session(self, session_id: str) -> bool:
        command = _build_codex_delete_command(
            codex_command=self.codex_command,
            session_id=session_id,
        )
        effective_env = build_effective_execution_env(
            env=self.env,
            inherit_parent_env=self.inherit_parent_env,
            run_dir=self.codex_home or Path.cwd(),
        )
        try:
            await run_bounded_subprocess(
                command=command,
                stdin_text="",
                cwd=self.cwd,
                env=effective_env,
                timeout_seconds=_CODEX_SESSION_DELETE_TIMEOUT_SECONDS,
                max_output_chars=1000,
                max_error_chars=500,
                timeout_message=(
                    f"Timed out deleting Codex session {session_id}."
                ),
                empty_stdout_message=(
                    f"Codex did not confirm deletion of session {session_id}."
                ),
                oversized_stdout_message=(
                    f"Codex returned oversized deletion output for session {session_id}."
                ),
                metadata={"operation": "delete_codex_session"},
                artifact_run_dir=None,
            )
        except (OSError, RuntimeError) as exc:
            logging.warning(
                "Failed to delete Codex session %s: %s",
                session_id,
                str(exc)[:500],
            )
            return False
        else:
            return True

    async def _retire_contaminated_session(self, session_id: str) -> None:
        deletion_task = asyncio.create_task(self._delete_codex_session(session_id))
        interrupted = False
        while not deletion_task.done():
            try:
                await asyncio.shield(deletion_task)
            except asyncio.CancelledError:
                interrupted = True

        deleted = deletion_task.result()
        if not deleted:
            logging.warning(
                "Detached contaminated Codex session %s; immediate deletion failed.",
                session_id,
            )
        if interrupted:
            raise asyncio.CancelledError()

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
            workspace = TicketExecutionWorkspace(
                artifact_dir=self.artifact_dir or None,
                prefix="ticket-codex-support-run-",
            )
            with workspace as run_dir:
                async with self.codex_runtime_lock:
                    ysupport_mcp_enabled = self._prepare_support_home()
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
                    await prepare_support_request_attachments(
                        support_request,
                        run_dir=run_dir,
                    )
                    bundle = _build_codex_support_execution_bundle(
                        support_request=support_request,
                        run_dir=run_dir,
                        codex_command=self.codex_command,
                        model=self.model,
                        reasoning_effort=self.reasoning_effort,
                        resume_session_id=session_record.session_id if session_record else None,
                    )
                    persist_session = True
                    export_workspace = True
                    exported_run_dir: Path | None = None
                    try:
                        execution_output = await self._run_with_auth_retry(
                            bundle=bundle,
                            run_dir=run_dir,
                            hooks=hooks,
                        )
                        try:
                            support_result = verify_support_turn_result(
                                SupportTurnResult.from_json(
                                    execution_output.final_response_text
                                ),
                                support_request,
                            )
                        except SignedTransactionSafetyViolation:
                            persist_session = False
                            export_workspace = False
                            logging.warning(
                                "Codex support output crossed the signed-transaction "
                                "safety boundary for %s; requesting one safe rewrite.",
                                conversation_key or "unkeyed conversation",
                            )
                            rewrite_session_id = (
                                self._extract_session_id_from_run_dir(run_dir)
                                or bundle.resumed_session_id
                            )
                            _discard_unsafe_execution_stdout(run_dir)
                            if rewrite_session_id is None:
                                raise
                            try:
                                if (
                                    self.session_manager is not None
                                    and conversation_key is not None
                                ):
                                    self.session_manager.reset(conversation_key)
                                rewrite_run_dir = (
                                    run_dir / "transaction-safety-rewrite"
                                )
                                rewrite_bundle = _build_codex_support_execution_bundle(
                                    support_request=support_request,
                                    run_dir=rewrite_run_dir,
                                    codex_command=self.codex_command,
                                    model=self.model,
                                    reasoning_effort=self.reasoning_effort,
                                    resume_session_id=rewrite_session_id,
                                    transaction_safety_rewrite=True,
                                )
                                rewrite_output = await self._run_with_auth_retry(
                                    bundle=rewrite_bundle,
                                    run_dir=rewrite_run_dir,
                                    hooks=hooks,
                                )
                                support_result = verify_support_turn_result(
                                    SupportTurnResult.from_json(
                                        rewrite_output.final_response_text
                                    ),
                                    support_request,
                                )
                            finally:
                                await self._retire_contaminated_session(
                                    rewrite_session_id
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
                        if export_workspace:
                            exported_run_dir = safe_export_workspace_copy(
                                workspace,
                                logger_name=__name__,
                                context="codex support execution",
                            )

                try:
                    response_json = support_result_to_transport_result(
                        support_result,
                        request,
                    ).to_json()
                except Exception as exc:
                    if (
                        self.session_manager is not None
                        and conversation_key is not None
                        and session_record is not None
                    ):
                        self.session_manager.record_failure(
                            conversation_key=conversation_key,
                            error_text=str(exc),
                            artifact_dir=(
                                str(exported_run_dir) if exported_run_dir else None
                            ),
                        )
                    raise

                if (
                    persist_session
                    and self.session_manager is not None
                    and conversation_key is not None
                ):
                    session_id = (
                        self._extract_session_id_from_run_dir(run_dir)
                        or bundle.resumed_session_id
                    )
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
                return response_json

    def _prepare_support_home(self) -> bool:
        if not (self.codex_home and self.ysupport_mcp_url and self.mcp_server_api_key):
            return False
        support_home = prepare_codex_support_home(
            codex_home=self.codex_home,
            repo_root=self.repo_root,
            mcp_container_name=self.ysupport_mcp_container,
            auth_source=self.codex_auth_source,
            auth_sync_source=self.codex_auth_sync_source,
            auth_link_source=self.codex_auth_link_source,
            ysupport_mcp_url=self.ysupport_mcp_url,
            mcp_server_api_key=self.mcp_server_api_key,
            web_search_mode=self.web_search_mode,
        )
        return support_home.ysupport_mcp_enabled

    def _prime_support_home_auth_state(self) -> None:
        if self.codex_home is None:
            return
        prepare_codex_auth_state(
            home_auth_path=self.codex_home / "auth.json",
            auth_source_path=self.codex_auth_source,
            auth_sync_source_path=self.codex_auth_sync_source,
            auth_link_source_path=self.codex_auth_link_source,
            preferred_source_path=self.codex_auth_sync_source,
        )

    async def _run_with_auth_retry(
        self,
        *,
        bundle: CodexSupportExecutionBundle,
        run_dir: Path,
        hooks: TicketExecutionHooks | None,
    ) -> CodexSupportExecutionOutput:
        last_error: Exception | None = None
        for attempt in range(2):
            self._prime_support_home_auth_state()
            if attempt > 0:
                self._prepare_support_home()
            try:
                return await run_codex_support_json_subprocess(
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
                        "auth_retry_attempt": attempt,
                    },
                    artifact_run_dir=run_dir,
                    progress_callback=(
                        hooks.send_progress_update
                        if hooks is not None
                        else None
                    ),
                )
            except Exception as exc:
                last_error = exc
                if attempt == 0 and _is_codex_auth_error_text(str(exc)):
                    continue
                raise
            finally:
                self._sync_support_home_auth_state()
        assert last_error is not None
        raise last_error

    def _sync_support_home_auth_state(self) -> None:
        if self.codex_home is None:
            return
        prepare_codex_auth_state(
            home_auth_path=self.codex_home / "auth.json",
            auth_source_path=self.codex_auth_source,
            auth_sync_source_path=self.codex_auth_sync_source,
            auth_link_source_path=self.codex_auth_link_source,
        )

    def _extract_session_id_from_run_dir(self, run_dir: Path) -> str | None:
        stdout_path = run_dir / "stdout.txt"
        if stdout_path.exists() and self.session_manager is not None:
            try:
                stdout_text = stdout_path.read_text(encoding="utf-8")
            except OSError:
                pass
            else:
                session_id = self.session_manager.extract_session_id_from_jsonl(
                    stdout_text
                )
                if session_id is not None:
                    return session_id
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


def _discard_unsafe_execution_stdout(run_dir: Path) -> None:
    try:
        (run_dir / "stdout.txt").unlink(missing_ok=True)
    except OSError as exc:
        logging.warning(
            "Could not remove unsafe Codex execution stdout from temporary workspace: %s",
            exc,
        )


def _build_codex_support_execution_bundle(
    *,
    support_request: SupportTurnRequest,
    run_dir: str | Path,
    codex_command: Sequence[str],
    model: str | None,
    reasoning_effort: str | None,
    resume_session_id: str | None = None,
    transaction_safety_rewrite: bool = False,
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
    if transaction_safety_rewrite:
        prompt_text = _codex_support_transaction_safety_rewrite_prompt(
            response_schema_path=response_schema_path,
        )
    else:
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
        image_paths=image_attachment_paths(support_request),
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
    if request.run_context.get("is_public_trigger"):
        conversation_owner_id = request.run_context.get("conversation_owner_id")
        if conversation_owner_id is not None:
            return f"public_user:{conversation_owner_id}"
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
    image_paths: list[Path],
    resume_session_id: str | None,
) -> list[str]:
    if resume_session_id:
        command = _build_codex_resume_command(
            codex_command=codex_command,
            session_id=resume_session_id,
            model=model,
            reasoning_effort=reasoning_effort,
            image_paths=image_paths,
        )
        command.extend(["--output-schema", str(response_schema_path), "-"])
        return command

    command = [
        arg for arg in codex_command
        if arg != "--ephemeral"
    ]
    if model:
        command.extend(["-m", model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    for image_path in image_paths:
        command.extend(["-i", str(image_path)])
    command.extend(
        [
            "--json",
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
    image_paths: list[Path],
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
    for image_path in image_paths:
        command.extend(["-i", str(image_path)])
    command.append("--json")
    return command


def _build_codex_delete_command(
    *,
    codex_command: Sequence[str],
    session_id: str,
) -> list[str]:
    if not _SESSION_UUID_RE.fullmatch(session_id):
        raise ValueError("Codex session deletion requires a UUID session ID.")
    exec_index = next(
        (index for index, token in enumerate(codex_command) if token == "exec"),
        None,
    )
    if exec_index is None:
        raise ValueError("Codex support command must include an exec subcommand.")
    return list(codex_command[:exec_index]) + [
        "delete",
        "--force",
        session_id,
    ]


def _expired_unreferenced_rollout_ids(
    codex_home: Path,
    *,
    active_session_ids: set[str],
    cutoff: datetime,
) -> list[str]:
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return []
    session_ids: set[str] = set()
    for path in sessions_dir.rglob("*.jsonl"):
        match = _ROLLOUT_SESSION_ID_RE.search(path.name)
        if match is None:
            continue
        session_id = match.group("session_id")
        if (
            session_id not in active_session_ids
            and datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) < cutoff
        ):
            session_ids.add(session_id)
    return sorted(session_ids)


def _codex_support_prompt(
    *,
    support_request_path: Path,
    response_schema_path: Path,
) -> str:
    support_request_path = support_request_path.resolve()
    response_schema_path = response_schema_path.resolve()
    return (
        "You are ySupport.\n\n"
        f"Read the support turn request from {support_request_path}.\n"
        f"Return only JSON matching {response_schema_path}.\n"
        "Use `current_turn_source` to determine who authored `current_user_message`.\n"
        "If `current_turn_source` is `internal_team`, treat `current_user_message` as an internal team update, not as the user speaking.\n"
        "If `current_turn_instruction` is present, follow it as the required output contract for this turn.\n"
        "Use `channel_type`, `channel_id`, `initial_button_intent`, and `requested_intent` as workflow context.\n"
        "Use `support_state` as explicit runtime state. Prefer it over re-inferring the same facts from transcript when they agree.\n"
        "If `attachments` includes screenshot or image files, inspect them before answering any question that depends on what they show.\n"
        "Use `support_state.workflow_context` as outer Discord/UI context. Follow its expected first actions before generic clarification.\n"
        "Treat `support_state.workflow_context.non_support_boundaries` as hard outer guardrails.\n"
        "If a boundary or off-scope message reaches you, reply briefly and stay on the boundary.\n"
        f"If the user opens with an ambiguous request to report a bug and has not said whether it is an ordinary product issue or a security vulnerability or bounty disclosure, begin the reply with {SECURITY_PROCESS_URL} and tell them not to post sensitive security details in Discord. Only after that, offer to accept ordinary product-bug details. Do not stop or request human handoff solely because the user used generic bug-report wording.\n"
        "Use only the tools listed in `constraints.allowed_tools`.\n"
        "If `ysupport_mcp` is listed, prefer it for Yearn-specific docs, repo context, vault context, and support facts.\n"
        "For docs/mechanics/product-explanation questions, synthesize a concise direct answer from the Yearn documentation excerpts and preserve relevant official links and YIP status.\n"
        "Treat Yearn documentation excerpts as the sole factual grounding for those answers when they resolve the question; do not add web search or external sources.\n"
        "Do not expose retrieval metadata or say that YIP status metadata is absent.\n"
        "No file writes.\n"
        f"{_TRANSACTION_SAFETY_INSTRUCTION}\n"
        f"{_GAS_SUFFICIENCY_INSTRUCTION}\n"
        "Do not tell the user to go to Discord or open a Discord ticket.\n"
        "If the exact fact is known, give it first.\n"
        "If the user asked multiple questions, answer them in order.\n"
        "For screenshot or metric-comparison questions, do not stop at a plausible explanation. Identify the exact labels and numbers in the user evidence, then compare them to current Yearn data only if needed.\n"
        "For unexplained portfolio value or redemption of a retired or non-redeemable receipt, reconcile displayed balances with presently realizable wallet positions and available history, including whether the value was already redeemed, migrated, distributed, or represented elsewhere. A receipt or share balance, PPS, vault accounting field, or zero redemption limit alone does not prove a current economic claim; if corresponding realizable value is unproven, report the discrepancy or uncertainty and do not claim the funds are safe, stuck, claimable, awaiting liquidity or operator action, or will later become redeemable.\n"
        "For investigation answers, use at least one Yearn-specific fact source and one metric-definition or repo/docs source when the question is about how displayed metrics differ.\n"
        "Before setting requires_human_handoff=true, exhaust the relevant available documentation, live-data, repository, web, and image evidence for the issue. Give every useful verified finding and troubleshooting step first.\n"
        "Inspect linked artifacts before handoff.\n"
        "A request for a human, moderator, admin, strategist, or team review does not by itself justify handoff. Continue the support investigation unless a concrete human-only action remains.\n"
        "The handoff reason must name that concrete remaining action, private fact, access change, recovery step, or decision. Do not hand off low-level support, vague reports, ordinary uncertainty, or issues the user can continue troubleshooting with the bot.\n"
        "When requires_human_handoff=true, set handoff_kind to exactly one of access_or_permission_action, fund_or_account_recovery, security_process, manual_strategy_action, private_internal_fact, or human_decision. Otherwise set handoff_kind=null.\n"
        "If the user is explicitly asking Yearn/team to dump rewards, swap rewards, sell rewards, compound, reinvest, or otherwise take a manual operator action on a strategy, vault, or pool, set requires_human_handoff=true.\n"
        "Keep the JSON internally consistent: if your answer says Yearn, the team, a strategist, or an operator must perform or review an action, requires_human_handoff must be true and handoff_reason must name that action. Never describe a required human or team action while returning requires_human_handoff=false.\n"
        "When handoff is required, explain the remaining human action but do not claim that you have escalated, handed off, or notified anyone. The outer runtime adds that confirmation only after Telegram delivery succeeds.\n"
        "Do not mention handoff if public evidence already answers the main question.\n"
        "If the remaining gap is only internal why/when context, answer and stop.\n"
        "If human review is truly needed, set requires_human_handoff=true and explain why briefly.\n"
        "Routine support: concise.\n"
        "Investigations and report triage: enough prose to explain conclusion, evidence, and remaining limit.\n"
        "Stop once the asked question is answered. No add-on sections.\n"
    )


def _codex_support_transaction_safety_rewrite_prompt(
    *,
    response_schema_path: Path,
) -> str:
    response_schema_path = response_schema_path.resolve()
    return (
        "Your previous response for this support turn exposed a transaction-sized "
        "serialized hex payload. Rewrite the response using only safe, read-only "
        "transaction troubleshooting. Keep the useful verified diagnosis and use "
        "transaction hashes, decoded fields, statuses, non-mutating calls or "
        "simulations, and official wallet or Yearn UI recovery flows as appropriate. "
        f"{_TRANSACTION_SAFETY_INSTRUCTION} "
        f"{_GAS_SUFFICIENCY_INSTRUCTION} "
        f"Return only JSON matching {response_schema_path}."
    )


def _is_codex_auth_error_text(text: str) -> bool:
    normalized = text.lower()
    return any(
        token in normalized
        for token in (
            "refresh_token_reused",
            "token_expired",
            "failed to refresh token",
            "provided authentication token is expired",
            "your access token could not be refreshed",
        )
    )
