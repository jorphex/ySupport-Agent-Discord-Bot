from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from time import monotonic
from typing import Sequence
from weakref import WeakValueDictionary

from ysupport.execution.codex_support_home import prepare_codex_support_home, prune_codex_support_caches
from ysupport.execution.codex_support_contract import (
    CODEX_SUPPORT_RESULT_SCHEMA,
    SignedTransactionSafetyViolation,
    SupportTurnRequest,
    SupportTurnResult,
    support_result_to_transport_result,
    verify_support_turn_result,
)
from ysupport.execution.codex_support_sessions import (
    CodexSupportSessionManager,
    conversation_key_for_request,
)
from ysupport.execution.codex_support_attachments import (
    image_attachment_paths,
    prepare_support_request_attachments,
)
from ysupport.execution.codex_support_subprocess import (
    CodexSupportExecutionOutput,
    run_codex_support_json_subprocess,
)
from ysupport.execution.subprocess_utils import (
    build_effective_execution_env,
    run_bounded_subprocess,
    safe_export_workspace_copy,
    validate_allowed_command_prefix,
)
from ysupport.execution.workspace import TicketExecutionWorkspace
from ysupport.execution.executor import (
    TicketExecutionHooks,
    TicketExecutionNonFallbackError,
)
from ysupport.execution.context import merge_explicit_evidence
from ysupport.execution.internal_rpc import InternalRpcServer
from ysupport.execution.transport import (
    build_smoke_transport_result,
    TicketExecutionTransportRequest,
    deserialize_investigation_job,
    deserialize_investigation_evidence,
    serialize_investigation_job,
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
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
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


class _RuntimeConcurrency:
    """Bound executions while reserving cache maintenance for idle periods."""

    def __init__(self, limit: int):
        if type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError("Concurrent support turns must be between 1 and 8.")
        self.slots = asyncio.Semaphore(limit)
        self.condition = asyncio.Condition()
        self.active = 0
        self.waiting = 0
        self.maintaining = False

    @asynccontextmanager
    async def turn(self):
        self.waiting += 1
        admitted = False
        try:
            async with self.slots:
                async with self.condition:
                    await self.condition.wait_for(lambda: not self.maintaining)
                    self.waiting -= 1
                    self.active += 1
                    admitted = True
                try:
                    yield
                finally:
                    self.active -= 1
        finally:
            if not admitted:
                self.waiting -= 1

    @asynccontextmanager
    async def idle_maintenance(self):
        async with self.condition:
            admitted = not self.active and not self.waiting and not self.maintaining
            if admitted:
                self.maintaining = True
        try:
            yield admitted
        finally:
            if admitted:
                async with self.condition:
                    self.maintaining = False
                    self.condition.notify_all()


class CodexSupportTicketExecutionJsonEndpoint:
    def __init__(
        self,
        *,
        codex_command: Sequence[str] | None = None,
        app_server_command: Sequence[str] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        codex_home: str | Path | None = None,
        codex_auth_link_source: str | Path | None = None,
        session_dir: str | Path | None = None,
        session_max_age_hours: int | None = None,
        ysupport_mcp_url: str | None = None,
        mcp_server_api_key: str | None = None,
        web_search_mode: str = "live",
        allowed_command_prefixes: Sequence[Sequence[str]] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_parent_env: bool = False,
        artifact_dir: str | None = None,
        timeout_seconds: float = 300.0,
        max_concurrent_turns: int = 2,
        max_output_chars: int = 200000,
        max_error_chars: int = 4000,
    ) -> None:
        command = list(codex_command or DEFAULT_CODEX_EXEC_COMMAND)
        if not command:
            raise ValueError("Codex support execution command cannot be empty.")
        self.codex_command = command
        self.app_server_command = list(app_server_command) if app_server_command else None
        if self.app_server_command:
            prefix = command[:command.index("exec")]
            if not prefix or self.app_server_command[:len(prefix)] != prefix:
                raise ValueError("Interactive Codex must use the approved execution prefix.")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex_home = Path(codex_home).resolve() if codex_home else None
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
        if self.codex_home is not None:
            self.env = {**(self.env or {}), "CODEX_HOME": str(self.codex_home)}
        self.inherit_parent_env = inherit_parent_env
        self.artifact_dir = artifact_dir
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_error_chars = max_error_chars
        self.execution_locks = _ConversationExecutionLocks()
        self.runtime = _RuntimeConcurrency(max_concurrent_turns)
        self.ysupport_mcp_enabled = self._prepare_support_home()

    async def prune_expired_sessions(self) -> int:
        if (
            self.session_manager is None
            or self.session_manager.max_age is None
            or self.codex_home is None
        ):
            return 0
        async with self.runtime.idle_maintenance() as admitted:
            if not admitted:
                return 0
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
                if self.runtime.waiting:
                    break
                if await self._delete_codex_session(session_id):
                    removed += 1
            if not self.runtime.waiting:
                await asyncio.to_thread(
                    prune_codex_support_caches, self.codex_home, cutoff=cutoff
                )
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
        if request.smoke_mode:
            return build_smoke_transport_result(
                request,
                endpoint_mode="codex_support_exec",
            ).to_json()
        conversation_key = _conversation_key_for_request(request)
        started = monotonic()
        timings = {"queue_ms": None, "setup_ms": None, "execution_ms": None}
        outcome = "error"
        try:
            async with self.execution_locks.acquire(conversation_key), self.runtime.turn():
                timings["queue_ms"] = round((monotonic() - started) * 1000)
                result = await self._execute_admitted_turn(request, hooks, conversation_key, timings)
                outcome = "completed"
                return result
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            if (
                hooks is not None and hooks.steering is not None
                and any(entry.submitted for entry in hooks.steering.inputs)
            ):
                raise TicketExecutionNonFallbackError(
                    "Interactive support output failed after follow-up submission."
                ) from exc
            raise
        finally:
            total_ms = round((monotonic() - started) * 1000)
            logging.info(
                "Support turn timings channel_id=%s conversation=%s outcome=%s "
                "queue_ms=%s setup_ms=%s execution_ms=%s total_ms=%s",
                request.run_context.get("channel_id"), conversation_key, outcome,
                timings["queue_ms"] if timings["queue_ms"] is not None else total_ms,
                timings["setup_ms"], timings["execution_ms"], total_ms,
            )

    async def _execute_admitted_turn(self, request, hooks, conversation_key, timings):
        setup_started = monotonic()
        workspace = TicketExecutionWorkspace(
            artifact_dir=self.artifact_dir or None,
            prefix="ticket-codex-support-run-",
        )
        with workspace as run_dir:
            async with InternalRpcServer(
                run_dir, timeout_seconds=self.timeout_seconds
            ) as internal_rpc:
                support_request = SupportTurnRequest.from_ticket_execution_request(
                    request,
                    ysupport_mcp_enabled=self.ysupport_mcp_enabled,
                )
                support_request.support_state["internal_rpc"] = internal_rpc.description
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
                timings["setup_ms"] = round((monotonic() - setup_started) * 1000)
                execution_started = monotonic()
                try:
                    execution_output = await self._run_codex(
                        bundle=bundle,
                        run_dir=run_dir,
                        hooks=hooks,
                    )
                    if hooks is not None and hooks.steering is not None:
                        if any(
                            entry.submitted and not entry.observed
                            for entry in hooks.steering.inputs
                        ):
                            persist_session = False
                            if self.session_manager is not None and conversation_key is not None:
                                self.session_manager.reset(conversation_key)
                        updated_job = deserialize_investigation_job(request.investigation_job)
                        for entry in hooks.steering.consumed:
                            request.aggregated_text += "\n" + entry.text
                            request.attachments.extend(entry.attachments)
                            support_request.attachments.extend(entry.prepared_attachments)
                            if entry.evidence is not None:
                                updated_job.evidence = deserialize_investigation_evidence(entry.evidence)
                            else:
                                merge_explicit_evidence(updated_job, entry.text)
                        request.investigation_job = serialize_investigation_job(updated_job)
                        support_request.current_user_message = request.aggregated_text
                        support_request.evidence = dict(request.investigation_job["evidence"])
                        if hooks.steering.consumed:
                            final_request = SupportTurnRequest.from_ticket_execution_request(
                                request, ysupport_mcp_enabled=self.ysupport_mcp_enabled,
                            )
                            support_request.support_state["known_targets"] = final_request.support_state["known_targets"]
                            support_request.support_state["current_turn_context"].extend(
                                hint for entry in hooks.steering.consumed for hint in entry.system_hints
                            )
                            (run_dir / "support_request.final.json").write_text(support_request.to_json())
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
                            rewrite_output = await self._run_codex(
                                bundle=rewrite_bundle,
                                run_dir=rewrite_run_dir,
                                hooks=(TicketExecutionHooks(send_progress_update=hooks.send_progress_update)
                                       if hooks else None),
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
                    timings["execution_ms"] = round((monotonic() - execution_started) * 1000)
                    await internal_rpc.close()
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
        if self.codex_home is None:
            return False
        support_home = prepare_codex_support_home(
            codex_home=self.codex_home,
            auth_link_source=self.codex_auth_link_source,
            ysupport_mcp_url=self.ysupport_mcp_url or "",
            mcp_server_api_key=self.mcp_server_api_key or "",
            web_search_mode=self.web_search_mode,
        )
        return support_home.ysupport_mcp_enabled

    async def _run_codex(
        self,
        *,
        bundle: CodexSupportExecutionBundle,
        run_dir: Path,
        hooks: TicketExecutionHooks | None,
    ) -> CodexSupportExecutionOutput:
        if self.app_server_command and hooks is not None and hooks.steering is not None:
            from ysupport.execution.codex_support_app_server import run_codex_support_app_server
            support_request = SupportTurnRequest(**json.loads(bundle.support_request_path.read_text()))
            if support_request.current_turn_source != "user" or support_request.channel_type != "ticket":
                raise TicketExecutionNonFallbackError("Only ordinary ticket-owner turns support steering.")
            return await run_codex_support_app_server(
                command=self.app_server_command, prompt_text=bundle.prompt_text,
                support_request=support_request, resume_session_id=bundle.resumed_session_id,
                model=self.model, reasoning_effort=self.reasoning_effort,
                cwd=self.cwd or str(run_dir),
                env=build_effective_execution_env(env=self.env, inherit_parent_env=self.inherit_parent_env, run_dir=run_dir),
                timeout_seconds=self.timeout_seconds, max_output_chars=self.max_output_chars,
                run_dir=run_dir, hooks=hooks,
            )
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
            },
            artifact_run_dir=run_dir,
            progress_callback=(
                hooks.send_progress_update if hooks is not None else None
            ),
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
            support_request=support_request,
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
    return conversation_key_for_request(request)


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
    support_request: SupportTurnRequest,
    support_request_path: Path,
    response_schema_path: Path,
) -> str:
    support_request_path = support_request_path.resolve()
    response_schema_path = response_schema_path.resolve()
    return (
        "Execute one ySupport turn using the standing instructions.\n"
        f"Support request: {support_request_path}\n"
        f"Response schema: {response_schema_path}\n"
        "Use the current JSON below directly; the files are identical snapshots.\n"
        f"Support request JSON:\n{support_request.to_json()}\n"
        f"Response schema JSON:\n{json.dumps(CODEX_SUPPORT_RESULT_SCHEMA)}\n"
        + (
            "ySupport MCP tools (use the matching callable; no full catalog listing needed):\n"
            "search_documentation: official docs; search_repo_context then fetch_repo_artifacts: source code.\n"
            "search_vaults: vault identity/exact details; support_dashboard_discover: current vault yields.\n"
            "support_dashboard_styfi: stYFI; support_dashboard_flex: Flex lending.\n"
            "support_dashboard_reports, support_dashboard_changes, support_dashboard_status: dashboard reports/changes/health.\n"
            "repo_context_status: repository index health.\n"
            if "ysupport_mcp" in support_request.constraints.get("allowed_tools", []) else ""
        )
        + "Use support_state.internal_rpc.command for chain reads; its operations and parameters are in the request. "
        "Batch independent reads in one shell invocation. Preserve exact integer units; decimal strings are accepted for integer contract arguments. "
        "Return only schema-valid JSON."
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
