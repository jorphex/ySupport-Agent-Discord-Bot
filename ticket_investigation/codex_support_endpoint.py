from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
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
from ticket_execution.subprocess_utils import (
    build_effective_execution_env,
    safe_export_workspace_copy,
    validate_allowed_command_prefix,
    write_execution_artifacts,
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
                    response_text = await _run_codex_support_json_subprocess(
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
                            hooks.send_progress_update
                            if hooks is not None
                            else None
                        ),
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
    command.append("--json")
    return command


def _codex_support_prompt(
    *,
    support_request_path: Path,
    response_schema_path: Path,
) -> str:
    return (
        "You are ySupport.\n\n"
        f"Read the support turn request from {support_request_path.name}.\n"
        f"Return only JSON matching {response_schema_path.name}.\n"
        "Use `channel_type`, `channel_id`, `initial_button_intent`, and `requested_intent` as workflow context.\n"
        "Use `support_state` as explicit runtime state. Prefer it over re-inferring the same facts from transcript when they agree.\n"
        "Use `support_state.workflow_context` as outer Discord/UI context. Follow its expected first actions before generic clarification.\n"
        "Treat `support_state.workflow_context.non_support_boundaries` as hard outer guardrails.\n"
        "If a boundary or off-scope message reaches you, reply briefly and stay on the boundary.\n"
        "Use only the tools listed in `constraints.allowed_tools`.\n"
        "If `ysupport_mcp` is listed, prefer it for Yearn-specific docs, repo context, vault context, and support facts.\n"
        "No file writes.\n"
        "Do not tell the user to go to Discord or open a Discord ticket.\n"
        "If the exact fact is known, give it first.\n"
        "If the user asked multiple questions, answer them in order.\n"
        "Before setting requires_human_handoff=true, do the strongest direct support work you can from the request and tools.\n"
        "Inspect linked artifacts before handoff.\n"
        "If the user asks for a human but also gives a concrete target or question, answer what you can first.\n"
        "Do not mention handoff if public evidence already answers the main question.\n"
        "If the remaining gap is only internal why/when context, answer and stop.\n"
        "If human review is truly needed, set requires_human_handoff=true and explain why briefly.\n"
        "Routine support: concise.\n"
        "Investigations and report triage: enough prose to explain conclusion, evidence, and remaining limit.\n"
        "Stop once the asked question is answered. No add-on sections.\n"
    )


async def _run_codex_support_json_subprocess(
    *,
    command: Sequence[str],
    stdin_text: str,
    cwd: str | None,
    env: dict[str, str],
    timeout_seconds: float,
    max_output_chars: int,
    max_error_chars: int,
    timeout_message: str,
    empty_stdout_message: str,
    oversized_stdout_message: str,
    metadata: dict[str, object],
    artifact_run_dir: Path | None,
    progress_callback,
) -> str:
    creation_kwargs = (
        {"creationflags": 0}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
        **creation_kwargs,
    )
    stdout_lines: list[str] = []
    stderr_chunks: list[str] = []
    final_response_text: str | None = None

    async def _feed_stdin() -> None:
        if process.stdin is None:
            return
        process.stdin.write(stdin_text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        try:
            await process.stdin.wait_closed()
        except Exception:
            return

    async def _read_stdout() -> None:
        nonlocal final_response_text
        if process.stdout is None:
            return
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            if not text:
                continue
            stdout_lines.append(text)
            event = _parse_json_event(text)
            if event is None:
                continue
            progress_text = _progress_update_from_codex_event(event)
            if progress_text and progress_callback is not None:
                await progress_callback(progress_text)
            response_text = _final_response_from_codex_event(event)
            if response_text:
                final_response_text = response_text

    async def _read_stderr() -> None:
        if process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _feed_stdin(),
                _read_stdout(),
                _read_stderr(),
            ),
            timeout=timeout_seconds,
        )
        await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        await _terminate_streamed_subprocess(process)
        write_execution_artifacts(
            artifact_run_dir,
            stdout_text="\n".join(stdout_lines),
            stderr_text="".join(stderr_chunks).strip(),
            metadata={**metadata, "timed_out": True},
        )
        raise RuntimeError(timeout_message) from exc

    stdout_text = "\n".join(stdout_lines).strip()
    stderr_text = "".join(stderr_chunks).strip()
    write_execution_artifacts(
        artifact_run_dir,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        metadata={**metadata, "returncode": process.returncode, "timed_out": False},
    )
    if process.returncode != 0:
        error_text = stderr_text or "Codex support execution exited without stderr output."
        raise RuntimeError(error_text[:max_error_chars])
    if final_response_text is None:
        if not stdout_text:
            raise RuntimeError(empty_stdout_message)
        raise RuntimeError("Codex support execution returned JSON events without a final agent message.")
    if len(final_response_text) > max_output_chars:
        raise RuntimeError(oversized_stdout_message)
    return final_response_text


def _parse_json_event(text: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _final_response_from_codex_event(event: dict[str, object]) -> str | None:
    if event.get("type") != "item.completed":
        return None
    item = event.get("item")
    if not isinstance(item, dict):
        return None
    if item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None


def _progress_update_from_codex_event(event: dict[str, object]) -> str | None:
    if event.get("type") != "item.started":
        return None
    item = event.get("item")
    if not isinstance(item, dict):
        return None
    item_type = str(item.get("type") or "").strip().lower()
    if not item_type or item_type in {"agent_message", "todo_list", "file_change"}:
        return None
    if item_type == "command_execution":
        return _progress_from_command_execution(str(item.get("command") or ""))
    if "web" in item_type or "search" in item_type:
        return "Checking external references"
    if "mcp" in item_type or item.get("tool_name") or item.get("server") or item.get("name"):
        return _progress_from_tool_item(item)
    return None


def _progress_from_command_execution(command: str) -> str | None:
    normalized = command.lower()
    if (
        "notes.md" in normalized
        or "support_request.json" in normalized
        or "support_response_schema.json" in normalized
    ):
        return None
    if "http" in normalized or "curl " in normalized or "wget " in normalized:
        return "Reading linked references"
    return "Running a local check"


def _progress_from_tool_item(item: dict[str, object]) -> str:
    raw_name = " ".join(
        str(item.get(key) or "")
        for key in ("tool_name", "name", "server", "title")
    ).lower()
    if "harvest" in raw_name or "report" in raw_name:
        return "Checking recent harvests"
    if "search_vaults" in raw_name or "discover" in raw_name or "vault" in raw_name:
        return "Checking vault state"
    if "repo" in raw_name or "artifact" in raw_name:
        return "Checking repo context"
    if "document" in raw_name or "doc" in raw_name:
        return "Checking Yearn docs"
    return "Checking Yearn support data"


async def _terminate_streamed_subprocess(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(os.getpgid(process.pid), 9)
    except ProcessLookupError:
        pass
    finally:
        await process.wait()
