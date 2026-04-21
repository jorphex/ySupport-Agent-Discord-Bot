from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ticket_investigation.codex_bundle import (
    DEFAULT_CODEX_EXEC_COMMAND,
    build_codex_ticket_execution_bundle,
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
    TicketExecutionTransportResult,
)


class CodexExecTicketExecutionJsonEndpoint:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        codex_command: Sequence[str] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
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
            raise ValueError("Codex ticket execution command cannot be empty.")
        self.repo_root = Path(repo_root)
        self.codex_command = command
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.allowed_command_prefixes = [
            list(prefix) for prefix in (allowed_command_prefixes or [])
        ]
        validate_allowed_command_prefix(
            self.codex_command,
            self.allowed_command_prefixes,
            error_message=(
                "Codex ticket execution command is not in the allowed prefix list."
            ),
        )
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
        self.inherit_parent_env = inherit_parent_env
        self.artifact_dir = artifact_dir
        self.run_dir_root = run_dir_root
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.max_error_chars = max_error_chars

    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        request = TicketExecutionTransportRequest.from_json(request_json)
        if request.wants_bug_review_status and hooks is not None:
            if hooks.send_bug_review_status is not None:
                await hooks.send_bug_review_status()

        workspace = TicketExecutionWorkspace(
            artifact_dir=self.artifact_dir or None,
            run_dir_root=self.run_dir_root or None,
            prefix="ticket-codex-run-",
        )
        with workspace as run_dir:
            smoke_response_json = None
            if request.smoke_mode:
                smoke_response_json = build_smoke_transport_result(
                    request,
                    endpoint_mode="codex_exec",
                ).to_json()
            bundle = build_codex_ticket_execution_bundle(
                request_json=request_json,
                run_dir=run_dir,
                repo_root=self.repo_root,
                codex_command=self.codex_command,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                response_json_override=smoke_response_json,
            )
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
                        f"Codex ticket execution timed out after {self.timeout_seconds} seconds."
                    ),
                    empty_stdout_message="Codex execution returned empty stdout.",
                    oversized_stdout_message="Codex execution returned too much stdout.",
                    metadata={
                        "base_command": self.codex_command,
                        "command": bundle.command,
                        "cwd": self.cwd or str(run_dir),
                    },
                    artifact_run_dir=run_dir,
                )
            finally:
                safe_export_workspace_copy(
                    workspace,
                    logger_name=__name__,
                    context="codex ticket execution",
                )

            TicketExecutionTransportResult.from_json(response_text)
            return response_text


def build_codex_exec_allowed_prefixes(
    command: Sequence[str],
    extra_prefixes: Sequence[Sequence[str]],
) -> list[list[str]]:
    prefixes = [list(command)]
    for prefix in extra_prefixes:
        prefix_list = list(prefix)
        if prefix_list not in prefixes:
            prefixes.append(prefix_list)
    return prefixes
