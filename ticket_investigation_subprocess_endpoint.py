from collections.abc import Sequence
import json
from pathlib import Path

from ticket_execution_subprocess_utils import (
    build_effective_execution_env,
    run_bounded_subprocess,
    safe_export_workspace_copy,
    validate_allowed_command_prefix,
)
from ticket_execution_workspace import TicketExecutionWorkspace
from ticket_investigation_executor import TicketExecutionHooks
from ticket_investigation_transport import (
    TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA,
    TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA,
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)


class SubprocessTicketExecutionJsonEndpoint:
    def __init__(
        self,
        command: Sequence[str],
        *,
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
        if not command:
            raise ValueError("Subprocess ticket execution command cannot be empty.")
        self.command = list(command)
        self.allowed_command_prefixes = [
            list(prefix) for prefix in (allowed_command_prefixes or [])
        ]
        validate_allowed_command_prefix(
            self.command,
            self.allowed_command_prefixes,
            error_message=(
                "Ticket execution subprocess command is not in the allowed prefix list."
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
            prefix="ticket-subprocess-run-",
        )
        with workspace as run_dir:
            self._start_artifact_run(
                run_dir if self.artifact_dir else None,
                request_json,
            )
            try:
                response_text = await run_bounded_subprocess(
                    command=self.command,
                    stdin_text=request_json,
                    cwd=self.cwd,
                    env=build_effective_execution_env(
                        env=self.env,
                        inherit_parent_env=self.inherit_parent_env,
                        run_dir=run_dir,
                    ),
                    timeout_seconds=self.timeout_seconds,
                    max_output_chars=self.max_output_chars,
                    max_error_chars=self.max_error_chars,
                    timeout_message=(
                        f"Ticket execution subprocess timed out after {self.timeout_seconds} seconds."
                    ),
                    empty_stdout_message=(
                        "Ticket execution subprocess returned empty stdout."
                    ),
                    oversized_stdout_message=(
                        "Ticket execution subprocess returned too much stdout."
                    ),
                    metadata={
                        "command": self.command,
                        "cwd": self.cwd,
                    },
                    artifact_run_dir=run_dir if self.artifact_dir else None,
                )
            finally:
                safe_export_workspace_copy(
                    workspace,
                    logger_name=__name__,
                    context="ticket execution subprocess",
                )

            TicketExecutionTransportResult.from_json(response_text)
            return response_text

    def _start_artifact_run(self, run_dir: Path | None, request_json: str) -> Path | None:
        if run_dir is None or not self.artifact_dir:
            return run_dir
        (run_dir / "request.json").write_text(request_json, encoding="utf-8")
        (run_dir / "request_schema.json").write_text(
            json.dumps(TICKET_EXECUTION_TRANSPORT_REQUEST_SCHEMA, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "response_schema.json").write_text(
            json.dumps(TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return run_dir
