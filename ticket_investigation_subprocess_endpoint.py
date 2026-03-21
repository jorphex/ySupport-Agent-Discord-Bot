import asyncio
from collections.abc import Sequence
import os

from ticket_investigation_executor import TicketExecutionHooks
from ticket_investigation_transport import (
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
        self._validate_command_prefix()
        self.cwd = cwd
        self.env = dict(env) if env is not None else None
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

        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=self._effective_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request_json.encode("utf-8")),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"Ticket execution subprocess timed out after {self.timeout_seconds} seconds."
            ) from exc

        if process.returncode != 0:
            error_text = stderr.decode("utf-8", errors="replace").strip()
            if not error_text:
                error_text = "Subprocess exited without stderr output."
            raise RuntimeError(error_text[: self.max_error_chars])

        response_text = stdout.decode("utf-8", errors="replace").strip()
        if not response_text:
            raise RuntimeError("Ticket execution subprocess returned empty stdout.")
        if len(response_text) > self.max_output_chars:
            raise RuntimeError(
                "Ticket execution subprocess returned too much stdout."
            )

        TicketExecutionTransportResult.from_json(response_text)
        return response_text

    def _effective_env(self) -> dict[str, str] | None:
        if self.env is None:
            return None
        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    def _validate_command_prefix(self) -> None:
        if not self.allowed_command_prefixes:
            return
        for prefix in self.allowed_command_prefixes:
            if self.command[: len(prefix)] == prefix:
                return
        raise ValueError(
            "Ticket execution subprocess command is not in the allowed prefix list."
        )
