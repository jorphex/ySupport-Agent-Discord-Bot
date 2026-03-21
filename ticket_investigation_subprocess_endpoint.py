import asyncio
from collections.abc import Sequence
import json
import os
from pathlib import Path
import uuid

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
        self._validate_command_prefix()
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

        run_dir = self._start_run_dir()
        artifact_run_dir = self._start_artifact_run(run_dir, request_json)
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=self._effective_env(run_dir),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request_json.encode("utf-8")),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            self._write_artifacts(
                artifact_run_dir,
                stdout_text="",
                stderr_text="",
                metadata={
                    "command": self.command,
                    "cwd": self.cwd,
                    "timed_out": True,
                },
            )
            raise RuntimeError(
                f"Ticket execution subprocess timed out after {self.timeout_seconds} seconds."
            ) from exc

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        self._write_artifacts(
            artifact_run_dir,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            metadata={
                "command": self.command,
                "cwd": self.cwd,
                "returncode": process.returncode,
                "timed_out": False,
            },
        )

        if process.returncode != 0:
            error_text = stderr_text
            if not error_text:
                error_text = "Subprocess exited without stderr output."
            raise RuntimeError(error_text[: self.max_error_chars])

        response_text = stdout_text
        if not response_text:
            raise RuntimeError("Ticket execution subprocess returned empty stdout.")
        if len(response_text) > self.max_output_chars:
            raise RuntimeError(
                "Ticket execution subprocess returned too much stdout."
            )

        TicketExecutionTransportResult.from_json(response_text)
        return response_text

    def _effective_env(self, run_dir: Path | None) -> dict[str, str] | None:
        if self.env is None:
            if self.inherit_parent_env:
                env = None
            else:
                env = {}
        elif not self.inherit_parent_env:
            env = dict(self.env)
        else:
            env = dict(os.environ)
            env.update(self.env)

        if run_dir is None:
            return env
        if env is None:
            env = dict(os.environ)
        env["TICKET_EXECUTION_RUN_DIR"] = str(run_dir)
        return env

    def _validate_command_prefix(self) -> None:
        if not self.allowed_command_prefixes:
            return
        for prefix in self.allowed_command_prefixes:
            if self.command[: len(prefix)] == prefix:
                return
        raise ValueError(
            "Ticket execution subprocess command is not in the allowed prefix list."
        )

    def _start_run_dir(self) -> Path | None:
        run_root = self.artifact_dir or self.run_dir_root
        if not run_root:
            return None
        run_dir = Path(run_root) / f"run-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _start_artifact_run(self, run_dir: Path | None, request_json: str) -> Path | None:
        if run_dir is None or not self.artifact_dir:
            return run_dir
        (run_dir / "request.json").write_text(request_json, encoding="utf-8")
        return run_dir

    def _write_artifacts(
        self,
        run_dir: Path | None,
        *,
        stdout_text: str,
        stderr_text: str,
        metadata: dict,
    ) -> None:
        if run_dir is None:
            return
        (run_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
