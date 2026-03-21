from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
import uuid

from ticket_investigation_codex_bundle import (
    DEFAULT_CODEX_EXEC_COMMAND,
    build_codex_ticket_execution_bundle,
)
from ticket_investigation_executor import TicketExecutionHooks
from ticket_investigation_transport import (
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

        with self._run_workspace() as run_dir:
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
                response_json_override=smoke_response_json,
            )
            process = await asyncio.create_subprocess_exec(
                *bundle.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd or str(run_dir),
                env=self._effective_env(run_dir),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(bundle.prompt_text.encode("utf-8")),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.wait()
                self._write_artifacts(
                    run_dir,
                    stdout_text="",
                    stderr_text="",
                    metadata={
                        "base_command": self.codex_command,
                        "command": bundle.command,
                        "cwd": self.cwd or str(run_dir),
                        "timed_out": True,
                    },
                )
                raise RuntimeError(
                    f"Codex ticket execution timed out after {self.timeout_seconds} seconds."
                ) from exc

            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            self._write_artifacts(
                run_dir,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                metadata={
                    "base_command": self.codex_command,
                    "command": bundle.command,
                    "cwd": self.cwd or str(run_dir),
                    "returncode": process.returncode,
                    "timed_out": False,
                },
            )

            if process.returncode != 0:
                error_text = stderr_text or "Codex execution exited without stderr output."
                raise RuntimeError(error_text[: self.max_error_chars])

            response_text = stdout_text
            if not response_text:
                raise RuntimeError("Codex execution returned empty stdout.")
            if len(response_text) > self.max_output_chars:
                raise RuntimeError("Codex execution returned too much stdout.")

            TicketExecutionTransportResult.from_json(response_text)
            return response_text

    def _effective_env(self, run_dir: Path) -> dict[str, str] | None:
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

        if env is None:
            env = dict(os.environ)
        env["TICKET_EXECUTION_RUN_DIR"] = str(run_dir)
        return env

    def _validate_command_prefix(self) -> None:
        if not self.allowed_command_prefixes:
            return
        for prefix in self.allowed_command_prefixes:
            if self.codex_command[: len(prefix)] == prefix:
                return
        raise ValueError(
            "Codex ticket execution command is not in the allowed prefix list."
        )

    def _run_workspace(self) -> _TemporaryRunDir | _PersistentRunDir:
        run_root = self.artifact_dir or self.run_dir_root
        if not run_root:
            return _TemporaryRunDir()
        run_dir = Path(run_root) / f"run-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return _PersistentRunDir(run_dir)

    def _write_artifacts(
        self,
        run_dir: Path,
        *,
        stdout_text: str,
        stderr_text: str,
        metadata: dict,
    ) -> None:
        (run_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class _PersistentRunDir:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _TemporaryRunDir:
    def __init__(self) -> None:
        self._delegate = tempfile.TemporaryDirectory(prefix="ticket-codex-run-")

    def __enter__(self) -> Path:
        return Path(self._delegate.__enter__())

    def __exit__(self, exc_type, exc, tb) -> None:
        return self._delegate.__exit__(exc_type, exc, tb)


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
