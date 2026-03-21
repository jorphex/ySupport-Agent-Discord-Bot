from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence


def validate_allowed_command_prefix(
    command: Sequence[str],
    allowed_command_prefixes: Sequence[Sequence[str]],
    *,
    error_message: str,
) -> None:
    if not allowed_command_prefixes:
        return
    for prefix in allowed_command_prefixes:
        if list(command[: len(prefix)]) == list(prefix):
            return
    raise ValueError(error_message)


def build_effective_execution_env(
    *,
    env: dict[str, str] | None,
    inherit_parent_env: bool,
    run_dir: Path,
) -> dict[str, str]:
    if env is None:
        effective_env = dict(os.environ) if inherit_parent_env else {}
    elif inherit_parent_env:
        effective_env = dict(os.environ)
        effective_env.update(env)
    else:
        effective_env = dict(env)

    effective_env["TICKET_EXECUTION_RUN_DIR"] = str(run_dir)
    return effective_env


def write_execution_artifacts(
    run_dir: Path | None,
    *,
    stdout_text: str,
    stderr_text: str,
    metadata: dict[str, Any],
) -> None:
    if run_dir is None:
        return
    (run_dir / "stdout.txt").write_text(stdout_text, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def safe_export_workspace_copy(
    workspace,
    *,
    logger_name: str,
    context: str,
) -> Path | None:
    try:
        return workspace.export_copy()
    except Exception as exc:
        logging.getLogger(logger_name).warning(
            "Failed to export %s workspace copy: %s",
            context,
            exc,
        )
        return None


async def run_bounded_subprocess(
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
    metadata: dict[str, Any],
    artifact_run_dir: Path | None,
) -> str:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin_text.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        write_execution_artifacts(
            artifact_run_dir,
            stdout_text="",
            stderr_text="",
            metadata={**metadata, "timed_out": True},
        )
        raise RuntimeError(timeout_message) from exc

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    write_execution_artifacts(
        artifact_run_dir,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        metadata={**metadata, "returncode": process.returncode, "timed_out": False},
    )

    if process.returncode != 0:
        error_text = stderr_text or "Subprocess exited without stderr output."
        raise RuntimeError(error_text[:max_error_chars])
    if not stdout_text:
        raise RuntimeError(empty_stdout_message)
    if len(stdout_text) > max_output_chars:
        raise RuntimeError(oversized_stdout_message)
    return stdout_text
