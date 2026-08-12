from __future__ import annotations

import asyncio
import codecs
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
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


async def create_isolated_subprocess(
    *,
    command: Sequence[str],
    cwd: str | None,
    env: dict[str, str],
) -> tuple[asyncio.subprocess.Process, int | None]:
    creation_task = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
            **_subprocess_creation_kwargs(),
        )
    )
    try:
        process = await asyncio.shield(creation_task)
    except asyncio.CancelledError:
        process = await creation_task
        process_group_id = _capture_process_group_id(process)
        await _terminate_subprocess(process, process_group_id)
        raise
    return process, _capture_process_group_id(process)


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
    process, process_group_id = await create_isolated_subprocess(
        command=command,
        cwd=cwd,
        env=env,
    )
    stdout_capture = _BoundedStreamCapture(max_output_chars, fail_on_limit=True)
    stderr_capture = _BoundedStreamCapture(max_error_chars, fail_on_limit=False)
    tasks = {
        asyncio.create_task(_write_stdin(process, stdin_text)),
        asyncio.create_task(stdout_capture.read(process.stdout)),
        asyncio.create_task(stderr_capture.read(process.stderr)),
        asyncio.create_task(process.wait()),
    }
    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            task.result()
        if pending:
            raise _SubprocessTimedOut()
    except _SubprocessTimedOut as exc:
        await _terminate_subprocess(process, process_group_id)
        await _finish_tasks(tasks)
        write_execution_artifacts(
            artifact_run_dir,
            stdout_text=stdout_capture.text,
            stderr_text=stderr_capture.text,
            metadata={**metadata, "timed_out": True},
        )
        raise RuntimeError(timeout_message) from exc
    except _StreamLimitExceeded as exc:
        await _terminate_subprocess(process, process_group_id)
        tasks.add(asyncio.create_task(_drain_stream(process.stdout)))
        await _finish_tasks(tasks)
        write_execution_artifacts(
            artifact_run_dir,
            stdout_text=stdout_capture.text,
            stderr_text=stderr_capture.text,
            metadata={
                **metadata,
                "returncode": process.returncode,
                "timed_out": False,
                "stdout_limit_exceeded": True,
            },
        )
        raise RuntimeError(oversized_stdout_message) from exc
    except BaseException:
        await _terminate_subprocess(process, process_group_id)
        await _finish_tasks(tasks)
        raise

    stdout_text = stdout_capture.text.strip()
    stderr_text = stderr_capture.text.strip()
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
    return stdout_text


class _StreamLimitExceeded(Exception):
    pass


class _SubprocessTimedOut(Exception):
    pass


class _BoundedStreamCapture:
    def __init__(self, max_chars: int, *, fail_on_limit: bool) -> None:
        self.max_chars = max_chars
        self.fail_on_limit = fail_on_limit
        self._parts: list[str] = []
        self._captured_chars = 0

    @property
    def text(self) -> str:
        return "".join(self._parts)

    async def read(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := await stream.read(64 * 1024):
            self._capture(decoder.decode(chunk))
        self._capture(decoder.decode(b"", final=True))

    def _capture(self, text: str) -> None:
        if not text:
            return
        remaining = self.max_chars - self._captured_chars
        if remaining > 0:
            retained = text[:remaining]
            self._parts.append(retained)
            self._captured_chars += len(retained)
        if len(text) > remaining and self.fail_on_limit:
            raise _StreamLimitExceeded()


async def _write_stdin(
    process: asyncio.subprocess.Process,
    stdin_text: str,
) -> None:
    if process.stdin is None:
        return
    try:
        process.stdin.write(stdin_text.encode("utf-8"))
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        process.stdin.close()


async def _drain_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    while await stream.read(64 * 1024):
        pass


async def _finish_tasks(tasks: set[asyncio.Task[Any]]) -> None:
    _done, pending = await asyncio.wait(tasks, timeout=1)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _subprocess_creation_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }
    return {
        "start_new_session": True,
    }


def _capture_process_group_id(process: asyncio.subprocess.Process) -> int | None:
    if os.name == "nt":
        return None
    try:
        return os.getpgid(process.pid)
    except ProcessLookupError:
        return process.pid


async def _terminate_subprocess(
    process: asyncio.subprocess.Process,
    process_group_id: int | None,
) -> None:
    try:
        if os.name == "nt":
            if process.returncode is None:
                process.kill()
        else:
            os.killpg(process_group_id or process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        if process.returncode is None:
            await process.wait()
