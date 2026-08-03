from __future__ import annotations

import asyncio
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import signal
from typing import Awaitable, Callable, Sequence

from ticket_execution.subprocess_utils import write_execution_artifacts

_MAX_JSONL_EVENT_BYTES = 8 * 1024 * 1024
_MAX_STDOUT_CAPTURE_CHARS = 4 * 1024 * 1024
_MAX_STDERR_CAPTURE_CHARS = 256 * 1024


@dataclass
class CodexSupportExecutionOutput:
    final_response_text: str


async def run_codex_support_json_subprocess(
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
    progress_callback: Callable[[str], Awaitable[None]] | None,
) -> CodexSupportExecutionOutput:
    creation_kwargs = (
        {"creationflags": 0} if os.name == "nt" else {"start_new_session": True}
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
    process_group_id = process.pid if os.name != "nt" else None
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdout_capture_chars = 0
    stderr_capture_chars = 0
    stdout_truncated = False
    stderr_truncated = False
    saw_stdout = False
    final_response_text: str | None = None

    async def feed_stdin() -> None:
        if process.stdin is None:
            return
        process.stdin.write(stdin_text.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()
        try:
            await process.stdin.wait_closed()
        except Exception:
            return

    async def read_stdout() -> None:
        nonlocal final_response_text, saw_stdout, stdout_capture_chars, stdout_truncated
        if process.stdout is None:
            return

        async def handle_line(line: bytes) -> None:
            nonlocal final_response_text, saw_stdout, stdout_capture_chars, stdout_truncated
            text = line.decode("utf-8", errors="replace").rstrip("\r")
            if not text:
                return
            saw_stdout = True
            capture_text = ("\n" if stdout_capture_chars else "") + text
            previous_capture_chars = stdout_capture_chars
            stdout_capture_chars = _append_bounded_text(
                stdout_capture,
                capture_text,
                captured_chars=stdout_capture_chars,
                max_chars=_MAX_STDOUT_CAPTURE_CHARS,
            )
            stdout_truncated = stdout_truncated or (
                stdout_capture_chars - previous_capture_chars < len(capture_text)
            )
            event = parse_json_event(text)
            if event is None:
                return
            progress_text = progress_update_from_codex_event(event)
            if progress_text and progress_callback is not None:
                await progress_callback(progress_text)
            response_text = final_response_from_codex_event(event)
            if response_text:
                final_response_text = response_text

        pending = bytearray()
        while True:
            chunk = await process.stdout.read(64 * 1024)
            if not chunk:
                break
            pending.extend(chunk)
            if len(pending) > _MAX_JSONL_EVENT_BYTES and pending.find(b"\n") < 0:
                raise RuntimeError(
                    "Codex support execution returned an oversized JSONL event."
                )
            while True:
                separator_index = pending.find(b"\n")
                if separator_index < 0:
                    break
                if separator_index > _MAX_JSONL_EVENT_BYTES:
                    raise RuntimeError(
                        "Codex support execution returned an oversized JSONL event."
                    )
                line = bytes(pending[:separator_index])
                del pending[: separator_index + 1]
                await handle_line(line)
        if pending:
            if len(pending) > _MAX_JSONL_EVENT_BYTES:
                raise RuntimeError(
                    "Codex support execution returned an oversized JSONL event."
                )
            await handle_line(bytes(pending))

    async def read_stderr() -> None:
        nonlocal stderr_capture_chars, stderr_truncated
        if process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            previous_capture_chars = stderr_capture_chars
            stderr_capture_chars = _append_bounded_text(
                stderr_capture,
                text,
                captured_chars=stderr_capture_chars,
                max_chars=_MAX_STDERR_CAPTURE_CHARS,
            )
            stderr_truncated = stderr_truncated or (
                stderr_capture_chars - previous_capture_chars < len(text)
            )

    execution_tasks = [
        asyncio.create_task(feed_stdin()),
        asyncio.create_task(read_stdout()),
        asyncio.create_task(read_stderr()),
        asyncio.create_task(process.wait()),
    ]
    execution_future = asyncio.gather(*execution_tasks)
    try:
        await asyncio.wait_for(execution_future, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        await _terminate_streamed_subprocess(process, process_group_id)
        write_execution_artifacts(
            artifact_run_dir,
            stdout_text=stdout_capture.getvalue().strip(),
            stderr_text=stderr_capture.getvalue().strip(),
            metadata={
                **metadata,
                "timed_out": True,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
        )
        raise RuntimeError(timeout_message) from exc
    except asyncio.CancelledError:
        await _terminate_streamed_subprocess(process, process_group_id)
        raise
    except Exception:
        await _terminate_streamed_subprocess(process, process_group_id)
        raise
    finally:
        for task in execution_tasks:
            if not task.done():
                task.cancel()
        if not execution_future.done():
            execution_future.cancel()
        try:
            await execution_future
        except (Exception, asyncio.CancelledError):
            pass

    stdout_text = stdout_capture.getvalue().strip()
    stderr_text = stderr_capture.getvalue().strip()
    write_execution_artifacts(
        artifact_run_dir,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        metadata={
            **metadata,
            "returncode": process.returncode,
            "timed_out": False,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        },
    )
    if process.returncode != 0:
        error_text = (
            stderr_text or "Codex support execution exited without stderr output."
        )
        raise RuntimeError(error_text[:max_error_chars])
    if not saw_stdout:
        raise RuntimeError(empty_stdout_message)
    if final_response_text is None:
        raise RuntimeError(
            "Codex support execution returned JSON events without a final agent message."
        )
    if len(final_response_text) > max_output_chars:
        raise RuntimeError(oversized_stdout_message)
    return CodexSupportExecutionOutput(final_response_text=final_response_text)


def parse_codex_support_execution_output(
    stdout_text: str,
) -> CodexSupportExecutionOutput:
    final_response_text: str | None = None
    for line in stdout_text.splitlines():
        event = parse_json_event(line)
        if event is None:
            continue
        response_text = final_response_from_codex_event(event)
        if response_text:
            final_response_text = response_text
    if final_response_text is None:
        raise RuntimeError(
            "Codex support execution returned JSON events without a final agent message."
        )
    return CodexSupportExecutionOutput(final_response_text=final_response_text)


def parse_json_event(text: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def final_response_from_codex_event(event: dict[str, object]) -> str | None:
    if event.get("type") != "item.completed":
        return None
    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text.strip() if isinstance(text, str) and text.strip() else None


def progress_update_from_codex_event(event: dict[str, object]) -> str | None:
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
    if (
        "mcp" in item_type
        or item.get("tool_name")
        or item.get("server")
        or item.get("name")
    ):
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
        str(item.get(key) or "") for key in ("tool_name", "name", "server", "title")
    ).lower()
    if "view_image" in raw_name or "image" in raw_name or "attachment" in raw_name:
        return "Checking screenshots"
    if "harvest" in raw_name or "report" in raw_name:
        return "Checking recent harvests"
    if "search_vaults" in raw_name or "discover" in raw_name or "vault" in raw_name:
        return "Checking vault state"
    if "repo" in raw_name or "artifact" in raw_name:
        return "Checking repo context"
    if "document" in raw_name or "doc" in raw_name:
        return "Checking Yearn docs"
    return "Checking Yearn support data"


def _append_bounded_text(
    capture: io.StringIO,
    text: str,
    *,
    captured_chars: int,
    max_chars: int,
) -> int:
    remaining = max_chars - captured_chars
    if remaining <= 0:
        return captured_chars
    captured = text[:remaining]
    capture.write(captured)
    return captured_chars + len(captured)


async def _terminate_streamed_subprocess(
    process: asyncio.subprocess.Process,
    process_group_id: int | None,
) -> None:
    try:
        if os.name == "nt":
            if process.returncode is None:
                process.kill()
        elif process_group_id is not None:
            os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        await process.wait()
