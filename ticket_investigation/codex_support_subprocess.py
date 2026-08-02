from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from ticket_execution.subprocess_utils import write_execution_artifacts


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
    stdout_lines: list[str] = []
    stderr_chunks: list[str] = []
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
        nonlocal final_response_text
        if process.stdout is None:
            return

        async def handle_line(line: bytes) -> None:
            nonlocal final_response_text
            text = line.decode("utf-8", errors="replace").rstrip("\r")
            if not text:
                return
            stdout_lines.append(text)
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
            while True:
                separator_index = pending.find(b"\n")
                if separator_index < 0:
                    break
                line = bytes(pending[:separator_index])
                del pending[: separator_index + 1]
                await handle_line(line)
        if pending:
            await handle_line(bytes(pending))

    async def read_stderr() -> None:
        if process.stderr is None:
            return
        while True:
            chunk = await process.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk.decode("utf-8", errors="replace"))

    io_tasks = [
        asyncio.create_task(feed_stdin()),
        asyncio.create_task(read_stdout()),
        asyncio.create_task(read_stderr()),
    ]
    io_future = asyncio.gather(*io_tasks)
    try:
        await asyncio.wait_for(io_future, timeout=timeout_seconds)
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
    except asyncio.CancelledError:
        await _terminate_streamed_subprocess(process)
        raise
    except Exception:
        await _terminate_streamed_subprocess(process)
        raise
    finally:
        for task in io_tasks:
            if not task.done():
                task.cancel()
        if not io_future.done():
            io_future.cancel()
        try:
            await io_future
        except (Exception, asyncio.CancelledError):
            pass

    stdout_text = "\n".join(stdout_lines).strip()
    stderr_text = "".join(stderr_chunks).strip()
    write_execution_artifacts(
        artifact_run_dir,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        metadata={**metadata, "returncode": process.returncode, "timed_out": False},
    )
    if process.returncode != 0:
        error_text = (
            stderr_text or "Codex support execution exited without stderr output."
        )
        raise RuntimeError(error_text[:max_error_chars])
    if not stdout_text:
        raise RuntimeError(empty_stdout_message)
    execution_output = parse_codex_support_execution_output(stdout_text)
    if len(execution_output.final_response_text) > max_output_chars:
        raise RuntimeError(oversized_stdout_message)
    return execution_output


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
