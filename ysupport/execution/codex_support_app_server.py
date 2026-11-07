"""One owned Codex app-server process, thread and interactive support turn."""

from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic

from ysupport.execution.codex_support_contract import CODEX_SUPPORT_RESULT_SCHEMA, SupportTurnRequest
from ysupport.execution.subprocess_utils import (
    create_isolated_subprocess,
    write_execution_artifacts,
)
from ysupport.execution.codex_support_attachments import (
    image_attachment_paths,
    prepare_support_request_attachments,
)
from ysupport.execution.codex_support_subprocess import (
    CodexSupportExecutionOutput,
    _terminate_streamed_subprocess,
    _execution_metrics,
    _log_execution_metrics,
    progress_update_from_codex_event,
)
from ysupport.execution.executor import TicketExecutionNonFallbackError

_MAX_EVENT_BYTES = 8 * 1024 * 1024
_MAX_CAPTURE_CHARS = 4 * 1024 * 1024


class AppServerRequestError(RuntimeError):
    pass


async def run_codex_support_app_server(
    *,
    command,
    prompt_text,
    support_request,
    resume_session_id,
    model,
    reasoning_effort,
    cwd,
    env,
    timeout_seconds,
    max_output_chars,
    run_dir,
    hooks,
):
    inbox = hooks.steering
    process, group_id = await create_isolated_subprocess(
        command=command, cwd=cwd, env=env
    )
    pending = {}
    methods = {}
    sequence = 0
    stdout, stderr, receipts = [], [], []
    captured = 0
    final_text = None
    completed = asyncio.Event()
    thread_id = None
    turn_id = None
    turn_status = None
    started = monotonic()
    first_item_ms = None
    usage = None
    outcome = "error"
    tasks = []
    progress_queue = asyncio.Queue(maxsize=16)

    async def send(message):
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()

    async def request(method, params, *, timeout=None):
        nonlocal sequence
        sequence += 1
        ident = sequence
        future = asyncio.get_running_loop().create_future()
        pending[ident] = future
        methods[ident] = method
        try:
            await send({"id": ident, "method": method, "params": params})
            return await asyncio.wait_for(future, timeout) if timeout else await future
        finally:
            pending.pop(ident, None)
            methods.pop(ident, None)

    def item_event(item):
        nonlocal final_text
        if item.get("type") == "userMessage":
            for entry in inbox.inputs:
                if entry.message_id == item.get("clientId") and not entry.observed:
                    entry.observed = True
                    consumed = inbox.consumed
                    if consumed and inbox.on_observed is not None:
                        inbox.on_observed(consumed[-1])
                    # A draft published before this input cannot answer it.
                    final_text = None
                    receipts.append(
                        {"message_id": entry.message_id, "event": "observed"}
                    )
        elif item.get("type") == "agentMessage" and item.get("text"):
            phase = item.get("phase")
            if phase in (None, "final_answer"):
                final_text = item["text"]

    async def read_stdout():
        nonlocal captured, turn_status, thread_id, turn_id, first_item_ms, usage
        buffer = bytearray()
        while chunk := await process.stdout.read(65536):
            buffer.extend(chunk)
            while b"\n" in buffer:
                end = buffer.index(b"\n")
                if end > _MAX_EVENT_BYTES:
                    raise RuntimeError("Codex app-server event exceeds the size limit.")
                raw = bytes(buffer[:end])
                del buffer[: end + 1]
                event = json.loads(raw)
                if captured < _MAX_CAPTURE_CHARS:
                    text = raw.decode(errors="replace")
                    stdout.append(text[: _MAX_CAPTURE_CHARS - captured])
                    captured += len(stdout[-1])
                if "method" not in event:
                    future = pending.get(event.get("id"))
                    if future is not None and not future.done():
                        if "error" in event:
                            future.set_exception(
                                AppServerRequestError(str(event["error"]))
                            )
                        else:
                            result = event.get("result", {})
                            method = methods.get(event.get("id"))
                            if method in ("thread/start", "thread/resume"):
                                thread_id = result["thread"]["id"]
                            elif method == "turn/start":
                                turn_id = result["turn"]["id"]
                            future.set_result(result)
                    continue
                if "id" in event:
                    # This read-only service cannot answer approvals or interactive UI requests.
                    await send(
                        {
                            "id": event["id"],
                            "error": {
                                "code": -32601,
                                "message": "Interactive requests are disabled for support.",
                            },
                        }
                    )
                    continue
                method, params = event["method"], event.get("params", {})
                if params.get("threadId") != thread_id:
                    continue
                if (
                    method in ("item/started", "item/completed")
                    and params.get("turnId") == turn_id
                ):
                    if first_item_ms is None:
                        first_item_ms = round((monotonic() - started) * 1000)
                    item = params.get("item", {})
                    item_event(item)
                    if method == "item/started":
                        native_type = item.get("type")
                        normalized = {
                            **item,
                            "type": {
                                "commandExecution": "command_execution",
                                "mcpToolCall": "mcp_tool_call",
                            }.get(native_type, native_type),
                            "tool_name": item.get("tool"),
                        }
                        progress_text = progress_update_from_codex_event(
                            {"type": "item.started", "item": normalized}
                        )
                        if progress_text and not progress_queue.full():
                            progress_queue.put_nowait(progress_text)
                elif (
                    method == "thread/tokenUsage/updated"
                    and params.get("turnId") == turn_id
                ):
                    native_usage = params.get("tokenUsage", {}).get("total", {})
                    usage = {
                        key: native_usage.get(field)
                        for key, field in (
                            ("input_tokens", "inputTokens"),
                            ("cached_input_tokens", "cachedInputTokens"),
                            ("cache_write_input_tokens", "cacheWriteInputTokens"),
                            ("output_tokens", "outputTokens"),
                            ("reasoning_output_tokens", "reasoningOutputTokens"),
                        )
                    }
                elif (
                    method == "turn/completed"
                    and params.get("turn", {}).get("id") == turn_id
                ):
                    turn_status = params["turn"]["status"]
                    completed.set()
            if len(buffer) > _MAX_EVENT_BYTES:
                raise RuntimeError("Codex app-server event exceeds the size limit.")
        raise RuntimeError("Codex app-server closed before the support turn completed.")

    async def read_stderr():
        remaining = 256 * 1024
        while chunk := await process.stderr.read(4096):
            text = chunk.decode(errors="replace")[:remaining]
            if text:
                stderr.append(text)
                remaining -= len(text)

    async def progress():
        while True:
            text = await progress_queue.get()
            if hooks.send_progress_update is not None:
                await hooks.send_progress_update(text)

    async def steer():
        index = 0
        while not completed.is_set():
            inbox.changed.clear()
            while index < len(inbox.inputs) and not completed.is_set():
                entry = inbox.inputs[index]
                index += 1
                if not await inbox.allow_input(entry):
                    receipts.append(
                        {"message_id": entry.message_id, "event": "boundary_deferred"}
                    )
                    return
                update = SupportTurnRequest(
                    **{
                        **support_request.__dict__,
                        "attachments": list(entry.attachments),
                        "current_user_message": entry.text,
                    }
                )
                batch_dir = run_dir / "steering" / str(index)
                await prepare_support_request_attachments(update, run_dir=batch_dir)
                entry.prepared_attachments = update.attachments
                if completed.is_set():
                    return
                inputs = [
                    {
                        "type": "text",
                        "text_elements": [],
                        "text": json.dumps(
                            {
                                "current_turn_source": "user",
                                "discord_message_id": entry.message_id,
                                "current_user_message": entry.text,
                                "attachments": update.attachments,
                                "evidence": entry.evidence,
                                "system_hints": entry.system_hints,
                                "instruction": "Ticket owner follow-up. Update the current investigation and answer using this new input. Preserve relevant earlier context; explicit corrections supersede earlier assumptions.",
                            }
                        ),
                    }
                ]
                inputs.extend(
                    {"type": "localImage", "path": str(path)}
                    for path in image_attachment_paths(update)
                )
                entry.submitted = True
                receipts.append(
                    {
                        "message_id": entry.message_id,
                        "event": "submitted",
                        "input": inputs,
                    }
                )
                try:
                    result = await request(
                        "turn/steer",
                        {
                            "threadId": thread_id,
                            "expectedTurnId": turn_id,
                            "clientUserMessageId": entry.message_id,
                            "input": inputs,
                        },
                        timeout=min(5.0, timeout_seconds),
                    )
                except (AppServerRequestError, asyncio.TimeoutError) as exc:
                    receipts.append(
                        {
                            "message_id": entry.message_id,
                            "event": "deferred",
                            "reason": type(exc).__name__,
                        }
                    )
                    return
                if result.get("turnId") != turn_id:
                    raise RuntimeError(
                        "Codex app-server acknowledged a different active turn."
                    )
                receipts.append(
                    {"message_id": entry.message_id, "event": "acknowledged"}
                )
            await inbox.changed.wait()

    async def execute():
        nonlocal thread_id, turn_id
        await request(
            "initialize",
            {
                "clientInfo": {"name": "ysupport", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await send({"method": "initialized"})
        params = {
            "cwd": str(run_dir),
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        }
        if model:
            params["model"] = model
        if resume_session_id:
            params.update(threadId=resume_session_id, excludeTurns=True)
        response = await request(
            "thread/resume" if resume_session_id else "thread/start", params
        )
        thread_id = response["thread"]["id"]
        # Keep the existing session/audit reader contract.
        stdout.append(json.dumps({"type": "thread.started", "thread_id": thread_id}))
        inputs = [{"type": "text", "text": prompt_text, "text_elements": []}]
        inputs.extend(
            {"type": "localImage", "path": str(path)}
            for path in image_attachment_paths(support_request)
        )
        params = {
            "threadId": thread_id,
            "input": inputs,
            "outputSchema": CODEX_SUPPORT_RESULT_SCHEMA,
        }
        if model:
            params["model"] = model
        if reasoning_effort:
            params["effort"] = reasoning_effort
        response = await request("turn/start", params)
        turn_id = response["turn"]["id"]
        steering_task = asyncio.create_task(steer())
        tasks.append(steering_task)
        complete_task = asyncio.create_task(completed.wait())
        tasks.append(complete_task)
        while not complete_task.done():
            done, _ = await asyncio.wait(
                [complete_task, steering_task], return_when=asyncio.FIRST_COMPLETED
            )
            if steering_task in done:
                steering_task.result()
                await complete_task
        if turn_status != "completed":
            raise RuntimeError(f"Codex support turn ended with status {turn_status}.")
        if not final_text or len(final_text) > max_output_chars:
            raise RuntimeError(
                "Codex app-server returned empty or oversized support output."
            )
        return CodexSupportExecutionOutput(final_response_text=final_text)

    try:
        reader = asyncio.create_task(read_stdout())
        tasks.extend(
            [
                reader,
                asyncio.create_task(read_stderr()),
                asyncio.create_task(progress()),
            ]
        )
        execution = asyncio.create_task(execute())
        tasks.append(execution)
        done, _ = await asyncio.wait_for(
            asyncio.wait([reader, execution], return_when=asyncio.FIRST_COMPLETED),
            timeout=timeout_seconds,
        )
        if reader in done:
            reader.result()
        result = execution.result()
        outcome = "completed"
        return result
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception as exc:
        # A wrapper must not retry an original request after newer input was admitted.
        raise TicketExecutionNonFallbackError(
            f"Codex interactive support turn failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        inbox.closed = True

        async def close():
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            async def drain(stream):
                while await stream.read(65536):
                    pass

            drains = [
                asyncio.create_task(drain(stream))
                for stream in (process.stdout, process.stderr)
            ]
            process.stdin.close()
            await _terminate_streamed_subprocess(process, group_id)
            try:
                await process.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
            await asyncio.gather(*drains)

        closing = asyncio.create_task(close())
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        metrics = _execution_metrics(
            usage=usage,
            first_item_ms=first_item_ms,
            total_ms=round((monotonic() - started) * 1000),
        )
        _log_execution_metrics(outcome=outcome, metrics=metrics)
        try:
            write_execution_artifacts(
                run_dir,
                stdout_text="\n".join(stdout),
                stderr_text="".join(stderr),
                metadata={
                    "transport": "app-server",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    **metrics,
                    "steering": receipts,
                },
            )
        except OSError:
            if outcome != "cancelled" and not cancelled:
                raise
            logging.warning(
                "Could not save cancelled Codex support artifacts.", exc_info=True
            )
        finally:
            if cancelled:
                raise asyncio.CancelledError()
