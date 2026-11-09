"""Exercise cancellation and notice ordering through real local HTTP requests.

Production controllers, urllib and worker threads run normally. Only Telegram's
destination is redirected to a controlled local server; no live messages are sent.
Run: python -B -m tests.telegram_http_e2e --artifact .cache/verification/telegram-http-e2e.json
"""

import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib import request
from unittest.mock import patch

from openai import AsyncOpenAI

from tests.optimization_e2e import eventually
from ysupport import config
from ysupport.integrations import handoff
from ysupport.bot import state
from tests.test_ticket_intake import (
    _FakeDiscordChannel,
    _FakeOriginalMessage,
    _FakePublicChannel,
    _FakeTriggerMessage,
)
from tests.ticket_flow_test_support import FakeInvestigationExecutor
from ysupport.bot.ticket_channel_lifecycle import clear_deleted_ticket_channel
from ysupport.bot.views import stop_ticket_for_manual_support
from ysupport.bot.client import TicketBot
from ysupport.execution.contracts import TicketAgentFlowOutcome


class TelegramHttpFixture:
    def __init__(self, *, held_method="editMessageText", held_text=None, fail=False,
                 summary_error=False, summary_output=None):
        self.entered, self.release, self.finished = (threading.Event() for _ in range(3))
        self.requests, self.applied = [], []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                method = self.path.rsplit("/", 1)[-1]
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append((method, payload))
                held = method == held_method and (held_text is None or payload.get("text") == held_text)
                if held:
                    fixture.entered.set()
                    released = fixture.release.wait(8)
                else:
                    released = True
                failed = not released or (held and fail) or (method == "responses" and summary_error)
                if not failed:
                    fixture.applied.append((method, payload))
                if method == "responses":
                    result = {"id": "resp_fixture", "object": "response", "created_at": 0,
                        "model": "fixture", "status": "completed", "output": [
                            {"id": "msg_fixture", "type": "message", "role": "assistant",
                             "status": "completed", "content": [{"type": "output_text",
                                 "text": summary_output or "invalid fixture JSON", "annotations": []}]}]}
                else:
                    result = {"ok": not failed, "result": [] if method == "getUpdates" else (
                        {"message_id": 9000 + len(fixture.requests), "chat": {"id": 123}}
                        if method == "sendMessage" else True)}
                response = json.dumps(result).encode()
                self.send_response(503 if failed else 200)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                try:
                    self.wfile.write(response)
                finally:
                    if held:
                        fixture.finished.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.serving = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.original_urlopen = request.urlopen

    def __enter__(self):
        self.serving.start()
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(config, "TELEGRAM_BOT_TOKEN", "fixture"))
        self.patches.enter_context(patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"))
        self.patches.enter_context(patch.object(config, "TICKET_EXECUTION_MAX_CONCURRENT_TURNS", 1))
        self.patches.enter_context(patch.dict(os.environ, {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"}))
        self.patches.enter_context(patch('ysupport.integrations.handoff.request.urlopen', new=self.urlopen))
        self.patches.enter_context(patch('ysupport.bot.ticket_channel_lifecycle.discord.TextChannel', _FakeDiscordChannel))
        return self

    def urlopen(self, req, *, timeout):
        method = req.full_url.rsplit("/", 1)[-1]
        local = request.Request(f"http://127.0.0.1:{self.server.server_port}/{method}",
            data=req.data, headers=dict(req.headers), method="POST")
        return self.original_urlopen(local, timeout=timeout)

    def __exit__(self, *_args):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.serving.join(timeout=2)
        self.patches.close()


async def verify_terminal_notice(*, status, fail, stop=False):
    channel = _FakeDiscordChannel(7601)
    notice = state.TeamHandoffNotice(telegram_chat_id="123", telegram_message_id=channel.id,
        reason="HTTP fixture", message_text="Ticket needs a team reply", status=status,
        pending_reply_text="The issue is fixed.")
    held_text = (handoff.build_pending_delivery_handoff_notice(notice.message_text)
        if status == "pending_delivery" else handoff.build_closed_handoff_notice(notice.message_text))
    archived_text = handoff.build_archived_handoff_notice(notice.message_text)
    terminal_method = "deleteMessage" if stop else "editMessageText"

    def is_terminal(method, payload):
        return (method == terminal_method and payload.get("message_id") == channel.id
            and (stop or payload.get("text") == archived_text))
    tasks = []
    bot = TicketBot(intents=__import__("discord").Intents.none())
    bot.get_channel = lambda _channel_id: None
    with TelegramHttpFixture(held_text=held_text, fail=fail) as http:
        try:
            state.remember_team_handoff_notice(channel.id, notice)
            bot.telegram_handoffs._wake_delivery_worker()
            await eventually(http.entered.is_set)
            deletion = asyncio.create_task(stop_ticket_for_manual_support(channel.id)
                if stop else clear_deleted_ticket_channel(channel))
            tasks.append(deletion)
            await eventually(lambda: channel.id not in state.team_handoff_notice_by_channel)
            # Cancel the persistent worker too, exercising repeated child cancellation.
            closing = asyncio.create_task(bot.telegram_handoffs.close())
            tasks.append(closing)
            assert await asyncio.wait_for(handoff.edit_handoff_notice(
                chat_id="123", message_id=7602, message_text="Independent update"), timeout=1)
            queued = asyncio.create_task(handoff.edit_handoff_notice(
                chat_id="123", message_id=channel.id, message_text="Canceled queued update"))
            tasks.append(queued)
            await asyncio.sleep(.05)
            queued.cancel()
            queued_result, = await asyncio.gather(queued, return_exceptions=True)
            terminal_overtook = any(is_terminal(method, payload) for method, payload in http.requests)
            http.release.set()
            await asyncio.wait_for(asyncio.gather(deletion, closing), timeout=3)
            await eventually(http.finished.is_set)
            target_operations = [(method, payload) for method, payload in http.applied
                if payload.get("message_id") == channel.id]
            failures = []
            if terminal_overtook or not target_operations or not is_terminal(*target_operations[-1]):
                failures.append("Terminal notice operation overtook the in-flight edit")
            if not isinstance(queued_result, asyncio.CancelledError) or any(
                payload.get("text") == "Canceled queued update" for _, payload in http.requests
            ):
                failures.append("Canceled queued edit reached Telegram")
            state_path = state._TICKET_STATE_DIR / f"{channel.id}.json"
            if stop:
                assert json.loads(state_path.read_text())["stopped"] is True
            else:
                assert not state_path.exists()
            assert channel.id not in state.handoff_delivery_tasks
            assert not channel.sent_messages
            assert not any("Update delivery failed" in payload.get("text", "") for _, payload in http.requests)
            assert not failures, "; ".join(failures)
            return {"terminal_notice_operation_last": True, "independent_notice_progressed": True,
                "queued_cancellation_sent_nothing": True, "worker_and_child_reaped": True,
                "ticket_state_correct": True, "http_application_order": [
                    {"method": method, "text": payload.get("text")} for method, payload in target_operations]}
        finally:
            http.release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await bot.close()
            state.clear_ticket_channel_state(channel.id, delete_persisted=True)


async def verify_polling_cancellation():
    with TelegramHttpFixture(held_method="getUpdates") as http:
        task = asyncio.create_task(handoff.fetch_telegram_updates())
        try:
            await eventually(http.entered.is_set)
            task.cancel()
            result, = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)
            assert isinstance(result, asyncio.CancelledError)
            assert not http.release.is_set()
            assert await asyncio.wait_for(handoff.edit_handoff_notice(
                chat_id="123", message_id=7602, message_text="Independent update"), timeout=1)
            return {"polling_cancellation_prompt": True, "edits_independent_of_polling": True}
        finally:
            http.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await eventually(http.finished.is_set)


async def verify_handoff_presentation(*, public=True, metadata=True, summary_failure=None):
    channel_id, author_id, original_id = 7681, 7682, 7683
    question = "Is this vault retired? The keeper seems inactive."
    support_reply = (
        "The vault is not marked retired. Convex last reported Oct 19 and its "
        "harvest trigger is true. Why the keeper has not acted remains unknown."
    )
    action = "Keeper/strategist review is needed; decide whether to harvest."
    summary = "Convex last reported Oct 19 and its trigger is true; the keeper cause remains unknown."
    original = _FakeOriginalMessage(author_id, question)
    channel = _FakePublicChannel(channel_id, original) if public else _FakeDiscordChannel(channel_id)
    channel.category = SimpleNamespace(id=1)
    channel.name = 'vaults<&"' if public else 'ticket-1633'
    channel.guild = SimpleNamespace(id=734804446353031319, name="SERVER NAME MUST NOT APPEAR")
    if not metadata:
        channel.name = channel.guild = None
    bot = TicketBot(intents=__import__("discord").Intents.none())
    bot.get_channel = lambda _channel_id: channel
    executor = FakeInvestigationExecutor(result=SimpleNamespace(
        flow_outcome=TicketAgentFlowOutcome(
            raw_final_reply=support_reply,
            conversation_history=[{"role": "user", "content": question},
                                  {"role": "assistant", "content": support_reply}],
            completed_agent_key=None, requires_human_handoff=True, handoff_reason=action),
        updated_job=state.TicketInvestigationJob(channel_id=channel_id)))
    bot.investigation_executor = executor
    with TelegramHttpFixture(held_method=None, summary_error=summary_failure == "http",
            summary_output=json.dumps({"summary": summary}) if summary_failure is None else None) as http:
        async with AsyncOpenAI(api_key="fixture",
                base_url=f"http://127.0.0.1:{http.server.server_port}/v1", max_retries=0) as summary_client:
            with (
                patch.object(config, "OPENAI_API_KEY", "fixture"),
                patch.dict(os.environ, {"YSUPPORT_ALLOW_TEST_LLM": "1"}),
                patch.object(handoff, "_handoff_summary_async_client", summary_client),
                patch('ysupport.bot.client._outer_support_boundary_result',
                      return_value={"classification": "yearn_support", "tripwire_triggered": False}),
                patch('ysupport.bot.client.discord.TextChannel', _FakeDiscordChannel),
                patch.object(config, "PUBLIC_TRIGGER_USER_IDS", {"42"}),
                patch.object(config, "TRIGGER_CONTEXT_MAP", {"y": "yearn"}),
            ):
                try:
                    if public:
                        trigger = _FakeTriggerMessage(channel, reference_message_id=original_id)
                        trigger.author = SimpleNamespace(id=42, name="fixture", bot=False)
                        trigger.content = "y"
                        await bot.on_message(trigger)
                        assert trigger.deleted
                    else:
                        state.pending_messages[channel_id] = question
                        state.remember_ticket_owner_user_id(channel_id, author_id)
                        await bot.process_ticket_message(channel_id,
                            state.BotRunContext(channel_id=channel_id, project_context="yearn"),
                            skip_debounce=True)
                    requests = [payload for method, payload in http.requests if method == "responses"]
                    assert len(requests) == 1
                    context = json.loads(requests[0]["input"].split("Context JSON:\n", 1)[1])
                    assert context["support_reply"] == support_reply, "Completed findings missing from summary input"
                    assert context["latest_text"] == question and context["reason"] == action
                    sends = [payload for method, payload in http.requests if method == "sendMessage"]
                    assert len(sends) == 1
                    notice = sends[0]["text"]
                    assert "SERVER NAME" not in notice and "Ticket-7681" not in notice
                    assert "<b>Action</b>: " + action in notice
                    assert "<b>Issue</b>: " + (support_reply if summary_failure else summary) in notice
                    assert any("I've notified" in text for text in channel.sent_messages)
                    if metadata:
                        target = f"https://discord.com/channels/{channel.guild.id}/{channel.id}"
                        if public:
                            target += f"/{original_id}"
                            assert f"/{trigger.id}\"" not in notice
                        assert f'href="{target}"' in notice
                        assert ('Public support · #vaults&lt;&amp;&quot;' if public else '#ticket-1633') in notice
                    else:
                        assert 'href=' not in notice and "Public support · Channel 7681" in notice
                    if public:
                        assert "Respond in Discord. Telegram replies to this alert are ignored." in notice
                        assert "reply_markup" not in sends[0]
                        assert author_id not in state.public_conversations
                        assert not (state._PUBLIC_STATE_DIR / f"{author_id}.json").exists()
                        assert channel_id not in state.team_handoff_notice_by_channel
                        before = len(http.requests)
                        notice_id = 9000 + next(index for index, (method, _) in
                            enumerate(http.requests, start=1) if method == "sendMessage")
                        await bot.telegram_handoffs._handle_telegram_handoff_update({"message": {
                            "message_id": 9999, "chat": {"id": 123}, "text": "Please update the user.",
                            "reply_to_message": {"message_id": notice_id}}})
                        assert len(http.requests) == before and len(executor.calls) == 1
                    else:
                        assert "Reply to this message" in notice
                        assert sends[0]["reply_markup"]["inline_keyboard"][0][0]["callback_data"] == "dismiss_handoff"
                        saved = state.team_handoff_notice_by_channel[channel_id]
                        assert saved.message_text == notice and saved.status == "open"
                        await bot.telegram_handoffs._handle_telegram_handoff_update({"callback_query": {
                            "id": "dismiss-fixture", "data": "dismiss_handoff", "message": {
                                "message_id": saved.telegram_message_id, "chat": {"id": 123}}}})
                        assert channel_id not in state.team_handoff_notice_by_channel
                        assert channel_id in state.stopped_channels
                        assert any(method == "deleteMessage" for method, _ in http.requests)
                    return {"notice": notice, "findings_reached_summary_model": True,
                        "summary_fallback": summary_failure, "source_metadata_present": metadata,
                        "public_reply_ignored" if public else "ticket_dismissal_preserved": True}
                finally:
                    await bot.close()
                    state.clear_public_conversation(author_id)
                    state.clear_ticket_channel_state(channel_id, delete_persisted=True)


async def main(artifact):
    report = {"ok": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "scenarios": {}}
    for status, fail, stop in (("pending_delivery", False, False), ("pending_delivery", True, False),
                              ("delivered_pending_close", False, False), ("pending_delivery", False, True)):
        name = status + ("_http_error" if fail else "_stop" if stop else "")
        try:
            report["scenarios"][name] = {"ok": True, **await verify_terminal_notice(status=status, fail=fail, stop=stop)}
        except Exception as exc:
            report["scenarios"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        report["scenarios"]["polling"] = {"ok": True, **await verify_polling_cancellation()}
    except Exception as exc:
        report["scenarios"]["polling"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    for name, options in (
        ("public_notice", {}),
        ("public_summary_http_error", {"summary_failure": "http"}),
        ("public_missing_metadata_and_invalid_summary", {"metadata": False, "summary_failure": "invalid"}),
        ("ticket_notice", {"public": False}),
    ):
        try:
            report["scenarios"][name] = {"ok": True, **await verify_handoff_presentation(**options)}
        except Exception as exc:
            report["scenarios"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["ok"] = all(item["ok"] for item in report["scenarios"].values())
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        'ysupport/integrations/handoff.py', 'ysupport/bot/telegram_handoff_controller.py',
        'ysupport/bot/discord_support_runtime.py', 'ysupport/bot/client.py',
        'ysupport/execution/ysupport_codex_instructions.md', 'ysupport/bot/ticket_channel_lifecycle.py',
        'ysupport/bot/state.py', "tests/telegram_http_e2e.py")}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    artifact.chmod(0o600)
    print(json.dumps(report["scenarios"], indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/telegram-http-e2e.json"))
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
