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
import tempfile
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
from ysupport.execution.codex_support_contract import SupportTurnRequest
from ysupport.execution.codex_support_sessions import CodexSupportSessionManager
from ysupport.execution.contracts import TicketAgentFlowOutcome
from ysupport.execution.transport import TicketExecutionTransportRequest


class TelegramHttpFixture:
    def __init__(self, *, held_method="editMessageText", held_text=None, fail=False,
                 summary_error=False, summary_output=None):
        self.entered, self.release, self.finished = (threading.Event() for _ in range(3))
        self.requests, self.applied = [], []
        self.fail_methods = set()
        self.enforce_idempotence = False
        self.deleted_messages = set()
        self.updates = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                method = self.path.rsplit("/", 1)[-1]
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.requests.append((method, payload))
                request_index = len(fixture.requests)
                held = method == held_method and (held_text is None or payload.get("text") == held_text)
                if held:
                    fixture.entered.set()
                    released = fixture.release.wait(8)
                else:
                    released = True
                already_applied = fixture.enforce_idempotence and (method, payload) in fixture.applied
                deleted = (payload.get('chat_id'), payload.get('message_id')) in fixture.deleted_messages
                if fixture.enforce_idempotence and method in {'editMessageText', 'deleteMessage'} and (already_applied or deleted):
                    result = {'ok': False, 'error_code': 400, 'description': 'Bad Request: ' + (
                        ('message to edit not found' if deleted else 'message is not modified')
                        if method == 'editMessageText' else 'message to delete not found')}
                    response = json.dumps(result).encode()
                    self.send_response(400)
                    self.send_header('Content-Length', str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                failed = not released or (held and fail) or method in fixture.fail_methods or (method == "responses" and summary_error)
                if not failed:
                    fixture.applied.append((method, payload))
                    if method == 'deleteMessage':
                        fixture.deleted_messages.add((payload.get('chat_id'), payload.get('message_id')))
                if method == "responses":
                    result = {"id": "resp_fixture", "object": "response", "created_at": 0,
                        "model": "fixture", "status": "completed", "output": [
                            {"id": "msg_fixture", "type": "message", "role": "assistant",
                             "status": "completed", "content": [{"type": "output_text",
                                 "text": summary_output or "invalid fixture JSON", "annotations": []}]}]}
                else:
                    result = {"ok": not failed, "result": [u for u in fixture.updates
                        if u.get('update_id', 0) >= (payload.get('offset') or 0)] if method == "getUpdates" else (
                        {"message_id": 9000 + request_index, "chat": {"id": 123}}
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
            assert not any("Delivery needs review" in payload.get("text", "") for _, payload in http.requests)
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


async def verify_private_ticket_update(mode):
    import discord
    from ysupport.bot.discord_support_runtime import _record_waiting_for_team_followup

    channel = _FakeDiscordChannel(7610)
    channel.category = SimpleNamespace(id=1)
    owner = SimpleNamespace(id=7611, bot=False)
    bot_user = SimpleNamespace(id=7612, bot=True)
    staff = SimpleNamespace(id=7613, bot=False, guild_permissions=SimpleNamespace(administrator=True))
    messages = [SimpleNamespace(id=7620, author=owner, content="Check the old hash.", attachments=[])]
    entered, release = asyncio.Event(), asyncio.Event()
    image_modes = {"late_image", "late_image_summon", "late_image_reference", "late_image_text_reference", "staff_ack", "first_staff_ack",
                   "image_refresh_timeout", "image_refresh_deleted", "text_only_summon"}
    interrupted = {"restart", "ambiguous", "ambiguous_notice_retry", "recovery_gateway_ahead", "bounded_recovery",
                   "reconcile_timeout"}
    if mode in {"bounded_history", "bounded_recovery", "history_collision", "late_image_summon", "second_handoff_edited"}:
        messages[:0] = [SimpleNamespace(id=7500+i, author=owner, content=f"Earlier detail {i}", attachments=[])
                        for i in range(40)]
    if mode in {"staff_owner", "staff_owner_summon"}:
        owner.guild_permissions = staff.guild_permissions
    if mode == "older_recorded_edited":
        messages[0].content = "Later message 0"
    if mode == "assistant_revision":
        messages.append(SimpleNamespace(id=7621, author=bot_user, content="Approval complete.", attachments=[]))
    if mode == "native_context":
        messages.extend(SimpleNamespace(id=7621+i, author=staff, content=f"Staff note {i}", attachments=[])
                        for i in range(13))

    def image(attachment_id, name):
        return SimpleNamespace(id=attachment_id, filename=f"{name}.png", size=5, content_type="image/png",
                               url=f"https://cdn.discordapp.com/attachments/1/{attachment_id}/{name}.png")

    reconciliation_reads = 0

    async def history(*, limit, after=None, oldest_first=False):
        nonlocal reconciliation_reads
        if mode == "reconcile_timeout" and after is not None:
            reconciliation_reads += 1
            if reconciliation_reads == 1:
                raise asyncio.TimeoutError("First reconciliation read timed out")
        rows = [message for message in messages if after is None or message.id > after.id]
        for message in (rows if oldest_first else reversed(rows)):
            if limit <= 0:
                break
            yield message
            limit -= 1

    async def send(content, **kwargs):
        await _FakeDiscordChannel.send(channel, content, **kwargs)
        message = SimpleNamespace(id=(messages[-1].id if messages else 7620) + 1, author=bot_user,
                                  content=content, attachments=[], nonce=kwargs.get("nonce"),
                                  reference=kwargs.get("reference"))
        messages.append(message)
        if mode == "close_notice_retry":
            http.fail_methods.add("editMessageText")
        if mode == "notice_deleted":
            http.deleted_messages.add(("123", channel.id))
        if mode == "late_gateway_duplicate":
            _record_waiting_for_team_followup(channel.id, messages[1])
        if mode in image_modes and len(channel.sent_messages) == 1:
            followup = SimpleNamespace(id=message.id + 1, author=owner, content="See the new screenshot.", attachments=[
                SimpleNamespace(id=92, filename="late.png", size=5, content_type="image/png",
                                url="https://cdn.discordapp.com/attachments/1/2/late.png")])
            messages.append(followup)
            _record_waiting_for_team_followup(channel.id, followup)
        if mode in interrupted and len(channel.sent_messages) == 1:
            raise ConnectionError("Discord accepted the message before the connection ended")
        return message

    async def fetch_message(message_id):
        if mode == "summon_reference_timeout":
            raise asyncio.TimeoutError("Discord request timed out")
        if mode == "older_recorded_deleted":
            raise discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "Unknown message")
        if mode == "text_only_summon":
            raise TimeoutError("An unrelated saved screenshot is temporarily unavailable")
        if mode == "image_refresh_timeout" and executor.calls == 1 and not channel.sent_messages[1:]:
            raise TimeoutError("Controlled Discord screenshot timeout")
        if mode == "image_refresh_deleted":
            return SimpleNamespace(attachments=[])
        return next(message for message in messages if message.id == message_id)

    channel.history, channel.send, channel.fetch_message = history, send, fetch_message

    class Executor:
        calls = 0

        async def execute_turn(self, request, hooks=None):
            self.calls += 1
            self.last_request = request
            if mode == "stale_session":
                assert sessions.load(f"ticket:{channel.id}") is None, "Draft reused a pre-handoff native session"
            context = json.dumps(request.input_list)
            if self.calls == 1 and mode == "freshness":
                entered.set()
                await release.wait()
            if mode == "gateway_ahead" and self.calls == 1:
                _record_waiting_for_team_followup(channel.id, SimpleNamespace(
                    id=7623, content="The new hash replaces the old one.", attachments=[]))
            if mode in {"keeps_changing", "review_notice_retry"}:
                messages[0].content = f"The transaction changed again ({self.calls})."
                if mode == "review_notice_retry" and self.calls == 2:
                    http.fail_methods.add("editMessageText")
            if mode == "older_recorded_edited" and self.calls == 1:
                messages[1].content = "Corrected again after a new wallet check."
            if mode == "native_context":
                support = SupportTurnRequest.from_ticket_execution_request(
                    TicketExecutionTransportRequest.from_turn_request(request))
                answer = ("I can see the original question." if any(
                    row["role"] == "user" for row in support.recent_transcript
                ) else "The original question is missing.")
            elif mode == "assistant_revision":
                answer = "Approval is paused." if "Approval paused." in context else "Approval is complete."
                messages[1].content = "Approval paused."
            elif "corrected hash" in context:
                assert "Keeper is checking the new transaction" in context
                assert [item["filename"] for item in request.attachments] == ["current.png"]
                answer = "I checked the corrected hash."
            else:
                answer = "I checked the old hash."
            return SimpleNamespace(flow_outcome=TicketAgentFlowOutcome(
                raw_final_reply=answer, conversation_history=request.current_history + [
                    {"role": "assistant", "content": answer}], completed_agent_key=None,
                requires_human_handoff=False), updated_job=request.investigation_job)

    executor = Executor()

    def start_bot():
        bot = TicketBot(intents=discord.Intents.none())
        bot._connection.user = bot_user
        bot.get_channel = lambda _channel_id: channel
        bot.investigation_executor = executor
        return bot

    with (
        TelegramHttpFixture(held_method=None) as http,
        tempfile.TemporaryDirectory(prefix="ysupport-handoff-sessions-") as session_dir,
        patch.object(config, "TICKET_EXECUTION_CODEX_SESSION_DIR", session_dir),
    ):
        sessions = CodexSupportSessionManager(session_dir, max_age_hours=1)
        if mode == "stale_session":
            sessions.record_success(conversation_key=f"ticket:{channel.id}",
                session_id="550e8400-e29b-41d4-a716-446655440000", human_handoff_active=True)
        http.enforce_idempotence = mode == "notice_deleted"
        bot = start_bot()
        task = None
        try:
            state.remember_ticket_owner_user_id(channel.id, owner.id)
            initial_history = [{"role": "user", "content": item.content} for item in messages if item.author is owner]
            state.conversation_threads[channel.id] = list(initial_history)
            state.get_or_create_ticket_investigation_job(channel.id).mark_escalated_to_human()
            if mode != "first_staff_ack":
                state.mark_ticket_channel_stopped(channel.id, reason="staff_takeover")
            notice = state.TeamHandoffNotice("123", channel.id, "Review", message_text="Check ticket")
            state.remember_team_handoff_notice(channel.id, notice)
            with patch.object(bot.telegram_handoffs, "_wake_delivery_worker"):
                await bot.telegram_handoffs._handle_telegram_handoff_update({"message": {
                    "chat": {"id": 123}, "message_id": 200, "text": "Tell the user which transaction was checked.",
                    "reply_to_message": {"message_id": channel.id},
                }})
            if mode in {"deleted_before_draft", "deleted_user_message"}:
                messages.clear()
            if mode == "deleted_user_message":
                messages.append(SimpleNamespace(id=7621, author=staff, content="I'll check this.", attachments=[]))
            if mode == "late_gateway_duplicate":
                messages.append(SimpleNamespace(id=7621, author=owner, content="I found the withdrawal hash.", attachments=[]))
            if mode == "history_collision":
                messages.append(SimpleNamespace(id=7621, author=owner, content="Earlier detail 1", attachments=[]))
                messages.extend(SimpleNamespace(id=7622+i, author=staff, content=f"Later staff note {i}", attachments=[])
                                for i in range(29))
            if mode in {"gateway_masked", "gateway_masked_legacy"}:
                owner.name, staff.name = "owner", "staff"
                correction = SimpleNamespace(
                    id=7621, author=owner, channel=channel, reference=None, content="Correction with a screenshot from Base.",
                    attachments=[SimpleNamespace(id=95, filename="base.png", size=5, content_type="image/png",
                                                 url="https://cdn.discordapp.com/attachments/1/2/base.png")],
                )
                acknowledgement = SimpleNamespace(
                    id=7622, author=staff, channel=channel, reference=None,
                    content="Thanks, I'll take a look.", attachments=[],
                )
                messages.append(acknowledgement)
                with patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}), patch(
                    "ysupport.bot.client.discord.TextChannel", _FakeDiscordChannel,
                ):
                    await bot.on_message(correction)
                    await bot.on_message(acknowledgement)
                path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                payload = json.loads(path.read_text())
                if mode == "gateway_masked_legacy":
                    del payload["team_handoff_notice"]["discord_input_ids"]
                    payload["team_handoff_notice"]["latest_discord_input_id"] = acknowledgement.id
                await bot.close()
                state.clear_ticket_channel_state(channel.id, delete_persisted=False)
                path.write_text(json.dumps(payload))
                state.hydrate_ticket_state(channel.id)
                bot = start_bot()
            if mode in {"older_recorded_input", "recorded_image_priority", "older_recorded_edited",
                        "older_recorded_deleted", "older_recorded_image_missing", "image_without_mime", "second_handoff_edited"}:
                correction = SimpleNamespace(id=7621, author=owner, content="Use my corrected withdrawal details.",
                                             attachments=[image(101, "correction")])
                if mode == "image_without_mime":
                    correction.attachments[0].content_type = None
                messages.append(correction)
                _record_waiting_for_team_followup(channel.id, correction)
                if mode == "older_recorded_image_missing":
                    correction.attachments.clear()
                if mode not in {"recorded_image_priority", "second_handoff_edited"}:
                    messages.extend(SimpleNamespace(id=7622+i, author=owner, content=f"Later message {i}", attachments=[])
                                    for i in range(30))
                elif mode == "recorded_image_priority":
                    messages.extend(SimpleNamespace(id=7622+i, author=staff, content=f"Staff screenshot {i}",
                                                    attachments=[image(102+i, f"staff-{i}")]) for i in range(4))
                if mode == "older_recorded_edited":
                    path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                    persisted = json.loads(path.read_text())
                    if "history_discord_offsets" in persisted:
                        persisted["team_handoff_notice"]["discord_history_offsets"] = persisted.pop("history_discord_offsets")
                    await bot.close()
                    state.clear_ticket_channel_state(channel.id, delete_persisted=False)
                    path.write_text(json.dumps(persisted))
                    state.hydrate_ticket_state(channel.id)
                    bot = start_bot()
            task = asyncio.create_task(bot.telegram_handoffs._resume_pending_telegram_handoff_replies())
            if mode == "freshness":
                await asyncio.wait_for(entered.wait(), timeout=5)
                messages[0].content = "Check the corrected hash instead."
                messages.extend([
                    SimpleNamespace(id=7621, author=owner, content="", attachments=[SimpleNamespace(
                        id=91, filename="current.png", size=5, content_type="image/png",
                        url="https://cdn.discordapp.com/attachments/1/2/current.png?ex=updated")]),
                    SimpleNamespace(id=7622, author=staff, content="Keeper is checking the new transaction.", attachments=[]),
                ])
                release.set()
            await asyncio.wait_for(task, timeout=5)
            if mode in {"older_recorded_deleted", "older_recorded_image_missing"}:
                assert not channel.sent_messages and executor.calls == 0
            elif mode == "late_gateway_duplicate":
                assert len(channel.sent_messages) == 1 and executor.calls == 1
                saved = state.conversation_threads[channel.id]
                assert sum(row["content"] == messages[1].content for row in saved) == 1, saved
            elif mode == "second_handoff_edited":
                assert len(channel.sent_messages) == 1 and executor.calls == 1
                assert channel.id not in state.team_handoff_notice_by_channel
                path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                persisted = path.read_bytes()
                await bot.close()
                state.clear_ticket_channel_state(channel.id, delete_persisted=False)
                path.write_bytes(persisted)
                state.hydrate_ticket_state(channel.id)
                bot = start_bot()
                old_content = correction.content
                correction.content = "New wallet after revising the first correction."
                state.get_or_create_ticket_investigation_job(channel.id).mark_escalated_to_human()
                state.remember_team_handoff_notice(channel.id,
                    state.TeamHandoffNotice("123", channel.id + 1, "Review again", message_text="Second handoff"))
                with patch.object(bot.telegram_handoffs, "_wake_delivery_worker"):
                    await bot.telegram_handoffs._handle_telegram_handoff_update({"message": {
                        "chat": {"id": 123}, "message_id": 201, "text": "Update the user on the newer review.",
                        "reply_to_message": {"message_id": channel.id + 1},
                    }})
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                saved = json.loads(path.read_text())["history"]
                contents = [row["content"] for row in saved]
                assert len(channel.sent_messages) == 2 and executor.calls == 2
                assert old_content not in contents and contents.count(correction.content) == 1, contents
                assert all(row["content"] in contents for row in initial_history)
            elif mode in {"older_recorded_input", "recorded_image_priority", "older_recorded_edited", "image_without_mime"}:
                assert len(channel.sent_messages) == 1 and executor.calls == (2 if mode == "older_recorded_edited" else 1)
                request = SupportTurnRequest.from_ticket_execution_request(
                    TicketExecutionTransportRequest.from_turn_request(executor.last_request))
                assert any(row["content"] == correction.content for row in request.recent_transcript)
                assert "correction.png" in [item["filename"] for item in request.attachments]
                if mode == "older_recorded_edited":
                    path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                    saved = json.loads(path.read_text())["history"]
                    assert not any(row["content"] == "Use my corrected withdrawal details." for row in saved), saved
                    assert sum(row["content"] == correction.content for row in saved) == 1
                    assert sum(row["content"] == "Later message 0" for row in saved) == 2
            elif mode in {"older_summon_reference", "staff_owner_summon", "summon_reference_timeout"}:
                assert len(channel.sent_messages) == 1
                screenshot = SimpleNamespace(id=7622, author=owner, content="Please see my vault screenshot.",
                                             attachments=[image(101, "referenced")])
                messages.append(screenshot)
                if mode in {"older_summon_reference", "summon_reference_timeout"}:
                    messages.extend(SimpleNamespace(id=7623+i, author=staff, content=f"Staff note {i}", attachments=[])
                                    for i in range(30))
                staff.name = "staff"
                command = SimpleNamespace(id=messages[-1].id+1, author=staff, channel=channel, attachments=[],
                                          reference=SimpleNamespace(message_id=screenshot.id), content="y: Explain this.")
                messages.append(command)
                with patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}), patch(
                    "ysupport.bot.client.discord.TextChannel", _FakeDiscordChannel,
                ):
                    await bot.on_message(command)
                    await state.pending_tasks[channel.id]
                assert len(channel.sent_messages) == 2
                if mode == "summon_reference_timeout":
                    assert "referenced Discord message" in channel.sent_messages[-1] and executor.calls == 1
                else:
                    assert executor.calls == 2
                    assert {"role": "user", "content": screenshot.content} in executor.last_request.current_history
                    assert [item["source_author_role"] for item in executor.last_request.attachments
                            if item["filename"] == "referenced.png"] == ["ticket_user"]
            elif mode in interrupted:
                assert len(channel.sent_messages) == 1
                assert state.team_handoff_notice_by_channel[channel.id].status == "posting"
                if mode in {"ambiguous", "ambiguous_notice_retry"}:
                    messages.pop()
                await bot.close()
                path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                persisted = path.read_bytes()
                state.clear_ticket_channel_state(channel.id, delete_persisted=False)
                path.write_bytes(persisted)
                state.hydrate_ticket_state(channel.id)
                bot = start_bot()
                if mode == "recovery_gateway_ahead":
                    _record_waiting_for_team_followup(channel.id, SimpleNamespace(
                        id=7623, content="My latest follow-up is missing from this snapshot.", attachments=[]))
                if mode == "ambiguous_notice_retry":
                    http.fail_methods.add("editMessageText")
                if mode == "reconcile_timeout":
                    try:
                        await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                    except asyncio.TimeoutError:
                        pass
                    assert state.team_handoff_notice_by_channel[channel.id].status == "posting"
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert len(channel.sent_messages) == 1 and executor.calls == 1, "Interrupted send was retried"
                if mode == "reconcile_timeout":
                    assert reconciliation_reads == 2 and channel.id not in state.team_handoff_notice_by_channel
                    assert any("Replies closed." in payload.get("text", "") for _, payload in http.applied)
                if mode == "recovery_gateway_ahead":
                    saved = json.loads(path.read_text())["history"]
                    assert any("latest follow-up" in item["content"] for item in saved), saved
                if mode == "ambiguous_notice_retry":
                    http.fail_methods.clear()
                    await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                    assert state.team_handoff_notice_by_channel[channel.id].status == "review"
                    assert any("Delivery needs review" in payload.get("text", "") for _, payload in http.applied)
            elif mode == "close_notice_retry":
                assert len(channel.sent_messages) == 1
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                http.fail_methods.clear()
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert len(channel.sent_messages) == 1 and executor.calls == 1
                assert any("Replies closed." in payload.get("text", "") for _, payload in http.applied)
            elif mode in {"notice_deleted", "stale_session", "bounded_history", "history_collision", "staff_owner"} | image_modes:
                assert len(channel.sent_messages) == 1 and executor.calls == 1
            elif mode == "native_context":
                assert channel.sent_messages == ["I can see the original question."]
            elif mode == "assistant_revision":
                assert channel.sent_messages == ["Approval is paused."] and executor.calls == 2
            elif mode == "freshness":
                assert channel.sent_messages == ["I checked the corrected hash."] and executor.calls == 2
                assert "old hash" not in json.dumps(state.conversation_threads[channel.id])
            elif mode == "gateway_ahead":
                assert not channel.sent_messages, "A stale reply was posted while Discord history was behind"
                assert "new hash" in json.dumps(state.conversation_threads[channel.id])
            elif mode in {"gateway_masked", "gateway_masked_legacy"}:
                assert executor.calls == 0 and not channel.sent_messages
                assert any(row["content"] == correction.content for row in state.conversation_threads[channel.id])
                assert state.team_handoff_notice_by_channel[channel.id].followup_attachments[0]["filename"] == "base.png"
            elif mode in {"deleted_before_draft", "deleted_user_message"}:
                assert executor.calls == 0 and not channel.sent_messages, "Deleted ticket context was synthesized"
            else:
                assert not channel.sent_messages and executor.calls == 2
                assert state.team_handoff_notice_by_channel[channel.id].status == "review"
            if mode == "review_notice_retry":
                assert not any("Delivery needs review" in row.get("text", "") for _, row in http.applied)
                await bot.close()
                path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                persisted = path.read_bytes()
                state.clear_ticket_channel_state(channel.id, delete_persisted=False)
                path.write_bytes(persisted)
                state.hydrate_ticket_state(channel.id)
                bot = start_bot()
                http.fail_methods.clear()
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert any("Delivery needs review" in row.get("text", "") for _, row in http.applied)
                requests = len(http.requests)
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert len(http.requests) == requests, "The confirmed review notice was edited again"
                assert executor.calls == 2 and not channel.sent_messages
            if mode in {"bounded_history", "bounded_recovery", "history_collision"} | image_modes:
                path = state._TICKET_STATE_DIR / f"{channel.id}.json"
                payload = json.loads(path.read_text())
                if mode in image_modes:
                    assert not payload["team_handoff_notice"]
                    state.clear_ticket_channel_state(channel.id, delete_persisted=False)
                    path.write_text(json.dumps(payload))
                    state.hydrate_ticket_state(channel.id)
                    assert channel.id not in state.pending_messages
                    assert [item["filename"] for item in state.pending_attachments_by_channel.get(channel.id, [])] == ["late.png"]
                    assert any("new screenshot" in item["content"] for item in payload["history"])
                    if mode != "late_image":
                        summon = mode in {"late_image_summon", "late_image_reference", "late_image_text_reference", "image_refresh_timeout",
                                          "image_refresh_deleted", "text_only_summon"}
                        next_id = messages[-1].id + 1
                        messages.extend(SimpleNamespace(id=next_id+i, author=staff, content=f"Staff note {i}", attachments=[])
                                        for i in range(13))
                        command = SimpleNamespace(
                            id=next_id+13, author=staff, channel=channel, reference=None, attachments=[],
                            content="y: Explain the user's screenshot." if summon else "Thanks, I will take a look.")
                        if mode == "text_only_summon":
                            command.content = "y: Tell the user the team will post an update tomorrow."
                        if mode == "late_image_reference":
                            command.content = "y: Explain this."
                            command.reference = SimpleNamespace(
                                message_id=next(item.id for item in messages if item.author is owner and item.attachments)
                            )
                        elif mode == "late_image_text_reference":
                            command.reference = SimpleNamespace(message_id=messages[-1].id)
                        staff.name = "staff"
                        messages.append(command)
                        with patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}), patch(
                            "ysupport.bot.client.discord.TextChannel", _FakeDiscordChannel,
                        ):
                            await bot.on_message(command)
                            if summon:
                                await state.pending_tasks[channel.id]
                            if mode in {"image_refresh_timeout", "image_refresh_deleted"}:
                                assert executor.calls == 1 and "screenshot" in channel.sent_messages[-1].lower()
                                retry = SimpleNamespace(**{**vars(command), "id": command.id + 2})
                                if mode == "image_refresh_timeout":
                                    assert state.pending_attachments_by_channel[channel.id]
                                else:
                                    assert not state.pending_attachments_by_channel.get(channel.id)
                                    assert json.loads(path.read_text())["pending_ticket_input"] is None
                                    retry.attachments = next(item.attachments for item in messages if item.author is owner and item.attachments)
                                messages.append(retry)
                                await bot.on_message(retry)
                                await state.pending_tasks[channel.id]
                                assert executor.calls == 2
                            if mode == "text_only_summon":
                                assert executor.calls == 2 and not executor.last_request.attachments
                            elif summon:
                                assert [item["filename"] for item in executor.last_request.attachments] == ["late.png"]
                        after = json.loads(path.read_text())
                        assert after["history"][:len(payload["history"])] == payload["history"]
                        if not summon or mode == "text_only_summon":
                            assert after["pending_ticket_input"] == payload["pending_ticket_input"]
                elif mode == "history_collision":
                    assert [item for item in payload["history"] if item["role"] == "user"][:len(initial_history)] == initial_history
                else:
                    assert [item for item in payload["history"] if item["role"] == "user"] == initial_history
            assert state.stop_reasons_by_channel[channel.id] == "staff_takeover"
            review_modes = {"ambiguous", "ambiguous_notice_retry", "keeps_changing", "gateway_ahead", "gateway_masked",
                            "gateway_masked_legacy",
                            "deleted_before_draft", "deleted_user_message", "older_recorded_deleted",
                            "older_recorded_image_missing", "review_notice_retry"}
            assert (channel.id in state.team_handoff_notice_by_channel) == (mode in review_modes)
            if mode in review_modes:
                assert any("Delivery needs review" in payload.get("text", "") for _, payload in http.requests)
            return {"mode": mode, "discord_sends": len(channel.sent_messages), "native_turns": executor.calls}
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await bot.close()
            state.clear_ticket_channel_state(channel.id, delete_persisted=True)


async def verify_handoff_presentation(*, public=True, metadata=True, summary_failure=None):
    channel_id, author_id, original_id = 7681, 7682, 7683
    question = "Is this vault retired? The keeper seems inactive."
    support_reply = (
        "The vault is not marked retired. Convex last reported Oct 19 and its "
        "harvest trigger is true. Why the keeper has not acted remains unknown."
    )
    action = "Keeper/strategist review is needed; decide whether to harvest."
    summary = "Convex last reported Oct 19 and its trigger is true; the keeper cause remains unknown."
    proof = "Receipt on Ethereum at block 123. https://etherscan.io/tx/" + "0x" + "a" * 64
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
            completed_agent_key=None, requires_human_handoff=True, handoff_reason=action,
            verified_evidence=[proof]),
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
                        await eventually(lambda: bot.telegram_handoffs.public.records[original_id].status == "open")
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
                    notice = bot.telegram_handoffs.public.records[original_id].message_text if public else sends[0]["text"]
                    assert "SERVER NAME" not in notice and "Ticket-7681" not in notice
                    assert "<b>Action</b>: " + action in notice
                    assert "<b>Issue</b>: " + (support_reply if summary_failure else summary) in notice
                    assert ("<b>Verified evidence</b>" in notice) == (not public)
                    assert (proof in notice) == (not public)
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
                        assert "I will post one reply to their original Discord message." in notice
                        assert "Preparing handoff" in sends[0]["text"] and "reply_markup" not in sends[0]
                        assert any(method == 'editMessageText' and payload['text'] == notice
                            and payload['reply_markup']['inline_keyboard'] for method, payload in http.requests)
                        assert author_id not in state.public_conversations
                        assert not (state._PUBLIC_STATE_DIR / f"{author_id}.json").exists()
                        assert channel_id not in state.team_handoff_notice_by_channel
                        saved = bot.telegram_handoffs.public.records[original_id]
                        assert saved.history and saved.status == "open"
                        await bot.telegram_handoffs._handle_telegram_handoff_update({"callback_query": {
                            "id": "public-dismiss", "data": "dismiss_handoff", "message": {
                                "message_id": saved.telegram_message_id, "chat": {"id": 123}}}})
                        await eventually(lambda: original_id not in bot.telegram_handoffs.public.records)
                        assert channel_id not in state.stopped_channels and len(executor.calls) == 1
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
                    return {"notice": notice, "findings_reached_summary_model": True, "verified_evidence_in_notice": not public,
                        "summary_fallback": summary_failure, "source_metadata_present": metadata,
                        "public_context_saved_and_dismissed" if public else "ticket_dismissal_preserved": True}
                finally:
                    await bot.close()
                    state.clear_public_conversation(author_id)
                    state.clear_ticket_channel_state(channel_id, delete_persisted=True)


async def verify_ordinary_staff_path(*, summon):
    import discord

    channel = _FakeDiscordChannel(7690)
    channel.category = SimpleNamespace(id=1)
    owner = SimpleNamespace(id=7691, bot=False)
    staff = SimpleNamespace(id=7692, bot=False, name="staff", guild_permissions=SimpleNamespace(administrator=True))
    current = {"role": "user", "content": "Corrected details in Discord."}
    channel.history_messages.append(SimpleNamespace(id=7693, author=owner, content=current["content"], attachments=[]))
    state.remember_ticket_owner_user_id(channel.id, owner.id)
    state.conversation_threads[channel.id] = [{"role": "user", "content": "Earlier details."}]
    if not summon:
        state.pending_messages[channel.id] = "Queued follow-up."
        state.pending_attachments_by_channel[channel.id] = [{"filename": "pending.png"}]
        state.get_or_create_ticket_investigation_job(channel.id).remember_wallet("0x" + "11" * 20)
    bot = TicketBot(intents=discord.Intents.none())
    executor = FakeInvestigationExecutor(result=SimpleNamespace(flow_outcome=TicketAgentFlowOutcome(
        raw_final_reply="Updated reply.", conversation_history=[current, {"role": "assistant", "content": "Updated reply."}],
        completed_agent_key=None, requires_human_handoff=False), updated_job=state.TicketInvestigationJob(channel.id)))
    bot.investigation_executor = executor
    command = SimpleNamespace(id=7694, channel=channel, author=staff, attachments=[], reference=None,
                              content="y: Check the details." if summon else "I will take over.")
    try:
        with patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}), patch(
                "ysupport.bot.client.discord.TextChannel", _FakeDiscordChannel):
            await bot.on_message(command)
            if summon:
                await state.pending_tasks[channel.id]
        saved = json.loads((state._TICKET_STATE_DIR / f"{channel.id}.json").read_text())
        if summon:
            assert saved["history"] == executor.result.flow_outcome.conversation_history, saved["history"]
            assert executor.calls[0]["request"].current_history == [current]
        else:
            assert saved["stop_reason"] == "manual_stop" and not saved["history"], saved
            assert saved["pending_ticket_input"] is None and saved["investigation_job"] is None
        return {"ordinary_behavior_preserved": True}
    finally:
        await bot.close()
        state.clear_ticket_channel_state(channel.id, delete_persisted=True)


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
    for name, summon in (("ordinary_staff_takeover", False), ("ordinary_staff_summon", True)):
        try:
            report["scenarios"][name] = {"ok": True, **await verify_ordinary_staff_path(summon=summon)}
        except Exception as exc:
            report["scenarios"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    for mode in ("freshness", "restart", "ambiguous", "keeps_changing", "gateway_ahead", "gateway_masked",
                 "gateway_masked_legacy",
                 "deleted_before_draft", "review_notice_retry", "close_notice_retry",
                 "ambiguous_notice_retry", "recovery_gateway_ahead", "notice_deleted",
                 "deleted_user_message", "stale_session", "assistant_revision",
                 "bounded_history", "bounded_recovery", "late_image", "native_context", "staff_owner",
                 "history_collision", "late_image_summon", "late_image_reference", "late_image_text_reference",
                 "staff_ack", "first_staff_ack",
                 "image_refresh_timeout", "image_refresh_deleted", "text_only_summon",
                 "older_recorded_input", "older_recorded_edited", "older_recorded_deleted", "recorded_image_priority",
                 "older_recorded_image_missing", "older_summon_reference", "staff_owner_summon", "summon_reference_timeout",
                 "second_handoff_edited", "image_without_mime", "reconcile_timeout", "late_gateway_duplicate"):
        name = f"private_{mode}"
        try:
            report["scenarios"][name] = {"ok": True, **await verify_private_ticket_update(mode)}
        except Exception as exc:
            report["scenarios"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
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
        'ysupport/bot/state.py', 'ysupport/bot/utils.py', "tests/telegram_http_e2e.py")}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    artifact.chmod(0o600)
    print(json.dumps(report["scenarios"], indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/telegram-http-e2e.json"))
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
