"""Exercise concurrent native processes and ticket bursts without live messages.

Failure modes and acceptance are recorded in PLAN.md before runtime changes.
Run: python -B -m tests.latency_e2e --artifact .cache/verification/latency-e2e.json
"""

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.optimization_e2e import eventually
import config
import state
from codex_support_home import prune_codex_support_caches
from tests.test_ticket_intake import _FakeDiscordChannel
from ticket_investigation.codex_support_endpoint import CodexSupportTicketExecutionJsonEndpoint
from ticket_investigation.transport import build_smoke_transport_request
from ticket_investigation.executor import TransportTicketInvestigationExecutor
from ticket_investigation.json_endpoint import JsonEndpointTicketExecutionTransport
from views import stop_ticket_for_manual_support
from ysupport import TicketBot


MODEL = '''import json,os,sys,time,uuid
from pathlib import Path
root=Path(sys.argv[1]); directory=Path(os.environ["TICKET_EXECUTION_RUN_DIR"])
request=json.loads((directory/"support_request.json").read_text()); prompt=sys.stdin.read()
key=request["current_user_message"]
if key=="inline":
 assert '"current_user_message": "inline"' in prompt
 assert '"required"' in prompt and '"evidence_summary"' in prompt
 assert 'search_vaults' in prompt and 'search_documentation' in prompt
(root/(key+".started")).write_text(json.dumps({"request":request,"pid":os.getpid()}))
if key=="failure": raise RuntimeError("controlled execution failure")
deadline=time.monotonic()+12
while not (root/(key+".release")).exists():
 if time.monotonic()>deadline: raise RuntimeError("fixture gate timed out")
 time.sleep(.01)
result={"answer":key,"requires_human_handoff":False,"handoff_reason":None,"handoff_kind":None,"evidence_summary":"isolated native process", "used_tools":[]}
print(json.dumps({"type":"thread.started","thread_id":str(uuid.uuid4())}))
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":json.dumps(result)}}))
'''


async def verify_concurrency(root):
    root.mkdir()
    model = root / "model.py"
    model.write_text(MODEL)
    endpoint = CodexSupportTicketExecutionJsonEndpoint(
        codex_command=[sys.executable, str(model), str(root), "exec"],
        codex_home=root / "home", session_dir=root / "sessions",
        session_max_age_hours=168, max_concurrent_turns=2,
        ysupport_mcp_url="http://127.0.0.1:1/mcp", mcp_server_api_key="fixture",
        env={}, timeout_seconds=15,
    )
    tasks = []

    def launch(key, channel):
        request = replace(
            build_smoke_transport_request(), smoke_mode=None, aggregated_text=key,
            run_context={"channel_id": channel, "project_context": "yearn"},
        )
        task = asyncio.create_task(endpoint.execute_json_turn(request.to_json()))
        tasks.append(task)
        return task

    try:
        a = launch("a", 7101)
        await eventually(lambda: (root / "a.started").exists())
        b = launch("b", 7102)
        await eventually(lambda: (root / "b.started").exists(), timeout=2)
        same = launch("same", 7101)
        queued = launch("queued", 7103)
        await asyncio.sleep(.1)
        assert not (root / "same.started").exists()
        assert not (root / "queued.started").exists()
        assert await asyncio.wait_for(endpoint.prune_expired_sessions(), 1) == 0
        queued.cancel()
        await asyncio.gather(queued, return_exceptions=True)
        b_pid = json.loads((root / "b.started").read_text())["pid"]
        b.cancel()
        await asyncio.gather(b, return_exceptions=True)
        assert not Path(f"/proc/{b_pid}").exists() and not a.done()
        c = launch("c", 7103)
        await eventually(lambda: (root / "c.started").exists())
        sockets = [json.loads((root / f"{key}.started").read_text())["request"]["support_state"]["internal_rpc"]["command"][-1] for key in ("a", "b", "c")]
        assert len(set(sockets)) == 3
        (root / "a.release").touch()
        assert json.loads(await a)["flow_outcome"]["raw_final_reply"] == "a"
        await eventually(lambda: (root / "same.started").exists())
        for key in ("same", "c"):
            (root / f"{key}.release").touch()
        await asyncio.gather(same, c)
        assert endpoint.session_manager.load("ticket:7101").run_count == 1
        assert endpoint.session_manager.load("ticket:7103") is not None
        failure = launch("failure", 7104)
        result = await asyncio.gather(failure, return_exceptions=True)
        assert isinstance(result[0], RuntimeError)
        (root / "inline.release").touch()
        assert json.loads(await launch("inline", 7105))["flow_outcome"]["raw_final_reply"] == "inline"

        old = root / "home/.tmp/plugins/old"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("reproducible expired cache")
        timestamp = (datetime.now(timezone.utc) - timedelta(days=9)).timestamp()
        os.utime(old, (timestamp, timestamp))
        os.utime(old.parent, (timestamp, timestamp))
        entered, release = threading.Event(), threading.Event()

        def prune(home, *, cutoff):
            entered.set()
            assert release.wait(3)
            return prune_codex_support_caches(home, cutoff=cutoff)

        with patch("ticket_investigation.codex_support_endpoint.prune_codex_support_caches", new=prune):
            maintenance = asyncio.create_task(endpoint.prune_expired_sessions())
            tasks.append(maintenance)
            await eventually(entered.is_set)
            after = launch("after", 7106)
            await asyncio.sleep(.1)
            assert not (root / "after.started").exists()
            release.set()
            await maintenance
            await eventually(lambda: (root / "after.started").exists())
            (root / "after.release").touch()
            await after
        assert not old.exists()
        assert not (root / "queued.started").exists()
        return {"overlap": True, "bounded": True, "same_ticket_serial": True,
                "queue_and_active_cancellation": True, "private_rpc_isolated": True,
                "idle_maintenance_exclusive": True, "busy_cleanup_skips": True,
                "failure_releases_capacity": True, "inline_request_and_tool_map": True}
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def verify_maintenance_yield(root):
    root.mkdir()

    async def scenario(directory, cancel_waiter):
        directory.mkdir()
        model = directory / "model.py"
        model.write_text(MODEL)
        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=[sys.executable, str(model), str(directory), "exec"],
            codex_home=directory / "home", session_dir=directory / "records",
            session_max_age_hours=168, env={}, timeout_seconds=15)
        sessions = directory / "home/sessions"
        sessions.mkdir()
        identifiers = ["11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"]
        timestamp = (datetime.now(timezone.utc) - timedelta(days=9)).timestamp()
        for identifier in identifiers:
            path = sessions / f"rollout-{identifier}.jsonl"
            path.write_text("{}")
            os.utime(path, (timestamp, timestamp))
        cache = directory / "home/.tmp/plugins/old"
        cache.parent.mkdir(parents=True)
        cache.write_text("expired cache")
        os.utime(cache, (timestamp, timestamp))
        os.utime(cache.parent, (timestamp, timestamp))
        entered, release = asyncio.Event(), asyncio.Event()
        deleted = []

        async def delete(identifier):
            if not deleted:
                entered.set()
                await release.wait()
            deleted.append(identifier)
            (sessions / f"rollout-{identifier}.jsonl").unlink()
            return True

        endpoint._delete_codex_session = delete
        maintenance = asyncio.create_task(endpoint.prune_expired_sessions())
        turn = None
        try:
            await eventually(entered.is_set)
            (directory / "after.release").touch()
            request = replace(build_smoke_transport_request(), smoke_mode=None,
                aggregated_text="after", run_context={"channel_id": 7150, "project_context": "yearn"})
            turn = asyncio.create_task(endpoint.execute_json_turn(request.to_json()))
            await asyncio.sleep(.05)
            assert not (directory / "after.started").exists(), "Execution overlapped maintenance"
            if cancel_waiter:
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
            release.set()
            await maintenance
            if cancel_waiter:
                assert deleted == identifiers and not cache.exists(), "Canceled waiter prevented cleanup"
                assert not (directory / "after.started").exists()
            else:
                await turn
                assert deleted == identifiers[:1], "Cleanup started another deletion ahead of a queued ticket"
                assert cache.exists(), "Cache cleanup ran ahead of a queued ticket"
                await endpoint.prune_expired_sessions()
                assert deleted == identifiers and not cache.exists(), "Idle cleanup did not resume"
        finally:
            release.set()
            for task in (turn, maintenance):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(t for t in (turn, maintenance) if t), return_exceptions=True)

    await scenario(root / "waiting", False)
    await scenario(root / "canceled", True)
    return {"current_delete_finishes": True, "queued_ticket_precedes_remaining_cleanup": True,
            "cache_cleanup_deferred": True, "idle_cleanup_resumes": True,
            "canceled_waiter_does_not_block_cleanup": True}


async def verify_bursts(root):
    root.mkdir()
    bot = TicketBot(intents=__import__("discord").Intents.none())
    channels = {i: _FakeDiscordChannel(i) for i in (7201, 7202, 7203, 7204, 7205)}
    for channel in channels.values():
        channel.category = SimpleNamespace(id=1)
    bot.get_channel = channels.get
    requests, started, release = [], asyncio.Event(), asyncio.Event()
    canceled = []

    async def execute(request, hooks=None):
        requests.append(request)
        if request.aggregated_text == "withdraw":
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                canceled.append(request.run_context.channel_id)
                raise
            if request.run_context.channel_id == 7204:
                raise RuntimeError("controlled failed investigation with queued follow-up")
        answer = "Outdated draft" if request.aggregated_text == "withdraw" else "Use the verified withdrawal route."
        from ticket_investigation.contracts import TicketAgentFlowOutcome
        return SimpleNamespace(
            updated_job=request.investigation_job,
            flow_outcome=TicketAgentFlowOutcome(
                raw_final_reply=answer,
                conversation_history=[*request.current_history,
                    {"role": "user", "content": request.aggregated_text},
                    {"role": "assistant", "content": answer}],
                completed_agent_key=None,
                requires_human_handoff=(request.run_context.channel_id == 7205 and request.aggregated_text == "withdraw")),
        )

    bot.investigation_executor = SimpleNamespace(execute_turn=execute)
    owner = SimpleNamespace(id=777, name="owner", bot=False)

    def message(channel, content, message_id, image=False):
        attachments = [SimpleNamespace(id=message_id, filename=f"user-{message_id}.png",
            url=f"https://cdn.discordapp.com/attachments/1/2/user-{message_id}.png", content_type="image/png", size=123)] if image else []
        return SimpleNamespace(id=message_id, author=owner, content=content,
            channel=channels[channel], reference=None, attachments=attachments)

    try:
        with patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}), \
             patch.object(config, "COOLDOWN_SECONDS", .01), \
             patch("ysupport.discord.TextChannel", _FakeDiscordChannel), \
             patch("ysupport._outer_support_boundary_result", new=AsyncMock(return_value={"tripwire_triggered": False})):
            for channel in channels:
                state.remember_ticket_owner_user_id(channel, owner.id)
            await bot.on_message(message(7201, "withdraw", 1, image=True))
            await eventually(started.is_set)
            active = state.pending_tasks[7201]
            await bot.on_message(message(7201, "yvvbUSDT", 2))
            await bot.on_message(message(7201, "simulation failed", 3, image=True))
            assert state.pending_tasks[7201] is active and not canceled
            assert not any("Outdated draft" in text for text in channels[7201].sent_messages)
            release.set()
            await eventually(lambda: 7201 not in state.pending_tasks)
            assert len(requests) == 2 and not canceled
            assert "yvvbUSDT" in requests[-1].aggregated_text
            assert "simulation failed" in requests[-1].aggregated_text
            assert {a["filename"] for a in requests[-1].attachments} == {"user-1.png", "user-3.png"}
            assert not any("Outdated draft" in text for text in channels[7201].sent_messages)
            assert channels[7201].sent_messages[-1] == "Use the verified withdrawal route."
            assert any(item.get("role") == "user" and item["content"] == "withdraw" for item in requests[-1].current_history)
            for channel in (7202, 7203):
                started.clear()
                release.clear()
                await bot.on_message(message(channel, "withdraw", channel))
                await eventually(started.is_set)
                await bot.on_message(message(channel, "latest detail", channel + 10))
                if channel == 7202:
                    from views import take_over_ticket_for_staff
                    await take_over_ticket_for_staff(channel)
                else:
                    await stop_ticket_for_manual_support(channel)
                assert channel in canceled and channel not in state.pending_tasks
                assert all("Use the verified" not in text and "Outdated" not in text for text in channels[channel].sent_messages)
            for channel in (7204, 7205):
                started.clear()
                release.clear()
                with patch("ysupport._send_ticket_handoff_notice", new=AsyncMock()) as handoff:
                    await bot.on_message(message(channel, "withdraw", channel))
                    await eventually(started.is_set)
                    await bot.on_message(message(channel, "latest detail", channel + 10))
                    release.set()
                    await eventually(lambda: channel not in state.pending_tasks)
                    assert channels[channel].sent_messages[-1] == "Use the verified withdrawal route."
                    assert channel not in state.stopped_channels
                    handoff.assert_not_awaited()
        return {"active_work_preserved": True, "latest_text_and_image_used": True,
                "outdated_draft_suppressed": True, "staff_and_owner_stop_cancel": True,
                "original_and_latest_images_preserved": True,
                "failed_work_preserves_queued_followup": True,
                "overtaken_handoff_not_sent": True}
    finally:
        release.set()
        await bot.close()
        for channel in channels:
            state.clear_ticket_channel_state(channel, delete_persisted=True)


async def verify_staff_images(root):
    root.mkdir()
    bot = TicketBot(intents=__import__("discord").Intents.none())
    channel = _FakeDiscordChannel(7301)
    channel.category = SimpleNamespace(id=1)
    owner = SimpleNamespace(id=777, name="owner", bot=False)
    staff = SimpleNamespace(id=888, name="staff", bot=False,
        guild_permissions=SimpleNamespace(administrator=True))

    def message(identifier, author, filename):
        return SimpleNamespace(id=identifier, author=author, content="see screenshot",
            channel=channel, attachments=[SimpleNamespace(id=identifier, filename=filename,
                url=f"https://cdn.discordapp.com/attachments/1/2/{filename}",
                content_type="image/png", size=123)])

    messages = [message(1, owner, "user.png"), message(2, staff, "staff-old.png")]

    async def history(**kwargs):
        for item in reversed(messages):
            yield item

    channel.history = history
    captured = []

    async def execute(request, hooks=None):
        captured.append(request)
        from ticket_investigation.contracts import TicketAgentFlowOutcome
        return SimpleNamespace(updated_job=request.investigation_job,
            flow_outcome=TicketAgentFlowOutcome(raw_final_reply="Read the user's error.",
                conversation_history=[], completed_agent_key=None, requires_human_handoff=False))

    bot.investigation_executor = SimpleNamespace(execute_turn=execute)
    state.remember_ticket_owner_user_id(channel.id, owner.id)
    state.stop_ticket_channel(channel.id)
    command = message(3, staff, "staff-current.png")
    try:
        await bot._handle_ticket_staff_summon(command,
            state.BotRunContext(channel_id=channel.id, project_context="yearn"),
            "see user message and screenshot above", was_stopped=True)
        assert len(captured) == 1
        attachments = captured[0].attachments
        assert [a["filename"] for a in attachments] == ["user.png", "staff-current.png"]
        assert attachments[0]["source_author_role"] == "ticket_user"
        assert attachments[1]["source_author_role"] == "internal_team"
        assert channel.id in state.stopped_channels
        return {"recent_user_image": True, "old_staff_image_excluded": True,
                "current_staff_image_labeled": True, "remains_stopped": True}
    finally:
        await bot.close()
        state.clear_ticket_channel_state(channel.id, delete_persisted=True)


async def verify_telegram_concurrency(root):
    root.mkdir()
    model = root / "model.py"
    model.write_text(MODEL)
    bot = TicketBot(intents=__import__("discord").Intents.none())
    endpoint = CodexSupportTicketExecutionJsonEndpoint(
        codex_command=[sys.executable, str(model), str(root), "exec"],
        codex_home=root / "home", env={}, timeout_seconds=15)
    bot.investigation_executor = TransportTicketInvestigationExecutor(JsonEndpointTicketExecutionTransport(endpoint))
    channels = {i: _FakeDiscordChannel(i) for i in range(7401, 7410)}
    direct = None
    for channel in channels.values():
        channel.category = SimpleNamespace(id=1)
    bot.get_channel = channels.get

    api_calls = []
    async def api(method, payload):
        api_calls.append((method, dict(payload)))
        if method == "getUpdates":
            await asyncio.sleep(.05)
            return {"ok": True, "result": []}
        if method == "editMessageText" and str(payload.get("message_id")) == "7407":
            await asyncio.sleep(.03)
        return {"ok": True, "result": True}

    try:
        with patch.object(config, "TELEGRAM_BOT_TOKEN", "fixture"), \
             patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"), \
             patch.dict(os.environ, {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"}), \
             patch("handoff._telegram_api_call", new=api):
            for channel, key in zip(channels, ("a", "b")):
                state.remember_team_handoff_notice(channel, state.TeamHandoffNotice(
                    telegram_chat_id="123", telegram_message_id=channel, reason="fixture",
                    status="pending_delivery", pending_reply_text=key))
            bot.telegram_handoffs.start()
            await eventually(lambda: (root / "a.started").exists())
            await eventually(lambda: (root / "b.started").exists(), timeout=2)
            state.remember_team_handoff_notice(7403, state.TeamHandoffNotice(
                telegram_chat_id="123", telegram_message_id=7403, reason="new pending state",
                status="pending_delivery", pending_reply_text="c"))
            assert not (root / "c.started").exists()
            stopped = json.loads((root / "b.started").read_text())
            await stop_ticket_for_manual_support(7402)
            assert not Path(f"/proc/{stopped['pid']}").exists(), "Stopped handoff model is still running"
            assert not Path(stopped["request"]["support_state"]["internal_rpc"]["command"][-1]).exists()
            await eventually(lambda: (root / "c.started").exists())
            assert not (root / "a.release").exists(), "Unrelated handoff needed to finish first"
            assert not channels[7402].sent_messages
            (root / "a.release").touch()
            (root / "c.release").touch()
            await eventually(lambda: all(i not in state.team_handoff_notice_by_channel for i in channels))
            assert channels[7401].sent_messages == ["a"] and channels[7403].sent_messages == ["c"]
            request = replace(build_smoke_transport_request(), smoke_mode=None,
                aggregated_text="d", run_context={"channel_id": 7499, "project_context": "yearn"})
            direct = asyncio.create_task(endpoint.execute_json_turn(request.to_json()))
            await eventually(lambda: (root / "d.started").exists())
            for channel, key in ((7404, "e"), (7405, "queued")):
                state.remember_team_handoff_notice(channel, state.TeamHandoffNotice(
                    telegram_chat_id="123", telegram_message_id=channel, reason="queued fixture",
                    status="pending_delivery", pending_reply_text=key))
            await eventually(lambda: (root / "e.started").exists())
            await eventually(lambda: 7405 in state.handoff_delivery_tasks)
            assert not (root / "queued.started").exists()
            await stop_ticket_for_manual_support(7405)
            await stop_ticket_for_manual_support(7404)
            state.remember_team_handoff_notice(7406, state.TeamHandoffNotice(
                telegram_chat_id="123", telegram_message_id=7406, reason="next fixture",
                status="pending_delivery", pending_reply_text="f"))
            await eventually(lambda: (root / "f.started").exists())
            from views import take_over_ticket_for_staff
            await take_over_ticket_for_staff(7406)
            assert 7406 in state.handoff_delivery_tasks, "Staff takeover canceled the accepted Telegram reply"
            for key in ("d", "f"):
                (root / f"{key}.release").touch()
            await direct
            await eventually(lambda: 7406 not in state.team_handoff_notice_by_channel)
            assert channels[7406].sent_messages == ["f"] and 7406 in state.stopped_channels
            assert not channels[7404].sent_messages and not channels[7405].sent_messages
            assert not (root / "queued.started").exists()
            for channel, key in ((7407, "g"), (7408, "h")):
                state.remember_team_handoff_notice(channel, state.TeamHandoffNotice(
                    telegram_chat_id="123", telegram_message_id=channel, reason="lifecycle fixture",
                    status="pending_delivery", pending_reply_text=key))
            await eventually(lambda: (root / "g.started").exists() and (root / "h.started").exists())
            g = json.loads((root / "g.started").read_text())
            h = json.loads((root / "h.started").read_text())
            from ticket_channel_lifecycle import clear_deleted_ticket_channel
            with patch("ticket_channel_lifecycle.discord.TextChannel", _FakeDiscordChannel):
                await clear_deleted_ticket_channel(channels[7407])
            assert not Path(f"/proc/{g['pid']}").exists() and Path(f"/proc/{h['pid']}").exists()
            await bot.telegram_handoffs.close()
            assert not Path(f"/proc/{h['pid']}").exists()
            assert state.team_handoff_notice_by_channel[7408].status == "pending_delivery"
            early_notice = state.TeamHandoffNotice(telegram_chat_id="123", telegram_message_id=7409,
                reason="early stop fixture", status="pending_delivery", pending_reply_text="early")
            state.remember_team_handoff_notice(7409, early_notice)
            delivery = asyncio.create_task(bot.telegram_handoffs._run_handoff_delivery(
                channel_id=7409, notice=early_notice, team_reply_text="early"))
            stopping = asyncio.create_task(stop_ticket_for_manual_support(7409))
            await asyncio.gather(delivery, stopping)
            assert delivery.result() is None and not (root / "early.started").exists()
            assert all(channel not in state.handoff_delivery_tasks for channel in channels)
            assert not any(str(payload.get("message_id")) == "7407" and "Update delivery failed" in payload.get("text", "")
                           for method, payload in api_calls if method == "editMessageText"), "Deletion was reported as delivery failure"
            return {"handoff_replies_overlap": True, "bounded": True,
                    "stop_prevents_delivery": True, "each_reply_delivered_once": True,
                    "new_pending_state_processed_while_other_ticket_busy": True,
                    "stop_kills_native_process_and_socket": True,
                    "queued_handoff_never_starts_after_stop": True,
                    "worker_survives_stopped_child": True,
                    "staff_takeover_preserves_accepted_reply": True,
                    "deleted_ticket_cancels_its_own_model": True,
                    "shutdown_kills_child_and_preserves_pending_reply": True,
                    "stop_before_child_start_is_safe": True,
                    "delivery_task_ownership_released": True,
                    "cancellation_does_not_report_delivery_failure": True}
    finally:
        if direct and not direct.done():
            direct.cancel()
            await asyncio.gather(direct, return_exceptions=True)
        await bot.close()
        for channel in channels:
            state.clear_ticket_channel_state(channel, delete_persisted=True)


async def verify_deleted_handoff_races(root):
    from handoff import build_archived_handoff_notice
    from ticket_channel_lifecycle import clear_deleted_ticket_channel

    root.mkdir()
    channel = _FakeDiscordChannel(7501)
    bot = TicketBot(intents=__import__("discord").Intents.none())
    # discord.py removes the channel from its cache before dispatching deletion.
    bot.get_channel = lambda _channel_id: None
    edits = []
    failures = []
    cancel_entered, cancel_release = asyncio.Event(), asyncio.Event()
    archive_entered, archive_release = asyncio.Event(), asyncio.Event()
    archived_text = build_archived_handoff_notice("Queued ticket")

    async def pending_turn():
        try:
            await asyncio.Event().wait()
        finally:
            cancel_entered.set()
            await cancel_release.wait()

    async def api(method, payload):
        if method == "editMessageText":
            edits.append(dict(payload))
            if payload["text"] == archived_text:
                archive_entered.set()
                await archive_release.wait()
        return {"ok": True, "result": True}

    pending = asyncio.create_task(pending_turn())
    state.pending_tasks[channel.id] = pending
    deletion = None
    try:
        with patch.object(config, "TELEGRAM_BOT_TOKEN", "fixture"), \
             patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"), \
             patch.dict(os.environ, {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"}), \
             patch("handoff._telegram_api_call", new=api), \
             patch("ticket_channel_lifecycle.discord.TextChannel", _FakeDiscordChannel):
            state.remember_team_handoff_notice(channel.id, state.TeamHandoffNotice(
                telegram_chat_id="123", telegram_message_id=channel.id, reason="queued fixture",
                message_text="Queued ticket", status="pending_delivery", pending_reply_text="Resolved"))
            await asyncio.sleep(0)
            deletion = asyncio.create_task(clear_deleted_ticket_channel(channel))
            await eventually(cancel_entered.is_set)
            await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
            cancel_release.set()
            await eventually(archive_entered.is_set)
            await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
            archive_release.set()
            await deletion
            if [edit["text"] for edit in edits] != [archived_text]:
                failures.append("Queued delivery overwrote the deleted-ticket archive")
            assert not (state._TICKET_STATE_DIR / f"{channel.id}.json").exists()
            assert channel.id not in state.team_handoff_notice_by_channel
    finally:
        cancel_release.set()
        archive_release.set()
        await asyncio.gather(*(task for task in (pending, deletion) if task), return_exceptions=True)
        await bot.close()
        state.clear_ticket_channel_state(channel.id, delete_persisted=True)

    for status in ("pending_delivery", "delivered_pending_close"):
        directory = root / status
        directory.mkdir()
        model = directory / "model.py"
        model.write_text(MODEL)
        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=[sys.executable, str(model), str(directory), "exec"],
            codex_home=directory / "home", env={}, timeout_seconds=15)
        bot = TicketBot(intents=__import__("discord").Intents.none())
        bot.investigation_executor = TransportTicketInvestigationExecutor(JsonEndpointTicketExecutionTransport(endpoint))
        peer = _FakeDiscordChannel(7502)
        peer.category = SimpleNamespace(id=1)
        state.conversation_threads[peer.id] = [{"role": "user", "content": "Please check my ticket."}]
        bot.get_channel = lambda channel_id: peer if channel_id == peer.id else None
        edits = []

        async def api(method, payload):
            if method == "editMessageText":
                edits.append(dict(payload))
            return {"ok": True, "result": True}

        recovery = None
        try:
            with patch.object(config, "TELEGRAM_BOT_TOKEN", "fixture"), \
                 patch.object(config, "TELEGRAM_YSUPPORT_CHAT", "123"), \
                 patch.dict(os.environ, {"YSUPPORT_ALLOW_TEST_TELEGRAM": "1"}), \
                 patch("handoff._telegram_api_call", new=api), \
                 patch("ticket_channel_lifecycle.discord.TextChannel", _FakeDiscordChannel):
                for target, current_status, reply in ((peer, "pending_delivery", "peer"), (channel, status, "Resolved")):
                    state.remember_team_handoff_notice(target.id, state.TeamHandoffNotice(
                        telegram_chat_id="123", telegram_message_id=target.id, reason="snapshot fixture",
                        message_text="Queued ticket", status=current_status, pending_reply_text=reply))
                recovery = asyncio.create_task(bot.telegram_handoffs._resume_pending_telegram_handoff_replies())
                await eventually(lambda: (directory / "peer.started").exists())
                await clear_deleted_ticket_channel(channel)
                (directory / "peer.release").touch()
                await recovery
                if (state._TICKET_STATE_DIR / f"{channel.id}.json").exists():
                    failures.append(f"{status} snapshot recreated deleted ticket state")
                assert [edit["text"] for edit in edits if edit["message_id"] == channel.id] == [archived_text]
                assert peer.sent_messages == ["peer"]
                assert channel.id not in state.handoff_delivery_tasks
        finally:
            if recovery is not None and not recovery.done():
                recovery.cancel()
                await asyncio.gather(recovery, return_exceptions=True)
            await bot.close()
            for target in (channel, peer):
                state.clear_ticket_channel_state(target.id, delete_persisted=True)
    assert not failures, "; ".join(failures)
    return {"queued_handoff_invalid_during_cancellation": True,
            "archive_not_overwritten": True, "deleted_state_not_recreated_from_snapshot": True,
            "unrelated_native_delivery_preserved": True}


async def main(artifact):
    report = {"ok": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "scenarios": {}}
    with tempfile.TemporaryDirectory(prefix="ysupport-latency-e2e-") as temp:
        for name, check in (("native_concurrency", verify_concurrency),
                            ("maintenance_yield", verify_maintenance_yield),
                            ("owner_bursts", verify_bursts),
                            ("staff_images", verify_staff_images),
                            ("telegram_concurrency", verify_telegram_concurrency),
                            ("deleted_handoff_races", verify_deleted_handoff_races)):
            try:
                report["scenarios"][name] = {"ok": True, **await check(Path(temp) / name)}
            except Exception as exc:
                report["scenarios"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        report["ok"] = all(item["ok"] for item in report["scenarios"].values())
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        "tests/latency_e2e.py", "ysupport.py", "discord_support_runtime.py",
        "ticket_investigation/codex_support_endpoint.py", "config.py",
        "telegram_handoff_controller.py", "ysupport_codex_instructions.md",
        "state.py", "views.py", "ticket_channel_lifecycle.py")}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    artifact.chmod(0o600)
    print(json.dumps(report["scenarios"], indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/latency-e2e.json"))
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
