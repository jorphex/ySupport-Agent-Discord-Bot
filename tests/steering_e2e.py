"""Controller + stdio child-process steering checks; no live platform messages.

Failure cases are recorded in PLAN.md before implementation. The JSON artifact
contains assertions and source hashes so the checks can be repeated.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.optimization_e2e import eventually
from ysupport import config
from ysupport.bot import state
from tests.test_ticket_intake import _FakeDiscordChannel
from ysupport.execution.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ysupport.execution.executor import TransportTicketInvestigationExecutor
from ysupport.execution.json_endpoint import JsonEndpointTicketExecutionTransport
from ysupport.bot.views import stop_ticket_for_manual_support, take_over_ticket_for_staff
from ysupport.execution.subprocess_utils import write_execution_artifacts
from ysupport.bot.client import TicketBot

SERVER = r"""import json,os,sys,threading,time,uuid
from pathlib import Path
root=Path(sys.argv[1]); run=Path(os.environ['TICKET_EXECUTION_RUN_DIR'])
request=json.loads((run/'support_request.json').read_text()); key=request['current_user_message']; channel=str(request['channel_id'])
mode=(root/'mode').read_text(); lock=threading.Lock(); thread=str(uuid.uuid4()); turn='turn-'+str(uuid.uuid4()); updates=[]
start={'key':key,'pid':os.getpid(),'run':str(run),'method':None}
def send(obj):
 with lock: print(json.dumps(obj),flush=True)
def notify(method,params): send({'method':method,'params':dict(threadId=thread,**params)})
def finish():
 while key=='withdraw' and not (root/(channel+'.release')).exists(): time.sleep(.01)
 if mode in ('crash','crash-unobserved') and key=='withdraw': os._exit(2)
 answer=('Outdated' if key=='withdraw' and not updates else 'Updated')+': '+key+' '+ ' '.join(updates)
 result={'answer':answer,'requires_human_handoff':mode=='handoff','handoff_reason':'Needs team review' if mode=='handoff' else None,'handoff_kind':'human_decision' if mode=='handoff' else None,'evidence_summary':'isolated steering process','used_tools':[]}
 notify('item/completed',{'turnId':turn,'item':{'type':'agentMessage','id':'answer','phase':'final_answer','text':json.dumps(result)}})
 notify('turn/completed',{'turn':{'id':turn,'status':'completed','items':[]}})
for line in sys.stdin:
 msg=json.loads(line); method=msg.get('method'); params=msg.get('params',{}); ident=msg.get('id')
 if method=='initialize': send({'id':ident,'result':{'userAgent':'fixture'}})
 elif method in ('thread/start','thread/resume'):
  assert params['cwd']==str(run),params
  start['method']=method
  send({'id':ident,'result':{'thread':{'id':thread}}})
 elif method=='turn/start':
  (root/(channel+'.'+str(uuid.uuid4())+'.started')).write_text(json.dumps(start))
  send({'id':ident,'result':{'turn':{'id':turn,'status':'inProgress'}}})
  notify('item/completed',{'turnId':turn,'item':{'type':'agentMessage','id':'commentary','phase':'commentary','text':'UNVERIFIED INTERNAL DRAFT'}})
  notify('item/started',{'turnId':turn,'item':{'type':'mcpToolCall','id':'tool','server':'ysupport','tool':'search_documentation'}})
  threading.Thread(target=finish,daemon=True).start()
 elif method=='turn/steer':
  assert params['expectedTurnId']==turn
  record={'params':params,'run':str(run)}
  (root/(channel+'.'+str(uuid.uuid4())+'.steer')).write_text(json.dumps(record))
  if mode=='reject': send({'id':ident,'error':{'code':-32600,'message':'No active turn'}}); continue
  if mode in ('noack','crash-unobserved'): continue
  if mode!='unobserved':
   updates.extend(item.get('text','[native image]') for item in params['input'])
   notify('item/started',{'turnId':turn,'item':{'type':'userMessage','id':'user-'+str(ident),'clientId':params['clientUserMessageId'],'content':params['input']}})
  if mode!='observed-noack': send({'id':ident,'result':{'turnId':turn}})
"""


async def scenario(root, mode):
    root.mkdir()
    (root / "mode").write_text(mode)
    model = root / "server.py"
    model.write_text(SERVER)
    endpoint = CodexSupportTicketExecutionJsonEndpoint(
        codex_command=[sys.executable, str(model), str(root), "exec"],
        app_server_command=[sys.executable, str(model), str(root), "app-server"],
        codex_home=root / "home",
        session_dir=root / "sessions",
        session_max_age_hours=168,
        env={},
        cwd=str(root),
        timeout_seconds=6,
        max_concurrent_turns=2,
        artifact_dir=str(root / "artifacts"),
    )
    bot = TicketBot(intents=__import__("discord").Intents.none())
    bot.investigation_executor = TransportTicketInvestigationExecutor(
        JsonEndpointTicketExecutionTransport(endpoint)
    )
    channels = {i: _FakeDiscordChannel(i) for i in (7901, 7902)}
    for channel in channels.values():
        channel.category = SimpleNamespace(id=1)
    bot.get_channel = channels.get
    owner = SimpleNamespace(id=777, name="owner", bot=False)

    def message(channel, text, ident, image=False):
        attachments = (
            [
                SimpleNamespace(
                    id=ident,
                    filename=f"{ident}.png",
                    url=f"https://cdn.discordapp.com/attachments/1/2/{ident}.png",
                    content_type="image/png",
                    size=70,
                )
            ]
            if image
            else []
        )
        return SimpleNamespace(
            id=ident,
            author=owner,
            content=text,
            channel=channels[channel],
            reference=None,
            attachments=attachments,
        )

    downloads = []

    async def download(*, attachment, attachments_dir, index):
        target = attachments_dir / f"{index}.png"
        target.write_bytes(b"isolated fixture image")
        downloads.append(str(target))
        return target

    def records(suffix, channel=7901):
        return [json.loads(p.read_text()) for p in root.glob(f"{channel}.*.{suffix}")]

    async def boundary(text):
        return (
            {
                "tripwire_triggered": True,
                "message": "Boundary reply",
                "classification": "non_support_assistant",
            }
            if text.startswith("offtopic")
            else {"tripwire_triggered": False}
        )

    def write_artifacts(run_dir, **kwargs):
        request = json.loads((run_dir / "support_request.json").read_text())
        if mode == "stop-artifact-failure" and request["channel_id"] == 7901:
            raise OSError("controlled artifact write failure during Stop")
        write_execution_artifacts(run_dir, **kwargs)

    old_wallet = "0x" + "1" * 40
    new_wallet = "0x" + "2" * 40
    if mode == "wallet":
        state.last_wallet_by_channel[7901] = old_wallet
        state.get_or_create_ticket_investigation_job(7901).remember_wallet(old_wallet)
        state.pending_wallet_confirmation_by_channel[7901] = new_wallet
    try:
        with (
            patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
            patch.object(config, "COOLDOWN_SECONDS", 0.01),
            patch('ysupport.bot.client.discord.TextChannel', _FakeDiscordChannel),
            patch('ysupport.bot.client._outer_support_boundary_result', new=boundary),
            patch(
                'ysupport.execution.codex_support_app_server.write_execution_artifacts',
                new=write_artifacts,
            ),
            patch(
                'ysupport.bot.client._send_ticket_handoff_notice',
                new=AsyncMock(
                    return_value=state.TeamHandoffNotice(
                        telegram_chat_id="fixture",
                        telegram_message_id=1,
                        reason="Needs team review",
                    )
                ),
            ) as handoff,
            patch(
                'ysupport.tools.yearn_targets.resolve_yearn_address_target',
                new=AsyncMock(return_value=SimpleNamespace(kind="eoa_or_missing_code")),
            ),
            patch(
                'ysupport.execution.codex_support_attachments._download_attachment_image',
                new=download,
            ),
        ):
            for channel in channels:
                state.remember_ticket_owner_user_id(channel, owner.id)
            await bot.on_message(message(7901, "withdraw", 1, image=True))
            await eventually(lambda: bool(records("started")))
            await bot.on_message(
                message(
                    7901,
                    "offtopic"
                    if mode == "boundary"
                    else "yes"
                    if mode == "wallet"
                    else "Actually on katana",
                    2,
                )
            )
            await bot.on_message(message(7901, "", 3, image=True))
            if mode == "boundary":
                await asyncio.sleep(0.1)
                assert not records("steer")
            else:
                await eventually(lambda: bool(records("steer")))
            if mode == "normal":
                await bot.on_message(message(7901, "Actually on katana", 2))
            if mode in ("normal", "wallet", "handoff"):
                await eventually(lambda: len(records("steer")) == 2)
                await asyncio.sleep(0.05)
            if mode in ("stop", "staff", "delete", "stop-artifact-failure"):
                await bot.on_message(message(7902, "peer request", 4))
                await eventually(lambda: bool(records("started", 7902)))
                before_stop = list(channels[7901].sent_messages)
                if mode == "staff":
                    await take_over_ticket_for_staff(7901)
                elif mode == "delete":
                    from ysupport.bot.ticket_channel_lifecycle import clear_deleted_ticket_channel

                    await clear_deleted_ticket_channel(channels[7901])
                else:
                    await stop_ticket_for_manual_support(7901)
                await eventually(lambda: 7901 not in state.pending_tasks)
                assert not Path("/proc/" + str(records("started")[0]["pid"])).exists()
                assert channels[7901].sent_messages == before_stop
                assert not any(
                    t.startswith(("Updated:", "Outdated:"))
                    for t in channels[7901].sent_messages
                )
                await eventually(lambda: 7902 not in state.pending_tasks)
                assert (
                    channels[7902].sent_messages[-1].startswith("Updated: peer request")
                )
                return {"owned_process_stopped": True, "peer_completed": True}
            stored = json.loads((state._TICKET_STATE_DIR / "7901.json").read_text())
            if mode == "normal":
                import os
                import subprocess

                recovery_script = "import json;from ysupport.bot import state;state.hydrate_ticket_state(7901);print(json.dumps({'text':state.pending_messages[7901],'images':state.pending_attachments_by_channel[7901]}))"
                recovered = json.loads(
                    subprocess.check_output(
                        [sys.executable, "-B", "-c", recovery_script],
                        env=dict(os.environ),
                        text=True,
                    )
                )
                assert recovered["text"] == stored["pending_ticket_input"]["text"]
                assert len(recovered["images"]) == 2
            assert "Actually on katana" in stored["pending_ticket_input"][
                "text"
            ] or mode in ("boundary", "wallet")
            (root / "7901.release").touch()
            await eventually(lambda: 7901 not in state.pending_tasks, timeout=12)
            output = channels[7901].sent_messages
            assert not any("UNVERIFIED INTERNAL DRAFT" in t for t in output), output
            assert not any(t.startswith("Outdated:") for t in output), output
            starts = records("started")
            if mode == "normal":
                assert len(starts) == 1, starts
                assert any("Checking Yearn docs" in text for text in output)
                assert state.ticket_investigation_jobs[7901].evidence.chain == "katana"
                assert sum(t.startswith("Updated:") for t in output) == 1, output
                assert (
                    "Actually on katana" in output[-1]
                    and "[native image]" in output[-1]
                )
                assert len(downloads) == 2 and len(set(downloads)) == 2, downloads
                history = state.conversation_threads[7901]
                assert (
                    sum(
                        "Actually on katana" in str(h.get("content"))
                        for h in history
                        if h["role"] == "user"
                    )
                    == 1
                )
                inputs = [
                    item for r in records("steer") for item in r["params"]["input"]
                ]
                assert any(item["type"] == "localImage" for item in inputs)
            elif mode == "handoff":
                assert len(starts) == 1
                handoff.assert_awaited_once()
                assert "Actually on katana" in handoff.call_args.kwargs["summary"]
                assert (
                    handoff.call_args.kwargs["investigation_job"].evidence.chain
                    == "katana"
                )
                assert "Actually on katana" in str(state.conversation_threads[7901])
            elif mode == "wallet":
                assert len(starts) == 1
                assert (
                    state.ticket_investigation_jobs[7901].evidence.wallet == new_wallet
                )
                assert state.last_wallet_by_channel[7901] == new_wallet
                assert any(
                    new_wallet in str(r["params"]["input"]) for r in records("steer")
                )
            elif mode == "observed-noack":
                assert len(starts) == 2, (
                    starts
                )  # later image stayed queued behind ambiguous ACK
                assert len(records("steer")) == 1
                assert "Actually on katana" in str(state.conversation_threads[7901])
            elif mode == "boundary":
                assert output[-1] == "Boundary reply", output
            else:
                assert len(starts) == 2, starts
                assert "Actually on katana" in output[-1]
                assert sum(t.startswith("Updated:") for t in output) == 1, output
            assert (
                json.loads((state._TICKET_STATE_DIR / "7901.json").read_text()).get(
                    "pending_ticket_input"
                )
                is None
            )
            return {
                "same_turn": mode == "normal",
                "checked_queue_fallback": mode != "normal",
                "no_stale_reply": True,
                "native_images": mode == "normal",
                "history_and_evidence": True,
                "unfinished_input_persisted": True,
            }
    finally:
        for channel in channels:
            await state.cancel_pending_ticket_task(channel)
            state.clear_ticket_channel_state(channel, delete_persisted=True)
        await bot.close()


async def verify_intake_race(root, mode):
    """Race real controller admission/preparation against native completion."""
    root.mkdir()
    (root / "mode").write_text(
        {
            "rejected": "reject",
            "unobserved": "unobserved",
            "failed_observed": "crash",
            "failed_unobserved": "crash-unobserved",
        }.get(mode, "normal")
    )
    model = root / "server.py"
    model.write_text(SERVER)
    endpoint = CodexSupportTicketExecutionJsonEndpoint(
        codex_command=[sys.executable, str(model), str(root), "exec"],
        app_server_command=[sys.executable, str(model), str(root), "app-server"],
        codex_home=root / "home",
        session_dir=root / "sessions",
        env={},
        timeout_seconds=8,
        artifact_dir=str(root / "artifacts"),
    )
    bot = TicketBot(intents=__import__("discord").Intents.none())
    bot.investigation_executor = TransportTicketInvestigationExecutor(
        JsonEndpointTicketExecutionTransport(endpoint)
    )
    channel = _FakeDiscordChannel(7981)
    channel.category = SimpleNamespace(id=1)
    bot.get_channel = lambda _: channel
    owner = SimpleNamespace(id=777, name="owner", bot=False)

    def message(text, ident, image=False):
        attachments = (
            [
                SimpleNamespace(
                    id=ident,
                    filename=f"{ident}.png",
                    url=f"https://cdn.discordapp.com/attachments/1/2/{ident}.png",
                    content_type="image/png",
                    size=70,
                )
            ]
            if image
            else []
        )
        return SimpleNamespace(
            id=ident,
            author=owner,
            content=text,
            channel=channel,
            reference=None,
            attachments=attachments,
        )

    old_wallet, new_wallet = "0x" + "1" * 40, "0x" + "2" * 40
    state.last_wallet_by_channel[channel.id] = old_wallet
    state.pending_wallet_confirmation_by_channel[channel.id] = new_wallet
    state.get_or_create_ticket_investigation_job(channel.id).remember_wallet(old_wallet)
    state.remember_ticket_owner_user_id(channel.id, owner.id)
    preparing = asyncio.Event()
    downloads = 0

    async def download(*, attachment, attachments_dir, index):
        nonlocal downloads
        downloads += 1
        if downloads == 1 and mode in ("preparation", "partial", "stop"):
            preparing.set()
            await (
                asyncio.Event().wait()
            )  # Native completion or Stop cancels this download.
        path = attachments_dir / "fixture.png"
        path.write_bytes(b"controlled screenshot")
        return path

    try:
        with (
            patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
            patch.object(config, "COOLDOWN_SECONDS", 0.01),
            patch('ysupport.bot.client.discord.TextChannel', _FakeDiscordChannel),
            patch(
                'ysupport.bot.client._outer_support_boundary_result',
                new=AsyncMock(return_value={"tripwire_triggered": False}),
            ),
            patch(
                'ysupport.execution.codex_support_attachments._download_attachment_image',
                new=download,
            ),
        ):
            await bot.on_message(message("withdraw", 1))
            await eventually(lambda: bool(list(root.glob("7981.*.started"))))
            if mode == "partial":
                await bot.on_message(message("Actually on katana", 2))
                await eventually(
                    lambda: bool(bot._ticket_steering[channel.id][1].consumed)
                )
            await bot.on_message(message("yes", 3, image=True))
            if mode in ("preparation", "partial", "stop"):
                await eventually(preparing.is_set)
            else:
                await eventually(lambda: bool(list(root.glob("7981.*.steer"))))
                if mode == "failed_observed":
                    await eventually(
                        lambda: bool(bot._ticket_steering[channel.id][1].consumed)
                    )
                await asyncio.sleep(0.03)
            before_observation = (
                state.last_wallet_by_channel[channel.id],
                state.pending_wallet_confirmation_by_channel.get(channel.id),
            )
            if mode == "stop":
                await stop_ticket_for_manual_support(channel.id)
                assert channel.id in state.stopped_channels
                assert channel.id not in state.last_wallet_by_channel
                assert channel.id not in state.pending_wallet_confirmation_by_channel
                assert channel.id not in state.pending_tasks
                return {"stop_clears_staged_input": True}
            (root / "7981.release").touch()
            await eventually(lambda: channel.id not in state.pending_tasks, timeout=12)
            expected_before = (
                (new_wallet, None)
                if mode == "failed_observed"
                else (old_wallet, new_wallet)
            )
            assert before_observation == expected_before, (
                mode,
                before_observation,
                expected_before,
            )
            assert (
                state.ticket_investigation_jobs[channel.id].evidence.wallet
                == new_wallet
            )
            assert state.last_wallet_by_channel[channel.id] == new_wallet
            assert channel.id not in state.pending_wallet_confirmation_by_channel
            if mode == "partial":
                assert (
                    state.ticket_investigation_jobs[channel.id].evidence.chain
                    == "katana"
                )
            requests = [
                json.loads(path.read_text())
                for path in (root / "artifacts").rglob("support_request.json")
            ]
            assert len(requests) == 2, requests
            retry = next(
                request
                for request in requests
                if request["current_user_message"] != "withdraw"
            )
            assert retry["evidence"]["wallet"] == new_wallet, retry["evidence"]
            assert (
                sum(text.startswith("Updated:") for text in channel.sent_messages) == 1
            )
            assert not any(
                text.startswith("Outdated:") for text in channel.sent_messages
            )
            return {
                "unconsumed_intake_unchanged": mode != "failed_observed",
                "observed_intake_survives_failure": mode == "failed_observed",
                "retry_uses_confirmed_wallet": True,
                "single_reply": True,
            }
    finally:
        await state.cancel_pending_ticket_task(channel.id)
        state.clear_ticket_channel_state(channel.id, delete_persisted=True)
        await bot.close()


async def verify_reply_recovery(root, mode):
    """Restart after native completion while progress deletion still owns delivery."""
    import os
    import subprocess

    root.mkdir()
    (root / "mode").write_text("normal")
    model = root / "server.py"
    model.write_text(SERVER)
    endpoint = CodexSupportTicketExecutionJsonEndpoint(
        codex_command=[sys.executable, str(model), str(root), "exec"],
        app_server_command=[sys.executable, str(model), str(root), "app-server"],
        codex_home=root / "home",
        session_dir=root / "sessions",
        env={},
        timeout_seconds=8,
    )
    bot = TicketBot(intents=__import__("discord").Intents.none())
    bot.investigation_executor = TransportTicketInvestigationExecutor(
        JsonEndpointTicketExecutionTransport(endpoint)
    )
    channel = _FakeDiscordChannel(7982)
    channel.category = SimpleNamespace(id=1)
    bot.get_channel = lambda _: channel
    owner = SimpleNamespace(id=777, name="owner", bot=False)

    def message(text, ident, image=False):
        attachments = (
            [
                SimpleNamespace(
                    id=ident,
                    filename=f"{ident}.png",
                    url=f"https://cdn.discordapp.com/attachments/1/2/{ident}.png",
                    content_type="image/png",
                    size=70,
                )
            ]
            if image
            else []
        )
        return SimpleNamespace(
            id=ident,
            author=owner,
            content=text,
            channel=channel,
            reference=None,
            attachments=attachments,
        )

    async def download(*, attachment, attachments_dir, index):
        path = attachments_dir / "fixture.png"
        path.write_bytes(b"controlled screenshot")
        return path

    deleting, release = asyncio.Event(), asyncio.Event()
    close_count = 0

    async def close_progress(_):
        nonlocal close_count
        close_count += 1
        if close_count == 1:
            deleting.set()
            await release.wait()

    try:
        with (
            patch.object(config, "CATEGORY_CONTEXT_MAP", {1: "yearn"}),
            patch.object(config, "COOLDOWN_SECONDS", 0.01),
            patch('ysupport.bot.client.discord.TextChannel', _FakeDiscordChannel),
            patch(
                'ysupport.bot.client._outer_support_boundary_result',
                new=AsyncMock(return_value={"tripwire_triggered": False}),
            ),
            patch('ysupport.bot.client._DiscordProgressReporter.close', new=close_progress),
            patch(
                'ysupport.execution.codex_support_attachments._download_attachment_image',
                new=download,
            ),
        ):
            if mode == "wallet_shutdown":
                old_wallet, new_wallet = "0x" + "1" * 40, "0x" + "2" * 40
                state.last_wallet_by_channel[channel.id] = old_wallet
                state.pending_wallet_confirmation_by_channel[channel.id] = new_wallet
                state.get_or_create_ticket_investigation_job(
                    channel.id
                ).remember_wallet(old_wallet)
            state.remember_ticket_owner_user_id(channel.id, owner.id)
            await bot.on_message(message("withdraw", 1, image=True))
            await eventually(lambda: bool(list(root.glob("7982.*.started"))))
            followup = "yes" if mode == "wallet_shutdown" else "Actually on katana"
            await bot.on_message(message(followup, 2, image=True))
            await eventually(lambda: bool(bot._ticket_steering[channel.id][1].consumed))
            (root / "7982.release").touch()
            await eventually(deleting.is_set)
            await bot.on_message(message("latest detail", 3))
            snapshot = json.loads((state._TICKET_STATE_DIR / "7982.json").read_text())
            if mode in ("shutdown", "wallet_shutdown"):
                await bot.close()
                script = "import json;from ysupport.bot import state;state.hydrate_ticket_state(7982);print(json.dumps({'pending':state.pending_messages.get(7982),'images':state.pending_attachments_by_channel.get(7982),'history':state.conversation_threads.get(7982),'wallet':state.get_or_create_ticket_investigation_job(7982).evidence.wallet,'last_wallet':state.last_wallet_by_channel.get(7982)}))"
                recovered = json.loads(
                    subprocess.check_output(
                        [sys.executable, "-B", "-c", script],
                        env=dict(os.environ),
                        text=True,
                    )
                )
                assert (
                    recovered["pending"]
                    == f"withdraw\n{followup}\nlatest detail"
                ), recovered
                assert {item["attachment_id"] for item in recovered["images"]} == {1, 2}
                assert recovered["history"] == [], recovered
                if mode == "wallet_shutdown":
                    assert recovered["wallet"] == new_wallet
                    assert recovered["last_wallet"] == new_wallet
                assert not any(
                    text.startswith(("Updated:", "Outdated:"))
                    for text in channel.sent_messages
                )
            elif mode == "stop":
                await stop_ticket_for_manual_support(channel.id)
                stopped = json.loads(
                    (state._TICKET_STATE_DIR / "7982.json").read_text()
                )
                assert stopped["pending_ticket_input"] is None
                assert stopped["stopped"]
            elif mode == "delete":
                from ysupport.bot.ticket_channel_lifecycle import clear_deleted_ticket_channel

                await clear_deleted_ticket_channel(channel)
                assert not (state._TICKET_STATE_DIR / "7982.json").exists()
            else:
                release.set()
                await eventually(lambda: channel.id not in state.pending_tasks)
                committed = json.loads(
                    (state._TICKET_STATE_DIR / "7982.json").read_text()
                )
                assert committed["pending_ticket_input"] is None
                user_rows = [
                    row["content"]
                    for row in committed["history"]
                    if row.get("role") == "user"
                ]
                assert user_rows.count("withdraw\nActually on katana") == 1, user_rows
                assert user_rows.count("latest detail") == 1, user_rows
                assert (
                    sum(text.startswith("Updated:") for text in channel.sent_messages)
                    == 1
                )
            return {
                "mode": mode,
                "input_retained_until_commit": True,
                "images_preserved": True,
                "terminal_cleanup": mode in ("stop", "delete"),
                "saved_input_before_transition": snapshot.get("pending_ticket_input")
                is not None,
            }
    finally:
        release.set()
        await state.cancel_pending_ticket_task(channel.id)
        state.clear_ticket_channel_state(channel.id, delete_persisted=True)
        await bot.close()


async def main(artifact):
    report = {
        "ok": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": {},
    }
    with tempfile.TemporaryDirectory(prefix="ysupport-steering-e2e-") as temp:
        for mode in (
            "normal",
            "wallet",
            "handoff",
            "reject",
            "unobserved",
            "noack",
            "observed-noack",
            "crash",
            "boundary",
            "stop",
            "stop-artifact-failure",
            "staff",
            "delete",
        ):
            try:
                report["scenarios"][mode] = {
                    "ok": True,
                    **await scenario(Path(temp) / mode, mode),
                }
            except Exception as exc:
                report["scenarios"][mode] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        for mode in (
            "preparation",
            "partial",
            "rejected",
            "unobserved",
            "failed_observed",
            "failed_unobserved",
            "stop",
        ):
            name = "intake_" + mode
            try:
                report["scenarios"][name] = {
                    "ok": True,
                    **await verify_intake_race(Path(temp) / name, mode),
                }
            except Exception as exc:
                report["scenarios"][name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        for mode in ("shutdown", "wallet_shutdown", "commit", "stop", "delete"):
            name = "recovery_" + mode
            try:
                report["scenarios"][name] = {
                    "ok": True,
                    **await verify_reply_recovery(Path(temp) / name, mode),
                }
            except Exception as exc:
                report["scenarios"][name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    report["ok"] = all(s["ok"] for s in report["scenarios"].values())
    names = [
        'ysupport/bot/client.py',
        'ysupport/bot/state.py',
        'ysupport/bot/discord_support_runtime.py',
        'ysupport/bot/ticket_intake.py',
        'ysupport/execution/executor.py',
        'ysupport/execution/codex_support_endpoint.py',
        'ysupport/execution/codex_support_app_server.py',
        'ysupport/execution/steering.py',
        "tests/steering_e2e.py",
    ]
    report["source_sha256"] = {
        n: hashlib.sha256(Path(n).read_bytes()).hexdigest()
        for n in names
        if Path(n).exists()
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    artifact.chmod(0o600)
    print(json.dumps(report["scenarios"], indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=Path(".cache/verification/steering-e2e.json")
    )
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
