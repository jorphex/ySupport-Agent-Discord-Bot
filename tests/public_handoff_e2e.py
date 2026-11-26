"""Public trigger -> durable handoff -> Telegram reply -> original Discord reply.

Real controllers, native JSON subprocesses, session files and local Telegram HTTP;
only Discord and model/provider responses are fixtures. No live messages or models.
Covers delivery, persistence, restart, deletion, cancellation and context races.
"""

import argparse
import asyncio
import base64
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from discord.http import handle_message_parameters

from tests.telegram_http_e2e import TelegramHttpFixture
from tests.optimization_e2e import eventually
from tests.test_ticket_intake import _FakeDiscordChannel
from ysupport import config
from ysupport.bot import state
from ysupport.bot.client import TicketBot
from ysupport.execution.codex_support_endpoint import CodexSupportTicketExecutionJsonEndpoint
from ysupport.execution.executor import TransportTicketInvestigationExecutor
from ysupport.execution.json_endpoint import JsonEndpointTicketExecutionTransport
from ysupport.execution import codex_support_attachments


MODEL = '''import json,os,sys,uuid,time
from pathlib import Path
root=Path(sys.argv[1]); run=Path(os.environ["TICKET_EXECUTION_RUN_DIR"])
r=json.loads((run/"support_request.json").read_text()); key=str(uuid.uuid4())
r["fixture_argv"]=sys.argv
for a in r["attachments"]:
 if a.get("is_image"): assert Path(a["local_path"]).read_bytes().startswith(b"\\x89PNG")
(root/(key+".request.json")).write_text(json.dumps(r))
internal=r["current_turn_source"]=="internal_team"
text=r["current_user_message"]
if internal and text=="FAIL": raise RuntimeError("controlled model failure")
if internal and text=="BLOCK":
 (root/"blocked").touch()
 while not (root/"release").exists(): time.sleep(.01)
answer=("[NO_PUBLIC_UPDATE]" if text=="SKIP" else "Update: "+text) if internal else "The vault needs a keeper review; the cause is unknown."
if internal and text=="LONG": answer="x"*2500
result={"answer":answer,"requires_human_handoff":not internal,"handoff_kind":None if internal else "manual_strategy_action","handoff_reason":None if internal else "Keeper review is required.","evidence_summary":"Frozen public concern and contributor update.","used_tools":[]}
print(json.dumps({"type":"thread.started","thread_id":str(uuid.uuid4())}))
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":json.dumps(result)}}))
'''


class Message:
    def __init__(self, channel, ident, author_id, text, *, reply_to=None, staff=False):
        self.channel, self.id, self.content = channel, ident, text
        self.author = SimpleNamespace(id=author_id, name=str(author_id), bot=author_id == 999,
            roles=[SimpleNamespace(id=config.TICKET_CONTRIBUTOR_ROLE_ID)] if staff else [])
        self.reference = SimpleNamespace(message_id=reply_to) if reply_to else None
        self.attachments, self.nonce = [], None
        self.created_at = datetime.now(timezone.utc)

    async def reply(self, text, **kwargs):
        if not any(message.id == self.id for message in self.channel.messages):
            raise __import__('discord').NotFound(SimpleNamespace(status=404, reason='Not found'), 'Deleted fixture')
        self.channel.post_started.set()
        if self.channel.post_gate is not None and not self.channel.hold_after_post:
            await self.channel.post_gate.wait()
        # Model Discord's nonce enforcement using the installed SDK's real payload.
        with handle_message_parameters(content=text, nonce=kwargs.get('nonce')) as params:
            payload = params.payload
        if payload.get('nonce') is not None:
            self.channel.post_payloads.append(payload)
            previous = next((m for m in self.channel.answers if m.nonce == payload['nonce']), None)
            if payload.get('enforce_nonce') and previous is not None:
                return previous
        result = Message(self.channel, max([10000, *(m.id for m in self.channel.messages)]) + 1, 999, text, reply_to=self.id)
        result.nonce = kwargs.get("nonce")
        self.channel.messages.append(result)
        self.channel.answers.append(result)
        self.channel.session_observations.extend(json.loads(p.read_text())['conversation_key']
            for p in self.channel.session_root.glob('*.json'))
        if self.channel.post_gate is not None and self.channel.hold_after_post:
            await self.channel.post_gate.wait()
        return result


class Channel(_FakeDiscordChannel):
    def __init__(self):
        super().__init__(8100)
        self.name, self.guild = "vaults", SimpleNamespace(id=900, name="DO NOT SHOW SERVER")
        self.messages, self.answers = [], []
        self.post_started, self.post_gate = asyncio.Event(), None
        self.hold_after_post, self.session_observations = False, []
        self.post_payloads = []

    async def send(self, text, **kwargs):
        self.sent_messages.append(text)
        reference = kwargs.get('reference')
        if reference is not None and reference.fail_if_not_exists and not any(m.id == reference.message_id for m in self.messages):
            raise __import__('discord').NotFound(SimpleNamespace(status=404, reason='Not found'), 'Deleted fixture')
        message = Message(self, max([10000, *(m.id for m in self.messages)]) + 1, 999, text,
            reply_to=reference.message_id if reference else None)
        self.messages.append(message)
        return message

    async def fetch_message(self, ident):
        import discord
        for m in self.messages:
            if m.id == ident:
                return m
        raise discord.NotFound(SimpleNamespace(status=404, reason="Not found"), "Deleted fixture")

    def history(self, *, after=None, limit=50, oldest_first=None):
        async def rows():
            items = [m for m in self.messages if after is None or m.id > after.id]
            ascending = after is not None if oldest_first is None else oldest_first
            for m in sorted(items, key=lambda m: m.id, reverse=not ascending)[:limit]:
                fetched = copy.copy(m)
                fetched.nonce = None  # REST fetches do not retain MESSAGE_CREATE nonces.
                yield fetched
        return rows()


class Workflow:
    def __init__(self, root, *, held_method=None):
        self.root, self.channel, self.bots = root, Channel(), []
        self.channel.session_root = root / 'sessions'
        self.http = TelegramHttpFixture(held_method=held_method)
        (root / "model.py").write_text(MODEL)

    def __enter__(self):
        self.patches = ExitStack()
        self.patches.enter_context(self.http)
        self.patches.enter_context(patch.object(config, "TRIGGER_CONTEXT_MAP", {"y": "yearn"}))
        self.patches.enter_context(patch.object(config, "PUBLIC_TRIGGER_USER_IDS", {"42"}))
        self.patches.enter_context(patch.object(config, "TICKET_EXECUTION_MAX_CONCURRENT_TURNS", 2))
        self.patches.enter_context(patch.object(config, "TICKET_EXECUTION_CODEX_SESSION_DIR", str(self.root / "sessions")))
        self.patches.enter_context(patch('ysupport.bot.client._outer_support_boundary_result',
            return_value={"classification": "yearn_support", "tripwire_triggered": False}))
        self.patches.enter_context(patch('ysupport.bot.discord_support_runtime.summarize_handoff_summary', return_value=None))
        async def image(*, attachment, attachments_dir, index):
            codex_support_attachments._validate_attachment_image_source(attachment)
            path = attachments_dir / f"attachment_{index}.png"
            path.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBH0AAAAASUVORK5CYII='))
            return path
        self.patches.enter_context(patch.object(codex_support_attachments, '_download_attachment_image', new=image))
        return self

    def bot(self):
        bot = TicketBot(intents=__import__("discord").Intents.none())
        bot.get_channel = lambda ident: self.channel if ident == self.channel.id else None
        bot._connection.user = SimpleNamespace(id=999)
        endpoint = CodexSupportTicketExecutionJsonEndpoint(
            codex_command=[sys.executable, str(self.root / "model.py"), str(self.root), "exec"],
            codex_home=self.root / "home", session_dir=self.root / "sessions",
            env={}, max_concurrent_turns=2, timeout_seconds=15,
            ysupport_mcp_url="http://127.0.0.1:1/mcp", mcp_server_api_key="fixture")
        bot.investigation_executor = TransportTicketInvestigationExecutor(JsonEndpointTicketExecutionTransport(endpoint))
        self.bots.append(bot)
        return bot

    async def trigger(self, bot, ident=8200, user=43, *, wait_ready=True):
        source = next((m for m in self.channel.messages if m.id == ident), None)
        if source is None:
            source = Message(self.channel, ident, user, "Please check this vault's overdue harvest.")
            source.attachments = [SimpleNamespace(id=55, filename="observation.png", url="https://cdn.discordapp.com/attachments/8100/55/current.png",
                content_type="image/png", size=100)]
            self.channel.messages.append(source)
        trigger = Message(self.channel, ident + 1, 42, "y", reply_to=ident)
        async def delete():
            return None
        trigger.delete = delete
        await bot.on_message(trigger)
        if wait_ready and ident in bot.telegram_handoffs.public.records:
            await eventually(lambda: bot.telegram_handoffs.public.records.get(ident) is None
                or bot.telegram_handoffs.public.records[ident].status != 'activating')
        return source

    def requests(self):
        paths = sorted(self.root.glob("*.request.json"), key=lambda p: p.stat().st_mtime_ns)
        return [json.loads(p.read_text()) for p in paths]

    async def reply(self, bot, record, text="Keeper review is complete."):
        await bot.telegram_handoffs._handle_telegram_handoff_update({"message": {
            "message_id": 9900, "chat": {"id": 123}, "text": text,
            "reply_to_message": {"message_id": record.telegram_message_id}}})

    async def dismiss(self, bot, record):
        await bot.telegram_handoffs._handle_telegram_handoff_update({"callback_query": {
            "id": "fixture", "data": "dismiss_handoff", "message": {
                "message_id": record.telegram_message_id, "chat": {"id": 123}}}})

    async def close(self):
        for bot in self.bots:
            await bot.close()
        for user in (43, 44):
            state.clear_public_conversation(user)

    def __exit__(self, *_args):
        self.patches.close()


async def verify(root, mode):
    with Workflow(root, held_method='deleteMessage' if mode == 'slow_dismissal' else None) as w:
        bot = w.bot()
        try:
            if mode == 'snapshot_failure':
                with patch.object(bot.telegram_handoffs.public, '_save', return_value=False):
                    source = await w.trigger(bot)
                assert not bot.telegram_handoffs.public.records
                assert 43 in state.public_conversations
                assert "couldn't send" in w.channel.sent_messages[-1]
                assert not list(bot.telegram_handoffs.public.path(source.id).parent.glob('*.json'))
                return {'unsaved_snapshot_never_cleared_active_context': True}
            source = await w.trigger(bot)
            public = bot.telegram_handoffs.public
            record = public.records[source.id]
            path = public.path(source.id)
            saved = json.loads(path.read_text())
            assert saved['user_id'] == 43 and saved['channel_id'] == w.channel.id
            assert saved['message_id'] == source.id and saved['history'][-1]['role'] == 'assistant'
            assert saved['attachments'][0]['source_message_id'] == source.id
            assert 43 not in state.public_conversations
            assert not (state._PUBLIC_STATE_DIR / '43.json').exists()
            sends = [p for method,p in w.http.requests if method == 'sendMessage']
            assert 'Preparing handoff' in sends[0]['text'] and 'Public support · #vaults' in sends[0]['text']
            assert 'DO NOT SHOW SERVER' not in sends[0]['text']
            assert 'reply_markup' not in sends[0]
            assert any(method == 'editMessageText' and payload['text'] == record.message_text
                and payload['reply_markup']['inline_keyboard'] for method, payload in w.http.requests)
            if mode == 'deletion_events':
                other = await w.trigger(bot, 8300, user=44)
                await bot.on_raw_bulk_message_delete(SimpleNamespace(channel_id=w.channel.id, message_ids={source.id, other.id}))
                assert not public.records
                await w.trigger(bot, 8400)
                await w.trigger(bot, 8500, user=44)
                await bot.on_raw_thread_delete(SimpleNamespace(thread_id=w.channel.id))
                assert not public.records
                await w.trigger(bot, 8600)
                await w.trigger(bot, 8700, user=44)
                await bot.on_guild_channel_delete(w.channel)
                assert not public.records and not list(path.parent.glob('*.json'))
                assert w.channel.id not in state.stopped_channels and not w.channel.answers
                return {'bulk_thread_and_channel_deletion_cleaned_pending_context': True}
            if mode == 'slow_dismissal':
                await asyncio.wait_for(w.dismiss(bot, record), timeout=1)
                await eventually(w.http.entered.is_set)
                assert record.status == 'dismissed' and not record.history and not w.http.release.is_set()
                other = await w.trigger(bot, 8300, user=44)
                await w.reply(bot, public.records[other.id])
                await eventually(lambda: len(w.channel.answers) == 1)
                assert w.channel.answers[0].reference.message_id == other.id
                w.http.release.set()
                await eventually(lambda: not public.records)
                return {'dismissal_ack_prompt': True, 'independent_concern_delivered_during_cleanup': True}
            if mode in {'close_already_applied', 'dismiss_already_deleted'}:
                w.http.enforce_idempotence = True
                original_unlink = state._safe_unlink
                with patch.object(state, '_safe_unlink', side_effect=lambda p:
                        False if p == path else original_unlink(p)):
                    if mode == 'close_already_applied':
                        await w.reply(bot, record)
                    else:
                        await w.dismiss(bot, record)
                    await eventually(lambda: record.status in {'delivered_pending_close', 'dismissed'})
                    await eventually(lambda: record.delivery_nonce not in public.active)
                    assert path.exists()
                await bot.close()
                bot = w.bot()
                public = bot.telegram_handoffs.public
                public.hydrate()
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert source.id not in public.records and not path.exists(), 'Applied Telegram cleanup did not reconcile'
                assert len(w.channel.answers) == int(mode == 'close_already_applied')
                return {'already_applied_telegram_cleanup_reconciled': True}
            if mode == 'isolation':
                second = await w.trigger(bot, 8300, user=44)
                third = await w.trigger(bot, 8400, user=43)
                before = len(w.requests())
                await bot.on_message(Message(w.channel, 8500, 43, 'follow-up without trigger'))
                assert len(w.requests()) == before
                assert len(public.records) == 3
                await w.trigger(bot, source.id, user=43)
                assert len(w.requests()) == before and len(public.records) == 3
                await w.dismiss(bot, record)
                await eventually(lambda: source.id not in public.records)
                assert set(public.records) == {second.id, third.id}
                assert w.channel.id not in state.stopped_channels
                assert w.channel.id not in state.conversation_threads
                return {'three_concerns_isolated': True, 'dismissal_local': True, 'no_auto_reply': True}
            if mode in {'posting_recovered', 'posting_uncertain'}:
                w.channel.post_gate = asyncio.Event()
                w.channel.hold_after_post = mode == 'posting_recovered'
                await w.reply(bot, record)
                await eventually(w.channel.post_started.is_set)
                if w.channel.hold_after_post:
                    await eventually(lambda: len(w.channel.answers) == 1)
                assert json.loads(path.read_text())['status'] == 'posting'
                await bot.close()
                bot = w.bot()
                public = bot.telegram_handoffs.public
                public.hydrate()
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                if mode == 'posting_uncertain':
                    assert public.records[source.id].status == 'failed' and not w.channel.answers
                    await w.dismiss(bot, public.records[source.id])
                    await eventually(lambda: source.id not in public.records)
                assert source.id not in public.records and not path.exists()
                assert len(w.channel.answers) == int(mode == 'posting_recovered')
                assert len([r for r in w.requests() if r['current_turn_source'] == 'internal_team']) == 1
                return {'interrupted_send_never_blindly_reposted': True, 'confirmed_delivery': w.channel.hold_after_post}
            if mode == 'acceptance_failure':
                with patch.object(public, '_save', return_value=False):
                    try:
                        await w.reply(bot, record)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError('Unsaved acceptance must fail')
                assert record.status == 'open' and len(w.requests()) == 1
                assert json.loads(path.read_text())['status'] == 'open'
            if mode == 'restart':
                with patch.object(bot.telegram_handoffs, '_wake_delivery_worker'):
                    await w.reply(bot, record)
                assert json.loads(path.read_text())['status'] == 'pending_delivery'
                await bot.close()
                bot = w.bot()
                public = bot.telegram_handoffs.public
                public.hydrate()
                assert source.id in public.records
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
            elif mode in {'dismissal', 'generation_failure', 'deleted_source', 'long_answer', 'skip', 'linked_context', 'cancellation', 'close_failure', 'changed_during_synthesis', 'failure_notice_retry', 'busy_context'}:
                if mode == 'dismissal':
                    await w.dismiss(bot, record)
                    await eventually(lambda: source.id not in public.records)
                    assert not path.exists() and not w.channel.answers
                    assert w.channel.id not in state.stopped_channels
                    return {'dismissed_context_removed': True}
                if mode == 'linked_context':
                    w.channel.messages += [Message(w.channel, 8600, 44, 'UNRELATED CONCERN'),
                        Message(w.channel, 8601, 45, 'Keeper is checking this.', reply_to=source.id, staff=True),
                        Message(w.channel, 8602, 43, 'Here is more detail.', reply_to=8601, staff=True)]
                if mode == 'deleted_source':
                    w.channel.messages.remove(source)
                if mode == 'busy_context':
                    w.channel.messages.extend(Message(w.channel, 12000 + n, 43,
                        f'Related follow-up {n}', reply_to=source.id) for n in range(30))
                if mode in {'close_failure', 'failure_notice_retry'}:
                    w.http.fail_methods.add('editMessageText')
                text = {'generation_failure': 'FAIL', 'long_answer': 'LONG', 'skip': 'SKIP',
                        'cancellation': 'BLOCK', 'changed_during_synthesis': 'BLOCK',
                        'failure_notice_retry': 'FAIL'}.get(mode, 'Keeper review is complete.')
                await w.reply(bot, record, text)
                if mode == 'deleted_source':
                    await eventually(lambda: source.id not in public.records)
                    assert not path.exists() and not w.channel.answers
                    return {'deleted_source_context_removed': True}
                if mode == 'cancellation':
                    await eventually(lambda: (root / 'blocked').exists())
                    await bot.on_raw_message_delete(SimpleNamespace(message_id=source.id, channel_id=w.channel.id))
                    assert source.id not in public.records and not w.channel.answers and not path.exists()
                    return {'source_deletion_canceled_synthesis': True}
                if mode == 'changed_during_synthesis':
                    await eventually(lambda: (root / 'blocked').exists())
                    w.channel.messages.append(Message(w.channel, 8900, 45,
                        'A newer keeper update arrived.', reply_to=source.id, staff=True))
                    (root / 'release').touch()
                await eventually(lambda: record.status in {'failed', 'delivered_pending_close'} or source.id not in public.records)
                await eventually(lambda: record.delivery_nonce not in public.active)
                if mode == 'failure_notice_retry':
                    assert record.status == 'failed' and not record.failure_notice_shown
                    calls = len(w.requests())
                    w.http.fail_methods.clear()
                    await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                    assert record.failure_notice_shown and len(w.requests()) == calls
                if mode in {'generation_failure', 'deleted_source', 'long_answer', 'failure_notice_retry'}:
                    assert record.status == 'failed' and path.exists() and not w.channel.answers
                    assert any('Delivery' in p.get('text','') or 'delivery' in p.get('text','')
                        for method,p in w.http.requests if method == 'editMessageText')
                    await w.dismiss(bot, record)
                    await eventually(lambda: source.id not in public.records)
                    assert not path.exists()
                    return {'failed_safely_and_dismissible': True}
                if mode == 'close_failure':
                    assert len(w.channel.answers) == 1 and path.exists()
                    w.http.fail_methods.clear()
                    await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                if mode == 'skip':
                    await eventually(lambda: source.id not in public.records)
                    assert not w.channel.answers and not path.exists()
                    return {'obsolete_update_skipped': True}
            else:
                await w.reply(bot, record, 'ok')
                assert record.status == 'open'
                await w.reply(bot, record)
                await w.reply(bot, record, 'Duplicate must not win.')
            await eventually(lambda: source.id not in public.records)
            assert not path.exists() and len(w.channel.answers) == 1
            answer = w.channel.answers[0]
            assert answer.reference.message_id == source.id
            assert answer.content == ('Update: BLOCK' if mode == 'changed_during_synthesis'
                else 'Update: Keeper review is complete.')
            internal = [r for r in w.requests() if r['current_turn_source'] == 'internal_team']
            assert len(internal) == (2 if mode == 'changed_during_synthesis' else 1)
            assert internal[0]['channel_type'] == 'public'
            assert w.channel.session_observations == [f'public_handoff:{source.id}']
            if mode == 'changed_during_synthesis':
                assert 'A newer keeper update arrived.' in json.dumps(internal[-1]['recent_transcript'])
                assert 'resume' not in internal[-1]['fixture_argv']
            assert internal[0]['attachments'][0]['url'] == 'https://cdn.discordapp.com/attachments/8100/55/current.png'
            if mode == 'linked_context':
                text = json.dumps(internal[0]['recent_transcript'])
                assert 'UNRELATED CONCERN' not in text
                assert 'Keeper is checking this.' in text and 'Here is more detail.' in text
                assert 'Internal support staff' in text
                assert any(item['role'] == 'user' and 'Here is more detail.' in item['content']
                    for item in internal[0]['recent_transcript'])
            if mode == 'busy_context':
                transcript = json.dumps(internal[0]['recent_transcript'])
                assert source.content in transcript and 'the cause is unknown' in transcript
                assert 'Related follow-up 29' in transcript
            assert not list((root / 'sessions').glob('public_handoff_*.json'))
            assert w.channel.id not in state.conversation_threads and w.channel.id not in state.stopped_channels
            return {'one_original_message_reply': True, 'snapshot_removed': True, 'native_synthesis': True,
                'native_turns': [{key: row[key] for key in ('current_turn_source', 'current_user_message',
                    'channel_type', 'recent_transcript')} for row in internal],
                'session_keys_observed_at_post': w.channel.session_observations}
        finally:
            await w.close()


async def verify_review_fix(root, mode):
    from ysupport.bot import discord_support_runtime as runtime
    with Workflow(root) as w:
        bot = w.bot()
        public, controller = bot.telegram_handoffs.public, bot.telegram_handoffs
        tasks = []
        release = asyncio.Event()
        try:
            if mode == 'stale_worker_retrigger':
                first = await w.trigger(bot, 8200, user=43)
                second = await w.trigger(bot, 8300, user=44)
                old = public.records[second.id]
                with patch.object(controller, '_wake_delivery_worker'):
                    await w.reply(bot, public.records[first.id], 'BLOCK')
                    await w.reply(bot, old)
                slow = asyncio.create_task(public.resume())
                tasks.append(slow)
                await eventually(lambda: (root / 'blocked').exists())
                await asyncio.wait_for(public.resume(), 5)
                assert second.id not in public.records
                await w.trigger(bot, second.id, user=44)
                fresh = public.records[second.id]
                (root / 'release').touch()
                await asyncio.wait_for(slow, 5)
                assert public.records.get(second.id) is fresh, 'Stale worker erased the replacement handoff'
                assert json.loads(public.path(second.id).read_text())['telegram_message_id'] == fresh.telegram_message_id
                return {'replacement_record_and_disk_context_preserved': True}
            if mode in {'deleted_during_initial_turn', 'deleted_during_notice', 'early_reply'}:
                ready = asyncio.Event()
                if mode == 'deleted_during_initial_turn':
                    original = bot.investigation_executor.execute_turn
                    async def delayed(*args, **kwargs):
                        result = await original(*args, **kwargs)
                        ready.set()
                        await release.wait()
                        return result
                    target, name = bot.investigation_executor, 'execute_turn'
                else:
                    original = runtime.send_handoff_notice
                    async def delayed(*args, **kwargs):
                        result = await original(*args, **kwargs)
                        w.published_notice = result
                        ready.set()
                        await release.wait()
                        return result
                    target, name = runtime, 'send_handoff_notice'
                with patch.object(target, name, new=delayed):
                    task = asyncio.create_task(w.trigger(bot))
                    tasks.append(task)
                    await asyncio.wait_for(ready.wait(), 5)
                    if mode == 'early_reply':
                        notice = w.published_notice
                        await controller._handle_telegram_handoff_update({'message': {
                            'message_id': 9899, 'chat': {'id': 123}, 'text': 'Unrelated old alert reply.',
                            'reply_to_message': {'message_id': 8000, 'text': 'Alert only. Follow up in Discord.'}}})
                        w.http.updates = [{'update_id': 100, 'message': {
                            'message_id': 9900, 'chat': {'id': 123}, 'text': 'Keeper has fixed this.',
                            'reply_to_message': {'message_id': notice.message_id, 'text': notice.message_text}}}]
                        attempted = asyncio.Event()
                        handle = controller._handle_telegram_handoff_update
                        async def observed(update):
                            try:
                                return await handle(update)
                            finally:
                                attempted.set()
                        with patch.object(controller, '_handle_telegram_handoff_update', new=observed):
                            controller.task = asyncio.create_task(controller._telegram_handoff_reply_loop())
                            await asyncio.wait_for(attempted.wait(), 5)
                            assert controller._telegram_update_offset != 101, 'Early reply was acknowledged before registration'
                            release.set()
                            await asyncio.wait_for(task, 5)
                            await eventually(lambda: len(w.channel.answers) == 1 and 8200 not in public.records)
                        assert w.channel.answers[0].content == 'Update: Keeper has fixed this.'
                        return {'early_reply_retried_and_delivered': True}
                    w.channel.messages = [m for m in w.channel.messages if m.id != 8200]
                    await bot.on_raw_message_delete(SimpleNamespace(message_id=8200, channel_id=w.channel.id))
                    release.set()
                    await asyncio.wait_for(task, 5)
                assert 8200 not in public.records and not public.path(8200).exists(), 'Deleted source retained handoff context'
                if mode == 'deleted_during_initial_turn':
                    assert not any(method == 'sendMessage' for method, _ in w.http.requests)
                else:
                    assert any(method == 'deleteMessage' for method, _ in w.http.requests)
                assert not w.channel.answers
                return {'deleted_inflight_source_never_registered': True}
            if mode == 'activation_restart':
                w.http.fail_methods.add('editMessageText')
                source = await w.trigger(bot, wait_ready=False)
                await eventually(lambda: any(method == 'editMessageText' for method, _ in w.http.requests))
                await eventually(lambda: not public.active)
                assert json.loads(public.path(source.id).read_text())['status'] == 'activating'
                assert 'Preparing' in next(p['text'] for m, p in w.http.requests if m == 'sendMessage')
                await bot.close()
                w.http.fail_methods.clear()
                bot = w.bot()
                public, controller = bot.telegram_handoffs.public, bot.telegram_handoffs
                public.hydrate()
                await controller._resume_pending_telegram_handoff_replies()
                assert public.records[source.id].status == 'open'
                await w.reply(bot, public.records[source.id])
                await eventually(lambda: source.id not in public.records)
                assert len(w.channel.answers) == 1
                return {'durable_activation_recovered_after_restart': True}
            source = await w.trigger(bot)
            record = public.records[source.id]
            old_url = source.attachments[0].url
            new_url = 'https://cdn.discordapp.com/attachments/8100/66/replacement.png'
            def image(ident, url):
                return SimpleNamespace(id=ident, filename='observation.png', url=url, content_type='image/png', size=100)
            if mode == 'updated_source_image':
                source.attachments = [image(66, new_url)]
                await w.reply(bot, record)
            else:
                if mode == 'updated_linked_image':
                    linked = Message(w.channel, 12000, 43, 'Screenshot', reply_to=source.id)
                    linked.attachments = [image(66, new_url)]
                    w.channel.messages.append(linked)
                await w.reply(bot, record, 'BLOCK')
                await eventually(lambda: (root / 'blocked').exists())
                if mode == 'updated_linked_image':
                    new_url = 'https://cdn.discordapp.com/attachments/8100/67/newer.png'
                    linked.attachments = [image(67, new_url)]
                elif mode == 'signed_url_only':
                    source.attachments = [image(55, old_url + '?ex=new-signature')]
                else:
                    source.attachments = [image(66, new_url)]
                (root / 'release').touch()
            await eventually(lambda: source.id not in public.records)
            internal = [r for r in w.requests() if r['current_turn_source'] == 'internal_team']
            if mode == 'signed_url_only':
                assert len(internal) == 1, 'Signed URL refresh unnecessarily regenerated the draft'
            else:
                assert new_url in [a['url'] for a in internal[-1]['attachments']], 'Current screenshot missing from synthesis'
                assert len(internal) == (1 if mode == 'updated_source_image' else 2)
            assert len(w.channel.answers) == 1
            return {'current_screenshots_used': True, 'synthesis_turns': len(internal),
                'image_urls_by_turn': [[a['url'] for a in r['attachments']] for r in internal]}
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await w.close()


async def verify_delivery_fix(root, mode):
    with Workflow(root) as w:
        bot = w.bot()
        public, controller = bot.telegram_handoffs.public, bot.telegram_handoffs
        release, entered = asyncio.Event(), asyncio.Event()
        tasks = []
        try:
            if mode in {'announcement_order', 'announcement_restart', 'announcement_failure'}:
                execute = bot.investigation_executor.execute_turn
                async def progress(*args, **kwargs):
                    hooks = kwargs.get('hooks')
                    if hooks is not None:
                        await hooks.send_progress_update('Checking the vault.')
                    return await execute(*args, **kwargs)
                async def delayed_delete(_message):
                    entered.set()
                    await release.wait()
                send = w.channel.send
                async def delayed_send(text, **kwargs):
                    if kwargs.get('reference') is not None:
                        entered.set()
                        await release.wait()
                        raise RuntimeError('Controlled initial Discord send failure')
                    return await send(text, **kwargs)
                with ExitStack() as patches:
                    patches.enter_context(patch.object(bot.investigation_executor, 'execute_turn', new=progress))
                    if mode == 'announcement_failure':
                        patches.enter_context(patch.object(w.channel, 'send', new=delayed_send))
                    else:
                        patches.enter_context(patch.object(Message, 'delete', new=delayed_delete, create=True))
                    task = asyncio.create_task(w.trigger(bot, wait_ready=False))
                    tasks.append(task)
                    await asyncio.wait_for(entered.wait(), 5)
                    record = public.records[8200]
                    await w.reply(bot, record, 'Keeper has fixed the issue.')
                    assert record.status == 'pending_delivery'
                    assert json.loads(public.path(8200).read_text())['pending_reply_text'] == 'Keeper has fixed the issue.'
                    await eventually(lambda: not public.active)
                    await controller._resume_pending_telegram_handoff_replies()
                    assert not w.channel.answers, 'Team update overtook the unfinished initial announcement'
                    if mode == 'announcement_restart':
                        await bot.close()
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        bot = w.bot()
                        public, controller = bot.telegram_handoffs.public, bot.telegram_handoffs
                        public.hydrate()
                        await controller._resume_pending_telegram_handoff_replies()
                    else:
                        release.set()
                        await asyncio.wait_for(task, 5)
                    await eventually(lambda: 8200 not in public.records)
                replies = [m.content for m in w.channel.messages if m.author.bot and m.reference]
                assert replies[-1] == 'Update: Keeper has fixed the issue.'
                assert len([r for r in w.requests() if r['current_turn_source'] == 'internal_team']) == 1
                return {'early_reply_saved': True, 'public_replies_in_post_order': replies,
                    'restart_gate_released': mode == 'announcement_restart'}

            source = await w.trigger(bot)
            record = public.records[source.id]
            path = public.path(source.id)
            if mode == 'terminal_dismissal_overlap':
                from ysupport.integrations import handoff
                edit = handoff.edit_handoff_notice
                posting, canceled = asyncio.Event(), asyncio.Event()
                async def delayed_failure_notice(**kwargs):
                    if (kwargs['message_id'] == record.telegram_message_id
                        and 'Delivery needs manual review' in kwargs['message_text']):
                        posting.set()
                        try:
                            await asyncio.Event().wait()
                        except asyncio.CancelledError:
                            canceled.set()
                            await release.wait()
                            raise
                    return await edit(**kwargs)
                with patch.object(handoff, 'edit_handoff_notice', new=delayed_failure_notice):
                    await w.reply(bot, record, 'FAIL')
                    await asyncio.wait_for(posting.wait(), 5)
                    dismissal = asyncio.create_task(w.dismiss(bot, record))
                    tasks.append(dismissal)
                    await asyncio.wait_for(canceled.wait(), 5)
                    await w.trigger(bot)
                    fresh = public.records[source.id]
                    assert fresh is not record
                    w.channel.post_started.clear()
                    w.channel.post_gate = asyncio.Event()
                    await w.reply(bot, fresh, 'Keeper has completed the fix.')
                    await eventually(w.channel.post_started.is_set)
                    sessions = list((root / 'sessions').glob('public_handoff_*.json'))
                    assert len(sessions) == 1
                    release.set()
                    await asyncio.wait_for(dismissal, 5)
                    assert sessions[0].exists(), 'Old dismissal erased the replacement synthesis session'
                    assert public.records.get(source.id) is fresh
                    w.channel.post_gate.set()
                    await eventually(lambda: source.id not in public.records)
                assert len(w.channel.answers) == 1
                return {'old_cancellation_preserved_new_record_and_session': True}
            if mode.startswith('terminal_'):
                w.http.fail_methods.add('editMessageText')
            await w.reply(bot, record, 'Keeper review is pending signatures.')
            await eventually(lambda: record.status == 'delivered_pending_close')
            await eventually(lambda: not public.active)
            if mode.startswith('terminal_'):
                assert path.exists() and not record.history and not record.attachments
                if mode == 'terminal_move_failure':
                    with patch.object(Path, 'replace', side_effect=OSError('Controlled state move failure')):
                        await w.trigger(bot)
                    assert public.records.get(source.id) is record and path.exists()
                    assert len(w.requests()) == 2, 'Failed cleanup move lost pending state'
                elif mode == 'terminal_cleanup_overlap':
                    from ysupport.integrations import handoff
                    edit = handoff.edit_handoff_notice
                    async def delayed_edit(**kwargs):
                        if kwargs['message_id'] == record.telegram_message_id:
                            entered.set()
                            await release.wait()
                        return await edit(**kwargs)
                    with patch.object(handoff, 'edit_handoff_notice', new=delayed_edit):
                        cleanup = asyncio.create_task(public.resume())
                        tasks.append(cleanup)
                        await asyncio.wait_for(entered.wait(), 5)
                        await w.trigger(bot, wait_ready=False)
                        fresh = public.records[source.id]
                        assert fresh is not record, 'Completed cleanup blocked a new investigation'
                        await w.reply(bot, fresh, 'BLOCK')
                        await eventually(lambda: (root / 'blocked').exists())
                        w.http.fail_methods.clear()
                        release.set()
                        await asyncio.wait_for(cleanup, 5)
                        assert public.records.get(source.id) is fresh
                        assert json.loads(public.path(source.id).read_text())['telegram_message_id'] == fresh.telegram_message_id
                        (root / 'release').touch()
                        await eventually(lambda: source.id not in public.records)
                    assert len(w.channel.answers) == 2
                    return {'old_cleanup_preserved_new_record_and_synthesis': True}
                await w.trigger(bot, wait_ready=False)
                fresh = public.records[source.id]
                assert fresh is not record, 'Completed cleanup blocked a new investigation'
                assert len(w.requests()) == 3
                # Both generations must survive a restart, with independent notice cleanup.
                await bot.close()
                w.http.fail_methods.clear()
                bot = w.bot()
                public, controller = bot.telegram_handoffs.public, bot.telegram_handoffs
                public.hydrate()
                await controller._resume_pending_telegram_handoff_replies()
                fresh = public.records[source.id]
                await w.reply(bot, fresh, 'Keeper has completed the fix.')
                await eventually(lambda: source.id not in public.records)
                assert not list(path.parent.rglob('*.json')), 'Terminal cleanup was lost across restart'
                return {'fresh_trigger_investigated': True, 'both_notices_closed_after_restart': True,
                    'discord_updates': [m.content for m in w.channel.answers]}

            await eventually(lambda: source.id not in public.records)
            await w.trigger(bot)
            fresh = public.records[source.id]
            if mode == 'generation_recovery':
                w.channel.post_started.clear()
                w.channel.post_gate = asyncio.Event()
                w.channel.hold_after_post = True
            await w.reply(bot, fresh, 'Keeper has completed the fix.')
            if mode == 'generation_recovery':
                await eventually(w.channel.post_started.is_set)
                assert fresh.status == 'posting', 'Second generation reused a completed send instead of posting'
                await bot.close()
                bot = w.bot()
                public, controller = bot.telegram_handoffs.public, bot.telegram_handoffs
                public.hydrate()
                await controller._resume_pending_telegram_handoff_replies()
            await eventually(lambda: source.id not in public.records)
            updates = [m.content for m in w.channel.answers]
            assert updates == ['Update: Keeper review is pending signatures.', 'Update: Keeper has completed the fix.'], \
                'Discord deduplicated separate handoff generations'
            nonces = [p['nonce'] for p in w.channel.post_payloads]
            assert len(set(nonces)) == 2 and all(len(n) <= 25 for n in nonces)
            return {'discord_updates': updates, 'actual_sdk_payloads': w.channel.post_payloads,
                'recovery_did_not_repost': mode == 'generation_recovery'}
        finally:
            release.set()
            w.http.release.set()
            (root / 'release').touch()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await w.close()


async def verify_remote_recovery(root, mode):
    import discord
    with Workflow(root) as w:
        bot = w.bot()
        public = bot.telegram_handoffs.public
        try:
            source = await w.trigger(bot)
            record = public.records[source.id]
            path = public.path(source.id)
            if mode == 'delivery_no_prior_replies':
                w.channel.messages = [m for m in w.channel.messages if not m.author.bot]
                await w.reply(bot, record)
                await eventually(lambda: record.status == 'failed' or source.id not in public.records)
                assert source.id not in public.records and len(w.channel.answers) == 1, 'Empty reply history prevented delivery'
                return {'delivery_without_prior_replies': True}
            if mode.startswith('offline_'):
                if mode == 'offline_failed_source':
                    await w.reply(bot, record, 'FAIL')
                    await eventually(lambda: record.status == 'failed' and record.failure_notice_shown)
                await bot.close()
                bot = w.bot()
                public = bot.telegram_handoffs.public
                public.hydrate()
                recovered = public.records[source.id]
                if mode in {'offline_transient', 'offline_forbidden'}:
                    failure = (discord.Forbidden if mode == 'offline_forbidden' else discord.HTTPException)(
                        SimpleNamespace(status=403 if mode == 'offline_forbidden' else 503, reason='Controlled error'),
                        'Controlled permission or temporary failure')
                    with patch.object(w.channel, 'fetch_message', side_effect=failure):
                        await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                    assert recovered.status == 'open' and recovered.history and path.exists()
                if mode == 'offline_deleted_channel':
                    bot.get_channel = lambda _ident: None
                    async def deleted_channel(_ident):
                        raise discord.NotFound(SimpleNamespace(status=404, reason='Not found'), 'Unknown Channel')
                    bot.fetch_channel = deleted_channel
                else:
                    w.channel.messages = [m for m in w.channel.messages if m.id != source.id]
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert source.id not in public.records and not path.exists(), 'Offline deletion retained pending context'
                assert not recovered.history and not recovered.attachments and not recovered.pending_reply_text
                assert not w.channel.answers
                return {'deleted_source_context_cleared': True,
                    'temporary_errors_preserved_state': mode in {'offline_transient', 'offline_forbidden'}}
            if mode in {'completed_missing_notice', 'activation_missing_notice'}:
                if mode == 'activation_missing_notice':
                    await w.dismiss(bot, record)
                    await eventually(lambda: source.id not in public.records)
                    w.http.fail_methods.add('editMessageText')
                    await w.trigger(bot, wait_ready=False)
                    record = public.records[source.id]
                    await eventually(lambda: not public.active)
                else:
                    await w.reply(bot, record, 'BLOCK')
                    await eventually(lambda: (root / 'blocked').exists())
                w.http.enforce_idempotence = True
                w.http.deleted_messages.add((record.telegram_chat_id, record.telegram_message_id))
                w.http.fail_methods.clear()
                if mode == 'activation_missing_notice':
                    await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                    assert record.status == 'activating' and record.history and path.exists()
                    return {'missing_activation_notice_remains_an_error': True}
                (root / 'release').touch()
                await eventually(lambda: not public.active)
                await bot.close()
                bot = w.bot()
                public = bot.telegram_handoffs.public
                public.hydrate()
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert not path.exists() and not public.records, 'Missing completed notice retained terminal cleanup'
                before = len(w.http.requests)
                await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
                assert len(w.http.requests) == before and len(w.channel.answers) == 1
                return {'missing_terminal_notice_cleaned': True, 'no_further_cleanup_edits': True}

            if mode in {'recovery_previous_same_text', 'recovery_older_only'}:
                await w.reply(bot, record)
                await eventually(lambda: source.id not in public.records)
                await w.trigger(bot)
                record = public.records[source.id]
            w.channel.post_started.clear()
            w.channel.post_gate = asyncio.Event()
            w.channel.hold_after_post = mode not in {'recovery_wrong_author', 'recovery_older_only'}
            await w.reply(bot, record)
            await eventually(w.channel.post_started.is_set)
            expected = 'Update: Keeper review is complete.'
            if mode == 'recovery_ambiguous':
                extra = Message(w.channel, max(m.id for m in w.channel.messages) + 1, 999, expected, reply_to=source.id)
                w.channel.messages.append(extra)
            elif mode == 'recovery_wrong_author':
                w.channel.messages.append(Message(w.channel, 20000, 43, expected, reply_to=source.id))
            await bot.close()
            if mode == 'recovery_missing_metadata':
                payload = json.loads(path.read_text())
                payload.pop('delivery_digest', None)
                payload.pop('delivery_after_id', None)
                path.write_text(json.dumps(payload))
            bot = w.bot()
            public = bot.telegram_handoffs.public
            public.hydrate()
            recovered = public.records[source.id]
            count = len(w.channel.answers)
            await bot.telegram_handoffs._resume_pending_telegram_handoff_replies()
            assert len(w.channel.answers) == count, 'Recovery blindly resent an interrupted update'
            if mode == 'recovery_previous_same_text':
                assert source.id not in public.records and not path.exists(), 'REST history without nonce did not reconcile'
                assert recovered.response_id == w.channel.answers[-1].id, 'Recovery claimed an earlier generation'
                return {'same_text_generations_reconciled_separately': True, 'delivered_updates': count}
            assert recovered.status == 'failed' and path.exists(), 'Ambiguous or missing evidence was treated as delivered'
            return {'uncertain_send_requires_manual_review': True, 'no_resend': True, 'delivered_updates': count}
        finally:
            (root / 'release').touch()
            await w.close()


async def main(artifact):
    report = {'ok': False, 'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'scenarios': {}}
    modes = ('delivery', 'isolation', 'restart', 'dismissal', 'linked_context', 'skip',
             'generation_failure', 'deleted_source', 'long_answer', 'cancellation', 'close_failure',
             'posting_recovered', 'posting_uncertain', 'snapshot_failure', 'acceptance_failure',
             'changed_during_synthesis', 'close_already_applied', 'dismiss_already_deleted',
             'slow_dismissal', 'failure_notice_retry', 'busy_context', 'deletion_events')
    review_modes = ('stale_worker_retrigger', 'early_reply', 'deleted_during_initial_turn', 'deleted_during_notice',
        'updated_source_image', 'updated_source_image_during_synthesis', 'updated_linked_image',
        'signed_url_only', 'activation_restart')
    delivery_modes = ('repeated_handoff_nonce', 'generation_recovery', 'announcement_order',
        'announcement_restart', 'announcement_failure', 'terminal_cleanup_retrigger',
        'terminal_cleanup_overlap', 'terminal_move_failure', 'terminal_dismissal_overlap')
    remote_modes = ('delivery_no_prior_replies', 'offline_deleted_message', 'offline_deleted_channel', 'offline_failed_source',
        'offline_transient', 'offline_forbidden', 'completed_missing_notice', 'activation_missing_notice',
        'recovery_previous_same_text', 'recovery_ambiguous', 'recovery_wrong_author',
        'recovery_older_only', 'recovery_missing_metadata')
    with tempfile.TemporaryDirectory(prefix='ysupport-public-handoff-') as directory:
        for mode in modes + review_modes + delivery_modes + remote_modes:
            root = Path(directory) / mode
            root.mkdir()
            with patch.object(config, 'TICKET_EXECUTION_STATE_ROOT', root / 'state'):
                try:
                    runner = (verify_remote_recovery if mode in remote_modes else verify_delivery_fix
                        if mode in delivery_modes else verify_review_fix if mode in review_modes else verify)
                    report['scenarios'][mode] = {'ok': True, **await runner(root, mode)}
                except Exception as exc:
                    report['scenarios'][mode] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                        'traceback': traceback.format_exc()}
    report['ok'] = all(v['ok'] for v in report['scenarios'].values())
    names = ('ysupport/bot/public_handoffs.py', 'ysupport/bot/telegram_handoff_controller.py',
        'ysupport/bot/client.py', 'ysupport/bot/state.py', 'ysupport/integrations/handoff.py',
        'ysupport/execution/codex_support_sessions.py', 'ysupport/execution/transport.py',
        'ysupport/bot/discord_support_runtime.py', 'ysupport/bot/utils.py',
        'tests/telegram_http_e2e.py', 'tests/public_handoff_e2e.py')
    report['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in names if Path(name).exists()}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2)+'\n')
    artifact.chmod(0o600)
    print(json.dumps(report['scenarios'], indent=2))
    return int(not report['ok'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=Path('.cache/verification/public-handoff-e2e.json'))
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
