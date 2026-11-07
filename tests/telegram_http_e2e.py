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
from urllib import request
from unittest.mock import patch

from tests.optimization_e2e import eventually
import config
import handoff
import state
from tests.test_ticket_intake import _FakeDiscordChannel
from ticket_channel_lifecycle import clear_deleted_ticket_channel
from views import stop_ticket_for_manual_support
from ysupport import TicketBot


class TelegramHttpFixture:
    def __init__(self, *, held_method="editMessageText", held_text=None, fail=False):
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
                failed = not released or (held and fail)
                if not failed:
                    fixture.applied.append((method, payload))
                response = json.dumps({"ok": not failed, "result": [] if method == "getUpdates" else True}).encode()
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
        self.patches.enter_context(patch("handoff.request.urlopen", new=self.urlopen))
        self.patches.enter_context(patch("ticket_channel_lifecycle.discord.TextChannel", _FakeDiscordChannel))
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
    report["ok"] = all(item["ok"] for item in report["scenarios"].values())
    report["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
        "handoff.py", "telegram_handoff_controller.py", "ticket_channel_lifecycle.py", "state.py", "tests/telegram_http_e2e.py")}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    artifact.chmod(0o600)
    print(json.dumps(report["scenarios"], indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/telegram-http-e2e.json"))
    raise SystemExit(asyncio.run(main(parser.parse_args().artifact)))
