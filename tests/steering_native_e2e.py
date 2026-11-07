"""Bounded real-Codex steering probe; no Discord, Telegram or provider RPC calls.

Run as the bot service identity inside its existing mount namespace. Use an
isolated state directory and the service-owned auth source, never root auth.
"""

from dataclasses import replace
import json
import struct
from unittest.mock import patch
import zlib
from pathlib import Path

from codex_support_contract import SupportTurnRequest
from ticket_investigation.codex_support_app_server import run_codex_support_app_server
from ticket_investigation.executor import TicketExecutionHooks
from ticket_investigation.steering import TicketSteeringInbox
from ticket_investigation.transport import build_smoke_transport_request


async def verify(root, *, command, env, model, reasoning_effort):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    request = replace(
        build_smoke_transport_request(),
        smoke_mode=None,
        aggregated_text="Which block explorer should I use for a Yearn vault on Ethereum?",
        run_context={"channel_id": 7999, "project_context": "yearn"},
    )
    support = SupportTurnRequest.from_ticket_execution_request(
        request, ysupport_mcp_enabled=False
    )
    (root / "support_request.json").write_text(support.to_json())

    async def allow(_):
        return True

    # A deterministic 64px red PNG verifies that native vision receives the image.
    def png_chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    image = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 64) * 64))
        + png_chunk(b"IEND", b"")
    )

    async def download(*, attachment, attachments_dir, index):
        target = attachments_dir / "probe.png"
        target.write_bytes(image)
        return target

    inbox = TicketSteeringInbox(allow_input=allow)
    # Enqueue before initialization to exercise readiness without timing a model call.
    inbox.offer(
        "discord:probe-1",
        "Correction: my Yearn vault is on Katana, not Ethereum. Give the official explorer URL for Katana and name the solid color in this screenshot. Do not make any tool calls.",
        [
            {
                "url": "https://cdn.discordapp.com/attachments/1/2/probe.png",
                "filename": "probe.png",
                "content_type": "image/png",
                "is_image": True,
            }
        ],
    )
    prompt = (
        "This is a bounded support transport verification. Do not call tools. "
        "Return JSON matching the supplied output schema. Answer the latest ticket-owner input. "
        "Current support request: " + support.to_json()
    )
    with patch(
        "ticket_investigation.codex_support_attachments._download_attachment_image",
        new=download,
    ):
        output = await run_codex_support_app_server(
            command=command,
            prompt_text=prompt,
            support_request=support,
            resume_session_id=None,
            model=model,
            reasoning_effort=reasoning_effort,
            cwd=str(root),
            env=env,
            timeout_seconds=120,
            max_output_chars=200000,
            run_dir=root,
            hooks=TicketExecutionHooks(steering=inbox),
        )
    result = json.loads(output.final_response_text)
    metadata = json.loads((root / "metadata.json").read_text())
    assert len(inbox.consumed) == 1, metadata["steering"]
    assert "katanascan.com" in result["answer"].lower(), result
    assert "red" in result["answer"].lower(), result
    resumed_dir = root / "resumed"
    resumed_dir.mkdir(exist_ok=True)
    resumed_inbox = TicketSteeringInbox(allow_input=allow)
    resumed_output = await run_codex_support_app_server(
        command=command,
        prompt_text="Do not call tools. From our previous turn, what chain did I correct to and what color was the screenshot? Return a concise answer in the required JSON schema.",
        support_request=support,
        resume_session_id=metadata["thread_id"],
        model=model,
        reasoning_effort=reasoning_effort,
        cwd=str(resumed_dir),
        env=env,
        timeout_seconds=120,
        max_output_chars=200000,
        run_dir=resumed_dir,
        hooks=TicketExecutionHooks(steering=resumed_inbox),
    )
    resumed_result = json.loads(resumed_output.final_response_text)
    resumed_metadata = json.loads((resumed_dir / "metadata.json").read_text())
    assert resumed_metadata["thread_id"] == metadata["thread_id"]
    assert (
        "red" in resumed_result["answer"].lower()
        and "katana" in resumed_result["answer"].lower()
    ), resumed_result
    return {
        "ok": True,
        "answer": result["answer"],
        "same_turn": True,
        "native_vision": True,
        "resumed_context": True,
        "resumed_answer": resumed_result["answer"],
        "observed_native_user_message": True,
        "thread_id": metadata["thread_id"],
        "turn_id": metadata["turn_id"],
        "total_ms": metadata["total_ms"],
    }
