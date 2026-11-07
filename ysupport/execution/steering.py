"""Owned, ordered follow-ups for a single ordinary ticket execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from ysupport.bot.ticket_intake import TicketIntakeState


@dataclass
class TicketSteeringInput:
    message_id: str
    text: str
    attachments: list[dict]
    prepared_attachments: list[dict] = field(default_factory=list)
    evidence: dict | None = None
    intake_state: TicketIntakeState | None = None
    system_hints: list[str] = field(default_factory=list)
    submitted: bool = False
    observed: bool = False


@dataclass
class TicketSteeringInbox:
    allow_input: Callable[[TicketSteeringInput], Awaitable[bool]]
    on_observed: Callable[[TicketSteeringInput], None] | None = None
    inputs: list[TicketSteeringInput] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    def offer(self, message_id: str, text: str, attachments: list[dict]) -> bool:
        if self.closed or any(item.message_id == message_id for item in self.inputs):
            return False
        self.inputs.append(TicketSteeringInput(message_id, text, list(attachments)))
        self.changed.set()
        return True

    @property
    def consumed(self) -> list[TicketSteeringInput]:
        result = []
        for item in self.inputs:
            if not item.observed:
                break
            result.append(item)
        return result
