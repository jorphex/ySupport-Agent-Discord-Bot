from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from state import TicketInvestigationJob
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)

if TYPE_CHECKING:
    from ticket_investigation.contracts import TicketTurnRequest


@dataclass
class TicketExecutionHooks:
    send_progress_update: Callable[[str], Awaitable[None]] | None = None


class TicketExecutionNonFallbackError(ValueError):
    """Raised when retrying on a weaker endpoint would bypass a safety boundary."""


@dataclass
class TicketExecutionResult:
    flow_outcome: object
    updated_job: TicketInvestigationJob


class TicketInvestigationExecutor(Protocol):
    async def execute_turn(
        self,
        request: TicketTurnRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketExecutionResult:
        ...


class TicketExecutionTransport(Protocol):
    async def execute_transport_turn(
        self,
        request: TicketExecutionTransportRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketExecutionTransportResult:
        ...


class TransportTicketInvestigationExecutor:
    def __init__(self, transport: TicketExecutionTransport) -> None:
        self.transport = transport

    async def execute_turn(
        self,
        request: TicketTurnRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketExecutionResult:
        transport_request = TicketExecutionTransportRequest.from_turn_request(request)
        transport_result = await self.transport.execute_transport_turn(
            transport_request,
            hooks=hooks,
        )
        flow_outcome, updated_job = transport_result.to_execution_parts()
        return TicketExecutionResult(
            flow_outcome=flow_outcome,
            updated_job=updated_job,
        )
