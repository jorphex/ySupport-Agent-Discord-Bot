from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Protocol

from ticket_investigation_runtime import TicketTurnRequest
from ticket_investigation_worker import TicketInvestigationWorker, TicketWorkerResult


@dataclass
class TicketExecutionHooks:
    send_bug_review_status: Callable[[], Awaitable[None]] | None = None


class TicketInvestigationExecutor(Protocol):
    async def execute_turn(
        self,
        request: TicketTurnRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketWorkerResult:
        ...


class LocalTicketInvestigationExecutor:
    def __init__(self, worker: TicketInvestigationWorker) -> None:
        self.worker = worker

    async def execute_turn(
        self,
        request: TicketTurnRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketWorkerResult:
        effective_request = request
        if hooks and hooks.send_bug_review_status is not None:
            effective_request = replace(
                request,
                send_bug_review_status=hooks.send_bug_review_status,
            )
        return await self.worker.execute_turn(effective_request)
