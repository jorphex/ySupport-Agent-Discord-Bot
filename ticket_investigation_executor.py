from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Protocol

from state import TicketInvestigationJob
from ticket_investigation_runtime import TicketTurnRequest
from ticket_investigation_worker import TicketInvestigationWorker, TicketWorkerResult


@dataclass
class TicketExecutionHooks:
    send_bug_review_status: Callable[[], Awaitable[None]] | None = None


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


class LocalTicketInvestigationExecutor:
    def __init__(self, worker: TicketInvestigationWorker) -> None:
        self.worker = worker

    async def execute_turn(
        self,
        request: TicketTurnRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketExecutionResult:
        updated_job = deepcopy(request.investigation_job)
        effective_request = replace(request, investigation_job=updated_job)
        if hooks and hooks.send_bug_review_status is not None:
            effective_request = replace(
                effective_request,
                send_bug_review_status=hooks.send_bug_review_status,
            )
        worker_result: TicketWorkerResult = await self.worker.execute_turn(effective_request)
        return TicketExecutionResult(
            flow_outcome=worker_result.flow_outcome,
            updated_job=updated_job,
        )
