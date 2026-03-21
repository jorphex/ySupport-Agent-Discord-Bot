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


class LoopbackTransportTicketInvestigationExecutor:
    def __init__(self, delegate: TicketInvestigationExecutor) -> None:
        self.delegate = delegate

    async def execute_turn(
        self,
        request: TicketTurnRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketExecutionResult:
        from ticket_investigation_transport import (
            TicketExecutionTransportRequest,
            TicketExecutionTransportResult,
        )

        transport_request = TicketExecutionTransportRequest.from_turn_request(
            request,
            wants_bug_review_status=bool(hooks and hooks.send_bug_review_status is not None),
        )
        hydrated_request = transport_request.to_turn_request()
        hydrated_hooks = None
        if transport_request.wants_bug_review_status and hooks is not None:
            hydrated_hooks = TicketExecutionHooks(
                send_bug_review_status=hooks.send_bug_review_status,
            )
        result = await self.delegate.execute_turn(
            hydrated_request,
            hooks=hydrated_hooks,
        )
        transport_result = TicketExecutionTransportResult.from_execution_result(result)
        return transport_result.to_execution_result()
