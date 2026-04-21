from __future__ import annotations

from ticket_investigation_executor import LocalTicketInvestigationExecutor
from ticket_investigation_json_endpoint import ExecutorBackedTicketExecutionJsonEndpoint


def build_local_ticket_investigation_executor() -> LocalTicketInvestigationExecutor:
    from agents import Runner

    from ticket_investigation_runtime import TicketInvestigationRuntime
    from ticket_investigation_worker import TicketInvestigationWorker

    runtime = TicketInvestigationRuntime(Runner)
    worker = TicketInvestigationWorker(runtime)
    return LocalTicketInvestigationExecutor(worker)


def build_local_ticket_execution_json_endpoint() -> ExecutorBackedTicketExecutionJsonEndpoint:
    return ExecutorBackedTicketExecutionJsonEndpoint(build_local_ticket_investigation_executor())
