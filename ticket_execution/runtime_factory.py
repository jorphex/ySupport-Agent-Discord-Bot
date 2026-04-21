from __future__ import annotations

from ticket_investigation.executor import LocalTicketInvestigationExecutor
from ticket_investigation.json_endpoint import ExecutorBackedTicketExecutionJsonEndpoint


def build_local_ticket_investigation_executor() -> LocalTicketInvestigationExecutor:
    from agents import Runner

    from ticket_investigation.runtime import TicketInvestigationRuntime
    from ticket_investigation.worker import TicketInvestigationWorker

    runtime = TicketInvestigationRuntime(Runner)
    worker = TicketInvestigationWorker(runtime)
    return LocalTicketInvestigationExecutor(worker)


def build_local_ticket_execution_json_endpoint() -> ExecutorBackedTicketExecutionJsonEndpoint:
    return ExecutorBackedTicketExecutionJsonEndpoint(build_local_ticket_investigation_executor())
