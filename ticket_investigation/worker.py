"""State-transition wrapper for the explicit legacy local runtime."""

from dataclasses import dataclass

from ticket_investigation.contracts import TicketAgentFlowOutcome, TicketTurnRequest
from ticket_investigation.runtime import TicketInvestigationRuntime


@dataclass
class TicketWorkerResult:
    flow_outcome: TicketAgentFlowOutcome


class TicketInvestigationWorker:
    def __init__(self, runtime: TicketInvestigationRuntime) -> None:
        self.runtime = runtime

    async def execute_turn(self, request: TicketTurnRequest) -> TicketWorkerResult:
        request.investigation_job.begin_investigating()
        flow_outcome = await self.runtime.run_turn(request)
        request.investigation_job.complete_specialist_turn(flow_outcome.completed_agent_key)
        if flow_outcome.requires_human_handoff:
            request.investigation_job.mark_escalated_to_human()
        else:
            request.investigation_job.mark_waiting_for_user()
        return TicketWorkerResult(flow_outcome=flow_outcome)
