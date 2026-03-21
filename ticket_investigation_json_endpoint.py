import sys
from collections.abc import Sequence
from typing import Protocol

import config
from ticket_investigation_executor import (
    TicketExecutionHooks,
    TicketInvestigationExecutor,
)
from ticket_investigation_subprocess_endpoint import SubprocessTicketExecutionJsonEndpoint
from ticket_investigation_transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)


class TicketExecutionJsonEndpoint(Protocol):
    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        ...


class ExecutorBackedTicketExecutionJsonEndpoint:
    def __init__(self, delegate: TicketInvestigationExecutor) -> None:
        self.delegate = delegate

    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        transport_request = TicketExecutionTransportRequest.from_json(request_json)
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
        return TicketExecutionTransportResult.from_execution_parts(
            result.flow_outcome,
            result.updated_job,
        ).to_json()


class JsonEndpointTicketExecutionTransport:
    def __init__(self, endpoint: TicketExecutionJsonEndpoint) -> None:
        self.endpoint = endpoint

    async def execute_transport_turn(
        self,
        request: TicketExecutionTransportRequest,
        hooks: TicketExecutionHooks | None = None,
    ) -> TicketExecutionTransportResult:
        response_json = await self.endpoint.execute_json_turn(
            request.to_json(),
            hooks=hooks,
        )
        return TicketExecutionTransportResult.from_json(response_json)


def build_ticket_execution_json_endpoint(
    delegate: TicketInvestigationExecutor,
) -> TicketExecutionJsonEndpoint:
    if config.TICKET_EXECUTION_ENDPOINT == "local":
        return ExecutorBackedTicketExecutionJsonEndpoint(delegate)
    if config.TICKET_EXECUTION_ENDPOINT == "subprocess":
        return SubprocessTicketExecutionJsonEndpoint(
            _subprocess_command(),
        )
    raise ValueError(
        "Unsupported TICKET_EXECUTION_ENDPOINT value: "
        f"{config.TICKET_EXECUTION_ENDPOINT}"
    )


def _subprocess_command() -> Sequence[str]:
    if config.TICKET_EXECUTION_SUBPROCESS_COMMAND:
        return config.TICKET_EXECUTION_SUBPROCESS_COMMAND
    return [sys.executable, "-m", "ticket_investigation_worker_cli"]
