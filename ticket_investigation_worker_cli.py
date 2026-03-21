import asyncio
import sys

from agents import Runner, set_default_openai_key
from dotenv import load_dotenv

import config
from ticket_investigation_executor import LocalTicketInvestigationExecutor
from ticket_investigation_json_endpoint import ExecutorBackedTicketExecutionJsonEndpoint
from ticket_investigation_runtime import TicketInvestigationRuntime
from ticket_investigation_worker import TicketInvestigationWorker


def build_local_ticket_execution_json_endpoint() -> ExecutorBackedTicketExecutionJsonEndpoint:
    runtime = TicketInvestigationRuntime(Runner)
    worker = TicketInvestigationWorker(runtime)
    executor = LocalTicketInvestigationExecutor(worker)
    return ExecutorBackedTicketExecutionJsonEndpoint(executor)


async def execute_request_json(request_json: str) -> str:
    endpoint = build_local_ticket_execution_json_endpoint()
    return await endpoint.execute_json_turn(request_json)


async def _main() -> int:
    load_dotenv()
    set_default_openai_key(config.OPENAI_API_KEY)

    request_json = sys.stdin.read()
    if not request_json.strip():
        print("Expected ticket execution JSON on stdin.", file=sys.stderr)
        return 1

    response_json = await execute_request_json(request_json)
    sys.stdout.write(response_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
