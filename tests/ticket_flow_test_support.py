import unittest
from dataclasses import dataclass
from unittest.mock import patch

from support_agents import TicketTriageDecision


@dataclass
class FakeResult:
    final_output: str | None
    last_agent: object | None
    _history: list
    _decision: TicketTriageDecision | None = None

    def final_output_as(self, model_type):
        if self._decision is None:
            raise AssertionError("No structured decision configured.")
        return self._decision

    def to_input_list(self):
        return self._history


class FakeRunner:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if not self._results:
            raise AssertionError("No fake result available for runner call.")
        return self._results.pop(0)


class FakeInvestigationExecutor:
    def __init__(self, *, result=None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.calls = []

    async def execute_turn(self, request, hooks=None):
        self.calls.append({"request": request, "hooks": hooks})
        if self.exc is not None:
            raise self.exc
        return self.result


class TicketFlowTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        boundary_patcher = patch(
            "ysupport._outer_support_boundary_result",
            return_value={
                "classification": "yearn_support",
                "tripwire_triggered": False,
            },
        )
        boundary_patcher.start()
        self.addCleanup(boundary_patcher.stop)
        runtime_boundary_patcher = patch(
            "ticket_investigation.runtime.evaluate_support_boundary",
            return_value={
                "classification": "yearn_support",
                "tripwire_triggered": False,
            },
        )
        runtime_boundary_patcher.start()
        self.addCleanup(runtime_boundary_patcher.stop)
        agent_boundary_patcher = patch(
            "support_agents.evaluate_support_boundary",
            return_value={
                "classification": "yearn_support",
                "tripwire_triggered": False,
            },
        )
        agent_boundary_patcher.start()
        self.addCleanup(agent_boundary_patcher.stop)
        summary_patcher = patch(
            "discord_support_runtime.summarize_handoff_summary",
            return_value=None,
        )
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)
