import unittest
from unittest.mock import patch


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
        summary_patcher = patch(
            "discord_support_runtime.summarize_handoff_summary",
            return_value=None,
        )
        summary_patcher.start()
        self.addCleanup(summary_patcher.stop)
