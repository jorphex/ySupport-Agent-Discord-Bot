import tests as _test_environment  # noqa: F401

import asyncio
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from unittest.mock import patch


from ticket_execution.subprocess_utils import run_bounded_subprocess
from ticket_execution.workspace import TicketExecutionWorkspace
from ticket_investigation.json_endpoint import (
    ExecutorBackedTicketExecutionJsonEndpoint,
    FailoverTicketExecutionJsonEndpoint,
    JsonEndpointTicketExecutionTransport,
)
from ticket_investigation.subprocess_endpoint import SubprocessTicketExecutionJsonEndpoint
from ticket_investigation.executor import (
    LocalTicketInvestigationExecutor,
    LoopbackTicketExecutionTransport,
    LoopbackTransportTicketInvestigationExecutor,
    TicketExecutionTransport,
    TicketExecutionHooks,
    TicketExecutionNonFallbackError,
    TransportTicketInvestigationExecutor,
)
from ticket_investigation.transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)
from ticket_investigation.runtime import TicketAgentFlowOutcome, TicketTurnRequest
from ticket_investigation.worker import TicketWorkerResult
from state import BotRunContext, TicketInvestigationJob


@dataclass
class _FakeWorker:
    requests: list

    async def execute_turn(self, request):
        self.requests.append(request)
        request.investigation_job.begin_investigating()
        request.investigation_job.complete_specialist_turn("docs")
        return TicketWorkerResult(
            flow_outcome=TicketAgentFlowOutcome(
                raw_final_reply="ok",
                conversation_history=[],
                completed_agent_key="docs",
                requires_human_handoff=False,
            )
        )


class TicketExecutorTests(unittest.IsolatedAsyncioTestCase):
    def test_ticket_execution_workspace_is_ephemeral_without_artifact_dir(self) -> None:
        workspace = TicketExecutionWorkspace(prefix="test-ephemeral-ticket-run-")

        with workspace as run_dir:
            (run_dir / "sensitive.txt").write_text("temporary", encoding="utf-8")
            self.assertIsNone(workspace.export_copy())

        self.assertFalse(run_dir.exists())

    def test_ticket_execution_workspace_removes_partial_export_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_dir:
            workspace = TicketExecutionWorkspace(
                artifact_dir=artifact_dir,
                prefix="test-failed-ticket-export-",
            )
            with workspace as run_dir:
                (run_dir / "sensitive.txt").write_text(
                    "temporary",
                    encoding="utf-8",
                )
                with patch.object(
                    workspace,
                    "_make_read_only",
                    side_effect=OSError("chmod failed"),
                ):
                    with self.assertRaises(OSError):
                        workspace.export_copy()

            self.assertEqual(os.listdir(artifact_dir), [])

    async def test_executor_backed_json_endpoint_short_circuits_smoke_request(self) -> None:
        class _ExplodingExecutor:
            async def execute_turn(self, request, hooks=None):
                raise AssertionError("Smoke requests should not reach the delegate.")

        endpoint = ExecutorBackedTicketExecutionJsonEndpoint(_ExplodingExecutor())
        response_json = await endpoint.execute_json_turn(
            TicketExecutionTransportRequest(
                aggregated_text="smoke",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 104,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 104,
                    "mode": "idle",
                    "evidence": {"tx_hashes": []},
                },
                workflow_name="tests.endpoint.smoke",
                smoke_mode="ping",
            ).to_json()
        )

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "ticket_execution_smoke_ok:local")
        self.assertEqual(updated_job.channel_id, 104)

    async def test_local_executor_copies_job_before_worker_mutates_it(self) -> None:
        worker = _FakeWorker(requests=[])
        executor = LocalTicketInvestigationExecutor(worker)

        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context=BotRunContext(channel_id=92, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=92),
            workflow_name="tests.executor",
        )
        result = await executor.execute_turn(request)

        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(request.investigation_job.mode, "idle")
        self.assertIsNone(request.investigation_job.current_specialty)
        self.assertEqual(result.updated_job.mode, "investigating")
        self.assertEqual(result.updated_job.current_specialty, "docs")

    async def test_loopback_transport_executor_round_trips_request_and_result(self) -> None:
        worker = _FakeWorker(requests=[])
        executor = LoopbackTransportTicketInvestigationExecutor(
            LocalTicketInvestigationExecutor(worker)
        )
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context=BotRunContext(channel_id=93, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=93),
            workflow_name="tests.executor",
        )

        result = await executor.execute_turn(request)

        self.assertEqual(result.flow_outcome.raw_final_reply, "ok")
        self.assertEqual(result.updated_job.current_specialty, "docs")
        self.assertEqual(result.updated_job.mode, "investigating")

    async def test_transport_executor_uses_transport_boundary(self) -> None:
        @dataclass
        class _FakeTransport:
            requests: list
            hooks: list

            async def execute_transport_turn(self, request, hooks=None):
                self.requests.append(request)
                self.hooks.append(hooks)
                updated_job = TicketInvestigationJob(
                    channel_id=request.investigation_job["channel_id"]
                )
                updated_job.begin_investigating()
                updated_job.complete_specialist_turn("bug")
                return TicketExecutionTransportResult.from_execution_parts(
                    TicketAgentFlowOutcome(
                        raw_final_reply="transport-ok",
                        conversation_history=[],
                        completed_agent_key="bug",
                        requires_human_handoff=False,
                    ),
                    updated_job,
                )

        async def fake_send_progress_update(_message: str) -> None:
            return None

        transport: TicketExecutionTransport = _FakeTransport(requests=[], hooks=[])
        executor = TransportTicketInvestigationExecutor(transport)
        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context=BotRunContext(channel_id=95, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=95),
            workflow_name="tests.executor.transport",
        )

        result = await executor.execute_turn(
            request,
            hooks=TicketExecutionHooks(
                send_progress_update=fake_send_progress_update,
            ),
        )

        self.assertEqual(len(transport.requests), 1)
        self.assertIs(
            transport.hooks[0].send_progress_update,
            fake_send_progress_update,
        )
        self.assertEqual(result.flow_outcome.raw_final_reply, "transport-ok")
        self.assertEqual(result.updated_job.current_specialty, "bug")
        self.assertEqual(result.updated_job.mode, "investigating")

    async def test_loopback_transport_rehydrates_request_for_delegate(self) -> None:
        worker = _FakeWorker(requests=[])
        transport = LoopbackTicketExecutionTransport(
            LocalTicketInvestigationExecutor(worker)
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 96,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 96,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.transport.loopback",
        )

        result = await transport.execute_transport_turn(request)
        flow_outcome, updated_job = result.to_execution_parts()

        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(worker.requests[0].run_context.channel_id, 96)
        self.assertEqual(flow_outcome.raw_final_reply, "ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_executor_backed_json_endpoint_round_trips_request_and_result(self) -> None:
        worker = _FakeWorker(requests=[])
        endpoint = ExecutorBackedTicketExecutionJsonEndpoint(
            LocalTicketInvestigationExecutor(worker)
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 97,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 97,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.json",
        )

        response_json = await endpoint.execute_json_turn(request.to_json())
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(len(worker.requests), 1)
        self.assertEqual(worker.requests[0].run_context.channel_id, 97)
        self.assertEqual(flow_outcome.raw_final_reply, "ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_json_endpoint_transport_uses_json_boundary(self) -> None:
        @dataclass
        class _FakeJsonEndpoint:
            requests: list
            hooks: list

            async def execute_json_turn(self, request_json, hooks=None):
                self.requests.append(request_json)
                self.hooks.append(hooks)
                request = TicketExecutionTransportRequest.from_json(request_json)
                updated_job = TicketInvestigationJob(
                    channel_id=request.investigation_job["channel_id"]
                )
                updated_job.begin_investigating()
                updated_job.complete_specialist_turn("docs")
                return TicketExecutionTransportResult.from_execution_parts(
                    TicketAgentFlowOutcome(
                        raw_final_reply="json-ok",
                        conversation_history=[],
                        completed_agent_key="docs",
                        requires_human_handoff=False,
                    ),
                    updated_job,
                ).to_json()

        transport = JsonEndpointTicketExecutionTransport(
            _FakeJsonEndpoint(requests=[], hooks=[])
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 98,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 98,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.transport",
        )

        result = await transport.execute_transport_turn(request)
        flow_outcome, updated_job = result.to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "json-ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_subprocess_json_endpoint_round_trips_response(self) -> None:
        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "request=json.loads(sys.stdin.read()); "
                    "response={"
                    "'flow_outcome':{"
                    "'raw_final_reply':'subprocess-ok',"
                    "'conversation_history':[],"
                    "'completed_agent_key':'docs',"
                    "'requires_human_handoff':False"
                    "},"
                    "'updated_job':{"
                    "'channel_id':request['investigation_job']['channel_id'],"
                    "'requested_intent':request['investigation_job'].get('requested_intent'),"
                    "'mode':'investigating',"
                    "'current_specialty':'docs',"
                    "'last_specialty':'docs',"
                    "'evidence':request['investigation_job'].get('evidence',{})"
                    "}"
                    "}; "
                    "sys.stdout.write(json.dumps(response))"
                ),
            ]
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[{"role": "user", "content": "help"}],
            current_history=[],
            run_context={
                "channel_id": 99,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 99,
                "requested_intent": "investigate_issue",
                "mode": "collecting",
                "evidence": {"wallet": None, "chain": "katana", "tx_hashes": []},
            },
            workflow_name="tests.endpoint.subprocess",
        )

        response_json = await endpoint.execute_json_turn(request.to_json())
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "subprocess-ok")
        self.assertEqual(updated_job.channel_id, 99)
        self.assertEqual(updated_job.current_specialty, "docs")
        self.assertEqual(updated_job.evidence.chain, "katana")

    def test_subprocess_json_endpoint_rejects_disallowed_command(self) -> None:
        with self.assertRaises(ValueError):
            SubprocessTicketExecutionJsonEndpoint(
                [sys.executable, "-c", "print('nope')"],
                allowed_command_prefixes=[["codex", "exec"]],
            )

    async def test_subprocess_json_endpoint_rejects_oversized_stdout(self) -> None:
        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "exec(\"while True:\\n sys.stdout.write('x' * 4096); "
                    "sys.stdout.flush()\")"
                ),
            ],
            max_output_chars=10,
            timeout_seconds=5,
        )

        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "too much stdout"):
            await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 101,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 101,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_oversized",
                ).to_json()
            )
        self.assertLess(time.monotonic() - started, 2)

    async def test_run_bounded_subprocess_kills_spawned_child_processes_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            child_pid_path = os.path.join(temp_dir, "child_pid.txt")
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys, time; "
                    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                    f"pathlib.Path({child_pid_path!r}).write_text(str(child.pid), encoding='utf-8'); "
                    "time.sleep(60)"
                ),
            ]

            with self.assertRaises(RuntimeError):
                await run_bounded_subprocess(
                    command=command,
                    stdin_text="",
                    cwd=None,
                    env=dict(os.environ),
                    timeout_seconds=0.2,
                    max_output_chars=1000,
                    max_error_chars=1000,
                    timeout_message="timed out",
                    empty_stdout_message="empty",
                    oversized_stdout_message="oversized",
                    metadata={},
                    artifact_run_dir=None,
                )

            deadline = time.time() + 5
            child_pid = None
            while time.time() < deadline:
                if os.path.exists(child_pid_path):
                    with open(child_pid_path, encoding="utf-8") as handle:
                        child_pid = int(handle.read().strip())
                    break
                await asyncio.sleep(0.05)

            self.assertIsNotNone(child_pid)
            assert child_pid is not None

            while time.time() < deadline:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.05)
            else:
                self.fail("Timed-out subprocess left a spawned child process running.")

    async def test_run_bounded_subprocess_kills_descendant_after_leader_exits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            child_pid_path = os.path.join(temp_dir, "child_pid.txt")
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys; "
                    "child = subprocess.Popen([sys.executable, '-c', "
                    "'import time; time.sleep(60)']); "
                    f"pathlib.Path({child_pid_path!r}).write_text("
                    "str(child.pid), encoding='utf-8')"
                ),
            ]

            with self.assertRaisesRegex(RuntimeError, "timed out"):
                await run_bounded_subprocess(
                    command=command,
                    stdin_text="",
                    cwd=None,
                    env=dict(os.environ),
                    timeout_seconds=0.2,
                    max_output_chars=1000,
                    max_error_chars=1000,
                    timeout_message="timed out",
                    empty_stdout_message="empty",
                    oversized_stdout_message="oversized",
                    metadata={},
                    artifact_run_dir=None,
                )

            with open(child_pid_path, encoding="utf-8") as handle:
                child_pid = int(handle.read().strip())
            await self._assert_process_exits(child_pid)

    async def test_run_bounded_subprocess_kills_descendants_on_cancellation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            child_pid_path = os.path.join(temp_dir, "child_pid.txt")
            command = [
                sys.executable,
                "-c",
                (
                    "import pathlib, subprocess, sys, time; "
                    "child = subprocess.Popen([sys.executable, '-c', "
                    "'import time; time.sleep(60)']); "
                    f"pathlib.Path({child_pid_path!r}).write_text("
                    "str(child.pid), encoding='utf-8'); "
                    "time.sleep(60)"
                ),
            ]
            task = asyncio.create_task(
                run_bounded_subprocess(
                    command=command,
                    stdin_text="",
                    cwd=None,
                    env=dict(os.environ),
                    timeout_seconds=30,
                    max_output_chars=1000,
                    max_error_chars=1000,
                    timeout_message="timed out",
                    empty_stdout_message="empty",
                    oversized_stdout_message="oversized",
                    metadata={},
                    artifact_run_dir=None,
                )
            )
            deadline = time.monotonic() + 5
            while not os.path.exists(child_pid_path):
                if time.monotonic() >= deadline:
                    self.fail("Subprocess did not write its child PID.")
                await asyncio.sleep(0.01)
            with open(child_pid_path, encoding="utf-8") as handle:
                child_pid = int(handle.read().strip())

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await self._assert_process_exits(child_pid)

    async def _assert_process_exits(self, pid: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.05)
        self.fail(f"Subprocess left descendant {pid} running.")

    async def test_subprocess_json_endpoint_uses_explicit_env_without_parent_inheritance(self) -> None:
        original_blocked = os.environ.get("BLOCKED_TEST_ENV")
        os.environ["BLOCKED_TEST_ENV"] = "blocked"
        try:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,os,sys; "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':f\"{os.getenv('ALLOWED_TEST_ENV')}:{os.getenv('BLOCKED_TEST_ENV')}\","
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':102,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                env={"ALLOWED_TEST_ENV": "allowed"},
                inherit_parent_env=False,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 102,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 102,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_env",
                ).to_json()
            )
        finally:
            if original_blocked is None:
                os.environ.pop("BLOCKED_TEST_ENV", None)
            else:
                os.environ["BLOCKED_TEST_ENV"] = original_blocked

        flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "allowed:None")

    async def test_subprocess_json_endpoint_uses_configured_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,os,sys; "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':os.getcwd(),"
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':103,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                cwd=temp_dir,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 103,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 103,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_cwd",
                ).to_json()
            )

        flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, temp_dir)

    async def test_subprocess_json_endpoint_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,sys; "
                        "request=json.loads(sys.stdin.read()); "
                        "print('worker-stderr', file=sys.stderr); "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':'artifact-ok',"
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':request['investigation_job']['channel_id'],"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':request['investigation_job'].get('evidence',{})"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                artifact_dir=artifact_dir,
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="help",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 104,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 104,
                    "mode": "idle",
                    "evidence": {"tx_hashes": []},
                },
                workflow_name="tests.endpoint.subprocess_artifacts",
            )

            response_json = await endpoint.execute_json_turn(request.to_json())

            artifact_entries = os.listdir(artifact_dir)
            self.assertEqual(len(artifact_entries), 1)
            run_dir = os.path.join(artifact_dir, artifact_entries[0])
            self.assertTrue(os.path.exists(os.path.join(run_dir, "request.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "stdout.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "stderr.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "metadata.json")))
            with open(os.path.join(run_dir, "stdout.txt"), encoding="utf-8") as stdout_file:
                stdout_text = stdout_file.read()
            with open(os.path.join(run_dir, "stderr.txt"), encoding="utf-8") as stderr_file:
                stderr_text = stderr_file.read()
            self.assertIn("artifact-ok", stdout_text)
            self.assertIn("worker-stderr", stderr_text)

        flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "artifact-ok")

    async def test_subprocess_json_endpoint_exports_explicit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = SubprocessTicketExecutionJsonEndpoint(
                [
                    sys.executable,
                    "-c",
                    (
                        "import json,os,pathlib,sys; "
                        "run_dir=os.getenv('TICKET_EXECUTION_RUN_DIR'); "
                        "pathlib.Path(run_dir, 'marker.txt').write_text('ok', encoding='utf-8'); "
                        "response={"
                        "'flow_outcome':{"
                        "'raw_final_reply':run_dir,"
                        "'conversation_history':[],"
                        "'completed_agent_key':'docs',"
                        "'requires_human_handoff':False"
                        "},"
                        "'updated_job':{"
                        "'channel_id':105,"
                        "'mode':'investigating',"
                        "'current_specialty':'docs',"
                        "'last_specialty':'docs',"
                        "'evidence':{'tx_hashes':[]}"
                        "}"
                        "}; "
                        "sys.stdout.write(json.dumps(response))"
                    ),
                ],
                artifact_dir=artifact_dir,
            )

            response_json = await endpoint.execute_json_turn(
                TicketExecutionTransportRequest(
                    aggregated_text="help",
                    input_list=[],
                    current_history=[],
                    run_context={
                        "channel_id": 105,
                        "project_context": "yearn",
                        "repo_last_search_artifact_refs": [],
                    },
                    investigation_job={
                        "channel_id": 105,
                        "mode": "idle",
                        "evidence": {"tx_hashes": []},
                    },
                    workflow_name="tests.endpoint.subprocess_run_dir",
                ).to_json()
            )

            flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
                response_json
            ).to_execution_parts()
            self.assertIsNotNone(flow_outcome.raw_final_reply)
            assert flow_outcome.raw_final_reply is not None
            self.assertFalse(flow_outcome.raw_final_reply.startswith(artifact_dir))
            self.assertFalse(os.path.exists(flow_outcome.raw_final_reply))
            artifact_entries = os.listdir(artifact_dir)
            self.assertEqual(len(artifact_entries), 1)
            exported_dir = os.path.join(artifact_dir, artifact_entries[0])
            self.assertTrue(os.path.isdir(exported_dir))
            self.assertTrue(os.path.exists(os.path.join(exported_dir, "marker.txt")))
            self.assertFalse(os.stat(exported_dir).st_mode & stat.S_IWUSR)
            self.assertFalse(
                os.stat(os.path.join(exported_dir, "marker.txt")).st_mode & stat.S_IWUSR
            )

    async def test_failover_json_endpoint_uses_fallback_after_runtime_failure(self) -> None:
        class _FailingJsonEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                raise RuntimeError("primary failed")

        class _FallbackJsonEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                request = TicketExecutionTransportRequest.from_json(request_json)
                response = {
                    "flow_outcome": {
                        "raw_final_reply": "fallback-ok",
                        "conversation_history": [],
                        "completed_agent_key": "docs",
                        "requires_human_handoff": False,
                    },
                    "updated_job": {
                        "channel_id": request.investigation_job["channel_id"],
                        "requested_intent": request.investigation_job.get("requested_intent"),
                        "mode": "investigating",
                        "current_specialty": "docs",
                        "last_specialty": "docs",
                        "evidence": request.investigation_job.get("evidence", {}),
                    },
                }
                return json.dumps(response)

        endpoint = FailoverTicketExecutionJsonEndpoint(
            _FailingJsonEndpoint(),
            _FallbackJsonEndpoint(),
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 107,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 107,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.failover",
        )

        response_json = await endpoint.execute_json_turn(request.to_json())

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "fallback-ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_failover_json_endpoint_falls_back_when_primary_returns_malformed_result(self) -> None:
        class _MalformedPrimaryEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                return "{}"

        class _FallbackJsonEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                request = TicketExecutionTransportRequest.from_json(request_json)
                return json.dumps(
                    {
                        "flow_outcome": {
                            "raw_final_reply": "fallback-ok",
                            "conversation_history": [],
                            "completed_agent_key": "docs",
                            "requires_human_handoff": False,
                        },
                        "updated_job": {
                            "channel_id": request.investigation_job["channel_id"],
                            "requested_intent": request.investigation_job.get(
                                "requested_intent"
                            ),
                            "mode": "investigating",
                            "current_specialty": "docs",
                            "last_specialty": "docs",
                            "evidence": request.investigation_job.get("evidence", {}),
                        },
                    }
                )

        endpoint = FailoverTicketExecutionJsonEndpoint(
            _MalformedPrimaryEndpoint(),
            _FallbackJsonEndpoint(),
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 109,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 109,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.failover_malformed_primary",
        )

        response_json = await endpoint.execute_json_turn(request.to_json())

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "fallback-ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_failover_does_not_bypass_non_fallback_failure(self) -> None:
        fallback_called = False

        class _PolicyFailureEndpoint:
            async def execute_json_turn(self, request_json: str, hooks=None) -> str:
                raise TicketExecutionNonFallbackError("unsafe primary result")

        class _FallbackEndpoint:
            async def execute_json_turn(self, request_json: str, hooks=None) -> str:
                nonlocal fallback_called
                fallback_called = True
                raise AssertionError("Policy failures must not reach fallback.")

        endpoint = FailoverTicketExecutionJsonEndpoint(
            _PolicyFailureEndpoint(),
            _FallbackEndpoint(),
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={"channel_id": 210, "project_context": "yearn"},
            investigation_job={
                "channel_id": 210,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.failover_policy",
        )

        with self.assertRaisesRegex(
            TicketExecutionNonFallbackError,
            "unsafe primary result",
        ):
            await endpoint.execute_json_turn(request.to_json())
        self.assertFalse(fallback_called)

    async def test_subprocess_json_endpoint_returns_success_even_if_export_copy_fails(self) -> None:
        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "request=json.loads(sys.stdin.read()); "
                    "response={"
                    "'flow_outcome':{"
                    "'raw_final_reply':'subprocess-export-ok',"
                    "'conversation_history':[],"
                    "'completed_agent_key':'docs',"
                    "'requires_human_handoff':False"
                    "},"
                    "'updated_job':{"
                    "'channel_id':request['investigation_job']['channel_id'],"
                    "'mode':'investigating',"
                    "'current_specialty':'docs',"
                    "'last_specialty':'docs',"
                    "'evidence':request['investigation_job'].get('evidence',{})"
                    "}"
                    "}; "
                    "sys.stdout.write(json.dumps(response))"
                ),
            ],
            artifact_dir="/tmp/unused-artifact-dir",
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 110,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 110,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.subprocess_export_failure",
        )

        with self.assertLogs("ticket_investigation.subprocess_endpoint", level="WARNING") as logs:
            with patch(
                "ticket_investigation.subprocess_endpoint.TicketExecutionWorkspace.export_copy",
                side_effect=OSError("disk full"),
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "subprocess-export-ok")
        self.assertEqual(updated_job.current_specialty, "docs")
        self.assertTrue(
            any(
                "Failed to export ticket execution subprocess workspace copy" in line
                for line in logs.output
            )
        )
