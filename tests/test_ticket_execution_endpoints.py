import json
import os
import stat
import sys
import tempfile
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from ticket_investigation_json_endpoint import (
    ExecutorBackedTicketExecutionJsonEndpoint,
    FailoverTicketExecutionJsonEndpoint,
    JsonEndpointTicketExecutionTransport,
)
from ticket_investigation_codex_endpoint import CodexExecTicketExecutionJsonEndpoint
from ticket_investigation_subprocess_endpoint import SubprocessTicketExecutionJsonEndpoint
from ticket_investigation_executor import (
    LocalTicketInvestigationExecutor,
    LoopbackTicketExecutionTransport,
    LoopbackTransportTicketInvestigationExecutor,
    TicketExecutionTransport,
    TicketExecutionHooks,
    TransportTicketInvestigationExecutor,
)
from ticket_investigation_transport import (
    TicketExecutionTransportRequest,
    TicketExecutionTransportResult,
)
from ticket_investigation_runtime import TicketAgentFlowOutcome, TicketTurnRequest
from ticket_investigation_worker import TicketWorkerResult
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
                wants_bug_review_status=False,
                smoke_mode="ping",
            ).to_json()
        )

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "ticket_execution_smoke_ok:local")
        self.assertEqual(updated_job.channel_id, 104)

    async def test_local_executor_injects_hooks_into_request(self) -> None:
        worker = _FakeWorker(requests=[])
        executor = LocalTicketInvestigationExecutor(worker)

        async def fake_send_bug_review_status() -> None:
            return None

        request = TicketTurnRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context=BotRunContext(channel_id=92, project_context="yearn"),
            investigation_job=TicketInvestigationJob(channel_id=92),
            workflow_name="tests.executor",
        )
        result = await executor.execute_turn(
            request,
            hooks=TicketExecutionHooks(
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )

        self.assertEqual(len(worker.requests), 1)
        self.assertIs(worker.requests[0].send_bug_review_status, fake_send_bug_review_status)
        self.assertIsNone(request.send_bug_review_status)
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

        async def fake_send_bug_review_status() -> None:
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
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )

        self.assertEqual(len(transport.requests), 1)
        self.assertTrue(transport.requests[0].wants_bug_review_status)
        self.assertIs(transport.hooks[0].send_bug_review_status, fake_send_bug_review_status)
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
            wants_bug_review_status=False,
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
            wants_bug_review_status=False,
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
            wants_bug_review_status=False,
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
            wants_bug_review_status=False,
        )

        response_json = await endpoint.execute_json_turn(request.to_json())
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(flow_outcome.raw_final_reply, "subprocess-ok")
        self.assertEqual(updated_job.channel_id, 99)
        self.assertEqual(updated_job.current_specialty, "docs")
        self.assertEqual(updated_job.evidence.chain, "katana")

    async def test_subprocess_json_endpoint_sends_bug_review_hook_before_spawn(self) -> None:
        hook_calls: list[str] = []

        async def fake_send_bug_review_status() -> None:
            hook_calls.append("sent")

        endpoint = SubprocessTicketExecutionJsonEndpoint(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    "request=json.loads(sys.stdin.read()); "
                    "response={"
                    "'flow_outcome':{"
                    "'raw_final_reply':'hook-ok',"
                    "'conversation_history':[],"
                    "'completed_agent_key':'bug',"
                    "'requires_human_handoff':False"
                    "},"
                    "'updated_job':{"
                    "'channel_id':request['investigation_job']['channel_id'],"
                    "'mode':'investigating',"
                    "'current_specialty':'bug',"
                    "'last_specialty':'bug',"
                    "'evidence':request['investigation_job'].get('evidence',{})"
                    "}"
                    "}; "
                    "sys.stdout.write(json.dumps(response))"
                ),
            ]
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 100,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 100,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.subprocess_hook",
            wants_bug_review_status=True,
        )

        response_json = await endpoint.execute_json_turn(
            request.to_json(),
            hooks=TicketExecutionHooks(
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )
        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()

        self.assertEqual(hook_calls, ["sent"])
        self.assertEqual(flow_outcome.raw_final_reply, "hook-ok")
        self.assertEqual(updated_job.current_specialty, "bug")

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
                "import sys; sys.stdout.write('x' * 50)",
            ],
            max_output_chars=10,
        )

        with self.assertRaises(RuntimeError):
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
                    wants_bug_review_status=False,
                ).to_json()
            )

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
                    wants_bug_review_status=False,
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
                    wants_bug_review_status=False,
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
                wants_bug_review_status=False,
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

    async def test_subprocess_json_endpoint_exports_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as run_root:
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
                run_dir_root=run_root,
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
                    wants_bug_review_status=False,
                ).to_json()
            )

            flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
                response_json
            ).to_execution_parts()
            self.assertIsNotNone(flow_outcome.raw_final_reply)
            assert flow_outcome.raw_final_reply is not None
            self.assertFalse(flow_outcome.raw_final_reply.startswith(run_root))
            self.assertFalse(os.path.exists(flow_outcome.raw_final_reply))
            artifact_entries = os.listdir(run_root)
            self.assertEqual(len(artifact_entries), 1)
            exported_dir = os.path.join(run_root, artifact_entries[0])
            self.assertTrue(os.path.isdir(exported_dir))
            self.assertTrue(os.path.exists(os.path.join(exported_dir, "marker.txt")))
            self.assertFalse(os.stat(exported_dir).st_mode & stat.S_IWUSR)
            self.assertFalse(
                os.stat(os.path.join(exported_dir, "marker.txt")).st_mode & stat.S_IWUSR
            )

    async def test_codex_exec_json_endpoint_round_trips_response_and_writes_bundle(self) -> None:
        fake_codex = (
            "import json,os,pathlib,sys; "
            "prompt=sys.stdin.read(); "
            "cwd=pathlib.Path(os.getcwd()); "
            "request=json.loads((cwd/'request.json').read_text()); "
            "reply='codex-ok:{}:{}:{}:{}:{}'.format("
            "request['investigation_job'].get('evidence',{}).get('chain'),"
            "'--output-schema' in sys.argv,"
            "(cwd/'response_schema.json').exists(),"
            "(cwd/'codex_prompt.txt').exists(),"
            "'Read the ticket execution request from request.json.' in prompt"
            "); "
            "response={"
            "'flow_outcome':{"
            "'raw_final_reply':reply,"
            "'conversation_history':[],"
            "'completed_agent_key':'data',"
            "'requires_human_handoff':False"
            "},"
            "'updated_job':{"
            "'channel_id':request['investigation_job']['channel_id'],"
            "'requested_intent':request['investigation_job'].get('requested_intent'),"
            "'mode':'investigating',"
            "'current_specialty':'data',"
            "'last_specialty':'data',"
            "'evidence':request['investigation_job'].get('evidence',{})"
            "}"
            "}; "
            "print('codex-stderr', file=sys.stderr); "
            "sys.stdout.write(json.dumps(response))"
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = CodexExecTicketExecutionJsonEndpoint(
                repo_root="/root/bots/discord/ysupport",
                codex_command=[sys.executable, "-c", fake_codex],
                allowed_command_prefixes=[[sys.executable, "-c", fake_codex]],
                artifact_dir=artifact_dir,
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="investigate",
                input_list=[{"role": "user", "content": "investigate"}],
                current_history=[],
                run_context={
                    "channel_id": 106,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 106,
                    "requested_intent": "investigate_issue",
                    "mode": "collecting",
                    "evidence": {"wallet": None, "chain": "katana", "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_exec",
                wants_bug_review_status=False,
            )

            response_json = await endpoint.execute_json_turn(request.to_json())

            artifact_entries = os.listdir(artifact_dir)
            self.assertEqual(len(artifact_entries), 1)
            run_dir = os.path.join(artifact_dir, artifact_entries[0])
            self.assertTrue(os.path.exists(os.path.join(run_dir, "request.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "response_schema.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "codex_prompt.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "stdout.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "stderr.txt")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "metadata.json")))

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(
            flow_outcome.raw_final_reply,
            "codex-ok:katana:True:True:True:True",
        )
        self.assertEqual(updated_job.current_specialty, "data")

    def test_codex_exec_json_endpoint_rejects_disallowed_command(self) -> None:
        with self.assertRaises(ValueError):
            CodexExecTicketExecutionJsonEndpoint(
                repo_root="/root/bots/discord/ysupport",
                codex_command=[sys.executable, "-c", "print('nope')"],
                allowed_command_prefixes=[["codex", "exec"]],
            )

    async def test_codex_exec_json_endpoint_smoke_uses_expected_response_bundle(self) -> None:
        fake_codex = (
            "import pathlib,sys; "
            "cwd=pathlib.Path.cwd(); "
            "sys.stdout.write((cwd/'expected_response.json').read_text())"
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = CodexExecTicketExecutionJsonEndpoint(
                repo_root="/root/bots/discord/ysupport",
                codex_command=[sys.executable, "-c", fake_codex],
                allowed_command_prefixes=[[sys.executable, "-c", fake_codex]],
                artifact_dir=artifact_dir,
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="smoke",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 107,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 107,
                    "requested_intent": "smoke_probe",
                    "mode": "idle",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_smoke",
                wants_bug_review_status=False,
                smoke_mode="ping",
            )

            response_json = await endpoint.execute_json_turn(request.to_json())

            artifact_entries = os.listdir(artifact_dir)
            self.assertEqual(len(artifact_entries), 1)
            run_dir = os.path.join(artifact_dir, artifact_entries[0])
            self.assertTrue(os.path.exists(os.path.join(run_dir, "expected_response.json")))

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "ticket_execution_smoke_ok:codex_exec")
        self.assertEqual(updated_job.channel_id, 107)

    async def test_codex_exec_json_endpoint_exports_copy_of_temporary_run_dir(self) -> None:
        fake_codex = (
            "import json,os,pathlib,sys; "
            "run_dir=pathlib.Path(os.environ['TICKET_EXECUTION_RUN_DIR']); "
            "run_dir.joinpath('marker.txt').write_text('ok', encoding='utf-8'); "
            "response={"
            "'flow_outcome':{"
            "'raw_final_reply':str(run_dir),"
            "'conversation_history':[],"
            "'completed_agent_key':'data',"
            "'requires_human_handoff':False"
            "},"
            "'updated_job':{"
            "'channel_id':108,"
            "'mode':'investigating',"
            "'current_specialty':'data',"
            "'last_specialty':'data',"
            "'evidence':{'tx_hashes':[]}"
            "}"
            "}; "
            "sys.stdout.write(json.dumps(response))"
        )
        with tempfile.TemporaryDirectory() as artifact_dir:
            endpoint = CodexExecTicketExecutionJsonEndpoint(
                repo_root="/root/bots/discord/ysupport",
                codex_command=[sys.executable, "-c", fake_codex],
                allowed_command_prefixes=[[sys.executable, "-c", fake_codex]],
                artifact_dir=artifact_dir,
                cwd="/app",
            )
            request = TicketExecutionTransportRequest(
                aggregated_text="codex export",
                input_list=[],
                current_history=[],
                run_context={
                    "channel_id": 108,
                    "project_context": "yearn",
                    "repo_last_search_artifact_refs": [],
                },
                investigation_job={
                    "channel_id": 108,
                    "requested_intent": "investigate_issue",
                    "mode": "collecting",
                    "evidence": {"wallet": None, "chain": None, "tx_hashes": []},
                },
                workflow_name="tests.endpoint.codex_export",
                wants_bug_review_status=False,
            )

            response_json = await endpoint.execute_json_turn(request.to_json())

            flow_outcome, _updated_job = TicketExecutionTransportResult.from_json(
                response_json
            ).to_execution_parts()
            self.assertFalse(os.path.exists(flow_outcome.raw_final_reply))
            artifact_entries = os.listdir(artifact_dir)
            self.assertEqual(len(artifact_entries), 1)
            exported_dir = os.path.join(artifact_dir, artifact_entries[0])
            self.assertTrue(os.path.exists(os.path.join(exported_dir, "marker.txt")))
            self.assertFalse(os.stat(exported_dir).st_mode & stat.S_IWUSR)
            self.assertFalse(
                os.stat(os.path.join(exported_dir, "marker.txt")).st_mode & stat.S_IWUSR
            )

    async def test_failover_json_endpoint_uses_fallback_and_deduplicates_hook(self) -> None:
        hook_calls: list[str] = []

        async def fake_send_bug_review_status() -> None:
            hook_calls.append("sent")

        class _FailingJsonEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                if hooks is not None and hooks.send_bug_review_status is not None:
                    await hooks.send_bug_review_status()
                raise RuntimeError("primary failed")

        class _FallbackJsonEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                if hooks is not None and hooks.send_bug_review_status is not None:
                    await hooks.send_bug_review_status()
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
            wants_bug_review_status=True,
        )

        response_json = await endpoint.execute_json_turn(
            request.to_json(),
            hooks=TicketExecutionHooks(
                send_bug_review_status=fake_send_bug_review_status,
            ),
        )

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(hook_calls, ["sent"])
        self.assertEqual(flow_outcome.raw_final_reply, "fallback-ok")
        self.assertEqual(updated_job.current_specialty, "docs")

    async def test_failover_json_endpoint_falls_back_when_primary_returns_malformed_json(self) -> None:
        class _MalformedPrimaryEndpoint:
            async def execute_json_turn(
                self,
                request_json: str,
                hooks: TicketExecutionHooks | None = None,
            ) -> str:
                return "{not-json"

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
            wants_bug_review_status=False,
        )

        response_json = await endpoint.execute_json_turn(request.to_json())

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "fallback-ok")
        self.assertEqual(updated_job.current_specialty, "docs")

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
            run_dir_root="/tmp/unused-run-root",
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
            wants_bug_review_status=False,
        )

        with self.assertLogs("ticket_investigation_subprocess_endpoint", level="WARNING") as logs:
            with patch(
                "ticket_investigation_subprocess_endpoint.TicketExecutionWorkspace.export_copy",
                side_effect=OSError("disk full"),
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "subprocess-export-ok")
        self.assertEqual(updated_job.current_specialty, "docs")
        self.assertTrue(any("Failed to export ticket execution subprocess workspace copy" in line for line in logs.output))

    async def test_codex_exec_json_endpoint_returns_success_even_if_export_copy_fails(self) -> None:
        fake_codex = (
            "import json,sys; "
            "response={"
            "'flow_outcome':{"
            "'raw_final_reply':'codex-export-ok',"
            "'conversation_history':[],"
            "'completed_agent_key':'data',"
            "'requires_human_handoff':False"
            "},"
            "'updated_job':{"
            "'channel_id':111,"
            "'mode':'investigating',"
            "'current_specialty':'data',"
            "'last_specialty':'data',"
            "'evidence':{'tx_hashes':[]}"
            "}"
            "}; "
            "sys.stdout.write(json.dumps(response))"
        )
        endpoint = CodexExecTicketExecutionJsonEndpoint(
            repo_root="/root/bots/discord/ysupport",
            codex_command=[sys.executable, "-c", fake_codex],
            allowed_command_prefixes=[[sys.executable, "-c", fake_codex]],
            run_dir_root="/tmp/unused-codex-run-root",
        )
        request = TicketExecutionTransportRequest(
            aggregated_text="help",
            input_list=[],
            current_history=[],
            run_context={
                "channel_id": 111,
                "project_context": "yearn",
                "repo_last_search_artifact_refs": [],
            },
            investigation_job={
                "channel_id": 111,
                "mode": "idle",
                "evidence": {"tx_hashes": []},
            },
            workflow_name="tests.endpoint.codex_export_failure",
            wants_bug_review_status=False,
        )

        with self.assertLogs("ticket_investigation_codex_endpoint", level="WARNING") as logs:
            with patch(
                "ticket_investigation_codex_endpoint.TicketExecutionWorkspace.export_copy",
                side_effect=OSError("disk full"),
            ):
                response_json = await endpoint.execute_json_turn(request.to_json())

        flow_outcome, updated_job = TicketExecutionTransportResult.from_json(
            response_json
        ).to_execution_parts()
        self.assertEqual(flow_outcome.raw_final_reply, "codex-export-ok")
        self.assertEqual(updated_job.current_specialty, "data")
        self.assertTrue(any("Failed to export codex ticket execution workspace copy" in line for line in logs.output))
