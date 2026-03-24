import logging
import os
import unittest
import warnings

from agents import RunConfig, Runner

import config
from router import select_starting_agent
from state import (
    BotRunContext,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
)
from support_agents import (
    triage_agent,
    yearn_bug_triage_agent,
    yearn_data_agent,
    yearn_docs_qa_agent,
)
import tools_lib
from ticket_investigation_executor import (
    LocalTicketInvestigationExecutor,
    TransportTicketInvestigationExecutor,
)
from ticket_investigation_json_endpoint import (
    JsonEndpointTicketExecutionTransport,
    build_ticket_execution_json_endpoint,
)
from ticket_investigation_runtime import (
    TicketInvestigationRuntime,
    TicketTurnRequest,
    resolve_freeform_starting_agent,
)
from ticket_investigation_worker import TicketInvestigationWorker


RUN_LLM_E2E = bool(config.OPENAI_API_KEY) and os.getenv("RUN_LLM_E2E_TESTS") == "1"

AGENTS_BY_KEY = {
    "data": yearn_data_agent,
    "docs": yearn_docs_qa_agent,
    "bug": yearn_bug_triage_agent,
    "triage": triage_agent,
}

warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed transport <_SelectorSocketTransport.*",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    module=r"asyncio\.selector_events",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed <socket\.socket.*",
)
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
)


class _RepoContextNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            message.startswith("[CoreTool:search_repo_context] Repo context unavailable.")
            or message.startswith("[CoreTool:fetch_repo_artifacts] Repo context unavailable.")
        )


async def _close_agents_http_clients() -> None:
    try:
        from agents.models import openai_provider

        if openai_provider._http_client is not None:
            await openai_provider._http_client.aclose()
            openai_provider._http_client = None
    except Exception:
        pass

    try:
        await tools_lib.close_shared_openai_clients()
    except Exception:
        pass

    try:
        from agents.voice.models import openai_model_provider

        if openai_model_provider._http_client is not None:
            await openai_model_provider._http_client.aclose()
            openai_model_provider._http_client = None
    except Exception:
        pass


class LlmE2EBase(unittest.IsolatedAsyncioTestCase):
    maxDiff = None
    wallet_address = "0x1111111111111111111111111111111111111111"
    vault_address = "0x2222222222222222222222222222222222222222"
    _repo_context_noise_filter = _RepoContextNoiseFilter()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        logging.getLogger().addFilter(cls._repo_context_noise_filter)

    @classmethod
    def tearDownClass(cls) -> None:
        logging.getLogger().removeFilter(cls._repo_context_noise_filter)
        super().tearDownClass()

    async def asyncTearDown(self) -> None:
        await _close_agents_http_clients()
        await super().asyncTearDown()

    async def _run_support_turn(
        self,
        message: str,
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9001,
    ) -> tuple[str, BotRunContext, str]:
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        starting_agent_key = select_starting_agent(message, context)
        if starting_agent_key == "triage":
            starting_agent_key = await resolve_freeform_starting_agent(
                runner=Runner,
                input_list=message,
                run_context=context,
                workflow_name="tests.llm_e2e",
            )
        return await self._run_agent_turn(
            starting_agent_key,
            message,
            context=context,
        )

    async def _run_agent_turn(
        self,
        starting_agent_key: str,
        message,
        *,
        context: BotRunContext,
    ) -> tuple[str, BotRunContext, str]:
        starting_agent = AGENTS_BY_KEY[starting_agent_key]
        result = await Runner.run(
            starting_agent=starting_agent,
            input=message,
            context=context,
            max_turns=6,
            run_config=RunConfig(
                workflow_name="tests.llm_e2e",
                tracing_disabled=True,
            ),
        )
        return result.final_output or "", context, starting_agent_key

    async def _run_ticket_flow(
        self,
        message: str,
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9200,
        current_history=None,
        last_specialty: str | None = None,
        remembered_withdrawal_target: tuple[str, str] | None = None,
    ):
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = last_specialty
        if remembered_withdrawal_target is not None:
            investigation_job.remember_withdrawal_target(*remembered_withdrawal_target)
        history = current_history or []
        runtime = TicketInvestigationRuntime(Runner)
        for item in history:
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                runtime.merge_explicit_evidence(investigation_job, item["content"])
        runtime.merge_explicit_evidence(investigation_job, message)
        input_list = history + [{"role": "user", "content": message}]
        contextual_hints = runtime.build_contextual_hints(
            investigation_job,
            message,
            current_history=history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [
                {"role": "system", "content": " ".join(contextual_hints)}
            ] + [input_list[-1]]
        try:
            return await runtime.run_turn(
                TicketTurnRequest(
                    aggregated_text=message,
                    input_list=input_list,
                    current_history=history,
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

    async def _run_ticket_flow_through_executor(
        self,
        message: str,
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9300,
        current_history=None,
        last_specialty: str | None = None,
    ):
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = last_specialty
        history = current_history or []
        runtime = TicketInvestigationRuntime(Runner)
        for item in history:
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                runtime.merge_explicit_evidence(investigation_job, item["content"])
        runtime.merge_explicit_evidence(investigation_job, message)
        input_list = history + [{"role": "user", "content": message}]
        contextual_hints = runtime.build_contextual_hints(
            investigation_job,
            message,
            current_history=history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [
                {"role": "system", "content": " ".join(contextual_hints)}
            ] + [input_list[-1]]
        worker = TicketInvestigationWorker(runtime)
        local_executor = LocalTicketInvestigationExecutor(worker)
        endpoint = build_ticket_execution_json_endpoint(local_executor)
        transport = JsonEndpointTicketExecutionTransport(endpoint)
        executor = TransportTicketInvestigationExecutor(transport)
        try:
            return await executor.execute_turn(
                TicketTurnRequest(
                    aggregated_text=message,
                    input_list=input_list,
                    current_history=history,
                    run_context=context,
                    investigation_job=investigation_job,
                    workflow_name="tests.ticket_flow.executor",
                )
            )
        finally:
            clear_ticket_investigation_job(channel_id)

    async def _run_ticket_transcript(
        self,
        user_messages: list[str],
        *,
        initial_button_intent: str | None = None,
        channel_id: int = 9400,
        last_specialty: str | None = None,
    ) -> list:
        context = BotRunContext(
            channel_id=channel_id,
            project_context="yearn",
            initial_button_intent=initial_button_intent,
        )
        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        investigation_job.last_specialty = last_specialty
        runtime = TicketInvestigationRuntime(Runner)
        worker = TicketInvestigationWorker(runtime)
        history = []
        outcomes = []
        try:
            for index, message in enumerate(user_messages):
                if index > 0:
                    context.initial_button_intent = None
                runtime.merge_explicit_evidence(investigation_job, message)
                input_list = history + [{"role": "user", "content": message}]
                contextual_hints = runtime.build_contextual_hints(
                    investigation_job,
                    message,
                    current_history=history,
                )
                if contextual_hints:
                    input_list = input_list[:-1] + [
                        {"role": "system", "content": " ".join(contextual_hints)}
                    ] + [input_list[-1]]
                worker_result = await worker.execute_turn(
                    TicketTurnRequest(
                        aggregated_text=message,
                        input_list=input_list,
                        current_history=history,
                        run_context=context,
                        investigation_job=investigation_job,
                        workflow_name="tests.ticket_transcript",
                    )
                )
                outcomes.append(worker_result.flow_outcome)
                history = worker_result.flow_outcome.conversation_history
        finally:
            clear_ticket_investigation_job(channel_id)

        return outcomes

    @staticmethod
    def _get_agent_tool(agent, tool_name: str):
        for tool in agent.tools:
            if tool.name == tool_name:
                return tool
        raise AssertionError(f"Tool {tool_name} not found on agent {agent.name}.")

