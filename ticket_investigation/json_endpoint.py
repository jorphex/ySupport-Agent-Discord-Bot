import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from collections.abc import Sequence
import json
from typing import Protocol
from uuid import uuid4

import config
from ticket_investigation.codex_bundle import DEFAULT_CODEX_EXEC_COMMAND
from ticket_investigation.codex_endpoint import (
    CodexExecTicketExecutionJsonEndpoint,
    build_codex_exec_allowed_prefixes,
)
from ticket_investigation.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)
from ticket_investigation.executor import (
    TicketExecutionHooks,
    TicketInvestigationExecutor,
)
from ticket_investigation.subprocess_endpoint import SubprocessTicketExecutionJsonEndpoint
from ticket_investigation.transport import (
    build_smoke_transport_result,
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
        if transport_request.smoke_mode:
            return build_smoke_transport_result(
                transport_request,
                endpoint_mode="local",
            ).to_json()
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


class FailoverTicketExecutionJsonEndpoint:
    def __init__(
        self,
        primary: TicketExecutionJsonEndpoint,
        fallback: TicketExecutionJsonEndpoint,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        effective_hooks = _one_shot_hooks(hooks)
        try:
            response_json = await self.primary.execute_json_turn(
                request_json,
                hooks=effective_hooks,
            )
            TicketExecutionTransportResult.from_json(response_json)
            return response_json
        except Exception:
            try:
                response_json = await self.fallback.execute_json_turn(
                    request_json,
                    hooks=effective_hooks,
                )
                TicketExecutionTransportResult.from_json(response_json)
                return response_json
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Primary and fallback ticket execution endpoints both failed."
                ) from fallback_exc


class ShadowTicketExecutionJsonEndpoint:
    def __init__(
        self,
        primary: TicketExecutionJsonEndpoint,
        shadow: TicketExecutionJsonEndpoint,
        *,
        primary_mode: str,
        shadow_mode: str,
        artifact_dir: str | None = None,
    ) -> None:
        self.primary = primary
        self.shadow = shadow
        self.primary_mode = primary_mode
        self.shadow_mode = shadow_mode
        self.artifact_dir = artifact_dir or None
        self._shadow_tasks: set[asyncio.Task[None]] = set()

    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        response_json = await self.primary.execute_json_turn(
            request_json,
            hooks=hooks,
        )
        request = TicketExecutionTransportRequest.from_json(request_json)
        if request.smoke_mode:
            return response_json
        shadow_task = asyncio.create_task(
            self._run_shadow(request_json, response_json),
            name=f"ticket-execution-shadow:{self.shadow_mode}",
        )
        self._shadow_tasks.add(shadow_task)
        shadow_task.add_done_callback(self._shadow_tasks.discard)
        return response_json

    async def _run_shadow(
        self,
        request_json: str,
        primary_response_json: str,
    ) -> None:
        record: dict[str, object] = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_mode": self.primary_mode,
            "shadow_mode": self.shadow_mode,
            "request": json.loads(request_json),
            "primary_result": json.loads(primary_response_json),
        }
        try:
            shadow_response_json = await self.shadow.execute_json_turn(request_json)
            record["shadow_result"] = json.loads(shadow_response_json)
        except Exception as exc:
            record["shadow_error"] = repr(exc)
        self._write_shadow_record(record)

    def _write_shadow_record(self, record: dict[str, object]) -> None:
        if not self.artifact_dir:
            return
        artifact_root = Path(self.artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            f"-{self.shadow_mode}-{uuid4().hex}.json"
        )
        (artifact_root / filename).write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class ConditionalTicketExecutionJsonEndpoint:
    def __init__(
        self,
        default: TicketExecutionJsonEndpoint,
        canary: TicketExecutionJsonEndpoint,
        *,
        channel_ids: set[str],
        intents: set[str],
    ) -> None:
        self.default = default
        self.canary = canary
        self.channel_ids = set(channel_ids)
        self.intents = set(intents)

    async def execute_json_turn(
        self,
        request_json: str,
        hooks: TicketExecutionHooks | None = None,
    ) -> str:
        request = TicketExecutionTransportRequest.from_json(request_json)
        endpoint = self.canary if self._matches_canary(request) else self.default
        return await endpoint.execute_json_turn(request_json, hooks=hooks)

    def _matches_canary(self, request: TicketExecutionTransportRequest) -> bool:
        channel_id = request.run_context.get("channel_id")
        requested_intent = request.investigation_job.get("requested_intent")
        if channel_id is not None and str(channel_id) in self.channel_ids:
            return True
        if requested_intent and requested_intent in self.intents:
            return True
        return False


def build_ticket_execution_json_endpoint(
    delegate: TicketInvestigationExecutor,
) -> TicketExecutionJsonEndpoint:
    config.validate_ticket_execution_runtime_config()
    primary: TicketExecutionJsonEndpoint = _build_single_ticket_execution_json_endpoint(
        config.TICKET_EXECUTION_ENDPOINT,
        delegate,
    )
    fallback_mode = config.TICKET_EXECUTION_FALLBACK_ENDPOINT
    if not fallback_mode or fallback_mode == config.TICKET_EXECUTION_ENDPOINT:
        endpoint: TicketExecutionJsonEndpoint = primary
    else:
        fallback = _build_single_ticket_execution_json_endpoint(
            fallback_mode,
            delegate,
        )
        endpoint = FailoverTicketExecutionJsonEndpoint(primary, fallback)
    canary_mode = config.TICKET_EXECUTION_CANARY_ENDPOINT
    if canary_mode and (
        config.TICKET_EXECUTION_CANARY_CHANNEL_IDS
        or config.TICKET_EXECUTION_CANARY_INTENTS
    ):
        canary = _build_single_ticket_execution_json_endpoint(
            canary_mode,
            delegate,
        )
        endpoint = ConditionalTicketExecutionJsonEndpoint(
            endpoint,
            canary,
            channel_ids=config.TICKET_EXECUTION_CANARY_CHANNEL_IDS,
            intents=config.TICKET_EXECUTION_CANARY_INTENTS,
        )
    shadow_mode = config.TICKET_EXECUTION_SHADOW_ENDPOINT
    if shadow_mode:
        shadow = _build_single_ticket_execution_json_endpoint(
            shadow_mode,
            delegate,
        )
        return ShadowTicketExecutionJsonEndpoint(
            endpoint,
            shadow,
            primary_mode=config.TICKET_EXECUTION_ENDPOINT,
            shadow_mode=shadow_mode,
            artifact_dir=(
                config.TICKET_EXECUTION_SHADOW_ARTIFACT_DIR
                or config.TICKET_EXECUTION_ARTIFACT_DIR
                or None
            ),
        )
    return endpoint


def _build_single_ticket_execution_json_endpoint(
    mode: str,
    delegate: TicketInvestigationExecutor,
) -> TicketExecutionJsonEndpoint:
    if mode == "local":
        return ExecutorBackedTicketExecutionJsonEndpoint(delegate)
    if mode == "subprocess":
        command = list(_subprocess_command())
        return SubprocessTicketExecutionJsonEndpoint(
            command,
            allowed_command_prefixes=_allowed_subprocess_prefixes(command),
            cwd=config.TICKET_EXECUTION_SUBPROCESS_CWD,
            env=_subprocess_env(),
            artifact_dir=config.TICKET_EXECUTION_ARTIFACT_DIR or None,
            run_dir_root=config.TICKET_EXECUTION_RUN_DIR_ROOT or None,
        )
    if mode == "codex_exec":
        command = list(_codex_command())
        return CodexExecTicketExecutionJsonEndpoint(
            repo_root=config.BASE_DIR,
            codex_command=command,
            model=config.TICKET_EXECUTION_CODEX_MODEL,
            reasoning_effort=config.TICKET_EXECUTION_CODEX_REASONING_EFFORT,
            allowed_command_prefixes=build_codex_exec_allowed_prefixes(
                command,
                config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES,
            ),
            cwd=config.TICKET_EXECUTION_SUBPROCESS_CWD,
            env=_codex_env(),
            artifact_dir=config.TICKET_EXECUTION_ARTIFACT_DIR or None,
            run_dir_root=config.TICKET_EXECUTION_RUN_DIR_ROOT or None,
        )
    if mode == "codex_support_exec":
        command = list(_codex_command())
        return CodexSupportTicketExecutionJsonEndpoint(
            codex_command=command,
            model=config.TICKET_EXECUTION_CODEX_MODEL,
            reasoning_effort=config.TICKET_EXECUTION_CODEX_REASONING_EFFORT,
            repo_root=config.BASE_DIR,
            codex_home=config.TICKET_EXECUTION_CODEX_HOME,
            codex_auth_source=config.TICKET_EXECUTION_CODEX_AUTH_SOURCE,
            session_dir=config.TICKET_EXECUTION_CODEX_SESSION_DIR,
            session_max_age_hours=config.TICKET_EXECUTION_CODEX_SESSION_MAX_AGE_HOURS,
            ysupport_mcp_url=config.TICKET_EXECUTION_CODEX_YSUPPORT_MCP_URL,
            ysupport_mcp_container=config.TICKET_EXECUTION_CODEX_YSUPPORT_MCP_CONTAINER,
            mcp_server_api_key=config.MCP_SERVER_API_KEY,
            web_search_mode=config.TICKET_EXECUTION_CODEX_WEB_SEARCH_MODE,
            allowed_command_prefixes=build_codex_exec_allowed_prefixes(
                command,
                config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES,
            ),
            cwd=config.TICKET_EXECUTION_SUBPROCESS_CWD,
            env=_codex_env(),
            artifact_dir=config.TICKET_EXECUTION_ARTIFACT_DIR or None,
            run_dir_root=config.TICKET_EXECUTION_RUN_DIR_ROOT or None,
        )
    raise ValueError(f"Unsupported TICKET_EXECUTION_ENDPOINT value: {mode}")


def _subprocess_command() -> Sequence[str]:
    if config.TICKET_EXECUTION_SUBPROCESS_COMMAND:
        return config.TICKET_EXECUTION_SUBPROCESS_COMMAND
    return [sys.executable, "-m", "ticket_investigation_worker_cli"]


def _codex_command() -> Sequence[str]:
    if config.TICKET_EXECUTION_CODEX_COMMAND:
        return config.TICKET_EXECUTION_CODEX_COMMAND
    return DEFAULT_CODEX_EXEC_COMMAND


def resolve_ticket_execution_base_command(mode: str) -> list[str] | None:
    if mode == "subprocess":
        return list(_subprocess_command())
    if mode in {"codex_exec", "codex_support_exec"}:
        return list(_codex_command())
    return None


def _allowed_subprocess_prefixes(command: Sequence[str]) -> list[list[str]]:
    prefixes = [list(command)]
    for prefix in config.TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES:
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def _subprocess_env() -> dict[str, str]:
    allowed_keys = {
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "ALCHEMY_KEY",
        "MCP_SERVER_API_KEY",
        "GITHUB_TOKEN",
        "RUN_LLM_E2E_TESTS",
        "PATH",
        "HOME",
    }
    allowed_keys.update(config.TICKET_EXECUTION_SUBPROCESS_ENV_KEYS)

    allowed_prefixes = (
        "LLM_",
        "REPO_CONTEXT_",
        "TICKET_EXECUTION_",
    ) + tuple(config.TICKET_EXECUTION_SUBPROCESS_ENV_PREFIXES)

    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in allowed_keys or any(key.startswith(prefix) for prefix in allowed_prefixes):
            env[key] = value
    return env


def _codex_env() -> dict[str, str]:
    env = _subprocess_env()
    if config.TICKET_EXECUTION_CODEX_HOME:
        env["CODEX_HOME"] = config.TICKET_EXECUTION_CODEX_HOME
    return env


def _one_shot_hooks(hooks: TicketExecutionHooks | None) -> TicketExecutionHooks | None:
    if hooks is None or hooks.send_bug_review_status is None:
        return hooks
    sender = _OneShotHookSender(hooks.send_bug_review_status)
    return TicketExecutionHooks(
        send_bug_review_status=sender.send,
        send_progress_update=hooks.send_progress_update,
    )


class _OneShotHookSender:
    def __init__(self, delegate):
        self.delegate = delegate
        self.has_sent = False

    async def send(self) -> None:
        if self.has_sent:
            return
        self.has_sent = True
        await self.delegate()
