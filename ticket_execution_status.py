import asyncio
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import config
from repo_context import get_repo_context_status


def build_ticket_execution_status(*, include_smoke_probe: bool = False) -> dict[str, Any]:
    runtime_validation_error = None
    try:
        config.validate_runtime_environment_config()
        runtime_validation_ok = True
    except ValueError as exc:
        runtime_validation_ok = False
        runtime_validation_error = str(exc)

    validation_error = None
    try:
        config.validate_ticket_execution_runtime_config()
        validation_ok = True
    except ValueError as exc:
        validation_ok = False
        validation_error = str(exc)

    endpoint_build = _probe_endpoint_build()
    uses_codex = (
        config.TICKET_EXECUTION_ENDPOINT == "codex_exec"
        or config.TICKET_EXECUTION_FALLBACK_ENDPOINT == "codex_exec"
    )
    status = {
        "runtime_environment": {
            "validation_ok": runtime_validation_ok,
            "validation_error": runtime_validation_error,
            "issues": config.runtime_environment_validation_issues(),
        },
        "ticket_execution": {
            "primary_endpoint": config.TICKET_EXECUTION_ENDPOINT,
            "fallback_endpoint": config.TICKET_EXECUTION_FALLBACK_ENDPOINT or None,
            "uses_codex": uses_codex,
            "artifact_dir": config.TICKET_EXECUTION_ARTIFACT_DIR or None,
            "run_dir_root": config.TICKET_EXECUTION_RUN_DIR_ROOT or None,
            "sandbox_policy": {
                "workspace_mode": "temporary_per_turn",
                "export_mode": (
                    "read_only_copy"
                    if config.TICKET_EXECUTION_ARTIFACT_DIR
                    or config.TICKET_EXECUTION_RUN_DIR_ROOT
                    else "ephemeral_only"
                ),
                "scratch_env_var": "TICKET_EXECUTION_RUN_DIR",
            },
            "summary": config.ticket_execution_runtime_summary(),
            "warnings": config.ticket_execution_runtime_warnings(),
            "validation_ok": validation_ok,
            "validation_error": validation_error,
            "endpoint_build_ok": endpoint_build["ok"],
            "endpoint_build_error": endpoint_build["error"],
            "endpoint_class": endpoint_build["endpoint_class"],
            "primary_command_probe": _probe_command(config.TICKET_EXECUTION_ENDPOINT),
            "fallback_command_probe": _probe_command(
                config.TICKET_EXECUTION_FALLBACK_ENDPOINT
            ),
        },
        "repo_context": get_repo_context_status(),
    }
    if include_smoke_probe:
        status["ticket_execution"]["smoke_probe"] = _probe_endpoint_smoke()
    return status


def main() -> int:
    include_smoke_probe = "--smoke" in sys.argv[1:]
    status = build_ticket_execution_status(include_smoke_probe=include_smoke_probe)
    print(json.dumps(status, indent=2, sort_keys=True))
    exit_ok = (
        0
        if status["ticket_execution"]["validation_ok"]
        and status["runtime_environment"]["validation_ok"]
        else 1
    )
    if include_smoke_probe:
        smoke_probe = status["ticket_execution"].get("smoke_probe", {})
        if not smoke_probe.get("ok"):
            return 1
    return exit_ok


def _probe_command(mode: str) -> dict[str, Any] | None:
    if not mode:
        return None
    command = _resolve_base_command(mode)
    if not command:
        return None

    executable = command[0]
    resolved_path: str | None = None
    if "/" in executable:
        resolved_path = str(Path(executable).resolve()) if Path(executable).exists() else None
    else:
        resolved_path = shutil.which(executable)

    return {
        "mode": mode,
        "command": command,
        "available": resolved_path is not None,
        "resolved_path": resolved_path,
    }


def _probe_endpoint_build() -> dict[str, Any]:
    try:
        from ticket_investigation_json_endpoint import build_ticket_execution_json_endpoint

        endpoint = build_ticket_execution_json_endpoint(_StatusExecutor())
    except ModuleNotFoundError as exc:
        return {
            "ok": None,
            "error": f"Endpoint build probe unavailable: {exc}",
            "endpoint_class": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "endpoint_class": None,
        }
    return {
        "ok": True,
        "error": None,
        "endpoint_class": endpoint.__class__.__name__,
    }


def _probe_endpoint_smoke() -> dict[str, Any]:
    try:
        from ticket_investigation_json_endpoint import build_ticket_execution_json_endpoint
        from ticket_investigation_transport import (
            TicketExecutionTransportRequest,
            TicketExecutionTransportResult,
        )
    except ModuleNotFoundError as exc:
        return {
            "ok": None,
            "error": f"Smoke probe unavailable: {exc}",
            "raw_final_reply": None,
        }

    request = TicketExecutionTransportRequest(
        aggregated_text="ticket execution smoke probe",
        input_list=[],
        current_history=[],
        run_context={
            "channel_id": 0,
            "category_id": None,
            "is_public_trigger": False,
            "project_context": "yearn",
            "initial_button_intent": None,
            "repo_search_calls": 0,
            "repo_fetch_calls": 0,
            "repo_searches_without_fetch": 0,
            "repo_last_search_query": None,
            "repo_last_search_artifact_refs": [],
        },
        investigation_job={
            "channel_id": 0,
            "requested_intent": "smoke_probe",
            "mode": "idle",
            "current_specialty": None,
            "last_specialty": None,
            "evidence": {
                "wallet": None,
                "chain": None,
                "tx_hashes": [],
            },
        },
        workflow_name="ticket_execution_status.smoke_probe",
        wants_bug_review_status=False,
        smoke_mode="ping",
    )
    try:
        endpoint = build_ticket_execution_json_endpoint(_StatusExecutor())
        response_json = asyncio.run(endpoint.execute_json_turn(request.to_json()))
        response = TicketExecutionTransportResult.from_json(response_json)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "raw_final_reply": None,
        }

    raw_final_reply = response.flow_outcome.get("raw_final_reply")
    smoke_ok = isinstance(raw_final_reply, str) and raw_final_reply.startswith(
        "ticket_execution_smoke_ok:"
    )
    return {
        "ok": smoke_ok,
        "error": None if smoke_ok else "Smoke probe returned unexpected payload.",
        "raw_final_reply": raw_final_reply,
    }


class _StatusExecutor:
    async def execute_turn(self, request, hooks=None):
        raise RuntimeError("Status executor should not execute turns.")


def _resolve_base_command(mode: str) -> list[str] | None:
    if mode == "subprocess":
        if config.TICKET_EXECUTION_SUBPROCESS_COMMAND:
            return list(config.TICKET_EXECUTION_SUBPROCESS_COMMAND)
        return [sys.executable, "-m", "ticket_investigation_worker_cli"]
    if mode == "codex_exec":
        if config.TICKET_EXECUTION_CODEX_COMMAND:
            return list(config.TICKET_EXECUTION_CODEX_COMMAND)
        return [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--color",
            "never",
            "--ephemeral",
        ]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
