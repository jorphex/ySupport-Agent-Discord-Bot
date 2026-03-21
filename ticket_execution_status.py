import json
from typing import Any

import config
from repo_context import get_repo_context_status


def build_ticket_execution_status() -> dict[str, Any]:
    validation_error = None
    try:
        config.validate_ticket_execution_runtime_config()
        validation_ok = True
    except ValueError as exc:
        validation_ok = False
        validation_error = str(exc)

    uses_codex = (
        config.TICKET_EXECUTION_ENDPOINT == "codex_exec"
        or config.TICKET_EXECUTION_FALLBACK_ENDPOINT == "codex_exec"
    )
    return {
        "ticket_execution": {
            "primary_endpoint": config.TICKET_EXECUTION_ENDPOINT,
            "fallback_endpoint": config.TICKET_EXECUTION_FALLBACK_ENDPOINT or None,
            "uses_codex": uses_codex,
            "artifact_dir": config.TICKET_EXECUTION_ARTIFACT_DIR or None,
            "run_dir_root": config.TICKET_EXECUTION_RUN_DIR_ROOT or None,
            "summary": config.ticket_execution_runtime_summary(),
            "warnings": config.ticket_execution_runtime_warnings(),
            "validation_ok": validation_ok,
            "validation_error": validation_error,
        },
        "repo_context": get_repo_context_status(),
    }


def main() -> int:
    status = build_ticket_execution_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ticket_execution"]["validation_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
