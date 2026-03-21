from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from ticket_investigation_transport import TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA


DEFAULT_CODEX_EXEC_COMMAND = [
    "codex",
    "exec",
    "--skip-git-repo-check",
    "--color",
    "never",
    "--ephemeral",
]


@dataclass
class CodexTicketExecutionBundle:
    command: list[str]
    prompt_text: str
    prompt_path: Path
    request_path: Path
    response_schema_path: Path
    expected_response_path: Path | None
    run_dir: Path


def build_codex_ticket_execution_bundle(
    *,
    request_json: str,
    run_dir: str | Path,
    repo_root: str | Path,
    codex_command: Sequence[str] | None = None,
    model: str | None = None,
    response_json_override: str | None = None,
) -> CodexTicketExecutionBundle:
    run_dir_path = Path(run_dir)
    repo_root_path = Path(repo_root)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    request_path = run_dir_path / "request.json"
    response_schema_path = run_dir_path / "response_schema.json"
    prompt_path = run_dir_path / "codex_prompt.txt"
    expected_response_path = (
        run_dir_path / "expected_response.json"
        if response_json_override is not None
        else None
    )

    request_path.write_text(request_json, encoding="utf-8")
    response_schema_path.write_text(
        json.dumps(TICKET_EXECUTION_TRANSPORT_RESULT_SCHEMA, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if expected_response_path is not None:
        expected_response_path.write_text(response_json_override, encoding="utf-8")
    prompt_text = _codex_ticket_execution_prompt(
        request_path=request_path,
        response_schema_path=response_schema_path,
        repo_root=repo_root_path,
        expected_response_path=expected_response_path,
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")

    command = list(codex_command or DEFAULT_CODEX_EXEC_COMMAND)
    if model:
        command.extend(["-m", model])
    command.extend(
        [
            "--output-schema",
            str(response_schema_path),
            "-C",
            str(run_dir_path),
            "--add-dir",
            str(repo_root_path),
            "-",
        ]
    )

    return CodexTicketExecutionBundle(
        command=command,
        prompt_text=prompt_text,
        prompt_path=prompt_path,
        request_path=request_path,
        response_schema_path=response_schema_path,
        expected_response_path=expected_response_path,
        run_dir=run_dir_path,
    )


def _codex_ticket_execution_prompt(
    *,
    request_path: Path,
    response_schema_path: Path,
    repo_root: Path,
    expected_response_path: Path | None,
) -> str:
    if expected_response_path is not None:
        return (
            "You are running a deterministic smoke probe for the ticket execution boundary.\n\n"
            f"Return exactly the JSON contents of {expected_response_path.name}.\n"
            "Do not modify the JSON.\n"
            "Do not emit prose outside the JSON object.\n"
        )
    return (
        "You are an isolated ticket investigation worker.\n\n"
        f"Read the ticket execution request from {request_path.name}.\n"
        f"Return only JSON matching {response_schema_path.name}.\n"
        f"The Yearn repo root is available at {repo_root}.\n"
        "Use the provided run directory for temporary working files.\n"
        "Do not emit prose outside the final JSON object.\n"
    )
