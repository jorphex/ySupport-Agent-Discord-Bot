# ySupport Discord Bot

A Discord support bot for Yearn that handles ticket triage, public-trigger interactions, and structured follow‑ups. It uses tool-backed agents for vault lookups, deposit checks, and docs Q&A, and applies guardrails to detect and handle BD/PR/listing requests. Built to keep support fast and consistent across public and ticketed conversations.

- Triage support tickets and public triggers
- Provide vault info, deposit checks, and docs Q&A
- Detect and redirect BD/PR/listing requests

## Ticket Execution Modes

Ticket investigation turns run through a configurable execution boundary.

- `TICKET_EXECUTION_ENDPOINT=local`
  - default
  - runs the investigation worker in-process
- `TICKET_EXECUTION_ENDPOINT=subprocess`
  - runs the JSON worker contract in a separate process
- `TICKET_EXECUTION_ENDPOINT=codex_exec`
  - runs the JSON worker contract through the Codex bundle/command boundary

Optional failover:

- `TICKET_EXECUTION_FALLBACK_ENDPOINT=local`
- `TICKET_EXECUTION_FALLBACK_ENDPOINT=subprocess`

Operational policy:

- if any execution mode uses `codex_exec`, set either `TICKET_EXECUTION_ARTIFACT_DIR` or `TICKET_EXECUTION_RUN_DIR_ROOT`
- if `codex_exec` is the primary mode, a fallback endpoint is strongly recommended

Useful rollout env vars:

- `TICKET_EXECUTION_CODEX_COMMAND`
  - override the base Codex command
- `TICKET_EXECUTION_CODEX_MODEL`
  - optional model override for `codex_exec`
- `TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES`
  - extra allowed command prefixes for bounded external execution
- `TICKET_EXECUTION_ARTIFACT_DIR`
  - persist per-run request/prompt/stdout/stderr/metadata artifacts
- `TICKET_EXECUTION_RUN_DIR_ROOT`
  - persist per-run scratch directories without full artifact capture

Suggested rollout order:

1. `local`
2. `subprocess`
3. `codex_exec` with `TICKET_EXECUTION_FALLBACK_ENDPOINT=local`
4. `codex_exec` without fallback only after you trust the worker path

Health check:

- run `python -m ticket_execution_status`
- it prints JSON with:
  - execution primary/fallback
  - workspace/artifact policy
  - validation result and warnings
  - repo-context status
- exit code is nonzero when the ticket execution config is invalid
