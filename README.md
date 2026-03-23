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
- execution sandboxes are always per-run temporary workspaces; persistent roots only receive exported copies after the run completes
- exported sandbox copies are archived as read-only snapshots so they are not reused as mutable workspaces
- custom subprocess or Codex wrappers should use `TICKET_EXECUTION_RUN_DIR` for scratch access instead of assuming cwd is the sandbox root

Useful rollout env vars:

- `TICKET_EXECUTION_CODEX_COMMAND`
  - override the base Codex command
- `TICKET_EXECUTION_CODEX_MODEL`
  - optional model override for `codex_exec`
- `TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES`
  - extra allowed command prefixes for bounded external execution
- `TICKET_EXECUTION_ARTIFACT_DIR`
  - export per-run request/prompt/stdout/stderr/metadata artifacts from the disposable sandbox
- `TICKET_EXECUTION_RUN_DIR_ROOT`
  - export per-run scratch-directory copies without forcing full artifact capture

RPC env vars:

- `ETHEREUM_RPC_URL`
- `BASE_RPC_URL`
- `ARBITRUM_RPC_URL`
- `OPTIMISM_RPC_URL`
- `POLYGON_RPC_URL`
- `SONIC_RPC_URL`
- `KATANA_RPC_URL`
- if a per-chain RPC URL is unset, the code currently falls back to the shared `ALCHEMY_KEY` template where available, except Katana which falls back to its default public RPC

Suggested rollout order:

1. `local`
2. `subprocess`
3. `codex_exec` with `TICKET_EXECUTION_FALLBACK_ENDPOINT=local`
4. `codex_exec` without fallback only after you trust the worker path

Health check:

- run `python -m ticket_execution_status`
- it prints JSON with:
  - runtime environment validation
  - execution primary/fallback
  - workspace/artifact policy
  - validation result and warnings
  - repo-context status
- exit code is nonzero when the runtime environment or ticket execution config is invalid
- the Docker image also uses this command as its container `HEALTHCHECK`
- run `python -m ticket_execution_status --smoke` to execute a deterministic smoke turn through the currently configured endpoint and fallback path

## Ticket Transcript Helper

For regression capture and ticket review, you can fetch a read-only normalized transcript from Discord without copying chat messages manually.

- run `python -m ticket_transcript_fetch <channel_id_or_discord_link>`
- add `--limit N` to control how many recent messages to fetch
- add `--json` to emit normalized JSON instead of plain text
- the helper uses the configured `DISCORD_BOT_TOKEN` and only reads message history for channels the bot can already access

## Knowledge-Gap Worker

For offline support-quality review, the repo now includes a Discord-only knowledge-gap worker that reuses the same `DISCORD_BOT_TOKEN` as `ysupport`.

- run `python -m knowledge_gap_worker <channel_id_or_discord_link> --dry-run`
- add more ticket channels to analyze multiple tickets in one run
- add `--limit N` to control transcript size
- use `--report-channel-id <channel_id>` to override the private report sink
- use `--closed-only` to skip any ticket channel whose name does not start with `closed-`
- use `--state-path <json_path>` to persist already-posted report signatures for dedupe across runs
- use `--max-posts N` to cap how many reports can be posted in a single run after filtering and dedupe

Phase-1 behavior:

- reads Discord ticket history only
- grounds the ticket against existing Yearn docs and repo tools
- classifies whether the ticket should become a private internal report
- emits docs-gap / FAQ / bot-behavior / product-confusion / issue-draft style reports
- posts to `KNOWLEDGE_GAP_REPORT_CHANNEL_ID`; set it explicitly in the environment for the private destination channel

Phase-2 behavior:

- supports bounded multi-ticket runs
- can skip open/live tickets when you only want closed-ticket review
- dedupes reports within a run and optionally across runs via a persisted signature file
- can bound the number of reports posted in one run so a future scheduled job cannot spam the private channel

Phase-3 behavior:

- groups matching report signatures within a run into one private report
- keeps multiple affected ticket references in the same formatted message
- counts grouped themes against `--max-posts`, not raw ticket count

Intentional boundaries:

- same Discord bot token as `ysupport`
- separate offline workflow from the live support bot
- separate workflow from `ysupportreporter`
- no Telegram/internal-chat ingestion in phase 1
- no automatic public GitHub posting
