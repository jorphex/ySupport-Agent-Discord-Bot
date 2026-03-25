# ySupport Discord Bot

Discord support bot for Yearn.

Current scope:
- support tickets
- public trigger conversations
- docs/process answers
- vault/deposit/withdrawal help
- bounded repo/docs/onchain investigation
- offline knowledge-gap reporting from ticket transcripts

## Ticket Execution

Ticket investigations run through a configurable execution boundary.

Modes:
- `TICKET_EXECUTION_ENDPOINT=local`
  - default
  - runs the in-app investigation runtime in the bot process
- `TICKET_EXECUTION_ENDPOINT=subprocess`
  - runs the same investigation runtime in a separate worker process
- `TICKET_EXECUTION_ENDPOINT=codex_exec`
  - runs ticket execution through the Codex worker boundary

Optional fallback:
- `TICKET_EXECUTION_FALLBACK_ENDPOINT=local`
- `TICKET_EXECUTION_FALLBACK_ENDPOINT=subprocess`

Important execution settings:
- `TICKET_EXECUTION_CODEX_COMMAND`
- `TICKET_EXECUTION_CODEX_MODEL`
- `TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES`
- `TICKET_EXECUTION_ARTIFACT_DIR`
- `TICKET_EXECUTION_RUN_DIR_ROOT`

Execution behavior:
- sandboxes are per-run temporary workspaces
- persistent roots only receive exported copies after the run
- exported copies are read-only snapshots
- wrappers should use `TICKET_EXECUTION_RUN_DIR` for scratch access instead of assuming cwd is the live sandbox root

Health/status:
- `python -m ticket_execution_status`
- `python -m ticket_execution_status --smoke`

The status command reports:
- runtime validation
- execution primary/fallback mode
- sandbox/export policy
- repo-context status
- deterministic smoke result when `--smoke` is used

## RPC Configuration

Preferred per-chain RPC env vars:
- `ETHEREUM_RPC_URL`
- `BASE_RPC_URL`
- `ARBITRUM_RPC_URL`
- `OPTIMISM_RPC_URL`
- `POLYGON_RPC_URL`
- `SONIC_RPC_URL`
- `KATANA_RPC_URL`

Fallback behavior:
- if a per-chain RPC URL is unset, the code falls back to the shared `ALCHEMY_KEY` template where available
- Katana falls back to its default public RPC if unset

## Ticket Transcript Helper

Fetch a normalized Discord ticket transcript for review or regression work:

- `python -m ticket_transcript_fetch <channel_id_or_discord_link>`
- add `--limit N` to control history length
- add `--json` for normalized JSON output

The helper is read-only and only accesses channels the configured bot token can already read.

## Knowledge-Gap Worker

Offline support-quality worker for private internal reporting.

Typical dry run:
- `python -m knowledge_gap_worker <channel_id_or_discord_link> --dry-run`

Useful options:
- `--recent-closed N`
- `--preview-discovery`
- `--closed-only`
- `--limit N`
- `--report-channel-id <channel_id>`
- `--state-path <json_path>`
- `--max-posts N`

What it does:
- reads ticket transcripts from Discord
- grounds them against existing Yearn docs/repo tools
- classifies whether the ticket should become an internal report
- emits reports in these categories:
  - `docs_gap`
  - `faq_candidate`
  - `bot_behavior_gap`
  - `product_confusion`
  - `issue_draft_candidate`

Operational boundaries:
- reuses the same Discord bot token as `ysupport`
- separate offline workflow from the live support bot
- separate from `ysupportreporter`
- no Telegram/internal-chat ingestion
- no automatic public GitHub posting

## Notes

For local operator references, see:
- `WORKER_MAINTAINER_REFERENCE.md`
- `MODEL_MAINTAINER_REFERENCE.md`

These are intentionally local-only and gitignored.
