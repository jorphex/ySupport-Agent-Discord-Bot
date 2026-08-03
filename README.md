# ySupport Discord Bot

Discord support bot for Yearn.

It handles:
- support tickets
- public trigger conversations
- docs/process answers
- vault/deposit/withdrawal help
- bounded repo/docs/onchain investigation

The bot is built to keep support grounded:
- official-source-first for docs/process/product questions
- tool-grounded for tx/account/protocol/runtime issues
- explicit human handoff where the bot should not guess

The repo also contains:
- the live Discord bot runtime
- the ticket investigation runtime and execution boundary
- transcript-fetch tooling for ticket review
- an offline knowledge-gap worker for private internal reporting from support tickets

Documentation ingestion:
- `yearn_rag/update_docs.sh` is the tracked daily refresh entrypoint
- successful source fingerprints are stored under `.cache/docs_ingestion/`
- unchanged sources skip OpenAI embeddings and Pinecone writes only after the
  live namespace count and exact vector IDs are verified
- refresh runs are single-instance and abort on any missing or unreadable source
  instead of publishing a partial corpus
- repo-context builds cover the complete manifest and use stable artifact
  references across unchanged daily rebuilds
- set `DOCS_INGESTION_FORCE_REFRESH=1` for an explicit full repair
- refreshes upsert current vectors before removing stale IDs, so a provider
  failure does not empty the live namespace

Runtime boundaries:
- production support turns use `codex_support_exec`
- `support_boundary.py` is the live outer classifier shared with explicit replay
- `support_agents.py` and `ticket_investigation/runtime.py` are the legacy local backend retained for explicit local/subprocess replay and LLM evaluation
- `knowledge_gap_worker.py` and `knowledge_gap_reporting.py` are manual offline analysis tools and are not imported by live support turns

Host-native service mode

The bot can be run outside Docker under `systemd` while keeping the MCP server in Docker.

Recommended shape:
- keep `mcp_server.py` in Docker
- run `ysupport.py` on the host
- run the bot as a dedicated, unprivileged `ysupport` service user
- keep a dedicated bot `CODEX_HOME` for generated Codex config
- keep bot auth in service-owned state rather than exposing an operator's home directory

The MCP listener is host-local and bearer-authenticated. Its dedicated image
pins the base image and complete Python dependency graph, contains only the MCP
runtime, runs read-only as a non-root user, and receives only its explicitly
listed provider settings. Keep the existing `MCP_SERVER_API_KEY` in `.env`.

```sh
docker compose -f compose.mcp.yaml build
docker compose -f compose.mcp.yaml up -d
docker compose -f compose.mcp.yaml ps
```

`compose.mcp.yaml` publishes only `127.0.0.1:8001`, mounts only the generated
repo-context SQLite database read-only, and does not pass Discord or Telegram
credentials into MCP. HTTP requests without the configured bearer token are
rejected before tool execution. Rebuild after dependency or MCP source changes;
Compose retains the prior image locally for rollback. The tracked
`.dockerignore` is an allowlist containing only the files copied by
`Dockerfile.mcp`, so local credentials, caches, tests, and evidence assets never
enter the build context.

Important:
- do not point `TICKET_EXECUTION_CODEX_HOME` at your normal `~/.codex`
- the bot writes `config.toml` and instructions into its `CODEX_HOME`
- use `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to point at a `0600` auth file owned by the service user
- keep the checkout and Codex installation read-only; only the dedicated state directory and private temporary directory need writes

Minimal host-native setup:
- create a host venv and `pip install -r requirements.txt`
- ensure `codex exec` works for the service user
- leave `TICKET_EXECUTION_CODEX_HOME` on its dedicated bot path
- set `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to the service-owned auth file
- use `scripts/run_ysupport_host.sh` as the service entrypoint
- use `systemd/ysupport.service` as the hardened deployment unit
- if the checkout or Codex version path changes, update `WorkingDirectory`, `EnvironmentFile`, `ExecStart`, and the two `BindReadOnlyPaths` entries together
