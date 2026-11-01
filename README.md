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

When support staff joins a Discord ticket, automatic bot replies pause. Any pending
Telegram handoff stays open, with its ticket context preserved. A later substantive
Telegram reply produces one Discord update and closes that handoff; the bot stays
paused afterward. Telegram dismissal or an explicit Discord **Stop Bot** action
closes the handoff and prevents further team-reply delivery.
Accepted Telegram replies are persisted before their updates are acknowledged.
One background worker delivers them, so a slow support turn does not block
polling or dismissal of another ticket's handoff.

Documentation ingestion:
- `yearn_rag/update_docs.sh` is the tracked daily refresh entrypoint
- the refresh uses the repository `.venv` and fails closed if that environment is missing
- successful source fingerprints are stored under `.cache/docs_ingestion/`
- the generated Yearn docs checkout is shallow and sparse; refreshes preserve
  unchanged documentation timestamps and refuse to discard local edits/commits
- unchanged sources skip OpenAI embeddings and Pinecone writes only after the
  live namespace count and exact vector IDs are verified
- refresh runs are single-instance; documentation and repo-context stages run
  independently, while each refuses to publish an incomplete corpus
- any failed stage makes the job exit unsuccessfully; per-stage status and last
  successful completion are kept in `.cache/docs_ingestion/{docs,repo}.status`
- repo-context builds verify each source's current commit and cover the complete
  manifest; unchanged sources reuse verified cached files and indexed rows
- changed revisions, source settings, parser code, or damaged caches trigger a
  rebuild while artifact references remain stable for unchanged sections
- set `DOCS_INGESTION_FORCE_REFRESH=1` for an explicit full repair
- refreshes upsert current vectors before removing stale IDs, so a provider
  failure does not empty the live namespace
- the production cron entry overwrites `yearn_rag/update_docs.log` on each run
  instead of retaining unbounded historical output

Runtime boundaries:
- production support turns and the default configuration use `codex_support_exec`
- `support_boundary.py` owns the live outer classifier and its prompt; native evaluations use it too
- the local specialist engine and its JSON worker are retired; `local` configuration fails explicitly
- `subprocess` remains available for an external JSON backend only when
  `TICKET_EXECUTION_SUBPROCESS_COMMAND` names its command
- `ticket_execution.replay_cli` supports native or explicit external backends;
  replay sessions stay separate from live sessions by default
- RPC investigation stays inside the bot. Native turns use a private per-turn
  command for the existing transaction, contract, position and withdrawal helpers,
  plus a restricted set of basic balance/nonce/fee/block/storage reads.
  Provider credentials stay with the bot; MCP exposes no RPC tools and receives
  no RPC credentials. Calls, operations, log ranges and time are bounded.
- `ticket_transcript_fetch.py` is a manual read-only review tool and is not imported by live support turns
- documentation namespace discovery is shared for five minutes; answers and
  wallet observations are not cached by this metadata optimization
- idle session maintenance also removes expired generated plugin catalogs and
  staging snapshots using the configured session age; installed plugins, auth,
  and recent cache files are outside that cache cleanup
- `support_dashboard_flex` reads yHelper's Ethereum Flex lending data through eight
  fixed views: protocol, markets, market, redemption_priority, trove_health, history,
  activity and performance. It preserves source freshness, units and partial coverage,
  and bounds returned lists with explicit truncation. Trove health is market-wide;
  historical wallet-filtered performance does not establish current individual LTV.
- Prefer `support_dashboard_discover` for browsing/comparing current vault yields;
  use `search_vaults` for identification, exact details or known vaults outside
  dashboard analytics coverage. Search returns structured numeric rows and
  `pagination.next_offset`; repeat the same query/chain/sort for the next page.
  Broad searches use the cached catalog. Exact vault addresses get snapshot details;
  other searches can request `include_details=true`. APY sorts use `highest_apy` /
  `lowest_apy`, with the previous APR names retained as aliases.
  Raw snapshot fields never inherit cached catalog measurements; missing values stay
  null and incomplete snapshots are marked `partial`. Catalog age stays bound to
  the generation used by the call, and yBOLD reports its input sources separately.

Verification:

```sh
.venv/bin/python -B -m tests.optimization_e2e
```

This runs the real support controllers, native execution protocol, persistence,
Git checkout refresh, and SQLite indexing with controlled external responses.
It sends no live Discord/Telegram messages and calls no model. Repeatable JSON
evidence, including source hashes and scenario outcomes, is written to
`.cache/verification/optimization-e2e.json`.

The 44 former specialist LLM scenarios map to 42 distinct native cases. Equivalent
inputs share an evaluation while retaining their IDs and combined assertions.
Additional ticket regressions cover confirmed bridge delivery and unavailable
recipient-chain receipts without changing that original coverage map.
The runner uses real ticket controllers, the outer classifier, native execution,
and the production MCP tool schemas:

```sh
RUN_NATIVE_SUPPORT_E2E=1 .venv/bin/python -B -m tests.native_support_e2e \
  --output /private/native-support-e2e.json
```

Run as an unprivileged service user with an explicit, service-owned
`TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` and `OPENAI_API_KEY` for the classifier.
The runner imports runtime code from a temporary working directory, so it does
not read the checkout's protected `.env`; explicit environment settings are used.
Homes, sessions, ticket state and provider observations stay private. Relative
output paths are resolved before changing the working directory.

Only Discord/Telegram I/O and data providers are fixtures. The native agent must
issue a private internal read with the correct wallet, chain, vault or hash before
it receives an observation; report artifacts use a read-only HTTP fixture.
Documentation/repository/vault MCP calls must match the fixture subject. No lookup
answers are injected into the turn context. These scenarios verify collection and
interpretation; their helper responses are controlled fixtures.

```sh
.venv/bin/python -B -m tests.internal_rpc_e2e
```

This additionally exercises the real helpers against a local HTTP JSON-RPC node
and controlled Kong data, through the native execution protocol and private client.
It covers wallet/gauge/older-vault balances, withdrawals, receipts, partial scans,
balances/nonces, budgets, rejected writes and cancellation. Evidence is written to
`.cache/verification/internal-rpc-e2e.json`. Add `--native` under the service user
with explicit bot auth and `RUN_NATIVE_SUPPORT_E2E=1` to run a real-model
investigation through those same helpers/providers. No live RPC quota is used.

Multi-turn messages go through `TicketBot.on_message`. After a handoff, a user
follow-up must remain in persisted team context and make zero further native
calls. Cases also grade handoff flags, including the docs cases where escalation
is forbidden. JSON evidence records actual requests, Discord output, controller
transitions, source/fixture hashes, failures, and timings. Use `--case <id>` for a
scenario or alias. No live Discord/Telegram messages are sent.

The optimization E2E also exercises these gates with an external model fixture:
skipped/wrong lookups, unwanted handoffs, and a deliberately broken pending-team
guard must fail. This deterministic check calls no real model.

The offline suite runs with `.venv/bin/python -B -m unittest discover -s tests`.
OpenAI Agents remains a runtime dependency for the live outer classifier.

```sh
.venv/bin/python -B -m tests.flex_e2e
.venv/bin/python -B -m tests.vault_lookup_e2e
```

The Flex runner exercises the real authenticated MCP surface and dashboard HTTP client against
controlled yHelper responses, including identifiers, argument validation, errors,
freshness, units, pagination and partial/truncated results. Evidence is written to
`.cache/verification/flex-e2e.json`. Add `--native` to `tests.flex_e2e` as the unprivileged service user
with explicit bot auth and `RUN_NATIVE_SUPPORT_E2E=1` to verify real-model tool
selection and answer interpretation. No live messages or RPC providers are used.
The vault lookup E2E also checks paging, catalog-only broad queries, exact details,
fallbacks, numeric units and yBOLD APY provenance through authenticated MCP HTTP.
Its evidence is `.cache/verification/vault-lookup-e2e.json`.

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
rejected before tool execution. Before rebuilding after dependency or MCP source
changes, tag the current image as `ysupport-mcp:rollback`. The tracked
`Dockerfile.mcp.dockerignore` is an allowlist containing only the files copied by
`Dockerfile.mcp`, so local credentials, caches, tests, and evidence assets never
enter the build context.

Important:
- do not point `TICKET_EXECUTION_CODEX_HOME` at your normal `~/.codex`
- the bot writes `config.toml` and instructions into its `CODEX_HOME`
- use `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to point at a `0600` auth file owned by the service user
- keep the checkout and Codex installation read-only; only the dedicated state directory and private temporary directory need writes
- keep `.env` root-owned with mode `0600`; systemd reads it before dropping privileges and then hides it from the service namespace

Minimal host-native setup:
- create a host venv and `pip install -r requirements.txt`; direct dependencies are pinned to the reviewed deployment versions
- ensure `codex exec` works for the service user
- leave `TICKET_EXECUTION_CODEX_HOME` on its dedicated bot path
- set `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to the service-owned auth file
- use `scripts/run_ysupport_host.sh` as the service entrypoint
- use `systemd/ysupport.service` as the hardened deployment unit
- if the checkout or Codex version path changes, update `WorkingDirectory`, `EnvironmentFile`, `ExecStart`, and the two `BindReadOnlyPaths` entries together
