# ySupport Discord Bot

Yearn support for Discord tickets and public trigger conversations. The bot answers
docs/process questions and performs bounded investigation of vaults, deposits,
withdrawals, transactions and repository behavior. It prioritizes official sources
for product/process answers and tool evidence for stateful issues. Human help is
requested when the remaining action requires it. This repo includes the live bot,
ticket execution boundary and manual transcript-review tools.

Staff participation pauses automatic Discord replies. Pending Telegram handoffs
retain their context. A later substantive Telegram reply sends one Discord update
and closes the handoff; the bot stays paused. Telegram dismissal or Discord
**Stop Bot** closes the handoff and prevents further team-reply delivery. Accepted
Telegram replies are saved before acknowledgement. Bounded background workers
deliver independent handoffs; polling and dismissal remain responsive.
Explicit stop also cancels that handoff's queued or active investigation and
releases its capacity. Staff takeover keeps an accepted Telegram reply eligible.

## Documentation ingestion

- `yearn_rag/update_docs.sh` runs daily. It requires the repo `.venv` and permits
  one refresh at a time. Docs and repository stages run independently; either
  failure makes the job fail. Neither stage publishes an incomplete corpus.
- The generated docs checkout is shallow and sparse. Refresh preserves unchanged
  file timestamps and refuses to discard local edits or commits.
- Successful source fingerprints use `.cache/docs_ingestion/`. Unchanged sources
  skip OpenAI embeddings and Pinecone writes only after verification of the live
  namespace count and exact vector IDs. Current vectors are upserted before stale
  IDs are removed, so provider failure cannot empty the namespace.
- Repository indexing checks each source's current commit and the full manifest.
  Unchanged sources reuse verified cached files and rows. Changed revisions,
  settings, parser code or damaged caches trigger a rebuild. Artifact references
  stay stable for unchanged sections.
- Selected `yearn-metrics` and `yearn-envio` docs and source files provide accounting, product
  relationships and event-coverage context as `support_data`. Relevant sections
  remain eligible across product filters. Dated examples do not establish live
  balances or yields; generated data and unrelated operations files are excluded.
- Python, TypeScript, GraphQL, YAML and other text sources use overlapping chunks
  of at most 1,600 characters with a 200-character overlap. Later chunk titles
  identify line and character ranges; fetch returns the exact bounded chunk.
  Markdown, Solidity, Vyper and JSON keep their existing parsers.
  Broad support questions favor methodology docs over code chunks; explicit
  source filenames and identifiers retain code retrieval priority.
- Set `DOCS_INGESTION_FORCE_REFRESH=1` for a full repair. Per-stage status and last
  success use `.cache/docs_ingestion/{docs,repo}.status`. Production cron overwrites
  `yearn_rag/update_docs.log` each run to bound log growth.

## Runtime and tools

- Production and defaults use `codex_support_exec`. `support_boundary.py` owns the
  outer classifier and prompt; native evaluations use the same classifier.
- `TICKET_EXECUTION_MAX_CONCURRENT_TURNS` permits two investigations at once by
  default (range 1–8); production sets it to 8. Turns within a conversation remain
  serial. Session/cache maintenance starts only when idle and yields after its current deletion if
  support work queues. Remaining cleanup resumes on an idle pass. Telegram workers
  share the execution bound and claim each handoff once. Queue, setup and execution
  timings include ticket IDs and completed/error/canceled outcomes. Native token
  totals can include earlier turns in resumed sessions.
- Native prompts include the current request, response schema and compact tool
  map. Ordinary intake collects messages for two seconds. Owner follow-ups retain
  active work, suppress an outdated draft and run next without another collection
  delay. Original and new images remain available. Staff takeover and Stop Bot
  preserve their cancellation rules. `y:` also includes up to four recent user
  images from its bounded ticket history, with staff attachments labeled separately.
- The local specialist engine and JSON worker are retired; `local` fails explicitly.
  An external JSON backend requires `subprocess` and an explicit
  `TICKET_EXECUTION_SUBPROCESS_COMMAND`. `ticket_execution.replay_cli` supports
  native and explicit external backends; replay sessions are isolated by default.
- RPC investigation runs inside the bot through a private per-turn command.
  Transaction, contract, position and withdrawal helpers support bounded reads,
  plus allowed balance, nonce, fee, block and storage queries. Operations, provider
  calls, log ranges and time are limited. MCP has no RPC tools or credentials.
- `ticket_transcript_fetch.py` provides manual, read-only review; live turns do not
  import it. Docs namespace discovery is cached for five minutes; this cache does
  not store answers or wallet observations.
- Idle session cleanup also removes expired generated plugin catalogs and staging
  snapshots. It uses the configured session age and preserves auth, installed
  plugins and recent cache files.
- `support_dashboard_flex` reads yHelper's Ethereum Flex lending data through eight
  views: `protocol`, `markets`, `market`, `redemption_priority`, `trove_health`,
  `history`, `activity` and `performance`. Results retain freshness, units and partial
  coverage; bounded lists report truncation. Trove health is market-wide. Historical
  wallet-filtered performance does not establish current individual LTV.
- Prefer `support_dashboard_discover` for browsing vaults and comparing current
  yields. Use `search_vaults` for identity, exact details or known vaults outside
  dashboard coverage. Search returns numeric rows and `pagination.next_offset`;
  repeat the query, chain and sort for subsequent pages. Broad searches use the
  cached catalog. Exact vault addresses get snapshot details; other searches can
  set `include_details=true`. Sorts use `highest_apy` / `lowest_apy`; previous APR
  names remain aliases. Missing snapshot measurements stay null; incomplete
  snapshots are `partial`. Catalog age identifies the generation used by the call.
  yBOLD reports its APY input sources separately.

## Verification

Run offline checks with `.venv/bin/python -B -m unittest discover -s tests`.
OpenAI Agents remains a runtime dependency for the outer classifier.

Run each deterministic E2E module with `.venv/bin/python -B -m <module>`:

| Module | Coverage | JSON evidence under `.cache/verification/` |
| --- | --- | --- |
| `tests.optimization_e2e` | Support controllers, native execution protocol, persistence, Git refresh, SQLite indexing and evaluation fault detection | `optimization-e2e.json` |
| `tests.internal_rpc_e2e` | Real helpers and private client: wallet/gauge/older-vault balances, withdrawals, receipts, partial scans, nonces, budgets, rejected writes and cancellation | `internal-rpc-e2e.json` |
| `tests.latency_e2e` | Concurrent native processes, capacity and cancellation, idle cleanup, queued follow-ups, retained images and independent handoff delivery | `latency-e2e.json` |
| `tests.flex_e2e` | Authenticated MCP and HTTP: identifiers, validation, errors, freshness, units, pagination and partial/truncated results | `flex-e2e.json` |
| `tests.vault_lookup_e2e` | Authenticated MCP: pagination, catalog-only queries, exact details, fallbacks, units and yBOLD provenance | `vault-lookup-e2e.json` |

These runners use controlled external responses, send no live Discord/Telegram
messages and call no model or live RPC provider. Optimization probes must reject
skipped/wrong lookups, unwanted handoffs, stale discovery coverage and a broken
pending-team guard. Its evidence includes outcomes and source hashes.

Real-model native evaluation:

```sh
RUN_NATIVE_SUPPORT_E2E=1 .venv/bin/python -B -m tests.native_support_e2e \
  --output /private/native-support-e2e.json
```

Run as an unprivileged service user. Set an explicit, service-owned
`TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` and `OPENAI_API_KEY` for the classifier.
The runner imports code from a temporary working directory and uses explicit
settings, without reading the checkout's protected `.env`. Homes, sessions, ticket
state and provider observations stay private. Relative output paths are resolved
before the working directory changes.

The 44 former specialist LLM scenario IDs map to 42 distinct native cases, retaining
IDs and combined assertions for equivalent inputs. Two additional cases cover
confirmed bridge delivery and unavailable destination receipts. Use `--case <id>`
to select a scenario or alias.

Native cases use real controllers, the outer classifier, native execution and
production MCP schemas. Only Discord/Telegram I/O and data providers are fixtures.
An internal read must match the wallet, chain, vault or hash to obtain an observation;
report artifacts use read-only HTTP. Docs, repository and vault calls must match
the fixture subject. Lookup answers are never injected into turn context. These
cases verify collection and interpretation using controlled helper responses.

Multi-turn messages pass through `TicketBot.on_message`. After handoff, follow-ups
must stay in saved team context and make zero native calls. Checks cover handoff
flags, including docs cases that forbid escalation. JSON evidence records actual
requests, Discord output, state transitions, source/fixture hashes, failures and
timings. No live Discord/Telegram messages are sent.

For real-model helper/provider or Flex checks, add `--native` to
`tests.internal_rpc_e2e` or `tests.flex_e2e` under the same service-user, auth and
`RUN_NATIVE_SUPPORT_E2E=1` requirements. Internal RPC checks use a local HTTP
JSON-RPC node and controlled Kong data; Flex checks use controlled yHelper data.
Neither uses live RPC quota.

## Deployment

Production runs `ysupport.py` on the host under an unprivileged `ysupport` systemd
user. `mcp_server.py` runs in Docker. Use a dedicated bot `CODEX_HOME` and
service-owned auth; do not expose an operator's home directory.

MCP listens on `127.0.0.1:8001` and requires the configured bearer token before
executing tools. Its dedicated image pins the base image and full Python dependency
graph, contains only MCP runtime files, and runs read-only as a non-root user.
`compose.mcp.yaml` mounts only the generated repository SQLite database read-only
and passes only listed provider settings, without Discord/Telegram credentials.
Keep the existing `MCP_SERVER_API_KEY` in `.env`.

Before rebuilding for dependency or MCP source changes, tag the current image as
`ysupport-mcp:rollback`. `Dockerfile.mcp.dockerignore` permits only files copied by
`Dockerfile.mcp`; credentials, caches, tests and evidence cannot enter the build
context.

```sh
docker compose -f compose.mcp.yaml build
docker compose -f compose.mcp.yaml up -d
docker compose -f compose.mcp.yaml ps
```

Host setup:

1. Create a host venv; run `pip install -r requirements.txt`. Direct dependencies
   are pinned to reviewed versions. Verify `codex exec` works for the service user.
2. Keep `TICKET_EXECUTION_CODEX_HOME` on its dedicated bot path. The bot writes
   `config.toml` and instructions there; do not use your normal `~/.codex`.
3. Point `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` at a service-owned `0600` auth
   file. Keep `.env` root-owned with mode `0600`; systemd loads it before dropping
   privileges, then hides it from the service namespace.
4. Use `scripts/run_ysupport_host.sh` and the hardened `systemd/ysupport.service`.
   Keep the checkout and Codex installation read-only. Only dedicated state and
   private temporary directories need writes.
5. If checkout or Codex paths change, update `WorkingDirectory`, `EnvironmentFile`,
   `ExecStart` and both `BindReadOnlyPaths` entries together.
