# ySupport runtime and tool reference

## Execution settings

| Setting | Default or behavior |
| --- | --- |
| `TICKET_EXECUTION_ENDPOINT` | `codex_support_exec` |
| `TICKET_EXECUTION_CODEX_MODEL` | `gpt-6.2-sol` |
| `TICKET_EXECUTION_CODEX_REASONING_EFFORT` | `medium` |
| `TICKET_EXECUTION_MAX_CONCURRENT_TURNS` | `2`. Accepts `1` through `8`. The production host uses `8`. |
| `TICKET_EXECUTION_CODEX_SESSION_MAX_AGE_HOURS` | `168` |
| `TICKET_EXECUTION_SUBPROCESS_COMMAND` | Required when the endpoint is `subprocess`. The retired `local` endpoint fails validation. |

The concurrency limit applies across conversations. Turns within a conversation
run one at a time. Telegram delivery workers share the execution limit. Staff,
Telegram, and other noninteractive turns use one-shot Codex execution.

Session and cache cleanup starts when the bot is idle. If support work queues,
cleanup finishes the current deletion and yields. It expires generated plugin
catalogs and staging snapshots on the session schedule. It preserves auth,
installed plugins, and recent cache files.

Queue, setup, and execution logs include the ticket ID and outcome. Token counts
for a resumed session can include earlier turns. The native process receives
the bot virtual environment on `PATH`.

The host reads the Codex instruction template on each native turn.
[Runtime template](../ysupport/execution/ysupport_codex_instructions.md)

## MCP container settings

Compose exposes MCP on host loopback and passes only the listed provider
settings. The image uses a pinned base and dependency lock. It runs read-only
as a non-root user and mounts the SQLite index read-only. The build filter
includes only files used by the Dockerfile. [Compose file](../compose.mcp.yaml),
[build filter](../Dockerfile.mcp.dockerignore)

## Read-only MCP tools

| Tool | Use |
| --- | --- |
| `search_documentation` | Find Yearn and Flex docs. |
| `search_repo_context`, `fetch_repo_artifacts` | Find repository excerpts and fetch exact text by reference. |
| `repo_context_status` | Check the repository index for readiness and freshness. |
| `support_dashboard_discover` | Browse vaults and compare current yields. |
| `search_vaults` | Resolve a vault, inspect exact details, or find a vault outside dashboard coverage. |
| `support_dashboard_reports`, `support_dashboard_status`, `support_dashboard_changes` | Read reports, status, and changes from yHelper. |
| `support_dashboard_styfi`, `support_dashboard_flex` | Read stYFI data and Ethereum Flex lending data from yHelper. |

`search_vaults` returns numeric rows. To continue a page, use
`pagination.next_offset` with the same query, chain, and sort. Broad searches
read the cached catalog. An exact vault address loads details. You can also
request details with `include_details=true`. The APY sorts are `highest_apy` and
`lowest_apy`. The previous APR names still work as aliases.

Missing snapshot values return `null`. An incomplete snapshot reports
`partial`. Catalog age identifies the catalog version. yBOLD rows identify their
APY inputs. A false `ui_apy_observation.verified` leaves the displayed metric
and the cause of a displayed zero unresolved.

`support_dashboard_flex` supports `protocol`, `markets`, `market`,
`redemption_priority`, `trove_health`, `history`, `activity`, and `performance`.
Responses report freshness, units, incomplete coverage, and truncation.
`trove_health` describes the market. Wallet-filtered historical performance
cannot establish a wallet's current loan-to-value ratio.

## Private RPC reads

The bot supplies a private per-turn command for transaction, contract, position,
and withdrawal queries. It also supports bounded balance, nonce, fee, block, and
storage reads. The bot limits provider calls, log ranges, and elapsed time.
RPC credentials and operations stay inside the bot.

`read_contracts` accepts up to eight reads at one block. Set `tx_hash` to use the
receipt block. Block-end state can support a diagnosis. A trace or replay is
needed to prove the cause of a revert.

`get_wallet_history` checks supplied receipts or a wallet-filtered window of at
most 5,000 blocks. Kong supplies the latest 100 transfers for a contract as a
starting point. On Katana, Alchemy lacks indexed transfer history. yHelper
indexes harvests and reports there. Gaps in wallet history remain unresolved.

## Withdrawal evidence

For a confirmed payout, the current turn must find a positive underlying asset
transfer that matches the withdrawal receipt. `verify_payout` enables the extra
asset lookup. The bot can report a share burn separately when payout is unclear.
It keeps observed receipt hashes and tracks which hash a correction supersedes.
The chain, vault, and wallet in the request constrain proof selection.

A V3 explorer recipe needs a wallet-specific limit and a successful read-only
simulation at the same block. The recipe uses four-argument `redeem` with
`max_loss=0`. Partial capacity produces a partial recipe. An exact redemption
requires matching raw shares and recipient. A maximum redemption requires an
explicit full-balance request in the current turn. A failed simulation or unknown
vault version yields no recipe. The wallet must simulate again before submission.

The bot uses current-turn private tool results to render the verified amount,
block, and reference. Private Telegram notices can carry the verified evidence.

## Knowledge index

The pipeline refreshes a shallow, sparse Yearn checkout and fetches Flex docs.
It preserves unchanged files and protects local edits and commits. Document
indexing uses OpenAI `text-embedding-3-large` and Pinecone. For unchanged sources,
it checks the namespace count and vector IDs before skipping writes. It writes
current vectors before removing stale ones. Namespace metadata caches for five
minutes.

The repository index uses SQLite. It reuses verified files and rows for unchanged
source revisions. Changes to the revision, manifest, or parser trigger a rebuild,
as does a damaged cache. Artifact references stay stable for unchanged sections.

Python, TypeScript, GraphQL, YAML, and other text sources use chunks of up to
1,600 characters with 200 characters of overlap. Later chunk titles include the
line and character ranges. Markdown, Solidity, Vyper, and JSON use their own
parsers. Broad queries favor methodology docs. A source filename or identifier
gives matching code priority.

The source manifest selects accounting and event coverage from `yearn-metrics`
and `yearn-envio` as `support_data`. Relevant sections remain eligible across
product filters. Selected `governance-apps` files cover stYFI and stYFIx displays,
APR, Vote Boost, veYFI claims, and feature gates as `frontend_code`. The stYFI
source includes its root `deployment.json`. These selections omit generated
data, plans, mocks, and test evidence. For UI questions, matching topic
tokens give frontend excerpts priority and distinguish yvUSD from yvUSDC.
Contract questions use normal code ranking. Contracts describe protocol rules.
Live state and deployment checks establish current balances, yields, and feature
availability. [Source manifest](../ysupport/retrieval/repo_sources.json)
