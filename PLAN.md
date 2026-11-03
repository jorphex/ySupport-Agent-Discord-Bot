# Cross-product source scope repair

Goal: retain cross-product metrics/indexer retrieval without admitting unrelated
risk reports into scoped security or contract queries.

Acceptance and failure modes:
- Official disclosure guidance remains retrievable for the reproduced security
  query; unrelated risk documents cannot bypass its product scope.
- Apply the same restriction to metadata fallback. Keep risk reports available
  to unscoped queries, and preserve metrics/indexer retrieval and legacy filtering.
- Add regression coverage before implementation. Verify production-index search,
  authenticated MCP search/fetch, source hashes and deployment health.
- No manifest, index, parser, dependency, credential, RPC or bot restart changes.

- [x] Reproduce authority competition; cover metadata leakage through ingestion E2E.
- [x] Narrow the exception to metrics/indexer support data; verify existing coverage.
- [x] Review and deploy a verified MCP image; retain rollback and verify live tools.

Only `support_data` tagged `metrics` or `indexing` bypasses product scope, in both
section search and metadata fallback. Risk reports retain normal unscoped access.
The new ingestion E2E reproduced six competing risk results before repair, then
verified official guidance, rejected out-of-scope metadata and retained data access.

Optimization E2E, fourteen existing repo/MCP tests, Ruff and diff checks passed.
Four security and seven existing search/fetch cases passed on candidate and live
MCP. Auth, legacy filtering, unscoped risk access and attached-tool retrieval pass.
Final review found no further gap. The MCP-only fix is active; the bot PID, index
hash, manifest and bearer key are unchanged. Rollback is retained; temporary
candidate resources were removed.

Evidence: `.cache/verification/security-source-{before,e2e,search-fetch,candidate-mcp,production-mcp,deployment}.json`.

## Completed metrics and indexer repository context

Goal: ingest selected `yearn-metrics` and `yearn-envio` documentation as
`support_data`, with useful search/fetch results for normal support questions.

Acceptance and constraints:
- Pull explicit Markdown paths only. Exclude generated data, benchmarks, tests,
  dependencies and unrelated operations material. Keep daily incremental refresh.
- Make relevant support data eligible across product filters while retaining
  contract-code scoping, legacy filtering, result bounds and artifact references.
- Verify yvUSD product membership, fees/earnings, TVL overlap and indexed-event
  limits through real ingestion, search, fetch and authenticated MCP calls.
- Treat methodology and dated examples as source context, not live state.
- No parser expansion, new service/tool, RPC, dependency, credential or bot restart.

Failure modes recorded before implementation:
- Product filtering silently excludes relevant metrics/indexer documentation.
- Cross-product eligibility leaks unrelated contract code or legacy-only sources.
- Metadata fallback still applies the old filter and loses relevant support data.
- Broad globs ingest generated or operational noise; source documents are missing
  or produce no searchable sections; source changes break existing contract lookup.
- MCP serves a database and manifest from different versions after deployment.

- [x] Add E2E regression coverage and reproduce the filtering failure.
- [x] Add narrow source lists and correct support-data eligibility.
- [x] Build a candidate index; verify live-source search/fetch and existing queries.
- [x] Review, run relevant checks and activate a verified MCP/index pair.

Added 16 exact Markdown paths: thirteen metrics docs and three indexer docs,
107,159 characters and 79 sections. Generic READMEs were omitted because the
existing basename matcher also selected nested operations material. No parser
change was needed. The index now covers fifteen repositories and reused thirteen
unchanged source configurations. GitHub throttled repeat metadata lookups; the
narrowed build used commit revisions already resolved live in this pass.

Verification: the new product-filter regression failed before repair; optimization
E2E then passed, including metadata fallback and contract scoping. Fourteen existing
repo/MCP tests, Ruff and diff checks passed. Seven search/fetch cases passed against
the candidate and production MCP, including existing contract lookup. Legacy
filtering, eleven-tool discovery, unauthenticated rejection and attached-tool
search/fetch passed. Follow-up review found the broad support-data exception could
displace official security guidance; the repair is tracked above.

The verified MCP image and index are active. The previous image/index remain
available for rollback. The bot PID and MCP bearer key are unchanged; no RPC
credentials were added. Temporary candidate resources were removed.

Evidence: `.cache/verification/repo-sources-{before,optimization-e2e,build,search-fetch,candidate-mcp,production-mcp,deployment}.json`.

## Completed documentation revision

Goal: shorten maintained documentation without losing meaning, conditions,
technical identifiers, commands, limits or evidence references. Use direct verbs,
consistent terms and short sentences, guided by ASD-STE100.

Scope: README, privacy policy, this plan and the two local maintainer guides.
Preserve runtime instructions and historical ticket notes. No service, policy,
dependency, credential or ticket-data changes.

- [x] Inventory documents; check runtime claims against source and deployment evidence.
- [x] Remove repetition and simplify prose; preserve technical and privacy requirements.
- [x] Check commands, paths, limits and meaning; review the diff and resolve findings.

Acceptance: documentation is shorter, requirements remain explicit, code blocks
and local references are valid, and completed work has no stale pending actions.

Verified nine CLI modules and their documented flags, referenced source paths,
Markdown fences and shell-block syntax. Configuration identifiers and privacy
conditions are preserved. Corrected the completed restart status and the status
probe description. Runtime instructions and service state are unchanged.

## Completed discovery coverage repair

Commit `8ebf875` computes coverage from filtered rows before pagination. Empty
results have zero counts and a null ratio; known zero APY counts as available.
Fixture-match semantics and production tools are unchanged.

The empty-result regression reproduced stale two-vault coverage before repair.
All nineteen evaluation probes then behaved as expected, including the three
coverage cases and existing fault detection. Optimization E2E, Ruff and diff
checks passed. This test-only change required no restart.

Evidence: `.cache/verification/discovery-coverage-{before,e2e}.json`.

## Completed vault provenance and evaluation repair

Commit `332beca` separates catalog and snapshot observations. Missing snapshot
APY, TVL, fees, PPS or composition remain unknown; incomplete snapshots are
`partial`. Catalog timestamps identify the generation used by the call, even
during concurrent refreshes; reported age includes detail-fetch time. yBOLD
preserves its display rule and reports each input's source and time separately.

Native discovery cases use the production response projection over controlled
raw data and the exact eleven-tool catalog. Fixtures honor identity, chain,
filters and pagination. Answers requiring links, lifecycle flags or strategy
counts must fetch them; discovery fixtures cannot invent those fields.

All eleven HTTP/MCP scenarios passed on the host and in the final Docker image.
Four real-model native cases, 294 existing tests, sixteen fault probes and nine
Flex scenarios passed. Candidate reads verified lookup, yBOLD inputs, auth and
existing tools. Ruff, diff, Compose and deployed source comparisons passed.

The MCP-only release retained a rollback image, canonical auth and ticket state.
Live checks confirmed pagination, aliases, provenance and existing reads. The bot
stayed running; temporary evaluation and container resources were removed after
their evidence was saved.

Evidence: `.cache/verification/vault-provenance-{e2e,review,deployment}.json`.

## Completed compact lookup and Flex integration

- `0f0f3f5`: compact numeric vault rows, pagination, selective snapshot details
  and APY sort aliases. Evidence: `.cache/verification/vault-lookup-deployment.json`.
- `33d838d`: Ethereum Flex lending data through eight views. Ten live calls
  verified views, wallet filters and pagination; bot and eleven-tool MCP were
  healthy. Evidence: `.cache/verification/flex-deployment.json`.

## Completed support runtime cleanup

- Removed the stale marketing contact; retained the complete business policy.
- Replaced the local specialist engine and worker with native Codex execution.
  Preserved explicit external JSON backends and saved-request/replay compatibility.
- Added shallow/sparse docs refresh, verified reuse of unchanged sources,
  incremental repository indexing and cached namespace discovery.
- Persisted accepted Telegram replies before acknowledgement; a separate worker
  delivers them while preserving staff takeover and pending handoffs.
- Reclaimed idle locks and expired generated caches while preserving auth,
  installed plugins, recent data and active sessions.
- Kept transaction, position, withdrawal and basic chain reads internal, with
  per-turn operation, provider-call, log-range and time limits. Preserved both
  helper modules, chain resolution and the older-vault registry. MCP has no RPC
  credentials or RPC tools.
- Required evidence for each claimed cause. Relevant bridge delivery requires
  destination evidence or an explicit statement that arrival is unconfirmed.

The 44 former LLM scenario IDs map to 42 native cases. Two more cases cover
confirmed bridge arrival and unavailable destination receipts. The original
cases passed a full native run; targeted grounding checks passed four cases/six
turns and sixteen fault probes. Regrading accepted equivalent uncertainty wording
without changing execution inputs or model answers. Original grades and hashes
remain in the evidence. Temporary evaluation logs and proven orphaned bytecode
were removed; repeatable JSON evidence remains in `.cache/verification/`.

Runtime/MCP cleanup was deployed on October 30, 2026. The authorized November 1
bot restart completed. Discord reconnected, the configured MCP exposed all eleven
current tools, and live lookup returned both yvUSD variants.
Evidence: `.cache/verification/bot-restart-20261101.json`.
