# Compact vault lookup and explicit tool routing

Goal: keep Kong vault lookup distinct from dashboard yield browsing, reduce lookup
payloads and provider calls, and make all matched vaults reachable through paging.

Acceptance and constraints:
- Put the same conditional routing rule in both MCP descriptions: prefer dashboard
  discovery for browsing/comparing current yields; use Kong for identification,
  exact details and records outside dashboard analytics coverage. Manage tools here;
  the other server's skill is external and will receive a suggested sentence only.
- Return real structured MCP rows with numeric USD/APY values, numeric chain IDs,
  units, metric provenance, source timestamps and explicit total/offset/next-page data.
- Broad searches use the cached catalog without per-vault snapshots or strategies.
  Exact vault addresses and explicit details keep useful lifecycle, migration,
  staking, fees, PPS and strategy evidence; underlying-address searches remain broad.
- Add highest_apy/lowest_apy and retain the existing APR argument aliases. Preserve
  projection-before-oracle precedence, genuine zero and the official-address yBOLD
  display rule, including companion reuse and truthful base-only fallback.
- Preserve eleven tools and existing argument positions, active-first ordering,
  exact hidden/retired lookup and internal recommendation filters. No new dependency,
  ingestion, RPC access, skill file or bot-policy change. No live test messages.

Failure modes recorded before implementation:
- Broad lookup still makes snapshot requests or emits strategy cards; full catalog
  matches are claimed but unreachable, unstable ties skip rows, or filters change.
- Exact vault/chain gets confused with an underlying match; snapshot failure or
  wrong identity silently supplies another vault's details or looks current.
- Null/malformed/non-finite metrics become zero; APY becomes APR, percent scaling
  changes, yBOLD provenance/ranking breaks, or missing APY sorts as genuine zero.
- Source retrieval time is mistaken for metric time, or cached catalog ordering is
  confused with newer detail snapshots. Raw amounts lose precision.
- Actual structured MCP output breaks fixture callers, legacy tool arguments,
  native routing, recommendations, or exact retired-vault navigation.

- [x] Write HTTP/MCP E2E coverage and migrate obsolete prose fixtures before code.
- [x] Implement compact pageable lookup, selective details and explicit descriptions.
- [x] Verify E2E, relevant existing checks and native interpretation/routing; review.
- [x] Build and verify the hardened MCP candidate and prepare the release procedure.

Verified: eight vault HTTP/MCP scenarios, nine Flex scenarios, four real native
cases, sixteen existing fault probes and 294 existing tests pass. The retired
prose tests are covered by the provider/MCP scenarios. Review preserved unknown
strategy composition as null rather than zero and made the yield-question fixture
explicit rather than requiring an unsolicited APY quote. The live candidate matches
every runtime source file, retains eleven tools and rejects unauthenticated calls.
Default USDC text shrank from 11,613 to 6,245 characters, with 209 total matches
reachable by page. Evidence is in `.cache/verification/vault-lookup-review.json`.

Release procedure: commit the reviewed change, retain the current MCP image for
rollback, activate the verified candidate with existing Compose settings, and verify
live paging/details/aliases plus legacy reads. Keep the bot running: its policy and
runtime code are unchanged. Record health, commit/image, live responses and the
before/after payload measurement in the deployment artifact below. Remove temporary
candidate resources and evaluation output after preserving the evidence.

Verification evidence: `.cache/verification/vault-lookup-e2e.json`, native replay
artifacts and `.cache/verification/vault-lookup-deployment.json`.

## Completed Flex integration

Commit `33d838d` is deployed. All eight Flex views passed ten live calls, including
wallet filtering/pagination; bot and eleven-tool MCP were healthy. Evidence is in
`.cache/verification/flex-deployment.json`. Current changes preserve that tool.

## Completed support runtime cleanup

Goal: finish the authorized runtime, indexing and answer-grounding work while
preserving useful investigation tools, ticket data and existing deployment settings.

Completed:
- [x] Remove the stale marketing contact and show the complete business policy.
- [x] Use native Codex execution; retire the local specialist engine and worker.
- [x] Preserve explicit external JSON backends and saved/replay compatibility.
- [x] Refresh docs with a shallow/sparse checkout and reuse verified unchanged sources.
- [x] Incrementally index repository context; cache namespace discovery metadata.
- [x] Persist accepted Telegram replies before acknowledgement and deliver them in a
  separate worker; preserve staff takeover and pending-handoff behavior.
- [x] Reclaim idle locks and expired generated caches without touching auth,
  installed plugins, recent data or active sessions.
- [x] Keep transaction, position, withdrawal and basic chain-state reads internal
  through a private per-turn connection with operation/call/time/log bounds.
- [x] Preserve both helper modules, chain resolution and the older-vault registry.
  MCP retains ten docs/data tools and receives no RPC credentials.
- [x] Require evidence for the specific cause claimed; verify relevant bridge
  destination delivery or explicitly leave it unconfirmed.

Verification:
- Existing offline checks, helper/provider E2E and optimization fault probes use
  isolated state and controlled external services; add no unit tests.
- The 44 original LLM scenario IDs map to 42 distinct native cases. Two additional
  ticket-grounding cases cover confirmed arrival and unavailable destination receipts.
- The prior complete native run passed all original cases. The grounding pass
  verified four targeted cases/six turns and sixteen fault probes with real native
  answers and exact internal lookup identifiers. Equivalent uncertainty wording was
  accepted by regrading recorded answers; execution hashes and original grades remain
  in the evidence. No execution inputs or model answers were changed.
- Repeatable JSON artifacts and source hashes remain in `.cache/verification/`;
  temporary evaluation directories/logs and proven orphaned bytecode were removed.
- [x] Finish current offline checks, review the staged changes and commit.

Deployment constraint:
- The runtime/MCP changes were deployed and verified healthy on 2026-10-30.
- The latest two instruction refinements remain pending activation. Do not restart
  the bot or overwrite its loaded instructions until the user authorizes that step.
- This cleanup/commit pass sends no live messages and makes no deployment, credential,
  ticket-data or unrelated-service changes.
