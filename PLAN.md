# Flex lending data integration

Goal: expose yHelper's existing Ethereum Flex lending data to native investigations
through one MCP read tool, without adding providers, ingestion or RPC exposure.

Acceptance and constraints:
- Support protocol overview, market discovery/detail, redemption priority,
  aggregate trove health, history, activity and optional wallet-filtered performance.
- Use the existing dashboard HTTP client and a fixed view-to-route mapping; validate
  view-specific identifiers/arguments before I/O. Bound lists and retain truncation,
  cursors, freshness, source blocks, decimals and performance coverage.
- Preserve null/unknown values. Rates/LTV are decimal ratios; USD values are estimates.
  Lower-rate debt is not a guaranteed redemption outcome; aggregate trove health
  and historical PnL do not establish an individual wallet's current LTV or safety.
- Preserve the ten existing tools. Keep provider credentials/RPC internal. No new
  model, dependency, automatic routing rule or live test message.
- Write controlled HTTP/MCP E2E coverage before implementation; verify native tool
  selection and answer interpretation, live yHelper reads, and an isolated image.

Failure modes recorded before implementation:
- Wrong view/path/wallet/market reaches the provider, or irrelevant parameters are
  silently ignored; generic URLs, malformed addresses or cursor injection escape
  the intended endpoint set.
- Truncated curves/history/performance look complete; pagination is lost; source
  coverage is confused with returned rows; null/stale/missing data becomes zero.
- Decimal rates or raw token units are misread; independent source blocks are mixed;
  lower-rate debt or market-wide health becomes a personal redemption/safety promise.
- Provider failures/malformed payloads become successful empty data; existing MCP
  tools or auth regress; native tests use fake answers instead of observed reads.
- Tests call live RPC, touch production homes/state, leak credentials, or restart
  the bot/MCP outside the explicitly approved activation.

- [x] Add controlled HTTP/MCP and native replay cases.
- [x] Implement one bounded Flex tool using existing yHelper reads.
- [x] Run E2E/existing checks; review and address concrete findings.
- [x] Verify live reads and isolated image; retain evidence and clean probes.

The user has now authorized activation after final review and commit, followed by
live MCP tool verification. The release procedure is below.

Review findings addressed:
- Discovery already supplies market metrics; the native check accepts either
  discovery or exact detail plus the correct market's redemption-priority read.
- Failed E2E runs now retain recorded model answers/calls and source hashes.
- Truncated history retains the full window's first/last observations in addition
  to recent points, while preserving source coverage and explicit output limits.
- Provider responses must match requested market/status/event/time filters, wallet
  scope and activity page size; mismatches fail rather than silently changing scope.

Verified: nine controlled Flex HTTP/MCP E2E scenarios and two real Sol/medium
probes pass with no internal RPC reads; all 309 existing tests and sixteen existing
fault probes pass. Ruff, diff and Compose checks pass. The isolated candidate image
authenticates requests, retains the ten existing tools, and exposes the new Flex
tool; all eight views returned live yHelper data and legacy repo/stYFI reads passed.
Source-hashed evidence is in `.cache/verification/flex-review.json` and its linked
artifacts. The temporary evaluation directory/container/logs and unused intermediate
image were removed. The final candidate image is retained for activation.
Pre-activation checks preserved production bot/MCP IDs, loaded instructions/config
and canonical auth.

## Replay grading repair

Goal: make the Flex native replay reject incorrect amounts and unsupported wallet
claims while accepting equivalent, correct wording. Change evaluation only; keep
the bot, MCP, prompts and deployment unchanged.

Failure modes recorded before the repair:
- Amount substrings accept 2,500/1,000, fractional or negative amounts, wrong units,
  swapped idle-liquidity/debt values, or debt at a different rate. A separate clause
  mentioning 2% must not satisfy the requested threshold/value pair.
- A generic uncertainty word hides an invented current LTV, a liquidation-safety
  assurance, or numeric aggregate PnL despite incomplete position coverage. Both
  contracted assurances and an unrelated PnL denial must fail.
- Correct partial-position loss is confused with total wallet PnL, or a missing
  explicit refusal to infer LTV/safety still passes.
- Grading changes reject the original correct recorded answers or require a single
  exact sentence/Markdown layout, keyword, rate/value order or uncertainty wording.

- [x] Add offline negative controls to the existing Flex E2E runner.
- [x] Tighten the shared native answer checks and regrade saved correct answers.
- [x] Bind debt amount/rate in one positive fact and safety uncertainty to its predicate.
- [x] Run controlled E2E, both real-model probes and relevant static checks; review.

The grader checks exact amounts/units, decimal utilization versus percentages and
the requested debt amount/rate together, without crossing another percentage or
accepting a negated fact. Each affirmative safety predicate, including straight
and curly contractions, needs uncertainty that governs that predicate; unrelated
PnL negation cannot excuse it. Coverage verifies one covered position out of three.
It accepts the observed correct wording without requiring the keyword "partial".

Verified: all 25 targeted bad-answer controls are rejected, nine controlled E2E
scenarios and two fresh Sol/medium probes pass, with no internal RPC reads. Six
saved correct answers pass the current grader; original execution hashes/results,
including the correct answer rejected for wording, remain intact. All three review
reproductions are now rejected. Ruff and diff checks pass. Current source hashes
and evidence are in `.cache/verification/flex-grading-repair.json`. These evaluation
runs preserved production bot/MCP, loaded instructions/config and canonical auth.
Temporary evaluation data was removed.

## Approved activation

Goal: deploy the committed Flex integration and verify the real MCP surface.
Acceptance: the bot reconnects, MCP is healthy and exposes eleven tools, all eight
Flex views work with live yHelper data, prior tools/auth still work, and no RPC
credentials are exposed. Canonical auth and persisted ticket/handoff data remain
intact. No test messages are sent to Discord or Telegram.

Release procedure:
1. Review code/regression evidence; match every candidate-image runtime file to the
   checkout. Commit the five integration files and retain the prior image for rollback.
2. Activate the verified candidate through the existing Compose service, wait for
   authenticated readiness, then restart the host bot using its existing systemd unit.
3. Verify Discord readiness and loaded instructions, canonical auth, tool discovery,
   all eight Flex reads, explicit output bounds, freshness, pagination and legacy reads.
   Prefer the directly exposed tool when the connected client refreshes its catalog;
   otherwise exercise the same deployed tool over authenticated MCP HTTP.

Deployment results, commit/image identifiers and live-call evidence are recorded
in `.cache/verification/flex-deployment.json`.

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
