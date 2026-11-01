# Vault provenance and evaluation repair

Goal: fix the three reviewed gaps without mixing catalog/snapshot observations or
claiming native coverage from fields the real discovery tool never returns.

Acceptance and constraints:
- A matched snapshot uses its own observations. Missing metrics, fees, PPS or
  composition remain unknown; cached financial fields never acquire a snapshot
  timestamp. Partial snapshots preserve their available evidence explicitly.
- Bind catalog retrieval time to the actual catalog generation used by the call.
  Include elapsed detail-fetch time in its reported age, even across a concurrent
  cache refresh. Keep the existing cache policy and provider-call bounds.
- Native discovery cases exercise the real response projection over controlled raw
  data. They receive the same fields as production and make genuine metadata/link
  lookups when needed; no fabricated strategy/lifecycle/link evidence is injected.
- Preserve paging, APY aliases, yBOLD display behavior, eleven tools and data-only
  MCP access. No dependency, external skill, bot-policy or RPC changes.
- No live test messages. Deploy MCP only after checks; keep the bot running.

Failure modes recorded before implementation:
- An identity/block-only or otherwise partial HTTP 200 snapshot inherits catalog
  APY/TVL/fees/PPS and is presented as complete/newer snapshot evidence.
- During a delayed detail read, another request refreshes the catalog and supplies
  its timestamp for older returned rows; capturing age early also hides elapsed time.
- Native tests still return impossible discovery fields or bypass the actual
  projection, omit real tools from the controlled catalog, or ignore filters so
  unrelated rows satisfy the evaluation.
- Correct real discovery responses cannot support required URLs, lifecycle or
  strategy counts without a lookup; tests must verify those reads rather than
  weakening the expected answer or adding invented provider fields.

- [x] Add HTTP/MCP regression cases and migrate raw native fixtures before code.
- [x] Separate snapshot data, bind catalog provenance and run the real projection.
- [x] Run relevant E2E/native/existing checks; review and resolve concrete findings.
- [x] Verify the hardened candidate and prepare the MCP-only release procedure.

Snapshot rows now use the snapshot alone and mark missing main fields as partial;
raw measurements remain null rather than borrowing catalog values. yBOLD retains
its display rule with separate input-source/time metadata. Catalog provenance is
captured before detail I/O and its age includes the elapsed time. Native discovery
runs the real projection over raw fixtures, with scoped filters, an exact eleven-tool
catalog and real follow-up metadata reads. Controlled lookup fixtures also honor
identity/chain/paging instead of returning unrelated rows for every query.

Verified: all eleven HTTP/MCP scenarios pass both on the host and inside the final
Docker image. Four native cases pass with real projection and matching metadata
lookups. All 294 existing tests, sixteen fault probes and nine Flex scenarios pass.
Candidate live reads verify normal lookups plus separate yBOLD input provenance;
auth and legacy reads pass. Ruff/diff/Compose checks pass and all candidate runtime
sources match the checkout. Review evidence: `.cache/verification/vault-provenance-review.json`.

Release procedure: commit the reviewed fix, retain the previous image for rollback,
activate only the verified MCP image, and verify live lookup, source age, pagination,
aliases and legacy reads. Leave the bot running and preserve canonical auth and
state. Record results in the deployment artifact below, then remove temporary
evaluation/container resources after retaining their evidence.

Evidence: `.cache/verification/vault-provenance-e2e.json`, native/evaluation
artifacts and `.cache/verification/vault-provenance-deployment.json`.

## Completed compact lookup

Commit `0f0f3f5` is the deployed baseline. Paging, selective details, APY aliases
and structured output remain in scope. Its deployment evidence is
`.cache/verification/vault-lookup-deployment.json`; the subsequent provenance and
fixture findings are tracked above rather than treated as resolved by that run.

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
