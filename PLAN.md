# Support runtime cleanup

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
