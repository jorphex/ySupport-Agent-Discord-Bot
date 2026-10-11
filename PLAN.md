# Test signal review

Goal: remove tests that add no meaningful real-bug detection beyond the E2E and
stronger flow coverage, while retaining distinct behavioral regressions.

Acceptance: all test modules reviewed, deletions justified against actual coverage,
production and E2E code unchanged, remaining offline suite and Ruff pass.

Constraints: preserve unrelated files and existing historical `plan.md`; do not run
credentialed live E2E or change deployed services as part of test cleanup.

- [x] Inventory tests and assign disjoint parallel review scopes.
- [x] Review provider/docs/MCP tests and actual E2E boundaries.
- [x] Complete parallel Codex, flow, and execution test pruning.
- [x] Review combined diff and run remaining suite and relevant Ruff checks.
- [x] Record final coverage rationale and verification.

## Review outcome

Removed 78 tests: 48 routing/flow/policy duplicates and snapshots, 19 Codex/runtime
duplicates and snapshots, nine execution/serialization/replay duplicates, and two
provider-side checks (direct bearer-token acceptance duplicated by authenticated
HTTP tests; ingestion script/requirements text assertions that never execute locking).
Useful serialization assertions were moved into retained integration tests.

The 44 live E2E tests exercise model routing and answers, with many tool cores
mocked. They do not exercise Discord/Telegram controllers, durable state,
subprocess/session cleanup, MCP authentication, or ingestion. The E2E executor
helper has no callers. Retained tests protect these distinct boundaries, provider
error handling and calculations, authorization, cancellation, concurrent followups,
partial delivery, and withdrawal-target capture (which E2E seeds directly).

Baseline: 510 discovered, 466 passed, 44 skipped. Final full discovery: 432 tests,
388 passed, 44 live E2E skipped, in 67.5 seconds. Focused retained suites passed;
Ruff safe unused-import fixes, final `ruff check tests`, and `git diff --check`
passed. AST comparison confirms exactly 78 removed tests and no new tests.
Production and E2E files are unchanged. Live model behavior was not rerun.
