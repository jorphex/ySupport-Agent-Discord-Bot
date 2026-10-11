# Preserve pending Telegram handoffs during Discord staff takeover

Goal: staff can take over Discord while another teammate can still answer the
pending Telegram handoff; a delivered team update must leave Discord stopped.

Acceptance:
- Staff messages pause automatic replies and preserve handoff context and controls.
- Explicit Telegram replies work after takeover and restart, once per notice.
- Dismissal and explicit Stop Bot still block late delivery and clear the handoff.
- Processing/delivery failures preserve recoverable state and the stopped flag.
- Relevant tests, full offline suite, Ruff, and diff review pass.

Constraints: keep the prior test cleanup in its own commit; no live messages,
service restart, deployment, or changes to unrelated projects.

- [x] Commit prior test cleanup (c1e2c6a); baseline 432 tests, 44 live E2E skipped.
- [x] Separate staff takeover from explicit stop/dismissal and update Telegram guards.
- [x] Cover takeover, persistence, delivery, failures, and concurrent stop behavior.
- [x] Verify and commit the handoff change separately.

Verification: 80 focused tests passed. Full discovery ran 434 tests: 390 passed,
44 live E2E skipped. Relevant Ruff safe fixes and non-mutating checks passed;
final diff review and whitespace check passed. No deployment, restart, or live
messages were performed.

## Review fixes

Goal: keep pending handoff context intact across Discord staff summons and include
staff attachments when a later Telegram reply is composed.

Acceptance: a `y:` reply preserves older history and owner messages received during
generation/delivery; staff attachments survive restart and reach the Telegram turn.
Stopped state and explicit dismissal behavior remain covered by the existing tests.

- [x] Extend the existing flow tests to reproduce both context gaps.
- [x] Preserve live handoff history and collect staff attachments at message intake.
- [x] Run focused and full offline checks, review the diff, and commit separately.

Results: both extended regression tests failed before the fix and passed afterward.
The 80 focused tests passed. Full discovery: 434 tests, 390 passed and 44 live E2E
skipped. Ruff safe fixes, the final non-mutating check, and diff review passed.
No deployment, restart, or live messages were performed.
