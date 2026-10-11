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
