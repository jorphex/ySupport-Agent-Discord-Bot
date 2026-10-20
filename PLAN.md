# Prune low-signal unit tests

Goal: remove unit tests that duplicate E2E coverage or assert implementation
details without protecting a distinct, plausible bug.

Acceptance:
- Review every unit test against the actual E2E harness and remaining coverage.
- Remove prompt/configuration snapshots, mock echoes, and redundant checks.
- Preserve meaningful regressions for behavior E2E misses, including production
  execution isolation, provider failures, persistence, and ticket/handoff races.
- Remove helpers and imports made unused by deletions.
- Pass the surviving offline suite, relevant Ruff checks, and final diff review.

Constraints: test cleanup only; preserve E2E tests and production behavior.
No live messages, deployment, dependency changes, or new tests.

- [x] Map E2E coverage and divide independent test files among three reviewers.
- [x] Record the original offline suite result.
- [x] Review and prune all unit-test groups in parallel.
- [x] Review deletions for lost behavior coverage and unused scaffolding.
- [x] Verify the surviving suite and static checks; resolve concrete findings.

Verification:
- Original suite: 437 discovered, 393 passed, 44 opt-in LLM tests skipped.
- Three parallel reviewers audited native execution, ticket flows, and handoff/
  persistence tests; the parent reviewed docs, providers, MCP, and onchain tools.
- Review retained unique transport-state mapping, configured fallback wiring,
  malformed JSONL parsing, GitHub report parsing, and ENS/token-filter propagation.
  The real-LLM E2E harness does not cover those boundaries.
- Review also retained Flex/YIP source routing and direct retrieval's suppression
  of synthetic-query generation; E2E mocks that retrieval layer.
- Removed 72 unit tests and unused fixtures/imports. All remaining test IDs and
  bodies, E2E suites/harness, and production files are unchanged.
- Final suite: 365 discovered, 321 passed, 44 opt-in LLM tests skipped, 57.4s.
  Ruff on all changed Python files and the final diff check pass.
