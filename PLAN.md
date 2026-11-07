# Repository cleanup and release

Goal: commit the reviewed evidence/safety changes, remove superseded material
and duplicated tests, then push and verify the deployed source.

Acceptance and constraints:
- Preserve active tools, dynamic private RPC helpers, old-vault support, external
  JSON backends, credentials, state, sessions and unrelated services.
- Delete tests only with an explicit E2E replacement or an implementation-only
  assertion that adds no behavior coverage. Retain unique failure/edge cases.
- Preserve verifiable evaluation evidence. Remove only superseded local plans,
  stale notes, orphaned bytecode and this task's temporary resources.
- No new unit tests or dependencies. Run relevant E2Es, remaining tests, Ruff and
  path/import checks. Keep test-to-E2E mapping in the verification artifact.
- Split dependency-complete commits, push without force, deploy while idle and
  verify Discord/Telegram/MCP, source hashes and unchanged auth/ticket state.

- [x] Audit callers, current coverage, stale references and release state.
- [x] Verify E2E replacements, remove duplicate tests and superseded material.
  Eight tests map to actual MCP/RPC/controller E2Es. HTTP auth now explicitly
  rejects a wrong bearer token; vault fixtures include null/external origin.
  Unique YIP, precision, state, failure and old-vault tests remain.
- [x] Run checks and review the final diff; address concrete findings.
  106 E2E scenarios and 278 remaining tests pass. Ruff, diff and resource/path
  checks pass. Removed the now-empty warning wrapper found during review.

Release procedure: commit in dependency-complete groups, push, deploy while idle,
verify health/state, and remove the temporary checkout. Commit IDs and release
outcomes are recorded in `.cache/verification/cleanup-release-report.json`.

Verification artifacts: `.cache/verification/cleanup-release-*.json`.
Reviewed feature evidence remains in `evidence-fixes-report.json` and
`review-fixes-report.json` in that directory.
