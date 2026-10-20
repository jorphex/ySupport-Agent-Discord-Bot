# Restore automatic knowledge refresh and deploy projected APY correction

Goal: restore current repository/documentation evidence, prevent one refresh
failure from blocking the other, and deploy the reviewed APY correction.

Acceptance:
- Daily documentation and repository stages run independently and report their
  result and last successful completion without hiding failures.
- Diagnose the Pinecone verification mismatch and repair its concrete cause;
  retain exact corpus validation and safe update ordering.
- Rebuild repository context and verify search/fetch through the live MCP.
- Repeat the selected ticket's two-turn replay with current evidence and no
  Discord/Telegram delivery; remove temporary replay data afterward.
- Relevant tests, Ruff, final review, separate commits, push, and deployment pass.

Constraints: minimal changes; preserve credentials, live ticket state, pending
handoffs, and Icarus; retain the existing daily schedule and freshness guard.

- [x] Diagnose live namespace state and the refresh failure.
- [x] Repair independent stages and add minimal regression protection.
- [x] Refresh both corpora and verify live repo search/fetch.
- [x] Replay the ticket against candidate code with fresh evidence.
- [x] Review, verify, commit separately, push, deploy, and check live health.

Refresh results: pinned Pinecone 9.1.0 yields ListResponse/ListItem objects.
Stringifying ListItem rather than reading its ID prevented stale-vector deletion
and caused false missing-ID reports. Updating the existing fake to real SDK
response models reproduced the bug in the existing repair regressions. The fix
reads page.vectors[].id; the shell entrypoint independently runs documentation
and repository stages, preserves last-success times, and returns failure if
either stage fails. A shell integration regression exercises all six failure
points plus success.

The real refresh verified 2,695 Yearn and 60 Flex vectors and removed 29 obsolete
Yearn vectors. A second ingestion skipped both sources with zero embeddings,
upserts, or deletions. Repository context rebuilt all 13 configured sources;
authenticated production and candidate MCP search/fetch passed with fresh
context. The candidate returns the current 10.76% projection; production still
returns N/A until deployment. Full offline discovery ran 437 tests: 393 passed,
44 opt-in E2E skipped. Ruff, shell syntax, and diff checks passed.

The new exact two-turn replay completed in 46.9s/27.2s with four/two MCP calls
and no handoff. It preserved projected-versus-realized distinctions, current
data provenance, and the absence of a deposit-based fix. The model chose live
vault/documentation evidence and did not call repo search; repo readiness and
search/fetch were verified separately. Neither replay establishes the exact
September frontend cause. Two initial candidate-harness setup failures (working
directory dotenv loading and missing child CODEX_HOME) were corrected before
the successful run; production was unaffected.

Deployment: APY correction `496c3ba` and refresh repair `0d7d4fd` were pushed
separately. The rebuilt locked MCP image is healthy and matches checkout source
hashes; the previous production image is tagged `ysupport-mcp:rollback`.
Authenticated production APY and repo search/fetch checks pass, and missing
bearer authentication returns 401. The bot restarted while idle, reconnected to
Discord/Telegram with zero automatic restarts, and loaded instructions matching
the reviewed template. No source-review findings remain. Temporary candidate
units were stopped and all replay homes, transcripts, sessions, artifacts,
scripts, private environment files, and temporary test/build logs were removed.

Previous APY work and verification:

Goal: provide available Kong projections to the bot and distinguish estimated
yield, realized performance, and unverified frontend explanations.

Acceptance:
- Vault details and search ranking use available `performance.estimated.apy`.
- A valid zero projection is preserved; missing projections remain unavailable.
- Existing oracle-based vaults and the yBOLD display rule remain covered.
- The production instructions distinguish metrics and require evidence for claims
  about frontend fallbacks or deposit/harvest thresholds.
- A controlled two-turn production-backend replay covers the ticket question and
  follow-up; missing-estimate evidence is evaluated separately.
- Focused checks, full offline tests, Ruff, and final review pass.

The APY implementation was committed locally as `496c3ba`; deployment is now
authorized with the refresh repair. No live Discord/Telegram messages or retained
evaluation data are needed.

- [x] Add focused regressions that fail against the current implementation.
- [x] Share projected-APY selection across details and ranking; tighten grounding.
- [x] Run focused/full checks and attempt isolated production-backend evaluations.
- [x] Review the final change, fix findings, remove temporary evaluation data,
      and commit locally.

Results: the two added regression methods failed before the change. The initial
60 focused tests passed; full discovery ran 436 tests (392 passed, 44 opt-in live
E2E skipped). After review, the final 32 focused tests and Ruff/diff checks passed.
A read-only live Kong check confirmed that the actual projection now reaches
both the rendered estimate and ranking.

Initial model evaluations were partially blocked by HTTP 429. A subsequent
production-backend replay completed the exact two ticket messages against live
data and candidate MCP code in 57.1s and 40.6s, with five and three MCP calls.
It distinguished projected and historical APY and avoided deposit promises, but
the second turn could not use repository evidence because the index was stale.
The September UI cause remains unproven. Temporary replay artifacts were removed.
