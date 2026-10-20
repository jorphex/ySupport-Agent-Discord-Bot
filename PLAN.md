# Correct projected APY evidence for ticket support

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

Constraints: minimal source changes; no live Discord/Telegram messages,
production state changes, deployment, unrelated cleanup, or retained eval data.

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

Model evaluation used the production model/settings, isolated service-owned
homes/sessions, frozen formatter output, and no Discord/Telegram delivery. The
available-estimate conversation completed both turns; the missing-estimate case
completed its first turn. Review caught an unsupported UI-field guess and
tightened the grounding instruction. Completion of the missing follow-up and
retesting the final wording were blocked by persistent HTTP 429 responses after
cooldown/retries. This semantic validation remains incomplete; the source and
offline checks pass. Temporary replay state/artifacts were removed. No deployment
or service restart was performed.
