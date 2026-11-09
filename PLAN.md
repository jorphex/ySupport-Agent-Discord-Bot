# Clear Telegram handoff notices

Goal: identify the Discord source accurately and give contributors a concise,
grounded issue and action. Show channel names without server names.

Acceptance and constraints:
- Public headers identify the channel and link to the original user message,
  never the deleted trigger. Ticket headers use the actual channel name.
- Escape channel names and notice content for Telegram HTML. Missing metadata
  must fall back honestly, without fabricated ticket numbers or invalid links.
- Summaries receive the completed support answer, preserve uncertainty and
  distinguish user claims from findings. Summary failure must not block alerts.
- Public notices explicitly say Telegram replies are ignored. Preserve public
  state cleanup and ticket reply, takeover, stop and dismissal behavior.
- Tighten public-answer guidance without dropping decisive evidence or limits.
- No new dependencies, retention system, live chat messages or state changes.

Failure modes to verify before implementation: wrong source/message link;
invented ticket label; server-name exposure; malformed HTML; omitted findings;
unsupported summary claims; failed summarizer suppressing notification; public
replies accidentally delivered; ticket reply/dismissal regressions.

- [x] Inspect source metadata, summary inputs, callers and existing E2E coverage.
- [x] Extend local HTTP E2E coverage for source labels, links, summary inputs,
  fallback and public/ticket reply behavior; confirm the regression fails.
  All four new scenarios fail on missing support findings before runtime changes;
  all five existing cancellation/ordering scenarios pass.
- [x] Make the smallest presentation/context changes and update public guidance.
- [x] Run relevant E2Es, existing tests, Ruff and diff checks; review and resolve
  concrete findings. Save repeatable evidence under `.cache/verification/`.
  Nine local HTTP scenarios, six optimization E2Es and 278 existing tests pass.
  Reviewed metadata propagation, HTML escaping, summary limits/fallbacks and
  compatibility with saved notices. No remaining concrete findings.

Release: commit the verified changes, restart ySupport while idle, and verify
Discord/Telegram readiness, service imports and preserved auth/ticket state.
Record the commit and deployment outcome in
`.cache/verification/handoff-presentation-deployment.json`.

Verification evidence: `.cache/verification/handoff-presentation-report.json`.
