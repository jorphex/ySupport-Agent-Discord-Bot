# Improve wallet failure diagnosis and explorer guidance

Goal: address the verified response gaps in ticket 1622 with a minimal change
to production instructions, without adding routing rules or changing tools.

Acceptance:
- When gas is topped up but withdrawal still fails, request missing error
  details and offer concrete wallet/RPC checks distinct from contract simulation.
- Explorer guidance identifies overloaded functions unambiguously and includes
  verified raw share amounts without an extra user lookup.
- HTTP errors and nonce observations do not become unsupported claims about
  transaction submission or wallet queues.
- Historical production-backend replays cover the top-up follow-up, HTTP 401,
  and staff-requested explorer guidance without sending real ticket messages.
- Relevant checks and review pass; deploy while idle, verify live health,
  remove temporary replay data, and commit/push the reviewed result.

Constraints: prompt-only behavior change; retain the model, handoff/stop rules,
authentication, tools, and Icarus. No wording-only tests or evaluation archive.
Use historical evidence rather than the wallet's present-day zero-share balance.

- [x] Capture historical evidence and baseline replay behavior.
- [x] Add concise generic diagnostic and instruction-completeness guidance.
- [x] Replay the historical cases, review answers, and fix concrete gaps.
- [x] Run relevant checks, inspect the diff, and deploy while idle.
- [x] Verify loaded prompt/service health and remove temporary replay data.

Verification:
- Archive block 94,467,618 confirms 2 POL, the exact share balance, nonce 151,
  and a successful four-argument redemption with max_loss=1. Historical mempool
  and wallet-local queue state remain explicitly unknown.
- One matched baseline completed in 16.4s and omitted RPC troubleshooting.
  Additional baseline attempts hit provider 429s. After cooldown, final candidate
  cases completed in 26.3s (top-up), 51.0s (explorer), and 25.8s (401), all with
  no handoff. They used the production model/settings and isolated historical
  context; these timings exclude live chain-data retrieval and message delivery.
- Review caught unsupported submission-stage/request attribution and generic
  reset advice. The final rules require evidence for those claims and concrete
  RPC checks; all three cases were rerun against the final prompt. The 401 case
  includes the preceding explorer exchange and was tested as a staff-directed
  one-off, without changing stopped-ticket behavior.
- The relevant 42 native-execution, contract, and staff-flow tests, Ruff, and
  diff checks pass. No wording snapshots or redundant unit tests were added.
- Production restarted while idle, reconnected to Discord/Telegram with zero
  automatic restarts, and loaded instructions matching the reviewed template.
  MCP remains healthy. Temporary units and all replay transcripts, sessions,
  homes, artifacts, scripts, private environment files, and test logs are removed.
