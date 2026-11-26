# Verify a ySupport change

Run the checks from the repository root with the host dependencies installed.

## Run offline checks

Run the test suite first.

```sh
.venv/bin/python -B -m unittest discover -s tests
```

Run the end-to-end checks closest to your change. Use
`.venv/bin/python -B -m <module>` with a module from the table. Each runner saves
a JSON report under `.cache/verification/` unless you supply another path.
The runners use fixtures for Discord, Telegram, model output, and provider data.

| Module | Coverage | Default report |
| --- | --- | --- |
| `tests.layout_e2e` | Package imports, resource paths, entry points, host launch | `layout-e2e.json` |
| `tests.optimization_e2e` | Cache, ingestion, indexing, delivery, and replay grader failures | `optimization-e2e.json` |
| `tests.evidence_e2e` | Payouts, V3 redemption, scoped history, receipt blocks, and partial reads | `evidence-e2e.json` |
| `tests.internal_rpc_e2e` | Private tool access, reads, budgets, blocked writes, and cancellation | `internal-rpc-e2e.json` |
| `tests.latency_e2e` | Concurrent turns, cancellation, maintenance, queue, and handoff delivery | `latency-e2e.json` |
| `tests.steering_e2e` | Follow-up text and images, edited inputs, wallet intake, and restart recovery | `steering-e2e.json` |
| `tests.telegram_http_e2e` | Private notices, replies, dismissal, delivery order, and polling | `telegram-http-e2e.json` |
| `tests.public_handoff_e2e` | Public replies, context isolation, recovery, and cleanup | `public-handoff-e2e.json` |
| `tests.flex_e2e` | Flex tool schemas, identifiers, pagination, units, and partial data | `flex-e2e.json` |
| `tests.vault_lookup_e2e` | Vault lookup, detail snapshots, pagination, and yBOLD APY provenance | `vault-lookup-e2e.json` |

## Replay support cases with a real model

1. Run as an unprivileged service user. Set `OPENAI_API_KEY` for the classifier
   and `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to the user's `0600` bot-auth file.
2. Choose a private output path that the service user can write. Set
   `RUN_NATIVE_SUPPORT_E2E=1` and run the selected case. Omit `--case` to run the
   full catalog. You can use a scenario ID or its alias.

   ```sh
   RUN_NATIVE_SUPPORT_E2E=1 .venv/bin/python -B -m tests.native_support_e2e \
     --case bridge_followup_reads_each_supplied_receipt_on_its_chain \
     --output /private/native-support-e2e.json
   ```

3. Check `complete`, `passed`, and the per-case `failures` in the JSON. Use the
   recorded tool calls, Discord output, and state transitions to explain a
   failure. The report includes timings and source hashes.

The runner uses real controllers, the classifier, native Codex, and production
MCP schemas. It uses temporary state and fixture chat and provider I/O. A fixture
returns evidence only for a matching wallet, chain, vault, hash, or query. Report
artifacts use read-only HTTP. Multi-turn cases enter through `TicketBot.on_message`
and check that follow-ups awaiting the team make no native calls. The runner
uses the supplied environment and skips the checkout's `.env`.

For real-model checks of RPC or Flex tools, add `--native` to
`tests.internal_rpc_e2e` or `tests.flex_e2e` with the same auth and environment.
RPC checks use a local JSON-RPC node and controlled Kong data. Flex checks use
controlled yHelper data.

The [scenario catalog](../tests/native_support_cases.json) includes the retained
specialist aliases and cases for bridge delivery, withdrawals, migration, APY,
strategy behavior, and historical state.

Use the service identity and isolated state when checking app-server steering
with a real model. [Steering probe](../tests/steering_native_e2e.py)

## Replay a saved request

Use `ysupport.integrations.ticket_transcript_fetch` for manual, read-only
transcript review. Run with `--help` for the available fetch options.

Use the replay CLI with a saved ticket execution request. For the available
backends and session options, run the CLI help.

```sh
.venv/bin/python -m ysupport.execution.replay_cli --help
```

The replay CLI starts with an ephemeral session. Set `--session-dir` to an
isolated directory if you need continuity across requests. The `subprocess`
backend requires `TICKET_EXECUTION_SUBPROCESS_COMMAND`.
