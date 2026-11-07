# Remove E2E-covered unit tests and publish steering

Goal: remove tests whose meaningful assertions are covered by current E2Es, keep
uncovered failure/authority contracts, and publish focused commits without changing
verified runtime behavior.

- [x] Map redundant unit assertions to executable E2E cases; retain unique gaps.
- [x] Remove duplicates and unused imports; verify remaining tests and E2Es.
- [x] Commit native transport, controller/recovery integration, then test/docs cleanup.
- [x] Push main and verify the remote revision and clean tracked worktree.

Keep tests for acknowledgement-send races, pre-executor intake, late delivery,
transaction-safety rewrites, descendant cleanup, provider errors, token rejection,
persistence failures, and replay isolation: the E2Es do not cover all their cases.
No runtime edits or service restart are needed for this cleanup.

Removed six duplicate tests and 252 lines. Controller/native E2Es cover wallet
confirmation, queued follow-up handling, takeover during accepted team delivery
and deleted-ticket archiving; authenticated MCP E2Es cover missing and valid
bearer tokens. Keep the remaining distinct failure, authority and provider tests.

All 286 remaining tests, 25 steering scenarios, latency, real Telegram HTTP,
authenticated vault lookup and Flex E2Es pass. Runtime hashes still match the
verified deployment; this cleanup requires no restart. Removal-to-E2E mappings,
source hashes and publication verification are recorded in
`.cache/verification/test-overlap-cleanup.json`.

# Steering intake and recovery races

Goal: fix the two reproduced steering review findings without changing admission,
models, tools, deadlines, source roles or stop/staff/Telegram ownership.

Failure modes and acceptance (recorded before implementation):
- Intake preparation must stage wallet and confirmation state. Completion during
  image preparation, rejected/unobserved input, transport failure and cancellation
  must leave unconsumed confirmation available to the queued retry. Observed
  input must commit its matching intake/evidence snapshot in channel order.
  A failed execution explicitly admits prepared queued inputs to its fresh retry,
  preserving confirmation meaning when original and follow-up text are combined.
- Completed owner input, including observed follow-ups and image URLs, must remain
  in the durable recovery payload until reply/history/handoff commit. A new
  message during progress deletion followed by shutdown must recover all unsent
  input. Completed input must not be copied into both history and pending state.
- Stop/delete/staff commands must not retain or resurrect another task's recovery
  payload. Ordinary queue fallback, accepted Telegram delivery and ticket capacity
  must retain their current behavior.

Design: isolate the existing intake dictionaries for steering, commit observed
snapshots through an owned callback, and retain a separate owned recovery payload
for the short interval between native completion and controller commit. Keep
routing's existing active-executor and pre-executor payload semantics unchanged.

- [x] Add controller/stdio E2Es and reproduce both findings before runtime changes.
- [x] Stage intake writes; verify observation and fresh-retry admission cases.
- [x] Retain uncommitted recovery input until commit; verify shutdown and cleanup.
- [x] Review and fix concrete gaps; run required checks and activate while idle.

Steering preparation now uses isolated copies of the existing intake dictionaries.
The owning controller commits the matching evidence/intake snapshot for contiguous
observed input. If execution fails, it admits prepared queued input to the fresh
retry so combined text retains its confirmation meaning. Unconsumed preparation
cannot silently clear a confirmation while the original turn is still running.

Completed input transfers to an owned recovery payload until reply, deferred
history or handoff commit. New messages during progress cleanup retain the full
unsent text and image URLs. Ownership checks and terminal cleanup prevent stale
payloads from surviving Stop/delete or replacing another task's input.

Pre-fix E2Es reproduced preparation, partial consumption, rejection, unobserved
ACK, observed execution failure and shutdown data loss. Review also reproduced an
unobserved process failure and fixed its fresh-retry admission. Final verification:
25 controller/stdio scenarios, all 292 existing tests, latency (6), optimization
(6), real Telegram HTTP (5), Ruff and diff checks pass. Recovery is also checked
in an independent interpreter, including confirmed wallet state and both images.
No live test messages, provider RPC or new model calls were required.

Evidence: `.cache/verification/steering-races-{before,intake,e2e}.json`,
`steering-races-unobserved-failure-before.json`, sibling verification logs/reports
and `tests/steering_e2e.py`. Applied after an idle preflight. Discord reconnected, Telegram polling resumed,
MCP stayed healthy, concurrency remains 8 and ticket 1633 remains stopped.
Deployment and final evidence are in `steering-races-deployment.json` and
`steering-races-verification.json` in the same directory.

# Native ticket steering

Goal: let ordinary Discord ticket-owner follow-ups enter the active Codex turn,
with native text/image input, instead of waiting for the entire investigation.
Keep the model, authentication, tools, private RPC, eight-slot deployment limit,
per-ticket ordering, staff commands and Telegram delivery behavior unchanged.

Design and constraints:
- Use Codex app-server stdio for interactive owner turns; retain one-shot exec for
  noninteractive callers and safety rewrites. Derive the server command from the
  existing approved executable. No new dependency or backend configuration flag.
- Keep raw follow-ups queued until their correlated native user-message event is
  observed. A steer ACK alone is insufficient. Late/rejected/uncertain input uses
  the existing checked queue path; never blindly resend a timed-out steer.
- Check the outer support boundary before steering. Prepare screenshots once in
  separate batch directories. Preserve input source, order, evidence and IDs in
  the native request, history and exported replay artifacts.
- Persist unfinished active/queued owner input in existing ticket state. Recover
  on the next ticket interaction with a fresh native session; do not auto-send
  potentially duplicate replies after restart. Stop/delete discard this state.

Failure modes and acceptance (before implementation):
- Text bursts and image-only updates enter the same active native turn; one reply
  reflects all consumed input. Inputs arriving before readiness wait safely.
- Completion races, explicit rejection, missing ACK, process failure and unsafe
  output cannot lose input, publish a stale draft or retry an ambiguous steer.
- Stop/staff takeover/delete kill only their owned native process and release its
  slot. Other tickets and accepted Telegram updates retain existing contracts.
- Boundary-routed updates remain queued for normal handling. Updated history,
  evidence and handoff summaries include consumed input; replay data distinguishes
  submitted, acknowledged and observed inputs without recording credentials.
- Existing fallback transports remain functional. All relevant controller E2Es,
  existing checks and a bounded installed-Codex protocol probe must pass.

- [x] Add subprocess/controller E2E scenarios and record the initial failure.
- [x] Implement input ownership and native app-server transport; verify each layer.
- [x] Verify boundaries, recovery, cancellation and delivery with repeatable artifacts.
- [x] Review the full diff, fix concrete findings and document verification limits.

Owner turns now use one native app-server process and active-turn IDs. Source
roles, boundary/intake checks, native screenshots, the existing private RPC socket,
slot limits and deadlines remain enforced. Acknowledgements and observed user
items are recorded separately. Unobserved submissions detach their uncertain
session before queue fallback; accepted input updates evidence, history, handoff
summaries and a final replay request. Safety rewrites cannot accept fresh steering.

Review addressed wallet-confirmation preparation, stale known-target snapshots,
configured process versus turn working directories, unsafe commentary exposure,
weak failover after steering, ambiguous session reuse and cancellation cleanup.
Progress uses the existing generic tool labels. Process pipes are drained and
closed; diagnostic-write failures preserve Stop/cancellation authority.

Verification: 13 controller/subprocess E2E scenarios, all 292 existing tests,
latency (6 scenarios), real Telegram HTTP (5), optimization (6), private RPC (16),
Ruff and diff checks pass. The installed Codex 0.164.0 / production gpt-5.6-sol
probe consumed a native text/image steer, identified the image and retained the
correction after resume. The probe used service-owned auth and isolated state;
no live Discord/Telegram messages or provider RPC calls were made. It verifies
protocol/context behavior, not a production latency guarantee.

Evidence: `.cache/verification/steering-e2e.json` and sibling `steering-*` reports;
`tests.steering_e2e` repeats controller checks. `tests/steering_native_e2e.py` contains
the installed-Codex probe. Pre-fix artifacts retain reproduced progress-exposure
and canceled-artifact-write failures. Applied after an idle preflight, with rollback retained. Discord reconnected,
Telegram polling resumed, MCP stayed healthy and ticket 1633 remained stopped.
Deployment and final verification are recorded in `steering-deployment.json`
and `steering-verification.json` in the same evidence directory.

# Reference harness steering review

Goal: compare steering, follow-up queues, interruption and durable state in the
eight requested harness repositories, then identify patterns suitable for ySupport.

Acceptance and constraints:
- Pin source revisions and trace actual input admission, model/tool checkpoints,
  cancellation, completion races and recovery. Separate documented behavior from
  implementation findings and deployment-version limits.
- Inspect Pi/Pi Durable, DeepSeek Harness, OpenCode, Codex, Hermes, OpenMuse,
  CopilotKit and OpenBot. Record useful patterns and direct source references.
- Map findings to the current ySupport process/session/RPC boundaries. This pass
  is research and notes only; no runtime changes, dependencies or service restart.

- [x] Pin repositories and identify relevant source paths.
- [x] Trace steering and adjacent lifecycle patterns in all eight repositories.
- [x] Check ySupport integration constraints and record a grounded recommendation.

Reviewed all eight pinned repositories and verified 85 downloaded files against
their Git blob IDs. The installed Codex 0.164.0 release schemas also support native
steering, text/images, client message IDs and active-turn preconditions. Recommend
native steering for ordinary owner follow-ups, with checked queue fallback and
existing source, safety, budget and ownership contracts preserved. The current
one-shot exec transport must change; no model/harness replacement is required.

Detailed source references, useful patterns, limitations and future E2E acceptance
cases are saved in `.cache/verification/harness-steering/review.json`; durable
recommendations are in `NOTES.md`. Findings are from static source and official
documentation, not live timing measurements. No runtime code or services changed.

# Telegram HTTP cancellation

Goal: settle an in-flight Telegram edit before deletion or dismissal changes the
same notice. Keep independent notices concurrent and polling cancellation prompt.

Failure modes and acceptance (before implementation):
- Use the production urllib/thread transport against a local HTTP server. Hold an
  edit at the server, then delete its ticket: archiving must wait, finish last and
  leave no ticket state or delivery task. No live messages or credentials.
- Edits and deletion of the same Telegram message must remain ordered even when
  recovery/failure feedback is outside the native-delivery child. Other messages
  must continue without waiting for the held request.
- Repeated cancellation must not detach the HTTP thread or replace cancellation
  with a transport error. A canceled queued operation must never reach the server;
  a failed in-flight operation must release ownership for terminal cleanup.
- Preserve normal errors and the existing HTTP timeout. Read-only getUpdates
  cancellation must remain prompt. Per-message coordination must not retain idle
  locks. Existing native cancellation/staff takeover/deletion E2Es must still pass.

- [x] Add real-HTTP controller E2E and reproduce the reviewed failure.
- [x] Implement minimal per-message ordering and cancellation draining; verify.
- [x] Review the change, run required checks and apply while idle with rollback.

The HTTP transport now orders edits/deletion by Telegram chat and message ID.
Cancellation drains an admitted request before releasing its lock, preserving
cancellation even if HTTP fails or cancellation repeats. Weak lock ownership
avoids retaining idle entries; independent messages and polling remain concurrent.
The HTTP timeout, providers and dependencies are unchanged.

Real HTTP E2E reproduced three failures before the fix. All five final scenarios
pass, covering active and recovered closure, Stop Bot, repeated/queued cancellation,
server errors, independent messages and prompt polling cancellation. Latency E2E,
optimization E2E, all 292 existing tests, Ruff and diff checks pass. Source hashes
match the artifacts. Review found no further gap in this change.

Evidence: `.cache/verification/telegram-http-{before,e2e,latency-e2e,optimization-e2e}.json`
and `telegram-http-existing-tests.log`. The prior transport is retained in
`telegram-http-rollback.tar`.

Applied after confirming no active native processes, recent support activity or
accepted pending deliveries. Discord and Telegram reconnected, MCP remained
healthy and ticket 1633 stayed stopped. No live test messages were sent.
Deployment evidence: `.cache/verification/telegram-http-deployment.json`.

# Deleted-ticket handoff race

Goal: prevent queued delivery workers from overwriting a deleted ticket's archived
Telegram notice. Preserve staff takeover, active cancellation and ticket deletion.

Failure modes and acceptance (before implementation):
- Invalidate the handoff before waiting for ordinary-task cancellation or Telegram
  archiving. A newly available worker must not start delivery or send failure
  feedback during either wait, including when Discord has removed the channel.
- Workers can hold older notice snapshots while another ticket runs. Deleted or
  replaced notices must not send edits or recreate deleted persisted ticket state,
  including the delivered-pending-close path.
- Archive the captured notice once and remove ticket state. Preserve cancellation
  of active native work and unrelated deliveries. No unit tests or live messages.

- [x] Extend controller E2E and reproduce the queued/snapshot orderings.
- [x] Apply the smallest invalidation/eligibility fix and verify the regressions.
- [x] Review, run existing checks, and activate while idle with rollback retained.

The deletion handler captures and durably invalidates its notice before any wait.
Delivery workers check notice identity and eligibility before claiming work or
persisting recovered state. E2E reproduced both archive overwrite and deleted
state recreation before the fix; both now pass alongside native cancellation,
staff takeover and unrelated delivery. All 292 existing tests, optimization E2E,
Ruff safe-fix/non-mutating checks and diff checks pass. Runtime hashes match the
E2E evidence. Review found no further gap in this change.

Evidence: `.cache/verification/deleted-handoff-{before,e2e,optimization-e2e}.json`
and `deleted-handoff-existing-tests.log`. The two prior deployed runtime files are
verified and retained in `deleted-handoff-rollback.tar`.

Activated after a fresh idle check. Discord reconnected, Telegram polling resumed,
MCP stayed healthy and ticket 1633 stayed stopped. No live test messages were sent.
Deployment evidence: `.cache/verification/deleted-handoff-deployment.json`.

# Cancellation and maintenance review repairs

Goal: release investigation capacity when a Telegram update is stopped, and let
new support work proceed after the current maintenance operation finishes.
Preserve the existing latency changes, ticket state and staff takeover behavior.

Failure modes and acceptance (before implementation):
- Explicit Discord Stop Bot cancels only that handoff's task, kills its native
  process, closes its RPC socket and releases its slot. Another ticket stays live;
  queued work starts without releasing a test gate on the stopped model.
- A queued handoff can be canceled before native admission and must never start.
  Staff takeover still permits an accepted Telegram reply. Worker cancellation on
  shutdown cancels its child and retains durable pending-delivery state.
- Deletion and reinitialization cancel their handoff execution. Per-channel task
  ownership and delivery claims are released on success, error and cancellation.
  Cancellation must not overwrite archived notices with delivery-failure feedback.
- Maintenance finishes its current deletion, then yields if a support turn waits.
  It skips cache mutation while yielding, retains remaining cleanup for an idle
  pass, and still excludes execution while a deletion/cache operation is active.
  A canceled waiter must not leave a stale queue count or block later cleanup.
- Extend production-controller/native-process E2E first; reproduce both reviewed
  failures. No unit tests added, model/dependency/auth changes or live messages.

- [x] Add E2E assertions for active/queued handoff cancellation and cleanup yielding.
- [x] Implement separate handoff tasks and explicit cancellation paths; verify.
- [x] Track waiting executions and yield maintenance between operations; verify.
- [x] Review the diff, resolve findings and run relevant existing/E2E checks.
- [x] Activate only while idle, retain rollback, and verify the live service.

Separate per-channel handoff tasks now isolate explicit cancellation from their
persistent workers. Stop kills active native processes, closes private RPC sockets
and releases capacity; queued cancellation cannot start a model. Staff takeover
keeps accepted updates eligible. Deletion and shutdown cancel their children;
shutdown retains durable pending state. A review regression also prevents canceled
channel deletion from being mislabeled as a delivery failure in Telegram.

Maintenance counts waiting executions, yields after the current deletion and
skips cache work ahead of support. Remaining cleanup resumes when idle. Canceled
waiters release their count and do not defer idle cleanup indefinitely.

Expanded native-process/controller E2E and the optimization E2E pass, as do all
292 existing tests, Ruff safe fixes/non-mutating checks and diff checks. Before-fix
artifacts reproduce both findings and the related cancellation-status race.
Evidence: `.cache/verification/latency-repair-{before,e2e,optimization-e2e}.json` and
`latency-repair-existing-tests.log`. A verified copy of the prior deployed runtime
is retained in `latency-repair-rollback.tar`.

Activated after confirming no active processes, recent support turns or accepted
pending Telegram deliveries. Discord reconnected, Telegram polling resumed, MCP
remained healthy and ticket 1633 stayed stopped. No live test messages were sent.
Deployment evidence: `.cache/verification/latency-repair-deployment.json`.

# Support latency and concurrent investigations

Goal: remove measured investigation delays while preserving grounding, ticket
isolation, staff control and bounded internal RPC access. Cover all five reviewed
items: serial decisions, failed detours, burst messages, investigation scope and
cross-ticket concurrency, including independent Telegram delivery workers.

Constraints: no model change, dependency, public RPC tool, external message,
credential change or broad refactor. Preserve existing source-chunk work. Prepare
and verify changes before any runtime activation; do not interrupt live turns.

Failure modes and acceptance (recorded before implementation):
- Different tickets must overlap up to a configurable bound (default two); a
  third waits. Same-conversation turns remain serial and isolated. Cancellation
  while queued or active releases capacity and closes its own RPC/process only.
- Session/cache maintenance must never delete an active session or mutate caches
  during execution. Idle maintenance excludes new executions; busy cleanup skips
  until a later idle pass, rather than queueing ahead of live support.
- Initial requests/schema and concise tool routing are available in the native
  prompt without a separate file-read/model cycle. Retain request files, image
  arguments, output validation, current per-turn RPC socket and safety rewrite.
- Nullable catalog assets/symbols cannot abort filtered position scans. Receipt
  symbols and underlying symbols both resolve, and optional chain scope avoids
  unrelated chains. Failed/partial reads remain unknown, never zero.
- Katana guidance uses the official Katanascan URL. Preserve valid redemption
  signatures, raw units and loss semantics; a rounded maxRedeem difference alone
  must not be described as proof a full-balance redemption fails.
- Ordinary message collection is shorter; bug-report collection remains intact.
  Owner follow-ups queue behind active work without discarding observations.
  Suppress a draft/handoff overtaken by a follow-up, process the latest evidence
  immediately next, and preserve the fact an intermediate draft was not sent.
  Staff takeover and explicit Stop Bot still cancel work and prevent late replies.
- Staff summons include bounded recent ticket-user images and current command
  attachments, with source attribution; do not silently treat staff as the user.
- Routine support gives one evidence-backed next action and asks only for the next
  missing detail. No personality classifier, keyword router or hard accuracy cap.
- Emit ticket-correlated queue/setup/execution timings for completed, failed and
  canceled turns. Do not confuse first item timing or cumulative session tokens
  with provider-only latency or per-turn context size.
- E2E scenarios precede runtime edits. Exercise real controllers, native process
  boundaries and private RPC with controlled providers, saving repeatable JSON.
  Check existing suites and real-model selected cases, including simultaneous
  executions sharing a service-owned home; send no Discord/Telegram messages.

Steps:
- [x] Review this plan against callers, session lifecycle, cleanup and follow-ups.
- [x] Extend E2E coverage and reproduce concurrency, intake and metadata failures.
- [x] Implement bounded concurrency, idle maintenance and correlated timings;
  verify cancellation, isolation and cleanup before continuing.
- [x] Remove request-read/tool-discovery cycles and repair provider/explorer
  detours; run RPC and native-boundary E2E.
- [x] Queue owner follow-ups, shorten collection, preserve recent user images and
  tighten generic investigation instructions; verify full ticket transitions.
- [x] Review the complete diff, address concrete findings, run required checks and
  selected real-model replays; record measured outcomes and remaining limits.

Implementation checks: native-process E2E proves two overlapping executions,
a bounded third, same-ticket serialization, isolated sockets and cancellation.
Idle cleanup excludes work and busy cleanup skips. Private RPC E2E reproduces and
repairs nullable metadata, receipt-symbol filtering and exact decimal-string ABI
arguments; chain scoping and the official Katana explorer are verified. Ticket
E2E covers retained original/latest images, deferred drafts and handoffs, queued
follow-ups after failed execution, staff control and explicit stops. A second
serialization point in Telegram delivery is replaced by bounded workers with
per-channel claims; simultaneous delivery, one-time delivery and stop guards pass.

The first full suite exposed one obsolete assertion requiring global serialization;
it is retired in favor of the real-process concurrency E2E. Existing authentication,
same-conversation, safety-rewrite and delivery checks are retained. Four selected
real-model replays pass, including two terse withdrawal cases (17.0s and 26.0s).
Two real Codex executions sharing one service-owned home overlap for 11.94s with
separate sessions and correct private-provider results. These controlled replays
are evidence of behavior, not a production latency guarantee.

Final verification: 292 existing tests, latency E2E, private RPC E2E and the
broader optimization E2E pass. Retired two obsolete tests asserting one global
execution/worker count; end-to-end overlap, bounded capacity, fresh pending-state
recovery, same-ticket isolation and cancellation now verify the intended behavior.
Ruff safe fixes, non-mutating checks and diff checks pass. Reviewed runtime hashes
match the final deterministic evidence.

Activated only after a live-idle check, retaining
`.cache/verification/latency-runtime-rollback.tar`. The bot reconnected with
`codex_concurrency=2`, Telegram polling started, MCP remained healthy, and ticket
1633 remained stopped. No test messages were sent to Discord or Telegram.
Aggregate evidence: `.cache/verification/latency-deployment.json`; detailed
latency/RPC/optimization evidence is in the same directory. Real-model evidence:
`/var/lib/ysupport-codex/verification/latency-native-{rpc,replays}.json`.

# Full text-source retrieval

Goal: replace the 2,000-character fallback prefix with bounded, overlapping
source chunks, then add selected Envio handlers/schema and Metrics fee/TVL code.

Acceptance and failure modes:
- E2E first: search/fetch start, middle and end markers in Python, TypeScript,
  GraphQL and YAML, including long lines and evidence crossing a chunk boundary.
- Keep existing Markdown, Solidity, Vyper and JSON handlers. Preserve first-file
  references and deterministic chunk references across unchanged full rebuilds.
- Index all non-whitespace source content with bounded chunks and small overlap.
  Chunk locations must match the source; repeated text must not fetch other context.
- Preserve search/fetch result limits and snippet bounds. New chunk fetches return
  their exact bounded source excerpt; existing artifact fetch behavior is retained.
- Verify useful real-source queries, security-source priority, legacy filtering,
  auth and existing contract lookup through candidate and production MCP.
- Explanatory queries retain concise methodology docs when code chunks compete;
  explicit filenames/identifiers retain source retrieval priority.
- Add explicit source paths only. No generated code, broad source globs, dependency,
  model, service, RPC, credential or bot restart changes. Retain rollback.

- [x] Add ingestion/search/fetch regressions and reproduce the missing-tail failure.
- [x] Implement fallback chunks and bounded fetch; select useful source files.
- [x] Build and verify candidate index/image, growth, references and retrieval.
- [x] Review, resolve findings, deploy the verified pair and verify live tools.

Fallback sources now use at most 1,600 characters per chunk with 200-character
overlap and line-aware endings. Later chunks carry exact line/character ranges
and fetch their stored excerpt, avoiding ambiguous repeated-text anchors. First
file identities and existing specialized parsers are preserved.

Added eleven explicit source files: three Envio handler/schema files and eight
Metrics fee/product/TVL files. Fresh Git HEADs were resolved for all fifteen repos;
thirteen verified cached snapshots avoided redundant downloads. The index grew
from 50,532,352 to 54,644,736 bytes, with 393 files and 7,266 sections. All 7,076
prior section references remain valid. Full non-whitespace coverage and locations passed
for fourteen real text-source files.

Candidate verification caught code chunks displacing a concise fee-definition
document. A regression now preserves broad explanatory retrieval while explicit
source filenames/identifiers retain code priority. The specialized Markdown,
Solidity, Vyper and JSON parsers are unchanged.

E2E covered Python/TypeScript/GraphQL/YAML start/middle/end, long lines, overlap,
bounded responses, repeated text and deterministic full rebuilds. Fourteen existing
repo/MCP tests, Ruff and diff checks passed. Twenty-two authenticated search/fetch
cases passed against candidate and production, including four security queries and
existing contract lookups. Auth, legacy filters and attached deep-source retrieval
passed. Final review found no remaining gap.

The verified MCP image/index pair is active. The bot PID, bearer key, dependencies
and RPC boundary are unchanged. Rollback is retained; temporary candidate resources
were removed. Evidence: `.cache/verification/source-chunks-{before,ranking-before,e2e,build,search-fetch,candidate-mcp,production-mcp,deployment}.json`.

## Completed cross-product source scope repair

Goal: retain cross-product metrics/indexer retrieval without admitting unrelated
risk reports into scoped security or contract queries.

Acceptance and failure modes:
- Official disclosure guidance remains retrievable for the reproduced security
  query; unrelated risk documents cannot bypass its product scope.
- Apply the same restriction to metadata fallback. Keep risk reports available
  to unscoped queries, and preserve metrics/indexer retrieval and legacy filtering.
- Add regression coverage before implementation. Verify production-index search,
  authenticated MCP search/fetch, source hashes and deployment health.
- No manifest, index, parser, dependency, credential, RPC or bot restart changes.

- [x] Reproduce authority competition; cover metadata leakage through ingestion E2E.
- [x] Narrow the exception to metrics/indexer support data; verify existing coverage.
- [x] Review and deploy a verified MCP image; retain rollback and verify live tools.

Only `support_data` tagged `metrics` or `indexing` bypasses product scope, in both
section search and metadata fallback. Risk reports retain normal unscoped access.
The new ingestion E2E reproduced six competing risk results before repair, then
verified official guidance, rejected out-of-scope metadata and retained data access.

Optimization E2E, fourteen existing repo/MCP tests, Ruff and diff checks passed.
Four security and seven existing search/fetch cases passed on candidate and live
MCP. Auth, legacy filtering, unscoped risk access and attached-tool retrieval pass.
Final review found no further gap. The MCP-only fix is active; the bot PID, index
hash, manifest and bearer key are unchanged. Rollback is retained; temporary
candidate resources were removed.

Evidence: `.cache/verification/security-source-{before,e2e,search-fetch,candidate-mcp,production-mcp,deployment}.json`.

## Completed metrics and indexer repository context

Goal: ingest selected `yearn-metrics` and `yearn-envio` documentation as
`support_data`, with useful search/fetch results for normal support questions.

Acceptance and constraints:
- Pull explicit Markdown paths only. Exclude generated data, benchmarks, tests,
  dependencies and unrelated operations material. Keep daily incremental refresh.
- Make relevant support data eligible across product filters while retaining
  contract-code scoping, legacy filtering, result bounds and artifact references.
- Verify yvUSD product membership, fees/earnings, TVL overlap and indexed-event
  limits through real ingestion, search, fetch and authenticated MCP calls.
- Treat methodology and dated examples as source context, not live state.
- No parser expansion, new service/tool, RPC, dependency, credential or bot restart.

Failure modes recorded before implementation:
- Product filtering silently excludes relevant metrics/indexer documentation.
- Cross-product eligibility leaks unrelated contract code or legacy-only sources.
- Metadata fallback still applies the old filter and loses relevant support data.
- Broad globs ingest generated or operational noise; source documents are missing
  or produce no searchable sections; source changes break existing contract lookup.
- MCP serves a database and manifest from different versions after deployment.

- [x] Add E2E regression coverage and reproduce the filtering failure.
- [x] Add narrow source lists and correct support-data eligibility.
- [x] Build a candidate index; verify live-source search/fetch and existing queries.
- [x] Review, run relevant checks and activate a verified MCP/index pair.

Added 16 exact Markdown paths: thirteen metrics docs and three indexer docs,
107,159 characters and 79 sections. Generic READMEs were omitted because the
existing basename matcher also selected nested operations material. No parser
change was needed. The index now covers fifteen repositories and reused thirteen
unchanged source configurations. GitHub throttled repeat metadata lookups; the
narrowed build used commit revisions already resolved live in this pass.

Verification: the new product-filter regression failed before repair; optimization
E2E then passed, including metadata fallback and contract scoping. Fourteen existing
repo/MCP tests, Ruff and diff checks passed. Seven search/fetch cases passed against
the candidate and production MCP, including existing contract lookup. Legacy
filtering, eleven-tool discovery, unauthenticated rejection and attached-tool
search/fetch passed. Follow-up review found the broad support-data exception could
displace official security guidance; the repair is tracked above.

The verified MCP image and index are active. The previous image/index remain
available for rollback. The bot PID and MCP bearer key are unchanged; no RPC
credentials were added. Temporary candidate resources were removed.

Evidence: `.cache/verification/repo-sources-{before,optimization-e2e,build,search-fetch,candidate-mcp,production-mcp,deployment}.json`.

## Completed documentation revision

Goal: shorten maintained documentation without losing meaning, conditions,
technical identifiers, commands, limits or evidence references. Use direct verbs,
consistent terms and short sentences, guided by ASD-STE100.

Scope: README, privacy policy, this plan and the two local maintainer guides.
Preserve runtime instructions and historical ticket notes. No service, policy,
dependency, credential or ticket-data changes.

- [x] Inventory documents; check runtime claims against source and deployment evidence.
- [x] Remove repetition and simplify prose; preserve technical and privacy requirements.
- [x] Check commands, paths, limits and meaning; review the diff and resolve findings.

Acceptance: documentation is shorter, requirements remain explicit, code blocks
and local references are valid, and completed work has no stale pending actions.

Verified nine CLI modules and their documented flags, referenced source paths,
Markdown fences and shell-block syntax. Configuration identifiers and privacy
conditions are preserved. Corrected the completed restart status and the status
probe description. Runtime instructions and service state are unchanged.

## Completed discovery coverage repair

Commit `8ebf875` computes coverage from filtered rows before pagination. Empty
results have zero counts and a null ratio; known zero APY counts as available.
Fixture-match semantics and production tools are unchanged.

The empty-result regression reproduced stale two-vault coverage before repair.
All nineteen evaluation probes then behaved as expected, including the three
coverage cases and existing fault detection. Optimization E2E, Ruff and diff
checks passed. This test-only change required no restart.

Evidence: `.cache/verification/discovery-coverage-{before,e2e}.json`.

## Completed vault provenance and evaluation repair

Commit `332beca` separates catalog and snapshot observations. Missing snapshot
APY, TVL, fees, PPS or composition remain unknown; incomplete snapshots are
`partial`. Catalog timestamps identify the generation used by the call, even
during concurrent refreshes; reported age includes detail-fetch time. yBOLD
preserves its display rule and reports each input's source and time separately.

Native discovery cases use the production response projection over controlled
raw data and the exact eleven-tool catalog. Fixtures honor identity, chain,
filters and pagination. Answers requiring links, lifecycle flags or strategy
counts must fetch them; discovery fixtures cannot invent those fields.

All eleven HTTP/MCP scenarios passed on the host and in the final Docker image.
Four real-model native cases, 294 existing tests, sixteen fault probes and nine
Flex scenarios passed. Candidate reads verified lookup, yBOLD inputs, auth and
existing tools. Ruff, diff, Compose and deployed source comparisons passed.

The MCP-only release retained a rollback image, canonical auth and ticket state.
Live checks confirmed pagination, aliases, provenance and existing reads. The bot
stayed running; temporary evaluation and container resources were removed after
their evidence was saved.

Evidence: `.cache/verification/vault-provenance-{e2e,review,deployment}.json`.

## Completed compact lookup and Flex integration

- `0f0f3f5`: compact numeric vault rows, pagination, selective snapshot details
  and APY sort aliases. Evidence: `.cache/verification/vault-lookup-deployment.json`.
- `33d838d`: Ethereum Flex lending data through eight views. Ten live calls
  verified views, wallet filters and pagination; bot and eleven-tool MCP were
  healthy. Evidence: `.cache/verification/flex-deployment.json`.

## Completed support runtime cleanup

- Removed the stale marketing contact; retained the complete business policy.
- Replaced the local specialist engine and worker with native Codex execution.
  Preserved explicit external JSON backends and saved-request/replay compatibility.
- Added shallow/sparse docs refresh, verified reuse of unchanged sources,
  incremental repository indexing and cached namespace discovery.
- Persisted accepted Telegram replies before acknowledgement; a separate worker
  delivers them while preserving staff takeover and pending handoffs.
- Reclaimed idle locks and expired generated caches while preserving auth,
  installed plugins, recent data and active sessions.
- Kept transaction, position, withdrawal and basic chain reads internal, with
  per-turn operation, provider-call, log-range and time limits. Preserved both
  helper modules, chain resolution and the older-vault registry. MCP has no RPC
  credentials or RPC tools.
- Required evidence for each claimed cause. Relevant bridge delivery requires
  destination evidence or an explicit statement that arrival is unconfirmed.

The 44 former LLM scenario IDs map to 42 native cases. Two more cases cover
confirmed bridge arrival and unavailable destination receipts. The original
cases passed a full native run; targeted grounding checks passed four cases/six
turns and sixteen fault probes. Regrading accepted equivalent uncertainty wording
without changing execution inputs or model answers. Original grades and hashes
remain in the evidence. Temporary evaluation logs and proven orphaned bytecode
were removed; repeatable JSON evidence remains in `.cache/verification/`.

Runtime/MCP cleanup was deployed on October 30, 2026. The authorized November 1
bot restart completed. Discord reconnected, the configured MCP exposed all eleven
current tools, and live lookup returned both yvUSD variants.
Evidence: `.cache/verification/bot-restart-20261101.json`.
