# ySupport Discord Bot

Discord support bot for Yearn.

It handles:
- support tickets
- public trigger conversations
- docs/process answers
- vault/deposit/withdrawal help
- bounded repo/docs/onchain investigation

The bot is built to keep support grounded:
- official-source-first for docs/process/product questions
- tool-grounded for tx/account/protocol/runtime issues
- explicit human handoff where the bot should not guess

The repo also contains:
- the live Discord bot runtime
- the ticket investigation runtime and execution boundary
- transcript-fetch tooling for ticket review
- an offline knowledge-gap worker for private internal reporting from support tickets

Runtime boundaries:
- production support turns use `codex_support_exec`
- `support_boundary.py` is the live outer classifier shared with explicit replay
- `support_agents.py` and `ticket_investigation/runtime.py` are the legacy local backend retained for explicit local/subprocess replay and LLM evaluation
- `knowledge_gap_worker.py` and `knowledge_gap_reporting.py` are manual offline analysis tools and are not imported by live support turns

Host-native service mode

The bot can be run outside Docker under `systemd` while keeping the MCP server in Docker.

Recommended shape:
- keep `mcp_server.py` in Docker
- run `ysupport.py` on the host
- run the bot as a dedicated, unprivileged `ysupport` service user
- keep a dedicated bot `CODEX_HOME` for generated Codex config
- keep bot auth in service-owned state rather than exposing an operator's home directory

Important:
- do not point `TICKET_EXECUTION_CODEX_HOME` at your normal `~/.codex`
- the bot writes `config.toml` and instructions into its `CODEX_HOME`
- use `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to point at a `0600` auth file owned by the service user
- keep the checkout and Codex installation read-only; only the dedicated state directory and private temporary directory need writes

Minimal host-native setup:
- create a host venv and `pip install -r requirements.txt`
- ensure `codex exec` works for the service user
- leave `TICKET_EXECUTION_CODEX_HOME` on its dedicated bot path
- clear `TICKET_EXECUTION_CODEX_AUTH_SOURCE`
- clear `TICKET_EXECUTION_CODEX_AUTH_SYNC_SOURCE`
- set `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to the service-owned auth file
- use `scripts/run_ysupport_host.sh` as the service entrypoint
- use `systemd/ysupport.service` as the hardened deployment unit
- if the checkout or Codex version path changes, update `WorkingDirectory`, `EnvironmentFile`, `ExecStart`, and the two `BindReadOnlyPaths` entries together
