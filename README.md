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

Host-native service mode

The bot can be run outside Docker under `systemd` while keeping the MCP server in Docker.

Recommended shape:
- keep `mcp_server.py` in Docker
- run `ysupport.py` on the host
- keep a dedicated bot `CODEX_HOME` for generated Codex config
- link the bot auth to the machine's live Codex login instead of copying auth files

Important:
- do not point `TICKET_EXECUTION_CODEX_HOME` at your normal `~/.codex`
- the bot writes `config.toml` and instructions into its `CODEX_HOME`
- use `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to point at the live auth file instead

Minimal host-native setup:
- create a host venv and `pip install -r requirements.txt`
- ensure `codex exec` works for the service user
- leave `TICKET_EXECUTION_CODEX_HOME` on its dedicated bot path
- clear `TICKET_EXECUTION_CODEX_AUTH_SOURCE`
- clear `TICKET_EXECUTION_CODEX_AUTH_SYNC_SOURCE`
- set `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to the live host auth file, for example `${HOME}/.codex/auth.json`
- use `scripts/run_ysupport_host.sh` as the service entrypoint
- use `systemd/ysupport.service` as a sanitized template and replace the sample `User=` and `/opt/ysupport` paths with the real host values before installing it
