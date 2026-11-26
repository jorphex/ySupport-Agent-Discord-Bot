# ySupport

AI-assisted Yearn support for Discord tickets and public channels. ySupport
answers product questions, investigates vaults and transactions, and sends
questions that need a person to the team on Telegram.

## Find what you need

| If you want to | Read |
| --- | --- |
| Understand ticket replies, staff takeover, or Telegram handoffs | [Discord conversations](docs/conversations.md) |
| Set up the bot, restart it, or refresh its knowledge | [Operations guide](docs/operate.md) |
| Look up a runtime setting, tool, or evidence limit | [Runtime and tool reference](docs/reference.md) |
| Test a change or replay a ticket | [Verification guide](docs/verification.md) |
| Understand what ySupport stores or sends to providers | [Privacy policy](PRIVACY.md) |

## Repository layout

| Path | Contents |
| --- | --- |
| `ysupport/bot/` | Discord routing, ticket state, and handoffs |
| `ysupport/execution/` | Codex execution, private RPC, and replay commands |
| `ysupport/integrations/` | Discord and Telegram clients, transcript fetch, and MCP |
| `ysupport/tools/` | Vault data and internal on-chain reads |
| `ysupport/retrieval/` | Search, indexing, and the source manifest |
| `ysupport/ingestion/` | Documentation and repository refresh commands |
| `scripts/`, `systemd/`, `requirements/` | Launch scripts, the host unit, and MCP dependencies |
| `tests/`, `assets/` | Verification runners and screenshots |

Use `requirements.txt` for host dependencies. Run Python entry points as modules
from the repository root. Generated docs, indexes, and verification reports use
`.cache/`. The bot keeps ticket state and auth in their configured directories.
