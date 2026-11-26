# Run and maintain ySupport

Run the commands from the repository root.

## Set up the Discord service

1. Create the host virtual environment and install the pinned dependencies.

   ```sh
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. Set `TICKET_EXECUTION_CODEX_HOME` to a dedicated bot directory. The bot writes
   `config.toml` and its instructions there.
3. Set `TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE` to a service-owned auth file
   with mode `0600`. Keep `.env` root-owned with mode `0600`. Systemd loads
   `.env` before it drops privileges and hides the file from the service.
4. Check the paths in the [systemd unit](../systemd/ysupport.service). If the
   checkout or Codex installation moves, update `WorkingDirectory`,
   `EnvironmentFile`, `ExecStart`, `BindReadOnlyPaths`, and `InaccessiblePaths`
   as applicable. Keep the checkout and Codex installation read-only.
5. Start MCP first. Then install and enable `ysupport.service`. The unit runs
   `scripts/run_ysupport_host.sh` as `ysupport`.
6. Check the service state and the startup log. Look for the Discord login and
   `Telegram handoff reply loop initialized: True` when Telegram is configured.

   ```sh
   systemctl is-active ysupport.service
   journalctl -u ysupport.service -n 50 --no-pager
   ```

Run the status command with the service's runtime settings. Exit status `0`
means validation and the smoke request passed.

```sh
.venv/bin/python -m ysupport.execution.status --smoke
```

## Start or rebuild MCP

1. Set the provider variables in `.env` and keep the existing `MCP_SERVER_API_KEY`.
   [Compose file](../compose.mcp.yaml)
2. Before the first start, refresh the knowledge sources to create the SQLite
   database for the mount. [Refresh procedure](#refresh-the-knowledge-sources)
3. If you are replacing a running image, tag that image for rollback first.

   ```sh
   mcp_image_id=$(docker inspect --format '{{.Image}}' ysupport-mcp)
   docker image tag "$mcp_image_id" ysupport-mcp:rollback
   ```

4. Build and start MCP. Wait for `ysupport-mcp` to report `healthy`.

   ```sh
   docker compose -f compose.mcp.yaml build
   docker compose -f compose.mcp.yaml up -d
   docker compose -f compose.mcp.yaml ps
   ```

5. Connect to `http://127.0.0.1:8001/mcp` with the configured bearer token.
   Call `repo_context_status` and check that the index is ready and fresh.

See [MCP container settings](reference.md#mcp-container-settings).

## Restart after a host change

1. Confirm the checkout contains the intended revision. Wait for active support
   turns to finish before restarting.
2. Restart the Discord bot and check the startup log.

   ```sh
   systemctl restart ysupport.service
   systemctl is-active ysupport.service
   journalctl -u ysupport.service -n 50 --no-pager
   ```

3. If the release changes a file copied by the [MCP Dockerfile](../Dockerfile.mcp),
   rebuild MCP as well.

## Refresh the knowledge sources

1. With host dependencies installed, run the refresh script.

   ```sh
   scripts/update_docs.sh
   ```

2. Check `docs.status` and `repo.status` under `.cache/docs_ingestion/`. Both
   should report `status=succeeded`. Each file also shows the last success.
3. Use the verifier to check index freshness, search, and fetch the top result.

   ```sh
   .venv/bin/python -m ysupport.ingestion.verify_repo_context --query 'V3 vault withdrawal'
   ```

The script permits one refresh at a time. It runs both stages and exits nonzero
if either fails. Set `DOCS_INGESTION_FORCE_REFRESH=1` to force a full refresh.

The production cron runs daily at 03:00. It creates the log directory before
starting the script and overwrites the previous log on each run.

```cron
0 3 * * * mkdir -p /root/bots/discord/ysupport/.cache/docs_ingestion && /root/bots/discord/ysupport/scripts/update_docs.sh > /root/bots/discord/ysupport/.cache/docs_ingestion/update_docs.log 2>&1
```
