"""Real authenticated vault MCP calls against controlled Kong HTTP responses.

Run python -B -m tests.vault_lookup_e2e. Failure modes are recorded in PLAN.md.
No model, live provider, RPC, Discord or Telegram calls are made.
"""

import argparse
import asyncio
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import secrets
import socket
import tempfile
import threading
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parent.parent
ASSET = "0x" + "aa" * 20
RETIRED = "0x" + "ff" * 20
YBOLD = "0x9f4330700a36b29952869fac9b33f45eedd8a3d8"
STAKED = "0x23346b04a7f55b8760e5860aa5a77383d63491cd"


def vault(index, **overrides):
    return {
        "chainId": 1, "address": "0x" + f"{index:040x}", "origin": "yearn",
        "name": f"Yearn USDC {index}", "symbol": "yvUSDC", "apiVersion": "3.0.4",
        "asset": {"address": ASSET, "name": "USD Coin", "symbol": "USDC", "decimals": 6},
        "decimals": 6, "tvl": 1000-index,
        "performance": {"oracle": {"netAPY": 0.04}, "historical": {"net": 0.03}},
        "fees": {"performanceFee": 1000, "managementFee": None},
        "kind": "Multi Strategy", "strategiesCount": 17, "riskLevel": 1,
        "isRetired": False, "isHidden": False, "inclusion": {"isYearn": True},
        "migration": {"available": True, "target": RETIRED},
        "staking": {"available": True, "source": "VeYFI", "address": ASSET},
        "pricePerShare": "123456789012345678901234567890",
        **overrides,
    }


class Kong:
    def __init__(self):
        self.catalog = [vault(i) for i in range(1, 8)]
        self.catalog[0]["performance"]["estimated"] = {"apy": 0.12}
        self.catalog[1]["performance"]["estimated"] = {"apy": 0.0}
        self.catalog[1]["performance"]["oracle"]["netAPY"] = 0.30
        self.catalog[2]["performance"]["oracle"]["netAPY"] = None
        self.catalog[3].update(tvl="unknown", performance={"oracle": {"netAPY": "NaN"}})
        self.catalog += [
            vault(8, chainId=10), vault(9, address=RETIRED, isRetired=True),
            vault(10, name="Wrapper", asset={"address": RETIRED, "symbol": "OTHER"}, tvl=90000),
            vault(11, origin="external"),
            vault(12, address=YBOLD, name="yBOLD", symbol="yBOLD", asset={"symbol": "BOLD"}),
            vault(13, address=STAKED, name="Staked yBOLD", symbol="ysyBOLD", isHidden=True,
                  asset={"symbol": "BOLD"}, performance={"oracle": {"netAPY": 0.0347},
                    "historical": {"weeklyNet": 0.1772}, "estimated": {"apy": 0.99}}),
            vault(14, name="Yearn Saver", symbol="ysUSDC", kind="Single Strategy", strategiesCount=1),
            vault(15, origin=None),
        ]
        self.calls = []
        self.snapshot_status = 200
        self.catalog_status = 200
        self.wrong_identity = False
        self.omit_composition = False
        self.partial_snapshot = False
        self.snapshot_started = threading.Event()
        self.snapshot_release = None
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                fixture.calls.append(self.path)
                status = 200
                if self.path.startswith("/api/discover"):
                    value = {"filters": {"universe": "core"}, "rows": [{
                        "vault_address": RETIRED, "chain_id": 1, "symbol": "yvUSDC", "tvl_usd": 1000000,
                        "est_apy": 0.042, "realized_apy_30d": None, "momentum_7d_30d": None,
                        "market": "stablecoins", "token_symbol": "USDC",
                        "yearn_url": "must-not-leak", "strategy_count": 999, "is_retired": False,
                    }]}
                elif self.path == "/api/rest/list/vaults":
                    value = fixture.catalog
                    status = fixture.catalog_status
                elif self.path.startswith("/api/rest/snapshot/"):
                    if fixture.snapshot_release is not None:
                        fixture.snapshot_started.set()
                        fixture.snapshot_release.wait(5)
                    _, chain, address = self.path.rsplit("/", 2)
                    value = next((deepcopy(v) for v in fixture.catalog
                                  if v["chainId"] == int(chain) and v["address"].lower() == address.lower()), None)
                    status = fixture.snapshot_status if value else 404
                    if value:
                        value["meta"] = {key: value.pop(key) for key in (
                            "kind", "isRetired", "isHidden", "migration",
                        )}
                        value.pop("strategiesCount", None)
                        value.update(name="Exact snapshot " + value["name"], blockTime="1793537183", blockNumber="26320011")
                        value["composition"] = [{"address": ASSET, "name": f"Strategy {i}", "status": "active",
                            "currentDebtUsd": None, "lastReport": None,
                            "performance": {"historical": {"net": None}}} for i in range(17)]
                        if fixture.wrong_identity:
                            value["address"] = ASSET
                        if fixture.omit_composition:
                            value.pop("composition")
                        if fixture.partial_snapshot:
                            value = {key: value[key] for key in ("chainId", "address", "blockTime", "blockNumber")}
                else:
                    value, status = {}, 404
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


async def run(report):
    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="ysupport-vault-e2e-") as temp, ExitStack() as patches, Kong() as fixture:
        os.chdir(temp)
        patches.callback(os.chdir, old_cwd)
        patches.enter_context(patch.dict(os.environ, {
            "PYTHON_DOTENV_DISABLED": "1", "DISCORD_BOT_TOKEN": "", "TELEGRAM_BOT_TOKEN": "",
            "OPENAI_API_KEY": "", "PINECONE_API_KEY": "", "ALCHEMY_KEY": "", "GITHUB_TOKEN": "",
            "MCP_SERVER_API_KEY": "fixture", "TICKET_EXECUTION_STATE_ROOT": str(Path(temp)/"state"),
        }))
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        with patch("dotenv.load_dotenv", return_value=False):
            from ysupport.integrations import mcp_server
        from ysupport.tools import kong_vaults
        from ysupport.tools import vault_search_tools
        from ysupport import config
        import uvicorn

        logging.getLogger().setLevel(logging.ERROR)
        patches.enter_context(patch.object(kong_vaults, "KONG_BASE_URL", f"http://127.0.0.1:{fixture.server.server_port}"))
        patches.enter_context(patch.object(config, "SUPPORT_DASHBOARD_BASE_URL", f"http://127.0.0.1:{fixture.server.server_port}"))
        kong_vaults._VAULT_CATALOG_CACHE.update(value=None, fetched_at=0.0)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        token = secrets.token_hex(24)
        port = sock.getsockname()[1]
        server = mcp_server._build_mcp_server(host="127.0.0.1", port=port, api_key=token)
        for tool in await mcp_server.mcp.list_tools():
            server.add_tool(getattr(mcp_server, tool.name), annotations=tool.annotations)
        http = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="error", lifespan="on"))
        task = asyncio.create_task(http.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if http.started:
                    break
                if task.done():
                    await task
                await asyncio.sleep(0.01)
            assert http.started
            url = f"http://127.0.0.1:{port}/mcp"
            async with httpx.AsyncClient() as client:
                assert (await client.post(url)).status_code == 401
            async with streamablehttp_client(url, headers={"Authorization": "Bearer " + token}) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    discovered = {t.name: t for t in (await session.list_tools()).tools}
                    assert len(discovered) == 11
                    assert "prefer" in discovered["search_vaults"].description.lower()
                    assert "search_vaults" in discovered["support_dashboard_discover"].description

                    async def call(**args):
                        response = await session.call_tool("search_vaults", args)
                        text = "\n".join(c.text for c in response.content if hasattr(c, "text"))
                        assert not response.isError, text
                        value = response.structuredContent
                        assert isinstance(value, dict) and "rows" in value, text
                        assert json.loads(text) == value
                        report["calls"].append({"arguments": args, "result": value})
                        return value

                    first = await call(query="USDC", chain="ethereum", limit=3)
                    assert first["ui_apy_observation"]["verified"] is False
                    assert first["ui_apy_observation"]["displayed_metric"] is None
                    assert fixture.calls == ["/api/rest/list/vaults"]
                    rows = list(first["rows"])
                    page = first
                    while page["pagination"]["next_offset"] is not None:
                        page = await call(query="USDC", chain="ethereum", limit=3, offset=page["pagination"]["next_offset"])
                        rows += page["rows"]
                    assert len(rows) == first["pagination"]["total"] == 9
                    assert len({r["vault_address"] for r in rows}) == len(rows)
                    assert all("details" not in r and r["chain_id"] == 1 for r in rows)
                    assert fixture.calls == ["/api/rest/list/vaults"]
                    assert first["source"]["retrieved_at"] and first["source"]["cache_age_seconds"] >= 0
                    report["scenarios"]["structured_paging_uses_one_catalog_and_no_snapshots"] = True

                    values = {r["vault_address"]: r for r in rows}
                    assert values[fixture.catalog[0]["address"]]["estimated_apy"] == 0.12
                    assert values[fixture.catalog[1]["address"]]["estimated_apy"] == 0
                    assert values[fixture.catalog[2]["address"]]["estimated_apy"] is None
                    assert values[fixture.catalog[3]["address"]]["estimated_apy"] is None
                    assert values[fixture.catalog[3]["address"]]["tvl_usd"] is None
                    assert first["units"]["apy"] == "decimal_fraction"
                    for current, alias in (("highest_apy", "highest_apr"), ("lowest_apy", "lowest_apr")):
                        canonical = await call(query="all", sort_by=current, limit=25)
                        legacy = await call(query="all", sort_by=alias, limit=25)
                        assert [r["vault_address"] for r in canonical["rows"]] == [r["vault_address"] for r in legacy["rows"]]
                        assert canonical["filters"]["sort_by"] == legacy["filters"]["sort_by"] == current
                    assert canonical["rows"][0]["estimated_apy"] == 0
                    highest = await call(query="all", sort_by="highest_apy", limit=25)
                    assert highest["rows"][0]["vault_address"] == YBOLD
                    assert highest["rows"][0]["estimated_apy"] == 0.1772
                    assert next(r for r in highest["rows"] if r["vault_address"] == STAKED)["estimated_apy"] == 0.1772
                    assert highest["rows"][0]["apy_inputs"]["weekly_net_apy"] == 0.1772
                    assert highest["rows"][0]["apy_inputs"]["data_source"] == "catalog"
                    assert highest["rows"][0]["apy_inputs"]["catalog_retrieved_at"] == highest["source"]["retrieved_at"]
                    staked = await call(query=STAKED)
                    assert staked["rows"][0]["apy_inputs"]["data_source"] == "snapshot"
                    assert staked["rows"][0]["apy_inputs"]["block_time"] == staked["rows"][0]["block_time"]
                    report["scenarios"]["apy_aliases_null_zero_projection_and_ybold_ranking"] = True

                    before = len(fixture.calls)
                    exact = await call(query=RETIRED, limit=2)
                    assert exact["rows"][0]["vault_address"] == RETIRED
                    assert exact["rows"][0]["is_retired"]
                    assert fixture.calls[before:] == ["/api/rest/snapshot/1/" + RETIRED]
                    details = exact["rows"][0]["details"]
                    assert len(details["strategies"]) == 17
                    assert details["price_per_share_raw"] == "123456789012345678901234567890"
                    assert details["fees"]["management_fee_bps"] is None
                    assert details["strategies"][0]["current_debt_usd"] is None
                    assert details["migration"]["target"] == RETIRED
                    assert details["staking"]["available"]
                    assert exact["rows"][1]["detail_status"] == "not_requested"
                    underlying = await call(query=ASSET, limit=2)
                    assert all(r["detail_status"] == "not_requested" for r in underlying["rows"])
                    report["scenarios"]["exact_retired_identity_details_and_raw_units_survive"] = True

                    before = len(fixture.calls)
                    detailed = await call(query="USDC", limit=2, include_details=True)
                    assert len(fixture.calls[before:]) == 2
                    assert all(r["detail_status"] == "available" and "details" in r for r in detailed["rows"])
                    fixture.snapshot_status = 503
                    fallback = await call(query=RETIRED)
                    assert fallback["rows"][0]["detail_status"] == "unavailable"
                    assert "details" not in fallback["rows"][0]
                    fixture.snapshot_status = 200
                    fixture.wrong_identity = True
                    mismatch = await call(query=RETIRED)
                    assert mismatch["rows"][0]["vault_address"] == RETIRED
                    assert mismatch["rows"][0]["detail_status"] == "unavailable"
                    assert "details" not in mismatch["rows"][0]
                    fixture.wrong_identity = False
                    fixture.omit_composition = True
                    missing = await call(query=RETIRED)
                    assert missing["rows"][0]["strategy_count"] is None
                    assert missing["rows"][0]["details"]["strategies"] is None
                    fixture.omit_composition = False
                    report["scenarios"]["details_only_fetch_current_page_and_fail_closed"] = True

                    fixture.partial_snapshot = True
                    partial = (await call(query=RETIRED))["rows"][0]
                    assert partial["data_source"] == "snapshot" and partial["detail_status"] == "partial"
                    assert partial["block_time"] == "1793537183"
                    assert partial["tvl_usd"] is None and partial["estimated_apy"] is None
                    assert partial["strategy_count"] is None
                    assert partial["details"]["price_per_share_raw"] is None
                    assert partial["details"]["fees"]["performance_fee_bps"] is None
                    fixture.partial_snapshot = False
                    report["scenarios"]["partial_snapshot_never_inherits_catalog_financials"] = True

                    original_catalog = fixture.catalog
                    fixture.snapshot_status = 503
                    fixture.snapshot_release = threading.Event()
                    fixture.snapshot_started.clear()
                    kong_vaults._VAULT_CATALOG_CACHE.update(value=None, fetched_at=0.0)
                    with patch.object(kong_vaults, "_CACHE_TTL_SECONDS", 0.02):
                        waiting = asyncio.create_task(call(query=RETIRED))
                        try:
                            assert await asyncio.to_thread(fixture.snapshot_started.wait, 3)
                            original_time = kong_vaults._VAULT_CATALOG_CACHE["retrieved_at"]
                            await asyncio.sleep(0.05)
                            fixture.catalog = deepcopy(fixture.catalog)
                            next(v for v in fixture.catalog if v["address"] == RETIRED)["tvl"] = 5000
                            fresh = await call(query="USDC", limit=1)
                            assert fresh["source"]["retrieved_at"] != original_time
                        finally:
                            fixture.snapshot_release.set()
                        previous = await waiting
                        assert previous["rows"][0]["tvl_usd"] == 991
                        assert previous["source"]["retrieved_at"] == original_time
                        assert previous["source"]["cache_age_seconds"] >= 0.05
                    fixture.snapshot_release = None
                    fixture.snapshot_status = 200
                    fixture.catalog = original_catalog
                    kong_vaults._VAULT_CATALOG_CACHE.update(value=None, fetched_at=0.0)
                    report["scenarios"]["catalog_generation_and_elapsed_age_survive_concurrent_refresh"] = True

                    for args in ({"query": "USDC", "chain": "mainnet"}, {"query": "USDC", "limit": 0},
                                 {"query": "USDC", "offset": -1}, {"query": "USDC", "sort_by": "random"}, {"query": ""}):
                        before = len(fixture.calls)
                        response = await session.call_tool("search_vaults", args)
                        assert response.isError or response.structuredContent.get("error")
                        assert len(fixture.calls) == before
                    empty = await call(query="no-such-token")
                    assert empty["rows"] == [] and empty["pagination"]["total"] == 0
                    report["scenarios"]["invalid_inputs_before_io_and_explicit_empty_results"] = True

                    internal = await vault_search_tools.core_search_vaults("all", recommended_only=True)
                    assert all(not r["is_retired"] and not r["is_hidden"] for r in internal["rows"])
                    assert all(r["strategy_count"] > 1 for r in internal["rows"])
                    report["scenarios"]["internal_recommendation_filter_preserved"] = True

                    for oracle, weekly, expected in ((0.05, 0.04, 0.05), (0.05, None, 0.05)):
                        staked = next(v for v in fixture.catalog if v["address"] == STAKED)
                        staked["performance"]["oracle"]["netAPY"] = oracle
                        staked["performance"]["historical"]["weeklyNet"] = weekly
                        kong_vaults._VAULT_CATALOG_CACHE.update(value=None, fetched_at=0.0)
                        result = await call(query=YBOLD)
                        assert result["rows"][0]["estimated_apy"] == expected
                    fixture.catalog = [v for v in fixture.catalog if v["address"] != STAKED]
                    kong_vaults._VAULT_CATALOG_CACHE.update(value=None, fetched_at=0.0)
                    base = await call(query=YBOLD)
                    assert base["rows"][0]["estimated_apy"] == 0.04
                    assert "apy_inputs" not in base["rows"][0]
                    report["scenarios"]["ybold_companion_max_missing_and_base_fallback"] = True
                    fixture.catalog_status = 503
                    kong_vaults._VAULT_CATALOG_CACHE.update(value=None, fetched_at=0.0)
                    failure = await session.call_tool("search_vaults", {"query": "USDC"})
                    assert failure.structuredContent.get("error") and "rows" not in failure.structuredContent
                    report["scenarios"]["catalog_failure_is_error_not_empty_success"] = True
                    discover = await session.call_tool("support_dashboard_discover", {"token_symbol": "USDC"})
                    text = "\n".join(c.text for c in discover.content if hasattr(c, "text"))
                    projected = json.loads(text[text.index("{"):])
                    assert set(projected["rows"][0]) == {
                        "vault_address", "chain_id", "symbol", "tvl_usd", "est_apy",
                        "realized_apy_30d", "momentum_7d_30d", "market", "token_symbol",
                    }
                    assert projected["rows"][0]["est_apy"] == 0.042
                    report["scenarios"]["discovery_contract_excludes_unavailable_metadata"] = True
        finally:
            http.should_exit = True
            await task
            sock.close()


async def main(args):
    artifact = args.artifact.absolute()
    report = {"ok": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "scenarios": {}, "calls": []}
    try:
        await run(report)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["source_hashes"] = {n: hashlib.sha256((SOURCE_ROOT/n).read_bytes()).hexdigest() for n in (
            'ysupport/integrations/mcp_server.py', 'ysupport/tools/vault_search_tools.py', 'ysupport/tools/kong_vaults.py', 'ysupport/tools/support_dashboard_tools.py', "tests/vault_lookup_e2e.py",
        )}
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        artifact.chmod(0o600)
        print(json.dumps({"ok": report["ok"], "scenarios": len(report["scenarios"]), "artifact": str(artifact)}))


if __name__ == "__main__":
    import tests  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/vault-lookup-e2e.json"))
    asyncio.run(main(parser.parse_args()))
