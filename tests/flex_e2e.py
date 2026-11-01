"""Exercise real Flex MCP tools against controlled yHelper HTTP responses.

Run python -B -m tests.flex_e2e --artifact .cache/verification/flex-e2e.json.
Add --native as the service user with RUN_NATIVE_SUPPORT_E2E=1 and explicit bot
auth for real-model tool selection and interpretation. Never sends live messages.
Failure modes are recorded in PLAN.md before implementation.
"""

import argparse
import asyncio
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading
from urllib.parse import parse_qs, urlsplit
from unittest.mock import AsyncMock, patch


SOURCE_ROOT = Path(__file__).resolve().parent.parent
MARKET = "0x" + "22" * 20
WALLET = "0x" + "11" * 20
OTHER = "0x" + "33" * 20
OLD_TOOLS = {
    "search_documentation", "search_vaults", "search_repo_context",
    "fetch_repo_artifacts", "repo_context_status", "support_dashboard_discover",
    "support_dashboard_reports", "support_dashboard_status",
    "support_dashboard_changes", "support_dashboard_styfi",
}


def payloads():
    scope = {"chain_id": 1, "network": "ethereum", "source": "controlled_yhelper"}
    freshness = {"data_state": "ready", "block_number": 500, "age_seconds": 10,
                 "stale_after_seconds": 7200, "reconciliation_verdict": "ok"}
    token = {"address": OTHER, "symbol": "USDC", "decimals": 6}
    row = {
        "chain_id": 1, "label": "yvUSD/USDC", "status": "active",
        "endorsement_status": "endorsed", "contract_version": "1.1.0",
        "addresses": {"market": MARKET, "lender": OTHER},
        "collateral_token": {**token, "symbol": "yvUSD"}, "borrow_token": token,
        "metrics": {"debt": 750, "deposits": 1000, "idle_liquidity": 250,
                    "idle_liquidity_raw": "250000000", "utilization": 0.75,
                    "lender_apr": None, "average_borrow_rate": 0.02},
    }
    common = {"scope": scope, "freshness": freshness}
    summary = {"markets": {"total": 3, "active": 3}, "deposits_usd": 3000}
    other_market = {**deepcopy(row), "label": "yvWETH/WETH",
                    "addresses": {"market": OTHER, "lender": OTHER}}
    base = "/api/flex/markets/" + MARKET
    return {
        "/api/flex/protocol": {**common, "summary": summary},
        "/api/flex/markets": {**common, "filters": {"status": "active"},
                              "summary": summary, "rows": [row, other_market, deepcopy(other_market)]},
        base: {**common, "market": row, "risk": {"maximum_ltv": 0.9},
               "oracle": {"collateral_price_in_borrow_token": 1.02}},
        base + "/redemption-priority": {
            **common, "freshness": {"data_state": "ready", "source_block_number": 600,
                                    "source_age_seconds": 10, "stale_after_seconds": 7200},
            "market_address": MARKET, "borrow_token": token,
            "rate_scale": {"one_pct_raw": "10000", "unit": "annual_decimal_ratio"},
            "total_debt_raw": "750000000", "total_debt": 750,
            "idle_liquidity_raw": "250000000", "idle_liquidity": 250,
            "source_url": "https://flex.example/market/" + MARKET,
            "points": [{"annual_interest_rate": rate, "redeemable_before": debt,
                        "annual_interest_rate_raw": str(int(rate * 1000000)),
                        "redeemable_before_raw": str(debt * 1000000)}
                       for rate, debt in [(0.01, 0), (0.02, 100), (0.05, 500)]],
        },
        base + "/trove-health": {
            **common, "market_address": MARKET, "collateral_token": token,
            "borrow_token": token,
            "metrics": {"active_troves": 9, "median_ltv": 0.8,
                        "near_max_troves": 2, "debt_near_max_share": 0.1},
        },
        base + "/history": {
            **common, "market_address": MARKET, "filters": {"days": 7, "interval": "day"},
            "coverage": {"points": 3, "expected_points": 3, "coverage_ratio": 1},
            "points": [{"sampled_at": f"2026-10-{day}T00:00:00Z", "utilization": 0.75}
                       for day in (29, 30, 31)],
        },
        "/api/flex/activity": {
            **common, "filters": {"market_address": MARKET, "event": "borrow"},
            "pagination": {"limit": 2, "next_cursor": "500:2"},
            "rows": [{"event": "borrow", "tx_hash": "0x" + "55" * 32,
                      "market_address": MARKET, "actors": {"owner": WALLET},
                      "amounts": {"debt_raw": "100000000"}} for _ in range(2)],
        },
        "/api/flex/performance": {
            "scope": {**scope, "address": WALLET, "market_status": "active"},
            "freshness": {"data_state": "partial"},
            "period": {"days": 7, "start_block": 100, "end_block": 500},
            "borrowers": {"positions": 3, "covered_positions": 1, "pnl_usd": None,
                          "end_value_usd": None,
                          "rows": [{"market_address": MARKET, "pnl_usd": -12,
                                    "is_complete": True, "issue": None},
                                   *[{"market_address": MARKET, "pnl_usd": None,
                                    "is_complete": False, "issue": "Opening valuation unavailable"}
                                     for _ in range(2)]]},
            "lenders": {"positions": 0, "covered_positions": 0, "pnl_usd": None,
                        "end_value_usd": None, "rows": []},
        },
    }


class Dashboard:
    def __init__(self):
        self.responses = payloads()
        self.calls = []
        self.status = 200
        self.malformed = False
        self.wrong_wallet = False
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                query = {key: values[0] for key, values in parse_qs(url.query).items()}
                fixture.calls.append({"path": url.path, "query": query})
                value = deepcopy(fixture.responses.get(url.path))
                status = fixture.status if value is not None else 404
                if url.path == "/api/flex/performance" and isinstance(value, dict):
                    value["scope"]["address"] = OTHER if fixture.wrong_wallet else query.get("address")
                body = json.dumps([] if fixture.malformed else value).encode()
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


def text_result(result):
    return "\n".join(item.text for item in result.content if hasattr(item, "text"))


def check_native_answer(index, answer):
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", answer.lower().replace("’", "'"))
    text = re.sub(r"[*_`]", "", text)

    def require(pattern):
        assert re.search(pattern, text), f"Missing required Flex answer evidence ({pattern}): {answer}"

    if index == 0:
        require(r"\butilization\b[^.;\n]{0,40}(?<![\d.+-])(?:75(?:\.0+)?\s*%|0\.750*(?!\s*%))(?!\d|\.\d)")
        require(r"\bidle\b[^.;\n]{0,65}(?<![\w.+-])250(?:\.0+)?\s+usdc\b")
        # Keep the threshold and its value in one fact; never cross another rate.
        context = r"(?:(?!\b(?:not|unknown|unavailable|unresolved|unconfirmed)\b)[^.;\n%])"
        rate = rf"\b(?:below|lower)\b{context}{{0,35}}(?<![\d.])2(?:\.0+)?\s*%"
        amount = r"(?<![\w.+-])100(?:\.0+)?\s+usdc\b"
        link = r"\s*(?:(?:[:=]|is|equals?|totals?|amounts? to)\s*|[—–-]\s+)?"
        require(
            rf"\bdebt\b{context}{{0,100}}{rate}{link}{amount}"
            rf"|\bdebt\b{context}{{0,40}}{amount}{context}{{0,80}}{rate}"
        )
        return

    assert index == 1
    require(r"(?<![\d.])3\s+(?:active.market\s+)?(?:position\s+)?(?:positions|rows)\b")
    require(r"(?<![\d.])1(?:\s+(?:of|out of)\s+3)?(?:\s+(?:positions?|rows?))?(?:\s+(?:is|was|are|were))?\s+(?:fully\s+)?(?:covered|complete)\b")
    require(r"(?<![\w.])(?:-\s*\$?\s*12(?:\.0+)?|\$\s*-\s*12(?:\.0+)?|loss[^.;\n]{0,15}\$\s*12(?:\.0+)?)(?!\d|\.\d)")
    require(r"\bno\b[^.;\n]{0,25}\blender\b|\blender\b[^.;\n]{0,40}\b(?:no|zero|0)\b")
    clauses = re.split(r"(?<=[.!?;])\s+|\n+|\b(?:but|however|nevertheless)\b", text)
    denial = r"\b(?:cannot|can't|can not|does not|doesn't|do not|don't|is not enough|insufficient|unable to|not possible)\b"
    unresolved = denial + r"|\b(?:unknown|unavailable|unresolved|unconfirmed|undetermined|not known|not provided)\b"
    for topic in (r"\b(?:ltv|loan.to.value)\b", r"\bliquidation\b"):
        assert any(re.search(topic, clause) and re.search(unresolved, clause) for clause in clauses), answer
    assert any(re.search(r"\b(?:aggregate|total|full.wallet)\b.*\bpnl\b", clause)
               and re.search(r"\b(?:unavailable|unresolved|unknown|incomplete|not available)\b|" + denial, clause)
               for clause in clauses), answer
    # Uncertainty elsewhere must not excuse contradictory numeric/safety claims.
    for pattern in (
        r"\b(?:ltv|loan.to.value)\b\s*(?:(?:is|=|:|of)\s*)?[-+]?\d",
        r"\b(?:ltv|loan.to.value)\b[^.;\n]{0,30}\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s*%[^.;\n]{0,30}\b(?:ltv|loan.to.value)\b",
        r"\b(?:aggregate|total|full.wallet)\b[^.;\n]{0,60}\bpnl\s*(?:is|=|:|of)?\s*[-+]?\$?\d",
    ):
        assert not re.search(pattern, text), f"Unsupported Flex wallet claim: {answer}"
    safety_predicate = (
        r"(?:\b(?:is|are|am|remains?|seems?|looks?|will be)\b|\b\w+'(?:re|s|m)\b)"
        r"\s+(?:(?:\w+ly|not)\s+)*(?:safe|unsafe)\b|\b(?:no|zero)\s+(?:liquidation\s+)?risk\b"
    )
    safety_denial = (
        denial + r"\s+(?:(?:reliably|confidently|definitively|currently|yet|to)\s+){0,2}"
        r"(?:establish|confirm|determine|verify|assess|say|tell|know|conclude|prove|guarantee|show|judge|infer)\b"
        r"|\b(?:unknown|unclear|unconfirmed|undetermined)\s+(?:whether|if)\b"
    )
    # A denial about PnL, or before a new assertion, cannot govern safety.
    scope = r"(?:(?!\b(?:pnl|profit|loss|so|therefore|thus|hence|because|and|but|however)\b).){0,200}$"
    unresolved_safety = (
        r"\s+(?:from\s+liquidation\s+)?(?:is\s+)?"
        r"(?:unknown|unclear|unresolved|unconfirmed|undetermined|unavailable|cannot be (?:established|confirmed|determined))\b"
    )
    for clause in clauses:
        for claim in re.finditer(safety_predicate, clause):
            linked_denial = re.search(rf"(?:{safety_denial}){scope}", clause[:claim.start()])
            linked_unknown = re.match(unresolved_safety, clause[claim.end():])
            assert linked_denial or linked_unknown, f"Unsupported liquidation-safety claim: {answer}"


def check_grading_controls(report):
    market_answer = (
        "Utilization: 75%. Immediately idle liquidity: 250 USDC. "
        "Debt at annual rates strictly below 2%: 100 USDC."
    )
    wallet_answer = (
        "Borrower performance is partial: 1 of 3 positions is covered, showing -$12 PnL; "
        "aggregate PnL is unavailable. No lender positions were returned. "
        "This data cannot establish the wallet's current LTV or confirm that it is safe from liquidation."
    )
    for index, answer in (
        (0, market_answer), (1, wallet_answer),
        (0, market_answer.replace("75%", "0.75").replace("250 USDC", "**250.00 USDC**")),
        (1, wallet_answer.replace("-$12", "a loss of $12").replace("cannot", "can't")),
        (1, wallet_answer.replace("is partial", "has limited coverage").replace("unavailable", "unresolved")),
        (1, wallet_answer.replace(
            "This data cannot establish the wallet's current LTV or confirm that it is safe from liquidation.",
            "The wallet's current LTV is unknown. Liquidation safety cannot be established from this data.",
        )),
        (0, market_answer.replace("below 2%: 100 USDC", "below 2.00%: 100.00 USDC")),
        (0, market_answer.replace("Debt at annual rates strictly below 2%: 100 USDC", "Debt: 100 USDC at annual rates strictly below 2%")),
        (1, wallet_answer.replace("confirm that it is safe", "confirm whether you're safe")),
        (1, wallet_answer.replace("confirm that it is safe", "confirm whether you’re safe")),
        (1, wallet_answer.replace(
            "This data cannot establish the wallet's current LTV or confirm that it is safe from liquidation.",
            "The wallet's current LTV is unknown. Whether you're safe from liquidation is unknown.",
        )),
    ):
        check_native_answer(index, answer)
    invalid = [
        (0, market_answer.replace("250", "2,500")),
        (0, market_answer.replace("100", "1,000")),
        (0, market_answer.replace("250", "250.5")),
        (0, market_answer.replace("250", "-250")),
        (0, market_answer.replace("250 USDC", "250 USDT")),
        (0, market_answer.replace("250 USDC", "100 USDC").replace("2%: 100", "2%: 250")),
        (0, market_answer.replace("75%", "0.75%")),
        (0, market_answer.replace("below 2%", "below 3%") + " Debt below 2% is unavailable."),
        (0, market_answer.replace("below 2%: 100 USDC", "below 3%: 100 USDC, while debt below 2% is 200 USDC")),
        (0, market_answer.replace("below 2%: 100 USDC", "below 2% is not 100 USDC")),
        (0, market_answer.replace("Debt at annual rates strictly below 2%: 100 USDC", "Debt: 100 USDC is not at annual rates below 2%")),
        (1, "The historical data is partial. Your current trove LTV is 70%, and this "
            "proves you are safe from liquidation. Your total 7-day PnL is $0."),
        (1, wallet_answer + " Your current LTV is 70%."),
        (1, wallet_answer + " Your current LTV is 0.70."),
        (1, wallet_answer + " You are safe from liquidation."),
        (1, wallet_answer + " You're safe from liquidation."),
        (1, wallet_answer + " You’re safe from liquidation."),
        (1, wallet_answer + " I cannot calculate full PnL, so you are safe from liquidation."),
        (1, wallet_answer + " I cannot assess PnL and you're safe from liquidation."),
        (1, wallet_answer + " LTV cannot be established, but you are safe from liquidation."),
        (1, wallet_answer + " Your total 7-day PnL is $0."),
        (1, wallet_answer.replace("-$12", "$12")),
        (1, wallet_answer.replace("1 of 3", "3 of 3")),
        (1, wallet_answer.replace("1 of 3", "1 of 4")),
        (1, wallet_answer.replace(" or confirm that it is safe from liquidation", "")),
    ]
    report["grading_controls"] = []
    for index, answer in invalid:
        try:
            check_native_answer(index, answer)
        except AssertionError:
            report["grading_controls"].append({"question_index": index, "answer": answer, "rejected": True})
        else:
            raise AssertionError(f"Native replay accepted an incorrect answer: {answer}")
    report["scenarios"]["native_grading_rejects_incorrect_amounts_and_wallet_claims"] = True


async def run(args, report):
    check_grading_controls(report)
    runtime_env = dict(os.environ)
    auth = Path(runtime_env.get("TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE", ""))
    if args.native:
        if os.geteuid() == 0 or not auth.is_file():
            raise ValueError("Native probes require the unprivileged service user and explicit bot auth.")
        if auth.stat().st_uid != os.geteuid() or auth.stat().st_mode & 0o077:
            raise ValueError("Native auth must be service-owned and mode 0600.")
    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="ysupport-flex-e2e-") as temp, ExitStack() as patches, Dashboard() as fixture:
        root = Path(temp)
        os.chdir(root)
        patches.callback(os.chdir, old_cwd)
        patches.enter_context(patch.dict(os.environ, {
            "PYTHON_DOTENV_DISABLED": "1", "DISCORD_BOT_TOKEN": "",
            "TELEGRAM_BOT_TOKEN": "", "OPENAI_API_KEY": "", "PINECONE_API_KEY": "",
            "ALCHEMY_KEY": "", "GITHUB_TOKEN": "", "MCP_SERVER_API_KEY": "fixture",
            "TICKET_EXECUTION_STATE_ROOT": str(root / "state"),
        }))
        with patch("dotenv.load_dotenv", return_value=False):
            import config
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        import mcp_server
        import support_dashboard_tools
        import uvicorn

        logging.getLogger().setLevel(logging.ERROR)
        patches.enter_context(patch.object(config, "SUPPORT_DASHBOARD_BASE_URL",
                                          f"http://127.0.0.1:{fixture.server.server_port}"))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        token = secrets.token_hex(24)
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
                    raise RuntimeError("Fixture MCP failed to start")
                await asyncio.sleep(0.01)
            assert http.started
            url = f"http://127.0.0.1:{port}/mcp"
            async with httpx.AsyncClient() as client:
                assert (await client.post(url)).status_code == 401
            async with streamablehttp_client(url, headers={"Authorization": "Bearer " + token}) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                    assert set(tools) == OLD_TOOLS | {"support_dashboard_flex"}
                    assert tools["support_dashboard_flex"].annotations.readOnlyHint
                    report["scenarios"]["authenticated_discovery_preserves_existing_tools"] = True

                    async def call(**arguments):
                        result = await session.call_tool("support_dashboard_flex", arguments)
                        assert not result.isError, text_result(result)
                        text = text_result(result)
                        return json.loads(text[text.index("{"):])

                    views = [
                        ("protocol", {}, "/api/flex/protocol", {}),
                        ("markets", {"status": "active", "limit": 2}, "/api/flex/markets", {"status": "active"}),
                        ("market", {"market_address": MARKET}, "/api/flex/markets/" + MARKET, {}),
                        ("redemption_priority", {"market_address": MARKET, "limit": 2}, "/api/flex/markets/" + MARKET + "/redemption-priority", {}),
                        ("trove_health", {"market_address": MARKET}, "/api/flex/markets/" + MARKET + "/trove-health", {}),
                        ("history", {"market_address": MARKET, "days": 7, "interval": "day", "limit": 2}, "/api/flex/markets/" + MARKET + "/history", {"days": "7", "interval": "day"}),
                        ("activity", {"market_address": MARKET, "event": "borrow", "cursor": "501:3", "limit": 2}, "/api/flex/activity", {"market_address": MARKET, "event": "borrow", "cursor": "501:3", "limit": "2"}),
                        ("performance", {"wallet_address": WALLET, "days": 7, "limit": 2}, "/api/flex/performance", {"address": WALLET, "days": "7"}),
                    ]
                    results = {}
                    for view, arguments, path, query in views:
                        before = len(fixture.calls)
                        results[view] = await call(view=view, **arguments)
                        assert fixture.calls[before:] == [{"path": path, "query": query}]
                        assert results[view]["scope"]["chain_id"] == 1
                        assert results[view]["freshness"]["data_state"] in {"ready", "partial"}
                    report["scenarios"]["all_eight_views_use_exact_http_routes"] = True

                    assert results["market"]["market"]["metrics"]["lender_apr"] is None
                    assert results["redemption_priority"]["points"][0]["redeemable_before"] == 0
                    assert results["market"]["market"]["metrics"]["utilization"] == 0.75
                    assert results["redemption_priority"]["rate_scale"]["one_pct_raw"] == "10000"
                    assert results["market"]["freshness"]["block_number"] == 500
                    assert results["redemption_priority"]["freshness"]["source_block_number"] == 600
                    report["scenarios"]["null_zero_raw_units_and_decimal_rates_preserved"] = True
                    for view, key in (("markets", "rows"), ("redemption_priority", "points"), ("history", "points")):
                        assert len(results[view][key]) == 2
                        assert results[view]["output_limits"][key] == {"available": 3, "returned": 2, "truncated": True}
                    assert results["history"]["points"][0]["sampled_at"].startswith("2026-10-30")
                    assert results["history"]["coverage"]["points"] == 3
                    assert results["history"]["window_endpoints"]["first"]["sampled_at"].startswith("2026-10-29")
                    assert results["history"]["window_endpoints"]["last"]["sampled_at"].startswith("2026-10-31")
                    assert results["activity"]["pagination"]["next_cursor"] == "500:2"
                    assert results["performance"]["borrowers"]["pnl_usd"] is None
                    assert results["performance"]["borrowers"]["covered_positions"] == 1
                    assert results["performance"]["output_limits"]["borrowers.rows"]["truncated"]
                    assert results["performance"]["borrowers"]["rows"][0]["is_complete"] is True
                    assert results["performance"]["borrowers"]["rows"][1]["is_complete"] is False
                    report["scenarios"]["bounds_cursors_and_partial_coverage_are_explicit"] = True

                    market_path = "/api/flex/markets/" + MARKET
                    fixture.responses[market_path]["freshness"]["data_state"] = "stale"
                    assert (await call(view="market", market_address=MARKET))["freshness"]["data_state"] == "stale"
                    fixture.responses[market_path]["freshness"]["data_state"] = "ready"
                    report["scenarios"]["stale_source_state_not_rewritten"] = True

                    invalid = [
                        {"view": "https://example.invalid"}, {"view": "market"},
                        {"view": "market", "market_address": "../meta/status"},
                        {"view": "markets", "wallet_address": WALLET},
                        {"view": "protocol", "days": 7},
                        {"view": "activity", "cursor": "1:2&address=other"},
                        {"view": "activity", "limit": 51},
                        {"view": "history", "market_address": MARKET, "days": 8},
                    ]
                    for arguments in invalid:
                        before = len(fixture.calls)
                        result = await session.call_tool("support_dashboard_flex", arguments)
                        assert result.isError or "Error querying Flex data" in text_result(result)
                        assert len(fixture.calls) == before
                    report["scenarios"]["invalid_views_arguments_and_injection_rejected_before_io"] = True

                    fixture.responses[market_path]["market"]["addresses"]["market"] = OTHER
                    result = await session.call_tool("support_dashboard_flex", {"view": "market", "market_address": MARKET})
                    assert "Error querying Flex data" in text_result(result)
                    fixture.responses = payloads()
                    fixture.wrong_wallet = True
                    result = await session.call_tool("support_dashboard_flex", {"view": "performance", "wallet_address": WALLET})
                    assert "Error querying Flex data" in text_result(result)
                    fixture.wrong_wallet = False
                    mismatches = [
                        ("/api/flex/markets", "filters", "status", "all", {"view": "markets"}),
                        (market_path + "/history", "filters", "days", 30,
                         {"view": "history", "market_address": MARKET, "days": 7}),
                        ("/api/flex/activity", "filters", "market_address", OTHER,
                         {"view": "activity", "market_address": MARKET, "limit": 2}),
                        ("/api/flex/activity", "pagination", "limit", 50,
                         {"view": "activity", "market_address": MARKET, "limit": 2}),
                        ("/api/flex/performance", "period", "days", 30,
                         {"view": "performance", "wallet_address": WALLET, "days": 7}),
                    ]
                    for path, section, field, value, arguments in mismatches:
                        fixture.responses[path][section][field] = value
                        result = await session.call_tool("support_dashboard_flex", arguments)
                        assert "Error querying Flex data" in text_result(result), (path, section, field)
                        fixture.responses = payloads()
                    report["scenarios"]["mismatched_filters_period_and_activity_page_rejected"] = True
                    fixture.malformed = True
                    result = await session.call_tool("support_dashboard_flex", {"view": "protocol"})
                    assert "Error querying Flex data" in text_result(result)
                    fixture.malformed = False
                    fixture.status = 503
                    before = len(fixture.calls)
                    result = await session.call_tool("support_dashboard_flex", {"view": "protocol"})
                    assert "Error querying Flex data" in text_result(result) and len(fixture.calls) == before + 2
                    fixture.status = 200
                    report["scenarios"]["wrong_identity_malformed_payload_and_provider_errors_fail_closed"] = True

            if args.native:
                import docs_repo_tools
                import onchain_tools
                import vault_position_tools
                import chain_access
                import vault_search_tools
                from ticket_investigation.codex_support_endpoint import CodexSupportTicketExecutionJsonEndpoint
                from ticket_investigation.transport import build_smoke_transport_request

                for module, name in [
                    (docs_repo_tools, "core_search_docs_context"),
                    (docs_repo_tools, "core_search_repo_context"),
                    (docs_repo_tools, "core_fetch_repo_artifacts"),
                    (docs_repo_tools, "core_repo_context_status"),
                    (vault_search_tools, "core_search_vaults"),
                    *[(support_dashboard_tools, "core_" + name) for name in OLD_TOOLS if name.startswith("support_dashboard_")],
                ]:
                    patches.enter_context(patch.object(module, name, new=AsyncMock(return_value="No matching observation in this controlled fixture.")))
                rpc_probes = []
                for module, name in ((onchain_tools, "core_inspect_onchain"),
                                     (vault_position_tools, "core_check_all_deposits"),
                                     (vault_position_tools, "core_get_withdrawal_instructions"),
                                     (chain_access, "core_read_chain_state")):
                    probe = AsyncMock(side_effect=AssertionError("Unexpected internal RPC during Flex data replay"))
                    patches.enter_context(patch.object(module, name, new=probe))
                    rpc_probes.append(probe)
                child_env = {key: value for key, value in runtime_env.items() if key in {"PATH", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR"}}
                child_env.update(HOME=str(root), CODEX_HOME=str(root / "home"))
                endpoint = CodexSupportTicketExecutionJsonEndpoint(
                    codex_command=config.TICKET_EXECUTION_CODEX_COMMAND or None,
                    model=config.TICKET_EXECUTION_CODEX_MODEL,
                    reasoning_effort=config.TICKET_EXECUTION_CODEX_REASONING_EFFORT,
                    codex_home=root / "home", codex_auth_link_source=auth,
                    session_dir=root / "sessions", ysupport_mcp_url=url,
                    mcp_server_api_key=token, web_search_mode="disabled", env=child_env,
                    timeout_seconds=180,
                )
                questions = [
                    "For the Flex lending protocol's yvUSD/USDC market on Ethereum, find the market and report current utilization, immediately idle borrow-token liquidity, and the debt at strictly lower chosen annual rates than 2%.",
                    f"Check 7-day Flex borrower and lender performance for wallet {WALLET}. Can this data establish the wallet's current individual trove LTV or whether it is safe from liquidation?",
                ]
                for index, question in enumerate(questions):
                    before = len(fixture.calls)
                    request = replace(build_smoke_transport_request(), smoke_mode=None,
                                      aggregated_text=question,
                                      input_list=[{"role": "system", "content": "Controlled evaluation: use only the configured MCP and its local data. Do not query live sites or providers or write files."},
                                                  {"role": "user", "content": question}])
                    response = json.loads(await endpoint.execute_json_turn(request.to_json()))
                    answer = response["flow_outcome"]["raw_final_reply"]
                    calls = fixture.calls[before:]
                    probe = {"passed": False, "question": question, "answer": answer,
                             "provider_calls": calls, "model": endpoint.model,
                             "reasoning_effort": endpoint.reasoning_effort,
                             "rpc_helper_calls": sum(probe.call_count for probe in rpc_probes)}
                    report["native_probes"].append(probe)
                    if index == 0:
                        assert any(c["path"] in {market_path, "/api/flex/markets"} for c in calls), probe
                        assert any(c["path"] == market_path + "/redemption-priority" for c in calls)
                    else:
                        assert any(c["path"] == "/api/flex/performance" and c["query"].get("address") == WALLET for c in calls)
                    check_native_answer(index, answer)
                    assert not response["flow_outcome"]["requires_human_handoff"]
                    assert not any(probe.call_count for probe in rpc_probes)
                    probe["passed"] = True
        finally:
            http.should_exit = True
            await task
            sock.close()


async def main(args):
    artifact = args.artifact.absolute()
    report = {"ok": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scenarios": {}, "native_probes": []}
    try:
        await run(args, report)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["source_hashes"] = {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() for name in (
            "mcp_server.py", "support_dashboard_tools.py", "tests/flex_e2e.py",
            "ysupport_codex_instructions.md", "ticket_investigation/codex_support_endpoint.py",
        )}
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(report, indent=2) + "\n")
        artifact.chmod(0o600)
        print(json.dumps({"ok": report["ok"], "scenarios": len(report["scenarios"]), "native_probes": len(report["native_probes"]), "artifact": str(artifact)}))


if __name__ == "__main__":
    import tests  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/flex-e2e.json"))
    parser.add_argument("--native", action="store_true")
    asyncio.run(main(parser.parse_args()))
