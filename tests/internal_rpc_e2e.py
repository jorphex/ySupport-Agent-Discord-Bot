"""Exercise native execution -> private RPC client -> real helpers -> HTTP RPC.

Only the model, RPC node and Kong responses are fixtures. No live messages or
provider credentials are used. Evidence: .cache/verification/internal-rpc-e2e.json.
Failure cases are recorded in PLAN.md before implementing the internal connection.
"""

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import AsyncMock, patch

import tests  # noqa: F401

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from ysupport.tools import chain_access  # noqa: E402
from ysupport import config  # noqa: E402
from ysupport.tools import onchain_tools  # noqa: E402
from ysupport.tools import vault_position_tools  # noqa: E402
from ysupport.execution.codex_support_endpoint import CodexSupportTicketExecutionJsonEndpoint  # noqa: E402
from ysupport.execution.internal_rpc import InternalRpcServer  # noqa: E402
from ysupport.execution.transport import build_smoke_transport_request  # noqa: E402
from web3 import Web3  # noqa: E402


WALLET = "0x" + "11" * 20
VAULT = "0x" + "22" * 20
GAUGE = "0x" + "33" * 20
OLD_VAULT = "0x" + "44" * 20
TX = "0x" + "55" * 32
SECRET = "internal-rpc-fixture-secret"


class RpcNode:
    def __init__(self):
        self.calls = []
        self.fail_gauge = False
        self.delay = 0
        self.override = None
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fixture.calls.append(request)
                time.sleep(fixture.delay)
                method, params = request.get("method", "graphql"), request.get("params", []) if "method" in request else request
                error, result = None, None
                override = fixture.override(method, params) if fixture.override else None
                if override is not None:
                    result, error = override
                elif method == "web3_clientVersion":
                    result = "ysupport-fixture/1"
                elif method == "eth_chainId":
                    result = "0x1"
                elif method == "eth_blockNumber":
                    result = "0x1000"
                elif method in {"eth_getBalance", "eth_getTransactionCount"}:
                    if params[0].lower() != WALLET:
                        error = {"code": -32000, "message": "Unknown fixture wallet"}
                    elif method == "eth_getBalance":
                        result = hex(500_000_000_000_000_000)
                    else:
                        result = "0x6" if len(params) > 1 and params[1] == "pending" else "0x5"
                elif method == "eth_getCode":
                    result = "0x6000"
                elif method == "eth_call":
                    target = params[0]["to"].lower()
                    data = params[0].get("data", "")
                    balance = Web3.keccak(text="balanceOf(address)").hex().removeprefix("0x")[:8]
                    if target not in {VAULT, GAUGE, OLD_VAULT} or (data[2:10] == balance and data[-40:].lower() != WALLET[2:]):
                        error = {"code": -32000, "message": "Unknown fixture contract or wallet"}
                    elif target == GAUGE and fixture.fail_gauge:
                        error = {"code": -32000, "message": "Unavailable at /" + SECRET}
                    else:
                        convert = Web3.keccak(text="convertToAssets(uint256)").hex().removeprefix("0x")[:8]
                        value = (5_000_000 if data[2:10] == convert else 2_000_000) if target == GAUGE else 10_000_000 if target == VAULT else 1_000_000 if target == OLD_VAULT else 0
                        result = "0x" + hex(value)[2:].zfill(64)
                elif method == "eth_getTransactionReceipt":
                    result = {"transactionHash": TX, "transactionIndex": "0x0", "blockHash": "0x" + "66" * 32, "blockNumber": "0x7b", "from": WALLET, "to": VAULT, "cumulativeGasUsed": "0x5208", "gasUsed": "0x5208", "contractAddress": None, "logs": [], "logsBloom": "0x" + "00" * 256, "status": "0x1", "effectiveGasPrice": "0x1", "type": "0x2"}
                elif method == "eth_getTransactionByHash":
                    result = {"hash": TX, "nonce": "0x0", "blockHash": "0x" + "66" * 32, "blockNumber": "0x7b", "transactionIndex": "0x0", "from": WALLET, "to": VAULT, "value": "0x0", "gas": "0x5208", "gasPrice": "0x1", "input": "0x", "v": "0x1", "r": "0x1", "s": "0x1", "type": "0x2", "chainId": "0x1"}
                else:
                    error = {"code": -32601, "message": "Unexpected RPC method"}
                if override is None and method in {"eth_getTransactionReceipt", "eth_getTransactionByHash"} and params[0].lower() != TX:
                    result, error = None, {"code": -32000, "message": "Unknown fixture transaction"}
                payload = json.dumps(result if method == "graphql" else {"jsonrpc": "2.0", "id": request["id"], **({"error": error} if error else {"result": result})}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


async def main(artifact, *, native=False):
    if native and os.geteuid() == 0:
        raise ValueError("Run native probes as the unprivileged bot service user.")
    report = {"ok": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "scenarios": {}}
    logging.getLogger().setLevel(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix="ysupport-internal-rpc-e2e-") as temp, RpcNode() as node:
        root = Path(temp)
        model = root / "model.py"
        model.write_text('''import json,os,subprocess,sys
from pathlib import Path
request=json.loads((Path(os.environ["TICKET_EXECUTION_RUN_DIR"])/"support_request.json").read_text())
assert not any(k.endswith("_RPC_URL") or k=="ALCHEMY_KEY" for k in os.environ)
command=request["support_state"]["internal_rpc"]["command"]
results=[]
for query in json.loads(request["current_user_message"]):
 response=subprocess.run(command,input=json.dumps(query),text=True,capture_output=True,timeout=10)
 results.append(json.loads(response.stdout))
result={"answer":json.dumps(results),"requires_human_handoff":False,"handoff_kind":None,"handoff_reason":None,"evidence_summary":"Real helper/provider E2E","used_tools":["shell"] if results else []}
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":json.dumps(result)}}))
''')
        vault = {"address": VAULT, "chainId": 1, "origin": "yearn", "apiVersion": "3.0.0", "name": "Fixture USDC", "symbol": "yvUSDC", "decimals": 6, "asset": {"symbol": "USDC"}, "isRetired": True, "staking": {"available": True, "address": GAUGE}}
        old = {"address": OLD_VAULT, "name": "Older fixture vault", "symbol": "yUSDC", "decimals": 6}
        url = f"http://127.0.0.1:{node.server.server_port}/{SECRET}"
        with (
            patch.dict(os.environ, {"ALCHEMY_KEY": SECRET, "ETHEREUM_RPC_URL": url}),
            patch.object(config, "RPC_URLS", {"ethereum": url}),
            patch.object(chain_access, "WEB3_INSTANCES", {}),
            patch.object(chain_access, "_WEB3_BOOTSTRAPPED", False),
            patch.object(vault_position_tools, "V1_VAULTS", [old]),
            patch.object(vault_position_tools, "fetch_kong_vault_catalog", new=AsyncMock(return_value=[vault])),
            patch.object(vault_position_tools, "fetch_kong_vault_snapshot", new=AsyncMock(return_value=vault)),
        ):
            endpoint = CodexSupportTicketExecutionJsonEndpoint(codex_command=[sys.executable, str(model)], codex_home=root / "home", artifact_dir=str(root / "exports"), env={"PATH": os.environ["PATH"], "HOME": str(root)}, timeout_seconds=20)

            async def turn(queries):
                request = replace(build_smoke_transport_request(), smoke_mode=None, aggregated_text=json.dumps(queries))
                value = json.loads(await endpoint.execute_json_turn(request.to_json()))
                replies = json.loads(value["flow_outcome"]["raw_final_reply"])
                assert SECRET not in json.dumps(value)
                assert not list((root / "exports").rglob("*.sock"))
                return replies

            def inspect(**args):
                return {"operation": "inspect_onchain", "arguments": {"chain": "ethereum", **args}}
            scan = {"operation": "check_all_deposits", "arguments": {"user_address_or_ens": WALLET}}
            withdrawal = {"operation": "get_withdrawal_instructions", "arguments": {"chain": "ethereum", "vault_address": VAULT, "user_address_or_ens": WALLET}}
            contract_read = inspect(mode="call", to_address=VAULT, function_signature="balanceOf(address)", args_json=json.dumps([WALLET]), output_types_json='["uint256"]')
            before = len(node.calls)
            assert await turn([]) == [] and len(node.calls) == before
            report["scenarios"]["docs_without_rpc"] = True
            replies = await turn([inspect(mode="receipt", tx_hash=TX), scan, withdrawal, contract_read])
            assert all(r["ok"] for r in replies)
            assert TX in replies[0]["result"] and "status: 1" in replies[0]["result"], replies[0]
            assert "15.000000" in replies[1]["result"] and "1.000000" in replies[1]["result"], replies[1]
            assert "redeem" in replies[2]["result"] and "10000000" in replies[2]["result"]
            assert "10000000" in replies[3]["result"], replies[3]
            assert {c["method"] for c in node.calls} >= {"eth_call", "eth_getTransactionReceipt", "eth_getTransactionByHash"}
            report["scenarios"]["receipt_wallet_gauge_old_vault_and_withdrawal"] = True
            generic_withdrawal = {"operation": "get_withdrawal_instructions", "arguments": {"chain": " Ethereum ", "vault_address": VAULT}}
            before = len(node.calls)
            reply = (await turn([generic_withdrawal]))[0]
            assert reply["ok"] and json.loads(reply["result"])["status"] == "not_ready" and "instructions" not in json.loads(reply["result"]) and len(node.calls) == before, reply
            reply = (await turn([inspect(mode=" Receipt ", chain=" Ethereum ", tx_hash=TX)]))[0]
            assert reply["ok"] and "status: 1" in reply["result"], reply
            report["scenarios"]["optional_wallet_and_canonical_helper_names_preserved"] = True

            exact_integer = 1234567890123456789
            reply = (await turn([inspect(mode="call", to_address=VAULT,
                function_signature="previewRedeem(uint256)", args_json=json.dumps([str(exact_integer)]),
                output_types_json='["uint256"]')]))[0]
            assert reply["ok"] and "10000000" in reply["result"], reply
            encoded = next(c["params"][0]["data"] for c in reversed(node.calls) if c["method"] == "eth_call")
            assert int(encoded[-64:], 16) == exact_integer
            report["scenarios"]["decimal_integer_argument_preserves_exact_units"] = True

            # An unrelated incomplete catalog row must not abort receipt lookup.
            nullable = {"address": "0x" + "99" * 20, "chainId": 1,
                        "asset": None, "symbol": None}
            with patch.object(vault_position_tools, "fetch_kong_vault_catalog", new=AsyncMock(return_value=[nullable, vault])):
                filtered = {"operation": "check_all_deposits", "arguments": {
                    "user_address_or_ens": WALLET, "token_symbol": "yvUSDC"}}
                reply = (await turn([filtered]))[0]
                assert reply["ok"] and "15.000000" in reply["result"], reply
                filtered["arguments"]["token_symbol"] = "USDC"
                reply = (await turn([filtered]))[0]
                assert reply["ok"] and "15.000000" in reply["result"], reply
            report["scenarios"]["nullable_catalog_and_receipt_symbol_lookup"] = True

            katana = {**vault, "chainId": 747474}
            with patch.object(config, "RPC_URLS", {"ethereum": url, "katana": url}), \
                 patch.object(chain_access, "WEB3_INSTANCES", {}), \
                 patch.object(chain_access, "_WEB3_BOOTSTRAPPED", False), \
                 patch.object(vault_position_tools, "fetch_kong_vault_catalog", new=AsyncMock(return_value=[nullable, katana])), \
                 patch.object(vault_position_tools, "fetch_kong_vault_snapshot", new=AsyncMock(return_value=katana)):
                before = len(node.calls)
                scoped = {"operation": "check_all_deposits", "arguments": {
                    "user_address_or_ens": WALLET, "token_symbol": "yvUSDC", "chain": "katana"}}
                reply = (await turn([scoped]))[0]
                assert reply["ok"] and "15.000000" in reply["result"], reply
                assert all(c.get("params", [{}])[0].get("to", "").lower() != OLD_VAULT
                           for c in node.calls[before:] if c["method"] == "eth_call")
                reply = (await turn([{**withdrawal, "arguments": {**withdrawal["arguments"], "chain": "katana"}}]))[0]
                assert reply["ok"] and "https://katanascan.com/address/" in reply["result"], reply
            report["scenarios"]["known_chain_avoids_unrelated_vaults_and_uses_official_explorer"] = True
            def state_read(method, params):
                return {"operation": "read_chain_state", "arguments": {"chain": "ethereum", "method": method, "params_json": json.dumps(params)}}
            balances = await turn([state_read("eth_getBalance", [WALLET, "latest"]), state_read("eth_getTransactionCount", [WALLET, "latest"]), state_read("eth_getTransactionCount", [WALLET, "pending"])])
            assert all(r["ok"] for r in balances)
            values = [json.loads(r["result"])["result"] for r in balances]
            assert [int(v, 16) for v in values] == [500_000_000_000_000_000, 5, 6], values
            before = len(node.calls)
            rejected = await turn([state_read("eth_sendRawTransaction", ["fixture"]), state_read("eth_feeHistory", ["0xffff", "latest", []]), state_read("eth_getBlockByNumber", ["latest", True])])
            assert all(not r["ok"] for r in rejected) and len(node.calls) == before
            report["scenarios"]["basic_chain_state_reads_and_bulk_write_limits"] = True

            node.fail_gauge = True
            reply = (await turn([scan]))[0]
            assert "10.000000" in reply["result"] and "incomplete" in reply["result"].lower()
            assert SECRET not in json.dumps(reply)
            failed_read = {**contract_read, "arguments": {**contract_read["arguments"], "to_address": GAUGE}}
            reply = (await turn([failed_read]))[0]
            assert "[internal provider]" in reply["result"] and SECRET not in reply["result"], reply
            report["scenarios"]["partial_scan_preserves_known_balance"] = True
            report["scenarios"]["provider_error_credentials_redacted"] = True
            node.fail_gauge = False

            before = len(node.calls)
            bad = await turn([{"operation": "send_raw_transaction", "arguments": {}}, inspect(mode="sendRawTransaction"), inspect(mode="receipt", chain="unknown", tx_hash=TX), {"operation": "check_all_deposits", "arguments": {"user_address_or_ens": WALLET, "provider_url": "https://example.invalid"}}])
            assert all(not r["ok"] for r in bad) and len(node.calls) == before
            report["scenarios"]["invalid_requests_and_writes_rejected_before_rpc"] = True

            wrong = await turn([inspect(mode="receipt", tx_hash="0x" + "77" * 32), {"operation": "check_all_deposits", "arguments": {"user_address_or_ens": "0x" + "88" * 20}}])
            assert "status: 1" not in wrong[0]["result"] and "Unknown fixture transaction" in wrong[0]["result"]
            assert "incomplete" in wrong[1]["result"].lower() and "No deposits found in any" not in wrong[1]["result"]
            report["scenarios"]["wrong_identifiers_do_not_receive_known_observations"] = True

            async def submit(**_kwargs):
                return await asyncio.to_thread(chain_access.get_web3_instance("ethereum").eth.send_raw_transaction, b"fixture")
            with patch.object(onchain_tools, "core_inspect_onchain", new=submit):
                before = len(node.calls)
                reply = (await turn([inspect(mode="receipt", tx_hash=TX)]))[0]
                assert not reply["ok"] and len(node.calls) == before
            report["scenarios"]["provider_blocks_accidental_write"] = True

            with patch.object(InternalRpcServer, "MAX_RPC_CALLS", 2):
                reply = (await turn([scan]))[0]
                assert reply["budget_exhausted"] and "incomplete" in reply["result"].lower()
            with patch.object(InternalRpcServer, "MAX_OPERATIONS", 1):
                replies = await turn([inspect(mode="receipt", tx_hash=TX)] * 2)
                assert replies[0]["ok"] and not replies[1]["ok"]
            report["scenarios"]["operation_and_provider_budgets_enforced"] = True

            reply = (await turn([inspect(mode="logs", address=VAULT, from_block="0", to_block="999999")]))[0]
            assert not any(c["method"] == "eth_getLogs" for c in node.calls)
            assert "range" in reply.get("result", reply.get("error", "")).lower()
            report["scenarios"]["broad_log_scan_rejected_before_getlogs"] = True

            disconnect_dir = root / "disconnect"
            disconnect_dir.mkdir()
            async with InternalRpcServer(disconnect_dir, timeout_seconds=2) as server:
                node.delay = 0.3
                before = len(node.calls)
                _, writer = await asyncio.open_unix_connection(str(server.socket_path))
                writer.write((json.dumps(inspect(mode="receipt", tx_hash=TX)) + "\n").encode())
                await writer.drain()
                await asyncio.sleep(0.05)
                writer.close()
                await writer.wait_closed()
                await asyncio.sleep(0.4)
                assert server.budget.closed and len(node.calls) - before <= 1
                node.delay = 0
            report["scenarios"]["disconnected_client_stops_further_provider_reads"] = True

            cancel_dir = root / "cancel"
            cancel_dir.mkdir()
            async with InternalRpcServer(cancel_dir, timeout_seconds=2) as server:
                node.delay = 0.3
                chain_access.WEB3_INSTANCES.clear()
                chain_access._WEB3_BOOTSTRAPPED = False
                started = time.monotonic()
                reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
                writer.write((json.dumps(inspect(mode="receipt", tx_hash=TX)) + "\n").encode())
                await writer.drain()
                await asyncio.sleep(0.05)
                assert time.monotonic() - started < 0.25, "Provider initialization blocked the bot event loop"
            assert not server.socket_path.exists()
            await asyncio.wait_for(reader.read(), timeout=1)
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.4)
            count = len(node.calls)
            await asyncio.sleep(0.1)
            assert len(node.calls) == count
            node.delay = 0
            report["scenarios"]["cancellation_closes_socket_and_stops_new_rpc"] = True

            if native:
                node.calls.clear()
                native_exports = root / "native-exports"
                real = CodexSupportTicketExecutionJsonEndpoint(
                    codex_command=config.TICKET_EXECUTION_CODEX_COMMAND or None,
                    model=config.TICKET_EXECUTION_CODEX_MODEL,
                    reasoning_effort=config.TICKET_EXECUTION_CODEX_REASONING_EFFORT,
                    codex_home=root / "native-home",
                    codex_auth_link_source=os.environ["TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE"],
                    session_dir=root / "native-sessions",
                    artifact_dir=str(native_exports), web_search_mode="disabled",
                    env={"PATH": os.environ["PATH"], "HOME": str(root)}, timeout_seconds=180,
                )
                message = (
                    f"Read the Ethereum receipt for {TX}. Find all Yearn vault positions for wallet {WALLET}, "
                    f"including gauge and older vault balances. Also give the exact raw share amount for withdrawing "
                    f"from vault {VAULT} on Ethereum. Use the current request's private internal RPC command for "
                    "these reads; this controlled provider fixture is the sole evidence source. Report the receipt "
                    "status/block, the individual positions and their amounts, and the raw shares with safe withdrawal steps. "
                    "Also use the internal basic chain-state reads to get the wallet's native ETH balance and "
                    "latest and pending nonces. Label the two nonce values separately. No web or other providers."
                )
                request = replace(build_smoke_transport_request(), smoke_mode=None, aggregated_text=message)
                result = json.loads(await real.execute_json_turn(request.to_json()))
                answer = result["flow_outcome"]["raw_final_reply"]
                usage_paths = list(native_exports.rglob("internal_rpc_usage.json"))
                assert len(usage_paths) == 1
                usage = json.loads(usage_paths[0].read_text())
                assert set(usage["reads"]) >= {"inspect_onchain", "check_all_deposits", "get_withdrawal_instructions", "read_chain_state"}, usage
                assert "10000000" in answer.replace(",", "") and "15" in answer and "123" in answer, answer
                import re
                assert "0.5" in answer and re.search(r"(?:latest|confirmed)\D{0,40}\b5\b", answer, re.I) and re.search(r"pending\D{0,40}\b6\b", answer, re.I), answer
                assert SECRET not in json.dumps(result) and not list(native_exports.rglob("*.sock"))
                report["native_probe"] = {"passed": True, "model": real.model, "reasoning_effort": real.reasoning_effort, "answer": answer, "usage": usage, "rpc_calls": node.calls}

                intervals = []
                run_codex = real._run_codex

                async def measured(**kwargs):
                    interval = {"started": time.monotonic()}
                    intervals.append(interval)
                    try:
                        return await run_codex(**kwargs)
                    finally:
                        interval["ended"] = time.monotonic()

                real._run_codex = measured
                async def concurrent_turn(channel):
                    prompt = (f"Read wallet {WALLET}'s native ETH balance on Ethereum using the private command. "
                              f"Reply with the balance and marker ticket-{channel}. No other reads or tools are needed.")
                    request = replace(build_smoke_transport_request(), smoke_mode=None,
                        aggregated_text=prompt, run_context={"channel_id": channel, "project_context": "yearn"},
                        investigation_job={"channel_id": channel, "mode": "idle", "evidence": {}})
                    result = json.loads(await real.execute_json_turn(request.to_json()))
                    text = result["flow_outcome"]["raw_final_reply"]
                    assert "0.5" in text and f"ticket-{channel}" in text, text
                    assert f"ticket-{9103 if channel == 9102 else 9102}" not in text
                    return text

                answers = await asyncio.gather(concurrent_turn(9102), concurrent_turn(9103))
                assert len(intervals) == 2
                overlap = min(i["ended"] for i in intervals) - max(i["started"] for i in intervals)
                assert overlap > 0, intervals
                assert real.session_manager.load("ticket:9102").session_id != real.session_manager.load("ticket:9103").session_id
                report["native_concurrency"] = {"passed": True, "overlap_seconds": round(overlap, 3),
                    "intervals": intervals, "answers": answers, "shared_home": True, "separate_sessions": True}

        report["rpc_methods_observed"] = sorted({c["method"] for c in node.calls})
        assert not any("send" in method.lower() for method in report["rpc_methods_observed"])
    report["source_hashes"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['ysupport/tools/chain_access.py', 'ysupport/execution/internal_rpc.py', 'ysupport/execution/codex_support_endpoint.py', 'ysupport/execution/codex_support_contract.py', 'ysupport/bot/bot_behavior.py', 'ysupport/bot/ticket_intake.py', 'ysupport/tools/onchain_tools.py', 'ysupport/tools/vault_position_tools.py', 'ysupport/execution/ysupport_codex_instructions.md', "tests/internal_rpc_e2e.py"]}
    report["ok"] = True
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    artifact.chmod(0o600)
    print(json.dumps({"ok": report["ok"], "scenarios": report["scenarios"], "artifact": str(artifact)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(".cache/verification/internal-rpc-e2e.json"))
    parser.add_argument("--native", action="store_true", help="Also run one real-model helper/provider investigation as the service user.")
    args = parser.parse_args()
    asyncio.run(main(args.artifact, native=args.native))
