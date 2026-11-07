"""Real native stdio -> private socket -> helpers -> HTTP RPC evidence regressions.

Provider observations are frozen; no answer is injected. Run with --artifact to
retain source hashes, scenarios and the actual bounded RPC calls.
"""

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import tests  # noqa: F401

os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from eth_abi import encode, decode  # noqa: E402
from web3 import Web3  # noqa: E402
from tests.internal_rpc_e2e import RpcNode, VAULT, WALLET, TX, SECRET  # noqa: E402
from ysupport import config  # noqa: E402
from ysupport.tools import chain_access, vault_position_tools, kong_vaults  # noqa: E402
from ysupport.execution.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)  # noqa: E402
from ysupport.execution.transport import build_smoke_transport_request  # noqa: E402

ALPHA = "0x" + "ab" * 20
BURN_TX = "0x" + "77" * 32
ZERO = "0x" + "00" * 20
TRANSFER = "0x" + Web3.keccak(
    text="Transfer(address,address,uint256)"
).hex().removeprefix("0x")


def selector(sig):
    return Web3.keccak(text=sig).hex().removeprefix("0x")[:8]


def encoded(types, values):
    return "0x" + encode(types, values).hex()


class Observations:
    def __init__(self):
        self.mode = "liquid"

    def reply(self, method, params):
        if method == "graphql":
            assert params["variables"] == {"chain": 1, "address": VAULT}
            rows = []
            if self.mode == "indexed":
                rows = [
                    {
                        "chainId": 1,
                        "address": VAULT,
                        "sender": WALLET,
                        "receiver": ZERO,
                        "transactionHash": BURN_TX,
                        "blockNumber": "123",
                    },
                    {
                        "chainId": 747474,
                        "address": VAULT,
                        "sender": WALLET,
                        "receiver": ZERO,
                        "transactionHash": TX,
                        "blockNumber": "122",
                    },
                ]
            return {"data": {"transfers": rows}}, None
        if method == "eth_getBlockByNumber":
            number = int(params[0], 16)
            return {
                "number": hex(number),
                "timestamp": hex(1_700_000_000 + number),
                "hash": "0x" + "66" * 32,
                "transactions": [],
            }, None
        if method == "eth_getTransactionReceipt" and params[0].lower() in (TX, BURN_TX):
            burn = params[0].lower() == BURN_TX
            sender, receiver = (WALLET, ZERO) if burn else ("0x" + "55" * 20, WALLET)
            log = {
                "address": VAULT,
                "topics": [
                    TRANSFER,
                    "0x" + "0" * 24 + sender[2:],
                    "0x" + "0" * 24 + receiver[2:],
                ],
                "data": encoded(["uint256"], [10_000_000]),
                "blockNumber": "0x7b",
                "blockHash": "0x" + "66" * 32,
                "transactionHash": params[0],
                "transactionIndex": "0x0",
                "logIndex": "0x1",
                "removed": False,
            }
            receipt = {
                "transactionHash": params[0],
                "transactionIndex": "0x0",
                "blockHash": "0x" + "66" * 32,
                "blockNumber": "0x7b",
                "from": sender,
                "to": VAULT,
                "gasUsed": "0x5208",
                "cumulativeGasUsed": "0x5208",
                "logs": [log],
                "status": "0x1",
                "contractAddress": None,
                "logsBloom": "0x" + "00" * 256,
                "type": "0x2",
                "effectiveGasPrice": "0x1",
            }
            if burn:
                payout = 0 if self.mode == "zero_payout" else 12_000_000
                withdrawal = {
                    **log,
                    "logIndex": "0x2",
                    "topics": [
                        "0x"
                        + Web3.keccak(
                            text="Withdraw(address,address,address,uint256,uint256)"
                        )
                        .hex()
                        .removeprefix("0x"),
                        "0x" + "0" * 24 + WALLET[2:],
                        "0x" + "0" * 24 + WALLET[2:],
                        "0x" + "0" * 24 + WALLET[2:],
                    ],
                    "data": encoded(["uint256", "uint256"], [payout, 10_000_000]),
                }
                receipt["logs"].append(withdrawal)
                if payout:
                    receipt["logs"].append(
                        {
                            **log,
                            "address": "0x" + "66" * 20,
                            "logIndex": "0x3",
                            "topics": [
                                TRANSFER,
                                "0x" + "0" * 24 + VAULT[2:],
                                "0x" + "0" * 24 + WALLET[2:],
                            ],
                            "data": encoded(["uint256"], [payout]),
                        }
                    )
            if self.mode == "wrong_receipt":
                receipt["transactionHash"] = BURN_TX
            if self.mode == "history_failure" and burn:
                return None, {"code": -32000, "message": "archive receipt unavailable"}
            return receipt, None
        if method == "eth_getLogs":
            # The test node refuses broad history even if the client tries it.
            q = params[0]
            assert int(q["toBlock"], 16) - int(q["fromBlock"], 16) < 5000
            assert (
                [a.lower() for a in q["address"]] == [VAULT]
                if isinstance(q["address"], list)
                else q["address"].lower() == VAULT
            )
            if self.mode.startswith("range_"):
                direction = "incoming" if q["topics"][1] is None else "outgoing"
                if self.mode in (
                    "range_both_failure",
                    "range_" + direction + "_failure",
                ):
                    return None, {"code": -32000, "message": direction + " unavailable"}
                receipt, _ = self.reply(
                    "eth_getTransactionReceipt",
                    [TX if direction == "incoming" else BURN_TX],
                )
                return [receipt["logs"][0]], None
            return [], None
        if method != "eth_call":
            return None
        call = params[0]
        data = call.get("data", "")
        sig = data[2:10]
        balance = 0 if self.mode == "zero" else 10_000_000
        capacity = (
            0
            if self.mode == "illiquid"
            else 5_000_000
            if self.mode == "partial"
            else balance
        )
        values = {
            "balanceOf(address)": balance,
            "maxRedeem(address,uint256)": capacity,
            "maxWithdraw(address,uint256)": capacity * 6 // 5,
            "decimals()": 6,
        }
        for signature, value in values.items():
            if sig == selector(signature):
                if self.mode == "read_failure" and signature.startswith("balanceOf"):
                    return None, {"code": -32000, "message": "unavailable /" + SECRET}
                return encoded(["uint256"], [value]), None
        if sig == selector("maxWithdraw(address)"):
            return encoded(["uint256"], [123]), None
        if sig == selector("previewRedeem(uint256)"):
            shares = decode(["uint256"], bytes.fromhex(data[10:]))[0]
            return encoded(["uint256"], [shares * 6 // 5]), None
        if sig == selector("redeem(uint256,address,address,uint256)"):
            shares, receiver, owner, loss = decode(
                ["uint256", "address", "address", "uint256"], bytes.fromhex(data[10:])
            )
            assert (
                loss == 0
                and call.get("from", "").lower() == WALLET
                and receiver.lower() == owner.lower() == WALLET
            )
            if self.mode == "revert":
                return None, {"code": 3, "message": "execution reverted"}
            return encoded(
                ["uint256"], [shares * 6 // 5 - (1 if self.mode == "loss" else 0)]
            ), None
        if sig == selector("echo(string)"):
            value = decode(["string"], bytes.fromhex(data[10:]))[0]
            return encoded(["string"], [value]), None
        if sig == selector("nested(address[],(address,uint256))"):
            values = decode(
                ["address[]", "(address,uint256)"], bytes.fromhex(data[10:])
            )
            assert values[0][0].lower() == ALPHA and values[1][0].lower() == ALPHA
            return encoded(["uint256"], [45]), None
        return None, {"code": -32601, "message": "Unknown fixture function"}


async def run(artifact):
    root_path = Path(__file__).resolve().parents[1]
    report = {
        "ok": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": {},
    }
    with (
        tempfile.TemporaryDirectory(prefix="ysupport-evidence-e2e-") as tmp,
        RpcNode() as node,
    ):
        root = Path(tmp)
        observations = Observations()
        node.override = observations.reply
        model = root / "model.py"
        model.write_text("""import json,os,subprocess,sys
from pathlib import Path
request=json.loads((Path(os.environ['TICKET_EXECUTION_RUN_DIR'])/'support_request.json').read_text())
assert not any(k.endswith('_RPC_URL') or k=='ALCHEMY_KEY' for k in os.environ)
values=[]
for query in json.loads(request['current_user_message']):
 p=subprocess.run(request['support_state']['internal_rpc']['command'],input=json.dumps(query),text=True,capture_output=True,timeout=20)
 values.append(json.loads(p.stdout))
answer={'answer':json.dumps(values),'requires_human_handoff':False,'handoff_kind':None,'handoff_reason':None,'evidence_summary':'Frozen provider observations over real RPC','used_tools':['shell']}
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer)}}))
""")
        vault = {
            "origin": "yearn",
            "chainId": 1,
            "address": VAULT,
            "apiVersion": "3.0.0",
            "name": "Fixture USDC",
            "symbol": "yvUSDC",
            "decimals": 6,
            "asset": {"symbol": "USDC", "decimals": 6},
        }
        url = f"http://127.0.0.1:{node.server.server_port}/{SECRET}"
        with (
            patch.dict(os.environ, {"ALCHEMY_KEY": SECRET, "ETHEREUM_RPC_URL": url}),
            patch.object(config, "RPC_URLS", {"ethereum": url}),
            patch.object(
                kong_vaults,
                "KONG_BASE_URL",
                f"http://127.0.0.1:{node.server.server_port}",
            ),
            patch.object(chain_access, "WEB3_INSTANCES", {}),
            patch.object(chain_access, "_WEB3_BOOTSTRAPPED", False),
            patch.object(
                vault_position_tools,
                "fetch_kong_vault_snapshot",
                new=AsyncMock(return_value=vault),
            ),
        ):
            endpoint = CodexSupportTicketExecutionJsonEndpoint(
                codex_command=[sys.executable, str(model)],
                codex_home=root / "home",
                artifact_dir=str(root / "exports"),
                env={"PATH": os.environ["PATH"], "HOME": str(root)},
                timeout_seconds=30,
            )

            async def turn(query):
                request = replace(
                    build_smoke_transport_request(),
                    smoke_mode=None,
                    aggregated_text=json.dumps([query]),
                )
                result = json.loads(await endpoint.execute_json_turn(request.to_json()))
                assert SECRET not in json.dumps(result)
                return json.loads(result["flow_outcome"]["raw_final_reply"])[0]

            async def check(name, func):
                try:
                    report["scenarios"][name] = {"ok": True, **await func()}
                except Exception as exc:
                    report["scenarios"][name] = {"ok": False, "error": str(exc)}

            async def withdrawal(mode):
                observations.mode = mode
                vault["apiVersion"] = "" if mode == "unknown_version" else "3.0.0"
                before = len(node.calls)
                reply = await turn(
                    {
                        "operation": "get_withdrawal_instructions",
                        "arguments": {
                            "chain": "ethereum",
                            "vault_address": VAULT,
                            "user_address_or_ens": WALLET,
                        },
                    }
                )
                text = reply.get("result", "")
                assert reply["ok"], reply
                ready = mode in ("liquid", "partial")
                assert ("Click the **'Write'** button" in text) == ready, text
                if ready:
                    assert (
                        "max_loss" in text
                        and "0" in text
                        and ("5000000" if mode == "partial" else "10000000") in text
                    ), text
                    calls = node.calls[before:]
                    sim = [
                        c
                        for c in calls
                        if c["method"] == "eth_call"
                        and c["params"][0].get("data", "")[2:10]
                        == selector("redeem(uint256,address,address,uint256)")
                    ]
                    assert len(sim) == 1, calls
                    blocks = {
                        c["params"][1] for c in calls if c["method"] == "eth_call"
                    }
                    assert len(blocks) == 1 and "latest" not in blocks, blocks
                return {"ready": ready, "rpc_calls": len(node.calls) - before}

            for mode in (
                "liquid",
                "partial",
                "zero",
                "illiquid",
                "read_failure",
                "revert",
                "loss",
                "unknown_version",
            ):
                await check("withdrawal_" + mode, lambda mode=mode: withdrawal(mode))

            async def interpreter_case():
                from ysupport.execution.json_endpoint import _codex_env
                import subprocess

                result = subprocess.run(
                    ["python", "-c", "import sys;print(sys.prefix)"],
                    env=_codex_env(),
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
                assert result.returncode == 0 and result.stdout.strip() == sys.prefix, (
                    result.stderr
                )
                return {"venv_interpreter_available": True}

            await check("native_venv_interpreter", interpreter_case)
            observations.mode = "liquid"

            async def args_case():
                replies = []
                for signature, args, outputs in [
                    ("maxWithdraw(address)", [ALPHA], ["uint256"]),
                    ("echo(string)", [ALPHA], ["string"]),
                ]:
                    reply = await turn(
                        {
                            "operation": "inspect_onchain",
                            "arguments": {
                                "chain": "ethereum",
                                "mode": "call",
                                "to_address": VAULT,
                                "function_signature": signature,
                                "args_json": json.dumps(args),
                                "output_types_json": json.dumps(outputs),
                            },
                        }
                    )
                    assert reply["ok"] and "Error inspecting" not in reply["result"], (
                        reply
                    )
                    replies.append(reply)
                abi = {
                    "type": "function",
                    "name": "nested",
                    "stateMutability": "view",
                    "inputs": [
                        {"name": "a", "type": "address[]"},
                        {
                            "name": "b",
                            "type": "tuple",
                            "components": [
                                {"name": "owner", "type": "address"},
                                {"name": "n", "type": "uint256"},
                            ],
                        },
                    ],
                    "outputs": [{"type": "uint256"}],
                }
                reply = await turn(
                    {
                        "operation": "inspect_onchain",
                        "arguments": {
                            "chain": "ethereum",
                            "mode": "call",
                            "to_address": VAULT,
                            "function_abi_json": json.dumps(abi),
                            "args_json": json.dumps([[ALPHA], [ALPHA, "7"]]),
                        },
                    }
                )
                assert reply["ok"] and "result: 45" in reply["result"], reply
                return {
                    "nested_and_scalar_addresses": True,
                    "strings_preserved": ALPHA in replies[1]["result"],
                }

            await check("abi_normalization", args_case)

            async def batch_case():
                before = len(node.calls)
                reply = await turn(
                    {
                        "operation": "read_contracts",
                        "arguments": {
                            "chain": "ethereum",
                            "tx_hash": TX,
                            "calls_json": json.dumps(
                                [
                                    {
                                        "to_address": VAULT,
                                        "function_signature": "balanceOf(address)",
                                        "args_json": json.dumps([WALLET]),
                                        "output_types_json": '["uint256"]',
                                    },
                                    {
                                        "to_address": VAULT,
                                        "function_signature": "maxWithdraw(address)",
                                        "args_json": json.dumps([ALPHA]),
                                        "output_types_json": '["uint256"]',
                                    },
                                ]
                            ),
                        },
                    }
                )
                assert reply["ok"], reply
                data = json.loads(reply["result"])
                assert (
                    data["block_number"] == 123 and data["scope"] == "transaction_block"
                ), data
                assert all(row["ok"] for row in data["results"]), data
                assert all(
                    c["params"][1] == "0x7b"
                    for c in node.calls[before:]
                    if c["method"] == "eth_call"
                )
                return {"snapshot_block": 123, "single_operation": True}

            await check("historical_batch", batch_case)

            async def history_case(mode):
                observations.mode = mode
                reply = await turn(
                    {
                        "operation": "get_wallet_history",
                        "arguments": {
                            "chain": "ethereum",
                            "wallet_address": WALLET,
                            "token_address": VAULT,
                            "tx_hashes_json": json.dumps([TX, BURN_TX]),
                        },
                    }
                )
                assert reply["ok"], reply
                data = json.loads(reply["result"])
                assert data["coverage"]["history_complete"] is False, data
                if mode == "history_failure":
                    assert not data["coverage"]["complete"] and data["errors"], data
                    assert data["next_evidence"]["handoff_required"] is False, data
                else:
                    assert {r["direction"] for r in data["transfers"]} == {
                        "in",
                        "out",
                    } and any(r["is_burn"] for r in data["transfers"]), data
                if mode != "history_failure":
                    assert data["withdrawals"][0]["assets_raw"] == (
                        "0" if mode == "zero_payout" else "12000000"
                    ), data
                    assert bool(data["asset_transfers_in"]) == (
                        mode != "zero_payout"
                    ), data
                return {"coverage": data["coverage"], "records": len(data["transfers"])}

            for mode in ("liquid", "history_failure", "zero_payout"):
                await check("history_" + mode, lambda mode=mode: history_case(mode))

            async def indexed_case(mode):
                observations.mode = mode
                before = len(node.calls)
                reply = await turn(
                    {
                        "operation": "get_wallet_history",
                        "arguments": {
                            "chain": "ethereum",
                            "wallet_address": WALLET,
                            "token_address": VAULT,
                        },
                    }
                )
                assert reply["ok"], reply
                value = json.loads(reply["result"])
                assert (
                    value["coverage"]["complete"] is False
                    and value["coverage"]["history_complete"] is False
                ), value
                receipts = [
                    c["params"][0]
                    for c in node.calls[before:]
                    if c.get("method") == "eth_getTransactionReceipt"
                ]
                assert receipts == ([BURN_TX] if mode == "indexed" else []), receipts
                return {"verified_receipts": len(receipts), "history_complete": False}

            for mode in ("indexed", "index_empty"):
                await check(mode, lambda mode=mode: indexed_case(mode))

            async def range_case():
                before = len(node.calls)
                reply = await turn(
                    {
                        "operation": "get_wallet_history",
                        "arguments": {
                            "chain": "ethereum",
                            "wallet_address": WALLET,
                            "token_address": VAULT,
                            "from_block": 120,
                            "to_block": 125,
                        },
                    }
                )
                assert reply["ok"], reply
                value = json.loads(reply["result"])
                assert (
                    value["coverage"]["complete"]
                    and not value["coverage"]["history_complete"]
                ), value
                logs = [
                    c for c in node.calls[before:] if c.get("method") == "eth_getLogs"
                ]
                assert len(logs) == 2 and all(
                    any(WALLET[2:] in str(t) for t in c["params"][0]["topics"])
                    for c in logs
                ), logs
                return {"wallet_filtered_queries": 2, "scoped_complete_only": True}

            await check("bounded_history_window", range_case)

            async def directional_history_case(mode):
                observations.mode = mode
                reply = await turn(
                    {
                        "operation": "get_wallet_history",
                        "arguments": {
                            "chain": "ethereum",
                            "wallet_address": WALLET,
                            "token_address": VAULT,
                            "from_block": 120,
                            "to_block": 125,
                        },
                    }
                )
                assert reply["ok"], reply
                value = json.loads(reply["result"])
                failed = {
                    direction
                    for direction in ("incoming", "outgoing")
                    if mode in ("range_both_failure", "range_" + direction + "_failure")
                }
                expected = (
                    {TX, BURN_TX}
                    - ({TX} if "incoming" in failed else set())
                    - ({BURN_TX} if "outgoing" in failed else set())
                )
                assert {
                    row["transaction_hash"] for row in value["transfers"]
                } == expected, value
                assert value["coverage"]["complete"] == (not failed), value
                assert not value["coverage"]["history_complete"], value
                assert {e["direction"] for e in value["errors"]} == failed, value
                return {
                    "verified_transfers": len(value["transfers"]),
                    "failed_directions": sorted(failed),
                }

            for mode in (
                "range_healthy",
                "range_incoming_failure",
                "range_outgoing_failure",
                "range_both_failure",
            ):
                await check(mode, lambda mode=mode: directional_history_case(mode))
            observations.mode = "liquid"

            async def batch_integer_case():
                reply = await turn(
                    {
                        "operation": "read_contracts",
                        "arguments": {
                            "chain": "ethereum",
                            "block_identifier": 123,
                            "calls_json": json.dumps(
                                [
                                    {
                                        "to_address": VAULT,
                                        "function_signature": "balanceOf(address)",
                                        "args_json": json.dumps([WALLET]),
                                        "output_types_json": '["uint256"]',
                                    },
                                    {
                                        "to_address": VAULT,
                                        "function_signature": "unknown()",
                                        "output_types_json": '["uint256"]',
                                    },
                                ]
                            ),
                        },
                    }
                )
                assert reply["ok"], reply
                value = json.loads(reply["result"])
                assert value["block_number"] == 123 and not value["complete"], value
                assert [row["ok"] for row in value["results"]] == [True, False], value
                return {"integer_block": 123, "partial_errors_explicit": True}

            await check("integer_block_and_partial_batch", batch_integer_case)

            async def history_limits_case():
                for start, end in [(5000, 5001), (True, 125), (0, 5000)]:
                    reply = await turn(
                        {
                            "operation": "get_wallet_history",
                            "arguments": {
                                "chain": "ethereum",
                                "wallet_address": WALLET,
                                "token_address": VAULT,
                                "from_block": start,
                                "to_block": end,
                            },
                        }
                    )
                    assert (
                        not reply["ok"]
                        or not json.loads(reply["result"])["coverage"]["complete"]
                    ), reply
                return {"future_invalid_and_oversized_ranges_not_complete": True}

            await check("history_bounds", history_limits_case)

            async def wrong_receipt_case():
                observations.mode = "wrong_receipt"
                reply = await turn(
                    {
                        "operation": "read_contracts",
                        "arguments": {
                            "chain": "ethereum",
                            "tx_hash": TX,
                            "calls_json": json.dumps(
                                [
                                    {
                                        "to_address": VAULT,
                                        "function_signature": "balanceOf(address)",
                                        "args_json": json.dumps([WALLET]),
                                        "output_types_json": '["uint256"]',
                                    }
                                ]
                            ),
                        },
                    }
                )
                assert not reply["ok"], reply
                return {"mismatched_receipt_rejected": True}

            await check("transaction_receipt_identity", wrong_receipt_case)

        report["rpc_calls"] = node.calls
    report["source_sha256"] = {
        str(p.relative_to(root_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            root_path / "tests/evidence_e2e.py",
            *(root_path / "ysupport/tools").glob("*.py"),
            root_path / "ysupport/execution/internal_rpc.py",
            root_path / "ysupport/execution/json_endpoint.py",
        ]
    }
    report["ok"] = all(v["ok"] for v in report["scenarios"].values())
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "scenarios": report["scenarios"],
                "artifact": str(artifact),
            },
            indent=2,
        )
    )
    return int(not report["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path, default=Path(".cache/verification/evidence-e2e.json")
    )
    raise SystemExit(asyncio.run(run(parser.parse_args().artifact)))
