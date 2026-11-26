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
from ysupport.bot.state import TicketInvestigationJob  # noqa: E402
from ysupport.tools import chain_access, vault_position_tools, kong_vaults  # noqa: E402
from ysupport.execution.codex_support_endpoint import (
    CodexSupportTicketExecutionJsonEndpoint,
)  # noqa: E402
from ysupport.execution.context import merge_explicit_evidence  # noqa: E402
from ysupport.execution.transport import build_smoke_transport_request, serialize_investigation_job  # noqa: E402

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
            burn = params[0].lower() == BURN_TX or self.mode == "two_payouts"
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
        if sig == selector("asset()"):
            return encoded(["address"], ["0x" + ("55" if self.mode == "wrong_payout_token" else "66") * 20]), None
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
                and owner.lower() == WALLET
                and receiver.lower() == (ALPHA if self.mode == "exact_redemption" else WALLET)
            )
            if self.mode == "revert":
                return None, {"code": 3, "message": "execution reverted"}
            adjustment = {"loss": -1, "capacity_mismatch": 1}.get(self.mode, 0)
            return encoded(
                ["uint256"], [shares * 6 // 5 + adjustment]
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
try:
 instruction=json.JSONDecoder().raw_decode(request['current_user_message'])[0]
except json.JSONDecodeError:
 instruction=json.loads(Path(__file__).with_name('fixture_instruction.json').read_text())
for query in (instruction['queries'] if isinstance(instruction,dict) else instruction):
 p=subprocess.run(request['support_state']['internal_rpc']['command'],input=json.dumps(query),text=True,capture_output=True,timeout=20)
 values.append(json.loads(p.stdout))
Path(__file__).with_name('observations.json').write_text(json.dumps(values))
answer={'answer':'Fixture reads complete.','requires_human_handoff':False,'handoff_kind':None,'handoff_reason':None,'evidence_summary':'Frozen provider observations over real RPC','used_tools':['shell']}
if isinstance(instruction,dict):
 refs=[item['ref'] for value in values for item in value.get('evidence',[])]
 answer.update(answer=instruction['answer'],evidence_refs=(None if instruction.get('uncited') else refs if instruction['cite'] else ['fabricated']))
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
                raw = (root / "observations.json").read_text()
                assert SECRET not in raw
                return json.loads(raw)[0]

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
                "capacity_mismatch",
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
                before = len(node.calls)
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
                assert not any(call["method"] == "eth_call" and call["params"][0]["data"][2:10] == selector("asset()")
                               for call in node.calls[before:]), "Ordinary wallet history looked up the payout asset"
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

            async def verified_claim(mode):
                observations.mode = {"partial_fact": "zero_payout", "zero_shares": "zero"}.get(mode, mode)
                vault["apiVersion"] = "3.0.0"
                redemption = mode in {"redeem_ready", "revert", "wrong_wallet", "uncited_recipe",
                                      "multiline_recipe", "multiline_parameters", "zero_shares"}
                query = {"operation": "get_withdrawal_instructions", "arguments": {
                    "chain": "ethereum", "vault_address": VAULT, "user_address_or_ens": WALLET,
                }} if redemption else {"operation": "get_wallet_history", "arguments": {
                    "chain": "ethereum", "wallet_address": WALLET, "token_address": VAULT,
                    "tx_hashes_json": json.dumps([BURN_TX]), "verify_payout": True,
                }}
                answer = {
                    "payout_paraphrase": "The withdrawal returned 999 USDC.",
                    "multiline_recipe": "Open Write Contract.\n`redeem(999, wallet, wallet, 0)`.",
                    "multiline_parameters": "Open Write Contract and select redeem.\nshares = 999\nreceiver = your wallet\nmax_loss = 0.",
                    "partial_fact": "I can't tell whether you received a payout. The receipt shows a share burn. Please share the token transfer hash.",
                }.get(mode, "You received 999 USDC from the withdrawal." if mode in {"uncited", "wrong_amount"}
                      else "Call redeem(999, owner, owner, 0)." if mode == "uncited_recipe"
                      else "")
                ordinary = {
                    "ordinary_docs": ("How does the withdrawal queue work?", "It determines which strategies supply liquidity."),
                    "withdrawal_button": ("Where is the withdrawal button?", "Open the vault page and select Withdraw."),
                    "ordinary_airdrop": ("Did I receive an airdrop?", "Which airdrop are you asking about?"),
                    "ordinary_liquidation": (
                        "Where are my funds after my Flex position was liquidated?",
                        "The liquidation transaction shows which collateral was seized.",
                    ),
                }.get(mode)
                queries = [] if ordinary or mode in {"payout_paraphrase", "multiline_recipe", "multiline_parameters"} else [query]
                if ordinary:
                    answer = ordinary[1]
                instruction = {"queries": queries, "cite": mode != "forged",
                    "uncited": bool(ordinary) or mode in {"uncited", "uncited_recipe", "payout_paraphrase",
                                                        "multiline_recipe", "multiline_parameters"},
                    "answer": answer}
                request = replace(build_smoke_transport_request(), smoke_mode=None, aggregated_text=json.dumps(instruction))
                if mode == "payout_paraphrase":
                    request.aggregated_text += "\nWhat happened with my withdrawal?"
                elif mode == "partial_fact":
                    request.aggregated_text += "\nDid my withdrawal pay out?"
                elif mode == "redeem_ready":
                    request.aggregated_text += "\nRedeem all shares."
                elif mode == "corrected_hash":
                    request.aggregated_text += f"\nCorrection: use {TX} instead."
                elif mode in {"remembered_correction", "durable_correction"}:
                    (root / "fixture_instruction.json").write_text(json.dumps(instruction))
                    request.aggregated_text = "Did the withdrawal go through?"
                    request.current_history = [
                        {"role": "user", "content": f"Check {BURN_TX}."},
                        {"role": "user", "content": f"Correction: use {TX} instead."},
                        {"role": "assistant", "content": "I'll check the corrected hash."},
                    ]
                    request.investigation_job["evidence"]["tx_hashes"] = [BURN_TX, TX]
                    if mode == "durable_correction":
                        job = TicketInvestigationJob(request.run_context["channel_id"])
                        for item in request.current_history[:2]:
                            merge_explicit_evidence(job, item["content"])
                        request.investigation_job = serialize_investigation_job(job)
                        assert job.evidence.tx_hashes == [BURN_TX, TX]
                        request.current_history.extend(
                            {"role": "user", "content": f"Follow-up {i}."} for i in range(12)
                        )
                elif ordinary:
                    request.aggregated_text += "\n" + ordinary[0]
                if mode == "wrong_wallet":
                    request.investigation_job["evidence"]["wallet"] = ALPHA
                outcome = json.loads(await endpoint.execute_json_turn(request.to_json()))["flow_outcome"]
                answer = outcome["raw_final_reply"]
                proofs = outcome.get("verified_evidence", [])
                assert "999" not in answer, outcome
                if mode in {"liquid", "redeem_ready", "wrong_amount"}:
                    block = 4096 if redemption else 123
                    assert proofs and "12000000" in answer and f"block {block}" in answer, outcome
                    assert (BURN_TX in answer) != redemption, outcome
                    if redemption:
                        assert "max_loss=0" in answer and "10000000" in answer, outcome
                elif ordinary:
                    assert answer == ordinary[1] and not proofs
                elif mode == "partial_fact":
                    assert "burn" in answer.lower() and "couldn't verify" in answer.lower(), outcome
                    assert "transfer hash" in answer.lower() and "12000000" not in answer, outcome
                elif mode == "zero_shares":
                    assert "zero vault shares" in answer and "block 4096" in answer, outcome
                    assert proofs and "past holdings" in answer and "max_loss=" not in answer, outcome
                else:
                    assert all("share burn" in proof for proof in proofs) and "couldn't verify" in answer.lower(), outcome
                    assert "12000000" not in answer and "max_loss=0" not in answer, outcome
                return {"verified": proofs, "answer": answer}

            for mode in ("liquid", "zero_payout", "wrong_payout_token", "history_failure", "forged", "redeem_ready", "zero_shares",
                         "revert", "wrong_wallet", "wrong_amount", "uncited", "uncited_recipe",
                         "payout_paraphrase", "multiline_recipe", "multiline_parameters", "partial_fact",
                         "corrected_hash", "remembered_correction", "durable_correction", "ordinary_docs",
                         "withdrawal_button", "ordinary_airdrop", "ordinary_liquidation"):
                await check("claim_" + mode, lambda mode=mode: verified_claim(mode))

            async def claim_scope(mode):
                observations.mode = {
                    "failed_receipt": "history_failure", "implicit_receipt": "history_failure",
                    "two_receipts": "two_payouts", "two_wording": "two_payouts", "migration_burn": "zero_payout",
                    "restated_second": "two_payouts", "exact_redemption": "exact_redemption",
                    "return_to_owner": "exact_redemption",
                    "switch_to_full_balance": "exact_redemption",
                    "old_vault_recipient": "exact_redemption", "same_vault_exact": "exact_redemption",
                    "negated_recipient": "exact_redemption", "prior_full_overrides_exact": "exact_redemption",
                    "asset_followup": "exact_redemption", "new_owner_old_recipient": "exact_redemption",
                    "raw_amount_overridden": "exact_redemption", "recipient_wallet_label": "exact_redemption",
                    "restated_failed_receipt": "history_failure", "repeated_failed_receipt": "history_failure",
                }.get(mode, "liquid")
                compare_receipts = mode in {"two_receipts", "partial_receipts", "two_wording", "restated_second"}
                hashes = [BURN_TX, TX] if compare_receipts else [BURN_TX]
                history = {"operation": "get_wallet_history", "arguments": {
                    "chain": "ethereum", "wallet_address": WALLET, "token_address": VAULT,
                    "tx_hashes_json": json.dumps(hashes),
                }}
                chain_correction = mode in {"chain_only_correction", "stale_tag_after_chain_correction"}
                if compare_receipts or chain_correction or mode in {"matching_wallet", "explicit_wallet", "remembered_wallet"}:
                    history["arguments"]["verify_payout"] = True
                redemption = {"operation": "get_withdrawal_instructions", "arguments": {
                    "chain": "ethereum", "vault_address": VAULT,
                }}
                simulate = {**redemption, "arguments": {**redemption["arguments"], "user_address_or_ens": WALLET}}
                exact = {**simulate, "arguments": {**simulate["arguments"], "shares_raw": "100", "receiver_address": ALPHA}}
                third_party = {**simulate, "arguments": {**simulate["arguments"], "receiver_address": ALPHA}}
                other_vault = "0x" + "55" * 20
                vault_url = f"https://yearn.fi/vaults/1/{VAULT}"
                exact_question = (
                    f"How can I redeem only 100 raw shares from https://yearn.fi/vaults/1/{VAULT} "
                    f"with wallet {WALLET}, to recipient {ALPHA}?"
                )
                question, draft, queries, cite = {
                    "read_only_share_balance": (f"How many shares do I hold in {vault_url} with wallet {WALLET}? Only show the balance.",
                        "Your wallet holds 10,000,000 raw vault shares.", [simulate], False),
                    "additive_deposit_correction": (f"Correction: it was Base. Here is the second deposit transaction {TX}. Check both.",
                        "I'll compare both deposit transactions on Base.", [], False),
                    "remembered_wallet_after_correction": (f"Now redeem all from {vault_url}.", "", [simulate], True),
                    "confirmed_bare_wallet": (f"Now redeem all from {vault_url}.", "", [simulate], True),
                    "recipient_as_wallet_owner": (f"Redeem all from {vault_url} to wallet {WALLET}.", "", [simulate], True),
                    "recipient_wallet_label": (f"Redeem all from {vault_url} with wallet {WALLET} to wallet {ALPHA}.",
                        "", [third_party], True),
                    "raw_amount_overridden": (exact_question + "\nWithdraw only 75 USDC.", "", [exact], True),
                    "named_withdrawal_arguments": (f"How do I withdraw all from the V3 vault {vault_url}?",
                        f"On the explorer's Write Contract tab, open `withdraw`. Enter `assets=75`, "
                        f"`receiver={WALLET}`, `owner={WALLET}`, and `max_loss=0`, then click Write.", [], False),
                    "withdrawn_funds_destination": (f"I withdrew in {BURN_TX}. Where did the funds go?",
                        "The funds reached your wallet.", [{"operation": "get_receipt", "arguments": {
                            "chain": "ethereum", "tx_hash": BURN_TX,
                        }}], False),
                    "negated_recipient": (f"Redeem all from {vault_url} with wallet {WALLET}. Do not send the proceeds to {ALPHA}.",
                        "", [third_party], True),
                    "prior_full_overrides_exact": ("Can you verify that now?", "", [exact], True),
                    "asset_followup": ("Can I take out 75 USDC instead?", "", [exact], True),
                    "corrected_vault_in_batch": (f"Redeem all from {vault_url} with wallet {WALLET}.\n"
                        f"Correction: use https://yearn.fi/vaults/8453/{other_vault}.", "", [simulate], True),
                    "confirmed_vault_in_batch": (f"Redeem all from https://yearn.fi/vaults/8453/{other_vault}.\n"
                        f"Correction: redeem all from {vault_url} with wallet {WALLET}.", "", [simulate], True),
                    "corrected_amount_in_batch": (f"Redeem all from {vault_url} with wallet {WALLET}.\n"
                        "Actually, can I take out 75 USDC instead?", "", [simulate], True),
                    "corrected_wallet_stale_proof": (
                        f"Redeem all from {vault_url} with wallet {WALLET}.\n"
                        f"Correction: redeem all from {vault_url} with wallet {ALPHA}.", "", [simulate], True),
                    "new_owner_old_recipient": (f"Redeem all from {vault_url} with wallet {WALLET}.",
                        "", [third_party], True),
                    "new_owner_own_recipient": (f"Redeem all from {vault_url} with wallet {WALLET}.",
                        "", [simulate], True),
                    "contradictory_amounts_in_batch": (
                        f"Redeem all from {vault_url} with wallet {WALLET}.\nWithdraw only 75 USDC.", "", [simulate], True),
                    "multiline_withdraw": (f"Can I withdraw all from this V3 vault {vault_url}?",
                        f"On the explorer, write this call:\nwithdraw(\n  999,\n  {WALLET},\n  {WALLET},\n  0\n)", [], False),
                    "withdrawal_hash_plain_confirmation": (f"My withdrawal is {BURN_TX}. Can you confirm the tokens were sent?",
                        "Yes, the USDC reached your wallet.", [{"operation": "get_receipt", "arguments": {
                            "chain": "ethereum", "tx_hash": BURN_TX,
                        }}], False),
                    "unguarded_withdraw": (f"Can I withdraw all from this V3 vault {vault_url}?",
                        f"Use withdraw(999, {WALLET}, {WALLET}, 0) on the explorer.", [], False),
                    "unguarded_redeem": (f"Can I redeem all from this V3 vault {vault_url}?",
                        "Call the four-argument redeem function with your maxRedeem balance, your wallet as owner and receiver, and zero max loss.",
                        [], False),
                    "unguarded_payout": (f"Did the USDC reach my wallet after I withdrew? Transaction: {BURN_TX}.",
                        "Yes, the USDC arrived in your wallet.", [{"operation": "get_receipt", "arguments": {
                            "chain": "ethereum", "tx_hash": BURN_TX,
                        }}], False),
                    "general_withdrawal_asset": ("Which asset does a Yearn vault pay when you withdraw?",
                        "A withdrawal is paid in the vault's underlying token.", [], False),
                    "failed_receipt": (f"Did the tokens reach my wallet? Withdrawal transaction {BURN_TX}.",
                        "Yes. The USDC reached your wallet.", [history], False),
                    "prose_recipe": (f"How do I redeem from {VAULT}?",
                        "Open Write Contract and select the four-input redeem. Set _shares to 999 and _maxLoss to 0.", [], False),
                    "run_recipe": (f"How do I pull my shares out of https://yearn.fi/vaults/1/{VAULT}?",
                        "Run the four-argument redeem function with 999 shares, your address for both accounts, and zero as the final argument.",
                        [], False),
                    "implicit_receipt": ("Did the coins land in my account?",
                        "Yes, the USDC arrived in your wallet.", [history], False),
                    "restated_failed_receipt": (f"Did the coins land in my account? Tx: {BURN_TX}.",
                        "Yes, the USDC arrived in your wallet.", [history], False),
                    "repeated_failed_receipt": (f"Did the coins land in my account? Tx: {BURN_TX}.",
                        "Yes, the USDC arrived in your wallet.", [history], False),
                    "wrong_vault": (f"How can I redeem all shares from https://yearn.fi/vaults/1/{ALPHA} with wallet {WALLET}?",
                        "", [simulate], True),
                    "wrong_chain": (f"How can I redeem all shares from https://yearn.fi/vaults/8453/{VAULT} with wallet {WALLET}?",
                        "", [simulate], True),
                    "linked_vault": (f"How can I redeem all shares from {vault_url} with wallet {WALLET}?",
                        "", [simulate], True),
                    "unspecified_redemption_amount": (f"How can I redeem from {vault_url} with wallet {WALLET}?",
                        "", [simulate], True),
                    "spelled_amount": (f"Can I redeem fifty shares from {vault_url} with wallet {WALLET}?",
                        "", [simulate], True),
                    "asset_amount": (f"Can I take 100 USDC out of {vault_url}? My wallet is {WALLET}.",
                        "", [simulate], True),
                    "restricted_full_amount": (f"Redeem all except fifty shares from {vault_url}, wallet {WALLET}.",
                        "", [simulate], True),
                    "alternative_full_amount": (f"Should I redeem all shares or only half from {vault_url}, wallet {WALLET}?",
                        "", [simulate], True),
                    "old_vault_recipient": (f"New question: how do I withdraw all from {vault_url} with wallet {WALLET}?",
                        "", [third_party], True),
                    "new_vault_owner": (f"New question: how do I withdraw all from {vault_url} with wallet {WALLET}?",
                        "", [simulate], True),
                    "partial_target_override": (f"Use vault {VAULT} on the same network. Can I redeem all shares now?",
                        "", [simulate], True),
                    "same_vault_exact": ("Can you verify that same redemption now?", "", [exact], True),
                    "wrong_redemption_parameters": (exact_question, "", [simulate], True),
                    "exact_redemption": (exact_question, "", [exact], True),
                    "return_to_owner": ("Actually, send it to my own wallet.", "", [exact], True),
                    "switch_to_full_balance": ("Actually, withdraw all.", "", [exact], True),
                    "ambiguous_redemption_amount": (f"How can I redeem half my shares from https://yearn.fi/vaults/1/{VAULT}?",
                        "", [simulate], True),
                    "linked_vault_followup": ("Can I redeem all shares from that vault now?", "", [simulate], True),
                    "explicit_wallet": (f"Did {BURN_TX} pay this wallet {ALPHA}?",
                        "", [history], True),
                    "matching_wallet": (f"Did {BURN_TX} pay this wallet {WALLET}?",
                        "", [history], True),
                    "chain_only_correction": ("Correction, chain: Base. Did my withdrawal pay out?",
                        "", [history], True),
                    "stale_tag_after_chain_correction": ("Actually, it's on Base. Did my withdrawal pay out?",
                        "", [history], True),
                    "remembered_wallet": (f"Did the transaction {BURN_TX} pay out?",
                        "", [history], True),
                    "missing_wallet": (f"How do I withdraw from {VAULT}?",
                        "Share your wallet address so I can verify a safe withdrawal.", [redemption], False),
                    "two_receipts": ("Did both of those withdrawals pay out?", "", [history], True),
                    "two_wording": ("Did the two withdrawals pay out?", "", [history], True),
                    "restated_second": (f"Did both withdrawals pay out? The second is {TX}.", "", [history], True),
                    "partial_receipts": ("Did both of those withdrawals pay out?", "", [history], True),
                    "migration_read": (f"Which token did I receive during migration? The transfer is {BURN_TX}.",
                        "This receipt lookup is incomplete. Check the destination transfer.", [history], False),
                    "migration_burn": (f"Which replacement token did migration send in {BURN_TX}?",
                        "The source burn is visible. Check the destination transfer for the replacement token.", [history], False),
                    "migration_positive": (f"Which replacement token should I look for after migrating? Source transaction {BURN_TX}.",
                        "Check the replacement token identified by the migration docs at the destination.", [history], False),
                    "withdrawal_button_read": ("Where is the withdrawal button?",
                        "Open the vault page and select Withdraw.", [history], False),
                    "withdrawal_button_followup": ("Where is the withdrawal button?",
                        "Open the vault page and select Withdraw.", [history], False),
                }[mode]
                if mode in {"migration_read", "withdrawal_button_read", "withdrawal_button_followup"}:
                    observations.mode = "history_failure"
                (root / "fixture_instruction.json").write_text(json.dumps({
                    "queries": queries, "answer": draft, "cite": cite, "uncited": not cite,
                }))
                request = replace(build_smoke_transport_request(), smoke_mode=None, aggregated_text=question)
                request.run_context["is_public_trigger"] = True
                if chain_correction:
                    job = TicketInvestigationJob(request.run_context["channel_id"])
                    merge_explicit_evidence(job, f"My withdrawal was https://etherscan.io/tx/{BURN_TX}")
                    if mode == "chain_only_correction":
                        merge_explicit_evidence(job, question)
                        assert job.evidence.tx_chains == {BURN_TX: "base"}, job.evidence
                    request.investigation_job = serialize_investigation_job(job)
                if mode == "additive_deposit_correction":
                    job = TicketInvestigationJob(request.run_context["channel_id"])
                    merge_explicit_evidence(job, f"Here is the first deposit transaction https://arbiscan.io/tx/{BURN_TX}.")
                    merge_explicit_evidence(job, question)
                    request.investigation_job = serialize_investigation_job(job)
                    assert job.evidence.tx_hashes == [BURN_TX, TX] and job.evidence.chain == "base"
                    assert job.evidence.tx_chains == {BURN_TX: "arbitrum", TX: "base"}
                if mode in {"remembered_wallet_after_correction", "confirmed_bare_wallet", "recipient_as_wallet_owner"}:
                    request.investigation_job["evidence"]["wallet"] = ALPHA if mode == "recipient_as_wallet_owner" else WALLET
                if mode in {"remembered_wallet_after_correction", "confirmed_bare_wallet"}:
                    request.current_history = [{"role": "user", "content": f"Correction: my wallet is {ALPHA}."}]
                if mode == "confirmed_bare_wallet":
                    request.current_history += [
                        {"role": "user", "content": WALLET},
                        {"role": "assistant", "content": "Is this your wallet address?"},
                        {"role": "user", "content": "Yes."},
                    ]
                if mode == "remembered_wallet":
                    request.current_history = [{"role": "user", "content": f"My wallet is {ALPHA}."}]
                if mode in {"implicit_receipt", "restated_failed_receipt", "repeated_failed_receipt", "withdrawal_button_followup"}:
                    request.current_history = [{"role": "user", "content": f"My withdrawal transaction was {BURN_TX}."}]
                if mode == "repeated_failed_receipt":
                    request.current_history.append({"role": "user", "content": f"Here is the hash again: {BURN_TX}."})
                if mode == "linked_vault_followup":
                    request.current_history = [
                        {"role": "user", "content": f"I mean https://yearn.fi/vaults/8453/{VAULT}, wallet {WALLET}."},
                    ]
                if mode in {"return_to_owner", "switch_to_full_balance", "same_vault_exact", "prior_full_overrides_exact",
                            "asset_followup"}:
                    request.current_history = [{"role": "user", "content": exact_question}]
                if mode in {"new_owner_old_recipient", "new_owner_own_recipient"}:
                    request.current_history = [{"role": "user", "content": exact_question.replace(WALLET, other_vault)}]
                if mode == "prior_full_overrides_exact":
                    request.current_history.append({"role": "user", "content": "Actually, redeem everything."})
                if mode in {"old_vault_recipient", "new_vault_owner", "partial_target_override"}:
                    prior = exact_question.replace(VAULT, other_vault)
                    if mode == "partial_target_override":
                        prior = f"The correct network is Base. See https://yearn.fi/vaults/8453/{other_vault}."
                    request.current_history = [{"role": "user", "content": prior}]
                if compare_receipts:
                    request.current_history = [
                        {"role": "user", "content": f"Check this withdrawal: {BURN_TX}"},
                        {"role": "user", "content": f"Here is the second one: {TX}"},
                    ]
                outcome = json.loads(await endpoint.execute_json_turn(request.to_json()))["flow_outcome"]
                reply = outcome["raw_final_reply"]
                proofs = outcome.get("verified_evidence", [])
                if mode in {"two_receipts", "two_wording", "restated_second"}:
                    assert len(proofs) == 2 and all(tx in reply for tx in hashes), outcome
                elif mode == "partial_receipts":
                    assert len(proofs) == 1 and BURN_TX in reply and "couldn't verify" in reply.lower(), outcome
                elif mode == "matching_wallet":
                    assert len(proofs) == 1 and BURN_TX in reply, outcome
                elif mode in {"linked_vault", "new_vault_owner", "confirmed_vault_in_batch", "new_owner_own_recipient",
                              "confirmed_bare_wallet"}:
                    assert len(proofs) == 1 and "max_loss=0" in reply and VAULT in reply, outcome
                    assert f"receiver={WALLET}" in reply, outcome
                elif mode == "recipient_wallet_label":
                    assert len(proofs) == 1 and f"receiver={Web3.to_checksum_address(ALPHA)}, owner={WALLET}" in reply, outcome
                elif mode in {"exact_redemption", "same_vault_exact"}:
                    assert len(proofs) == 1 and f"shares=100, receiver={Web3.to_checksum_address(ALPHA)}" in reply, outcome
                    assert "120 raw underlying units" in reply and "shares=10000000" not in reply, outcome
                elif mode == "missing_wallet":
                    assert "share your wallet address" in reply.lower() and not proofs, outcome
                elif mode in {"migration_read", "migration_burn", "migration_positive",
                              "withdrawal_button_read", "withdrawal_button_followup", "general_withdrawal_asset",
                              "read_only_share_balance", "additive_deposit_correction"}:
                    assert reply == draft and not proofs, outcome
                else:
                    assert "couldn't verify" in reply.lower() and "999" not in reply and reply != draft and not proofs, outcome
                    assert "max_loss=0" not in reply, outcome
                return {"reply": reply, "proofs": len(proofs)}

            for mode in ("failed_receipt", "prose_recipe", "explicit_wallet", "matching_wallet", "remembered_wallet",
                         "missing_wallet", "two_receipts", "partial_receipts", "migration_read", "withdrawal_button_read",
                         "implicit_receipt", "run_recipe", "wrong_vault", "wrong_chain", "linked_vault", "two_wording",
                         "migration_burn", "wrong_redemption_parameters", "exact_redemption", "ambiguous_redemption_amount",
                         "linked_vault_followup", "return_to_owner", "switch_to_full_balance", "restated_second", "migration_positive",
                         "withdrawal_button_followup", "unspecified_redemption_amount", "spelled_amount", "asset_amount",
                         "restricted_full_amount", "alternative_full_amount",
                         "old_vault_recipient", "new_vault_owner", "partial_target_override", "same_vault_exact",
                         "restated_failed_receipt", "repeated_failed_receipt", "negated_recipient",
                         "prior_full_overrides_exact", "asset_followup", "corrected_vault_in_batch",
                         "confirmed_vault_in_batch", "corrected_amount_in_batch",
                         "unguarded_redeem", "unguarded_withdraw", "unguarded_payout", "general_withdrawal_asset",
                         "corrected_wallet_stale_proof", "new_owner_old_recipient", "new_owner_own_recipient", "contradictory_amounts_in_batch",
                         "multiline_withdraw", "withdrawal_hash_plain_confirmation", "remembered_wallet_after_correction",
                         "confirmed_bare_wallet", "recipient_as_wallet_owner", "recipient_wallet_label", "raw_amount_overridden",
                         "named_withdrawal_arguments", "withdrawn_funds_destination", "read_only_share_balance",
                         "additive_deposit_correction", "chain_only_correction", "stale_tag_after_chain_correction"):
                await check("scope_" + mode, lambda mode=mode: claim_scope(mode))

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
            root_path / "ysupport/execution/codex_support_contract.py",
            root_path / "ysupport/execution/codex_support_endpoint.py",
            root_path / "ysupport/execution/verified_evidence.py",
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
