"""Bounded, wallet-filtered transfer proofs owned by the private RPC runtime."""

import asyncio
import json
import re

from web3 import Web3

from ysupport import config
from ysupport.tools.chain_access import get_web3_instance
from ysupport.tools.kong_vaults import fetch_kong_recent_transfers

_TRANSFER = Web3.keccak(text="Transfer(address,address,uint256)")
_ZERO = "0x" + "0" * 40
_WITHDRAW = Web3.keccak(text="Withdraw(address,address,address,uint256,uint256)")


def _address(value):
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{40}", value):
        raise ValueError("A valid wallet/token address is required.")
    return Web3.to_checksum_address(value)


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{64}", value):
        raise ValueError(
            "History transaction hashes must be full transaction identifiers."
        )
    return value.lower()


def _transfer(log, *, wallet, token, tx_hash, block):
    if str(log.get("address", "")).lower() != token.lower() or log.get("removed"):
        return None
    topics = log.get("topics", [])
    if len(topics) != 3 or bytes(topics[0]) != bytes(_TRANSFER):
        return None
    if any(len(bytes(topic)) != 32 for topic in topics[1:]):
        raise ValueError("Malformed indexed transfer address.")
    sender = Web3.to_checksum_address(bytes(topics[1])[-20:].hex())
    receiver = Web3.to_checksum_address(bytes(topics[2])[-20:].hex())
    if wallet not in (sender, receiver):
        return None
    data = bytes(log.get("data", b""))
    if len(data) != 32:
        raise ValueError("Malformed ERC20 transfer amount.")
    return {
        "transaction_hash": tx_hash,
        "block_number": block,
        "log_index": log["logIndex"],
        "token": token,
        "from": sender,
        "to": receiver,
        "amount_raw": str(int.from_bytes(data, "big")),
        "direction": "self"
        if sender == receiver
        else "in"
        if receiver == wallet
        else "out",
        "is_mint": sender == _ZERO,
        "is_burn": receiver == _ZERO,
        "receipt_status": 1,
    }


def _withdrawal(log, *, wallet, token, tx_hash, block):
    topics = log.get("topics", [])
    if str(log.get("address", "")).lower() != token.lower() or log.get("removed"):
        return None
    if len(topics) != 4 or bytes(topics[0]) != bytes(_WITHDRAW):
        return None
    if any(len(bytes(topic)) != 32 for topic in topics[1:]):
        raise ValueError("Malformed indexed withdrawal address.")
    sender, receiver, owner = [
        Web3.to_checksum_address(bytes(topic)[-20:].hex()) for topic in topics[1:]
    ]
    if wallet not in (receiver, owner):
        return None
    data = bytes(log.get("data", b""))
    if len(data) != 64:
        raise ValueError("Malformed ERC4626 withdrawal quantities.")
    return {
        "transaction_hash": tx_hash,
        "block_number": block,
        "log_index": log["logIndex"],
        "vault": token,
        "sender": sender,
        "receiver": receiver,
        "owner": owner,
        "assets_raw": str(int.from_bytes(data[:32], "big")),
        "shares_raw": str(int.from_bytes(data[32:], "big")),
        "receipt_status": 1,
    }


async def core_get_wallet_history(
    *,
    chain: str,
    wallet_address: str,
    token_address: str,
    tx_hashes_json: str | None = None,
    from_block: str | int | None = None,
    to_block: str | int | None = None,
    max_results: int = 10,
) -> str:
    """Verify scoped transfer proofs, without asserting complete wallet history."""
    name = chain.strip().lower()
    web3 = get_web3_instance(name)
    if web3 is None or name not in config.CHAIN_NAME_TO_ID:
        raise ValueError("Unsupported history chain.")
    wallet, token = _address(wallet_address), _address(token_address)
    if type(max_results) is not int or not 1 <= max_results <= 25:
        raise ValueError("History output must contain at most 25 transfers.")
    if tx_hashes_json is not None and (from_block is not None or to_block is not None):
        raise ValueError("Choose supplied transactions or a block window.")
    if (from_block is None) != (to_block is None):
        raise ValueError("History windows require both bounded block numbers.")
    result = {
        "chain": name,
        "wallet": wallet,
        "token": token,
        "transfers": [],
        "withdrawals": [],
        "asset_transfers_in": [],
        "errors": [],
        "coverage": {"complete": False, "history_complete": False},
        "interpretation": "Only listed on-chain movements are verified. Completing supplied receipts or a block window is not complete wallet history. Burns alone do not prove payment; match withdrawal assets and incoming token transfers to the vault's underlying asset. A source receipt alone does not prove an unpaid claim. If coverage is incomplete, ask for a relevant transfer hash or block window; do not infer funds await liquidity or operator action.",
    }
    hashes = []
    candidate_truncated = False
    if tx_hashes_json is not None:
        supplied = json.loads(tx_hashes_json)
        if not isinstance(supplied, list) or not 1 <= len(supplied) <= 5:
            raise ValueError("History accepts one to five supplied transaction hashes.")
        hashes = list(dict.fromkeys(_hash(h) for h in supplied))
        result["coverage"]["scope"] = "provided_transactions"
    elif from_block is not None:

        def block_number(value):
            if type(value) is int:
                return value
            if isinstance(value, str) and re.fullmatch(
                r"(?:0x[0-9a-fA-F]+|[0-9]+)", value
            ):
                return int(value, 16 if value.startswith("0x") else 10)
            raise ValueError("History windows require numeric block numbers.")

        start, end = block_number(from_block), block_number(to_block)
        if start < 0 or end < start or end - start >= 5000:
            raise ValueError("History windows must contain at most 5000 blocks.")
        result["coverage"].update(scope="block_range", from_block=start, to_block=end)
        topic = "0x" + "0" * 24 + wallet[2:].lower()
        batches = await asyncio.gather(
            *(
                asyncio.to_thread(
                    web3.eth.get_logs,
                    {
                        "address": token,
                        "fromBlock": start,
                        "toBlock": end,
                        "topics": [Web3.to_hex(_TRANSFER), *direction],
                    },
                )
                for direction in ([None, topic], [topic])
            ),
            return_exceptions=True,
        )
        for direction, batch in zip(("incoming", "outgoing"), batches):
            try:
                if isinstance(batch, BaseException):
                    raise batch
                hashes.extend(
                    Web3.to_hex(log["transactionHash"]).lower() for log in batch
                )
            except Exception as exc:
                result["errors"].append(
                    {"scope": "block_range", "direction": direction, "error": str(exc)}
                )
        hashes = list(dict.fromkeys(hashes))
    else:
        result["coverage"].update(scope="recent_kong_transfers", index_limit=100)
        try:
            rows = await fetch_kong_recent_transfers(
                config.CHAIN_NAME_TO_ID[name], token
            )
            matching = [
                row
                for row in rows
                if row.get("chainId") == config.CHAIN_NAME_TO_ID[name]
                and str(row.get("address", "")).lower() == token.lower()
                and wallet.lower()
                in (
                    str(row.get("sender", "")).lower(),
                    str(row.get("receiver", "")).lower(),
                )
            ]
            hashes = list(
                dict.fromkeys(_hash(row["transactionHash"]) for row in matching)
            )
            result["coverage"].update(
                indexed_rows=len(rows), matching_candidates=len(matching)
            )
        except Exception as exc:
            result["errors"].append(
                {"scope": "recent_kong_transfers", "error": str(exc)}
            )
    # Bound receipt fan-out as well as the displayed rows. Never decode/profile unrelated contracts.
    if len(hashes) > max_results:
        hashes = hashes[:max_results]
        candidate_truncated = True
    head = await asyncio.to_thread(lambda: web3.eth.block_number)
    result["observed_block"] = head
    if from_block is not None and end > head:
        result["errors"].append(
            {
                "scope": "block_range",
                "error": "Requested range extends beyond the observed head.",
            }
        )
    header = await asyncio.to_thread(web3.eth.get_block, head)
    result["observed_timestamp"] = header.get("timestamp")
    limit = asyncio.Semaphore(4)

    async def prove(tx_hash):
        async with limit:
            try:
                receipt = await asyncio.to_thread(
                    web3.eth.get_transaction_receipt, tx_hash
                )
                if Web3.to_hex(
                    receipt["transactionHash"]
                ).lower() != tx_hash or receipt.get("status") not in (0, 1):
                    raise ValueError("Receipt identity/status could not be verified.")
                if receipt["status"] == 0:
                    return (
                        [],
                        [],
                        [],
                        {"transaction_hash": tx_hash, "status": "reverted"},
                        None,
                    )
                block = receipt["blockNumber"]
                if (
                    type(block) is not int
                    or block > head
                    or (from_block is not None and not start <= block <= end)
                ):
                    raise ValueError(
                        "Receipt block is outside the observed history scope."
                    )
                movements = [
                    _transfer(
                        log, wallet=wallet, token=token, tx_hash=tx_hash, block=block
                    )
                    for log in receipt["logs"]
                ]
                withdrawals = [
                    _withdrawal(
                        log, wallet=wallet, token=token, tx_hash=tx_hash, block=block
                    )
                    for log in receipt["logs"]
                ]
                payouts = []
                for log in receipt["logs"]:
                    if str(log.get("address", "")).lower() == token.lower():
                        continue
                    payout = _transfer(
                        log,
                        wallet=wallet,
                        token=_address(log.get("address")),
                        tx_hash=tx_hash,
                        block=block,
                    )
                    if (
                        payout is not None
                        and payout["from"] == token
                        and payout["to"] == wallet
                    ):
                        payouts.append(payout)
                return (
                    [row for row in movements if row is not None],
                    [row for row in withdrawals if row is not None],
                    payouts,
                    {
                        "transaction_hash": tx_hash,
                        "status": "success",
                        "block_number": block,
                    },
                    None,
                )
            except Exception as exc:
                return (
                    [],
                    [],
                    [],
                    None,
                    {"transaction_hash": tx_hash, "error": str(exc)},
                )

    proofs = await asyncio.gather(*(prove(h) for h in hashes))
    transactions = []
    transfers = {}
    withdrawals = {}
    payouts = {}
    for movements, exits, assets, transaction, error in proofs:
        if transaction:
            transactions.append(transaction)
        if error:
            result["errors"].append(error)
        for movement in movements:
            transfers[(movement["transaction_hash"], movement["log_index"])] = movement
        for event in exits:
            withdrawals[(event["transaction_hash"], event["log_index"])] = event
        for payout in assets:
            payouts[(payout["transaction_hash"], payout["log_index"])] = payout
    ordered = sorted(
        transfers.values(), key=lambda row: (row["block_number"], row["log_index"])
    )
    result["transfers"] = ordered[:max_results]
    result["transactions"] = transactions
    result["withdrawals"] = sorted(
        withdrawals.values(), key=lambda row: (row["block_number"], row["log_index"])
    )[:max_results]
    result["asset_transfers_in"] = sorted(
        payouts.values(), key=lambda row: (row["block_number"], row["log_index"])
    )[:max_results]
    result["coverage"].update(
        complete=not result["errors"]
        and not candidate_truncated
        and max(len(ordered), len(withdrawals), len(payouts)) <= max_results,
        truncated=candidate_truncated
        or max(len(ordered), len(withdrawals), len(payouts)) > max_results,
        verified_transactions=len(transactions),
    )
    if result["coverage"]["scope"] == "recent_kong_transfers":
        result["coverage"]["complete"] = False
    if not result["coverage"]["complete"]:
        result["next_evidence"] = {
            "kind": "public_transaction_hash_or_block_range",
            "handoff_required": False,
        }
    return json.dumps(result)
