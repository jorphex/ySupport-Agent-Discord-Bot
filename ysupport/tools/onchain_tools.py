from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional, Union

from web3 import Web3
from web3._utils.events import get_event_data

from ysupport.tools import chain_access


_inspect_contract_profile = chain_access.inspect_contract_profile
ensure_web3_instances = chain_access.ensure_web3_instances


def _json_loads_or_default(raw_value: Optional[str], default: Any) -> Any:
    if raw_value in (None, ""):
        return default
    return json.loads(raw_value)


def _normalize_chain(chain: str) -> str:
    return (chain or "").strip().lower()


def _parse_block_identifier(value: Optional[Union[str, int]]) -> Optional[Union[str, int]]:
    if value in (None, ""):
        return None
    if type(value) is int:
        if value < 0:
            raise ValueError("Block numbers must be non-negative.")
        return value
    if not isinstance(value, str):
        raise ValueError("Block identifiers must be numeric or named block tags.")
    normalized = value.strip().lower()
    relative_latest_match = re.fullmatch(r"latest([+-]\d+)", normalized)
    if relative_latest_match:
        raise ValueError(
            "Relative block identifiers like latest-50000 require a latest_block value."
        )
    if normalized in {"latest", "earliest", "pending", "safe", "finalized"}:
        return normalized
    number = int(normalized, 16) if normalized.startswith("0x") else int(normalized)
    if number < 0:
        raise ValueError("Block numbers must be non-negative.")
    return number


def _parse_block_identifier_with_latest(
    value: Optional[Union[str, int]],
    *,
    latest_block: Optional[int],
) -> Optional[Union[str, int]]:
    if not isinstance(value, str) or value == "":
        return _parse_block_identifier(value)
    normalized = value.strip().lower()
    relative_latest_match = re.fullmatch(r"latest([+-]\d+)", normalized)
    if relative_latest_match:
        if latest_block is None:
            raise ValueError(
                "Relative block identifiers like latest-50000 require a latest_block value."
            )
        return max(latest_block + int(relative_latest_match.group(1)), 0)
    return _parse_block_identifier(value)


def _parse_function_signature(signature: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", signature or "")
    if not match:
        raise ValueError(
            "Function signature must look like 'allowance(address,address)'."
        )
    name = match.group(1)
    inputs_text = match.group(2).strip()
    if not inputs_text:
        return name, []
    return name, [part.strip() for part in inputs_text.split(",")]


def _build_function_abi(
    *,
    function_abi_json: Optional[str],
    function_signature: Optional[str],
    output_types_json: Optional[str],
) -> dict[str, Any]:
    if function_abi_json:
        abi = json.loads(function_abi_json)
        if not isinstance(abi, dict) or abi.get("type") != "function":
            raise ValueError(
                "function_abi_json must be a JSON object describing a single function ABI."
            )
        return abi

    if not function_signature:
        raise ValueError(
            "Either function_abi_json or function_signature must be provided for mode='call'."
        )

    output_types = _json_loads_or_default(output_types_json, [])
    if not isinstance(output_types, list):
        raise ValueError(
            "output_types_json must be a JSON array of solidity output types."
        )

    function_name, input_types = _parse_function_signature(function_signature)
    return {
        "type": "function",
        "name": function_name,
        "stateMutability": "view",
        "inputs": [
            {"name": f"arg_{idx}", "type": solidity_type}
            for idx, solidity_type in enumerate(input_types)
        ],
        "outputs": [
            {"name": f"out_{idx}", "type": solidity_type}
            for idx, solidity_type in enumerate(output_types)
        ],
    }


def _coerce_abi_argument(field: dict[str, Any], value: Any) -> Any:
    kind = field.get("type", "")
    array = re.fullmatch(r"(.+)\[([0-9]*)\]", kind)
    if array:
        if not isinstance(value, (list, tuple)):
            raise ValueError("ABI array arguments must be arrays.")
        if array[2] and len(value) != int(array[2]):
            raise ValueError("ABI fixed array length does not match its type.")
        return [_coerce_abi_argument({**field, "type": array[1]}, item) for item in value]
    if kind == "tuple":
        components = field.get("components", [])
        if isinstance(value, dict):
            names = [component.get("name") for component in components]
            if set(value) != set(names):
                raise ValueError("ABI tuple fields do not match its components.")
            return tuple(_coerce_abi_argument(c, value[c["name"]]) for c in components)
        if not isinstance(value, (list, tuple)) or len(value) != len(components):
            raise ValueError("ABI tuple arguments do not match its components.")
        return tuple(_coerce_abi_argument(c, v) for c, v in zip(components, value))
    if kind == "address":
        return _checksum_address(value, label="address argument")
    if re.fullmatch(r"u?int\d*", kind) and isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return value


async def _contract_call_observation(
    web3_instance, *, to_address, function_signature=None, args_json=None,
    output_types_json=None, function_abi_json=None, block_identifier=None,
):
    checksum_to = _checksum_address(to_address, label="to_address")
    args = _json_loads_or_default(args_json, [])
    if not isinstance(args, list):
        raise ValueError("args_json must be a JSON array.")
    function_abi = _build_function_abi(
        function_abi_json=function_abi_json, function_signature=function_signature,
        output_types_json=output_types_json,
    )
    if len(args) != len(function_abi["inputs"]):
        raise ValueError("ABI argument count does not match its inputs.")
    args = [_coerce_abi_argument(field, value) for field, value in zip(function_abi["inputs"], args)]
    contract = web3_instance.eth.contract(address=checksum_to, abi=[function_abi])
    call = getattr(contract.functions, function_abi["name"])(*args)
    if block_identifier is None:
        result = await asyncio.to_thread(call.call)
    else:
        result = await asyncio.to_thread(call.call, block_identifier=block_identifier)
    return {"contract": checksum_to, "function": function_signature or function_abi["name"],
            "args": args, "result": result,
            "block_identifier": block_identifier if block_identifier is not None else "latest"}


async def core_read_contracts(
    *, chain: str, calls_json: str, block_identifier: Optional[str] = None,
    tx_hash: Optional[str] = None,
) -> str:
    """Read up to eight known contract fields at one explicit observation block."""
    chain_name = _normalize_chain(chain)
    web3_instance = ensure_web3_instances().get(chain_name)
    if web3_instance is None:
        raise ValueError("Unsupported batch-read chain.")
    calls = json.loads(calls_json)
    allowed = {"to_address", "function_signature", "args_json", "output_types_json", "function_abi_json"}
    if not isinstance(calls, list) or not 1 <= len(calls) <= 8 or any(
        not isinstance(call, dict) or set(call) - allowed or not call.get("to_address")
        or not (call.get("function_signature") or call.get("function_abi_json")) for call in calls
    ):
        raise ValueError("Batch reads require one to eight bounded contract queries.")
    if tx_hash and block_identifier is not None:
        raise ValueError("Choose either a transaction block or an explicit block.")
    transaction = None
    if tx_hash:
        if not re.fullmatch(r"0x[0-9a-fA-F]{64}", tx_hash):
            raise ValueError("Invalid transaction hash.")
        receipt = await asyncio.to_thread(web3_instance.eth.get_transaction_receipt, tx_hash)
        if Web3.to_hex(receipt["transactionHash"]).lower() != tx_hash.lower():
            raise ValueError("Transaction receipt identity could not be verified.")
        block = receipt["blockNumber"]
        transaction = {"hash": tx_hash, "status": receipt.get("status")}
        scope = "transaction_block"
    else:
        block = _parse_block_identifier(block_identifier) if block_identifier is not None else "latest"
        scope = "current_snapshot" if block == "latest" else "specified_block"
    header = await asyncio.to_thread(web3_instance.eth.get_block, block)
    number = header.get("number")
    if number is None:
        raise ValueError("A mined observation block is required; pending state cannot be pinned.")
    limit = asyncio.Semaphore(4)

    async def observe(call):
        async with limit:
            try:
                value = await _contract_call_observation(web3_instance, **call, block_identifier=number)
                return {"ok": True, **value}
            except Exception as exc:
                return {"ok": False, "request": call, "error": str(exc)}

    results = await asyncio.gather(*(observe(call) for call in calls))
    return json.dumps(_normalize_rpc_value({
        "chain": chain_name, "scope": scope, "block_number": number,
        "block_timestamp": header.get("timestamp"), "transaction": transaction,
        "complete": all(row["ok"] for row in results), "results": results,
        "causal_limit": "This is block-end state, not proof of a transaction's revert cause. Latest state does not explain a historical failure.",
    }))


def _normalize_rpc_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, (list, tuple)):
        return [_normalize_rpc_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_rpc_value(item) for key, item in value.items()}
    return value


def _format_structured_output(title: str, payload: dict[str, Any]) -> str:
    lines = [title]
    for key, value in payload.items():
        normalized = _normalize_rpc_value(value)
        if isinstance(normalized, (dict, list)):
            pretty = json.dumps(normalized, indent=2, sort_keys=True)
            lines.append(f"{key}: {pretty}")
        else:
            lines.append(f"{key}: {normalized}")
    return "\n".join(lines)


def _checksum_address(value: Optional[str], *, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required.")
    if not Web3.is_address(value):
        raise ValueError(f"{label} must be a valid EVM address.")
    return Web3.to_checksum_address(value)


def _parse_event_abis(event_abis_json: Optional[str]) -> list[dict[str, Any]]:
    parsed = _json_loads_or_default(event_abis_json, [])
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(
            "event_abis_json must be a JSON object or array of event ABI objects."
        )
    event_abis: list[dict[str, Any]] = []
    for event_abi in parsed:
        if not isinstance(event_abi, dict) or event_abi.get("type") != "event":
            raise ValueError("event_abis_json entries must be event ABI objects.")
        event_abis.append(event_abi)
    return event_abis


def _decode_logs_with_abis(
    web3_instance: Web3, logs: list[Any], event_abis: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not event_abis:
        return []
    decoded: list[dict[str, Any]] = []
    codec = web3_instance.codec
    for log in logs:
        matched = False
        for event_abi in event_abis:
            try:
                event_data = get_event_data(codec, event_abi, log)
            except Exception:
                continue
            decoded.append(
                {
                    "event": event_abi.get("name", "UnknownEvent"),
                    "address": log["address"],
                    "log_index": log.get("logIndex"),
                    "transaction_hash": log["transactionHash"].hex(),
                    "args": _normalize_rpc_value(dict(event_data["args"])),
                }
            )
            matched = True
            break
        if not matched:
            decoded.append(
                {
                    "event": None,
                    "address": log["address"],
                    "log_index": log.get("logIndex"),
                    "transaction_hash": log["transactionHash"].hex(),
                    "topics": [topic.hex() for topic in log["topics"]],
                    "data": log["data"],
                }
            )
    return decoded


def _standard_tx_event_abis() -> list[dict[str, Any]]:
    return [
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "from", "type": "address"},
                {"indexed": True, "name": "to", "type": "address"},
                {"indexed": False, "name": "value", "type": "uint256"},
            ],
            "name": "Transfer",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "owner", "type": "address"},
                {"indexed": True, "name": "spender", "type": "address"},
                {"indexed": False, "name": "value", "type": "uint256"},
            ],
            "name": "Approval",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "sender", "type": "address"},
                {"indexed": True, "name": "owner", "type": "address"},
                {"indexed": False, "name": "assets", "type": "uint256"},
                {"indexed": False, "name": "shares", "type": "uint256"},
            ],
            "name": "Deposit",
            "type": "event",
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "sender", "type": "address"},
                {"indexed": True, "name": "receiver", "type": "address"},
                {"indexed": True, "name": "owner", "type": "address"},
                {"indexed": False, "name": "assets", "type": "uint256"},
                {"indexed": False, "name": "shares", "type": "uint256"},
            ],
            "name": "Withdraw",
            "type": "event",
        },
    ]


def _format_token_amount(raw_value: Any, decimals: Any) -> str | None:
    if not isinstance(raw_value, int) or not isinstance(decimals, int):
        return None
    if decimals < 0 or decimals > 36:
        return None
    scaled_value = raw_value / (10**decimals)
    return f"{scaled_value:.18f}".rstrip("0").rstrip(".")


def _enrich_decoded_logs_with_profiles(
    decoded_logs: list[dict[str, Any]],
    profiles_by_address: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_logs: list[dict[str, Any]] = []
    for entry in decoded_logs:
        enriched = dict(entry)
        address = entry.get("address")
        profile = (
            profiles_by_address.get(Web3.to_checksum_address(address))
            if address and Web3.is_address(address)
            else None
        )
        if profile:
            enriched["contract_kind"] = profile.get("kind")
            enriched["token_symbol"] = profile.get("symbol")
            enriched["token_decimals"] = profile.get("decimals")
            if profile.get("asset"):
                enriched["asset"] = profile.get("asset")
                enriched["asset_symbol"] = profile.get("asset_symbol")
                enriched["asset_decimals"] = profile.get("asset_decimals")
        args = entry.get("args")
        if isinstance(args, dict):
            for value_field in ("value", "assets", "shares"):
                raw_value = args.get(value_field)
                if isinstance(raw_value, int):
                    decimals = None
                    if (
                        value_field == "shares"
                        and profile
                        and isinstance(profile.get("decimals"), int)
                    ):
                        decimals = profile["decimals"]
                    elif (
                        value_field == "assets"
                        and profile
                        and isinstance(profile.get("asset_decimals"), int)
                    ):
                        decimals = profile["asset_decimals"]
                    elif (
                        value_field == "value"
                        and profile
                        and isinstance(profile.get("decimals"), int)
                    ):
                        decimals = profile["decimals"]
                    formatted_value = _format_token_amount(raw_value, decimals)
                    if formatted_value is not None:
                        enriched[f"{value_field}_formatted"] = formatted_value
        enriched_logs.append(enriched)
    return enriched_logs


def _summarize_transaction_investigation(
    transaction: dict[str, Any],
    enriched_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    tx_from = transaction.get("from")
    tx_to = transaction.get("to")
    summary: dict[str, Any] = {
        "user_transfers_out": [],
        "user_transfers_in": [],
        "approvals": [],
        "deposits": [],
        "withdrawals": [],
        "unclassified_logs": 0,
        "notable_findings": [],
    }

    for entry in enriched_logs:
        event_name = entry.get("event")
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        if event_name == "Transfer":
            transfer_summary = {
                "token": entry.get("token_symbol"),
                "contract": entry.get("address"),
                "from": args.get("from"),
                "to": args.get("to"),
                "value": args.get("value"),
                "value_formatted": entry.get("value_formatted"),
                "contract_kind": entry.get("contract_kind"),
            }
            if tx_from and args.get("from") == tx_from:
                summary["user_transfers_out"].append(transfer_summary)
            if tx_from and args.get("to") == tx_from:
                summary["user_transfers_in"].append(transfer_summary)
        elif event_name == "Approval":
            summary["approvals"].append(
                {
                    "token": entry.get("token_symbol"),
                    "contract": entry.get("address"),
                    "owner": args.get("owner"),
                    "spender": args.get("spender"),
                    "value": args.get("value"),
                    "value_formatted": entry.get("value_formatted"),
                }
            )
        elif event_name == "Deposit":
            summary["deposits"].append(
                {
                    "vault": entry.get("address"),
                    "vault_symbol": entry.get("token_symbol"),
                    "sender": args.get("sender"),
                    "owner": args.get("owner"),
                    "assets": args.get("assets"),
                    "assets_formatted": entry.get("assets_formatted"),
                    "asset_symbol": entry.get("asset_symbol"),
                    "shares": args.get("shares"),
                    "shares_formatted": entry.get("shares_formatted"),
                }
            )
        elif event_name == "Withdraw":
            summary["withdrawals"].append(
                {
                    "vault": entry.get("address"),
                    "vault_symbol": entry.get("token_symbol"),
                    "sender": args.get("sender"),
                    "receiver": args.get("receiver"),
                    "owner": args.get("owner"),
                    "assets": args.get("assets"),
                    "assets_formatted": entry.get("assets_formatted"),
                    "asset_symbol": entry.get("asset_symbol"),
                    "shares": args.get("shares"),
                    "shares_formatted": entry.get("shares_formatted"),
                }
            )
        elif event_name is None:
            summary["unclassified_logs"] += 1

    if summary["user_transfers_out"]:
        summary["notable_findings"].append(
            f"Observed {len(summary['user_transfers_out'])} transfer(s) out from the tx sender."
        )
    if summary["user_transfers_in"]:
        summary["notable_findings"].append(
            f"Observed {len(summary['user_transfers_in'])} transfer(s) back to the tx sender."
        )
    if summary["deposits"]:
        summary["notable_findings"].append(
            f"Decoded {len(summary['deposits'])} explicit deposit event(s)."
        )
    if summary["withdrawals"]:
        summary["notable_findings"].append(
            f"Decoded {len(summary['withdrawals'])} explicit withdraw event(s)."
        )
    if summary["approvals"]:
        summary["notable_findings"].append(
            f"Decoded {len(summary['approvals'])} approval event(s)."
        )
    if not summary["deposits"] and not summary["withdrawals"]:
        summary["notable_findings"].append(
            "No explicit ERC4626 Deposit/Withdraw events were decoded from the transaction logs."
        )
    if tx_to:
        summary["notable_findings"].append(f"Primary transaction target was {tx_to}.")

    return summary


async def core_inspect_onchain(
    *,
    chain: str,
    mode: str,
    to_address: Optional[str] = None,
    function_signature: Optional[str] = None,
    args_json: Optional[str] = None,
    output_types_json: Optional[str] = None,
    function_abi_json: Optional[str] = None,
    tx_hash: Optional[str] = None,
    address: Optional[str] = None,
    topics_json: Optional[str] = None,
    from_block: Optional[str] = None,
    to_block: Optional[str] = None,
    event_abis_json: Optional[str] = None,
    block_identifier: Optional[str] = None,
    max_results: int = 10,
) -> str:
    chain_name = _normalize_chain(chain)
    web3_instances = ensure_web3_instances()
    if chain_name not in web3_instances:
        return f"Unsupported chain '{chain}'. Supported chains: {', '.join(sorted(web3_instances))}."

    web3_instance = web3_instances[chain_name]
    mode_normalized = (mode or "").strip().lower()
    max_results = max(1, min(max_results, 25))

    try:
        if mode_normalized == "call":
            observation = await _contract_call_observation(
                web3_instance, to_address=to_address,
                function_signature=function_signature, args_json=args_json,
                output_types_json=output_types_json, function_abi_json=function_abi_json,
                block_identifier=_parse_block_identifier(block_identifier),
            )
            return _format_structured_output("Onchain call result", {"chain": chain_name, **observation})

        if mode_normalized == "receipt":
            if not tx_hash:
                return "tx_hash is required for mode='receipt'."
            receipt = await asyncio.to_thread(
                web3_instance.eth.get_transaction_receipt, tx_hash
            )
            transaction = await asyncio.to_thread(
                web3_instance.eth.get_transaction, tx_hash
            )
            logs = list(receipt["logs"])
            event_abis = _parse_event_abis(event_abis_json)
            decoded_logs = _decode_logs_with_abis(
                web3_instance, logs[:max_results], event_abis
            )
            if not decoded_logs:
                decoded_logs = [
                    {
                        "address": log["address"],
                        "log_index": log.get("logIndex"),
                        "transaction_hash": log["transactionHash"].hex(),
                        "topics": [topic.hex() for topic in log["topics"]],
                        "data": log["data"],
                    }
                    for log in logs[:max_results]
                ]
            return _format_structured_output(
                "Transaction receipt",
                {
                    "chain": chain_name,
                    "transaction_hash": tx_hash,
                    "status": receipt.get("status"),
                    "block_number": receipt.get("blockNumber"),
                    "from": transaction.get("from"),
                    "to": transaction.get("to"),
                    "gas_used": receipt.get("gasUsed"),
                    "log_count": len(logs),
                    "logs_shown": min(len(logs), max_results),
                    "logs": decoded_logs,
                },
            )

        if mode_normalized in {"tx_summary", "tx_investigate"}:
            if not tx_hash:
                return f"tx_hash is required for mode='{mode_normalized}'."
            receipt = await asyncio.to_thread(
                web3_instance.eth.get_transaction_receipt, tx_hash
            )
            transaction = await asyncio.to_thread(
                web3_instance.eth.get_transaction, tx_hash
            )
            logs = list(receipt["logs"])
            summary_block_identifier = _parse_block_identifier(block_identifier)
            if summary_block_identifier is None:
                summary_block_identifier = receipt.get("blockNumber")
            event_limit = (
                len(logs) if mode_normalized == "tx_investigate" else max_results
            )
            decoded_logs = _decode_logs_with_abis(
                web3_instance,
                logs[:event_limit],
                _standard_tx_event_abis(),
            )
            unique_addresses: list[str] = []
            seen_addresses: set[str] = set()
            for log in logs:
                address = log.get("address")
                if not address or not Web3.is_address(address):
                    continue
                checksum_address = Web3.to_checksum_address(address)
                if checksum_address in seen_addresses:
                    continue
                seen_addresses.add(checksum_address)
                unique_addresses.append(checksum_address)
            profile_limit = min(
                len(unique_addresses), 12 if mode_normalized == "tx_investigate" else 8
            )
            contract_profiles = [
                await _inspect_contract_profile(
                    web3_instance,
                    address,
                    block_identifier=summary_block_identifier,
                )
                for address in unique_addresses[:profile_limit]
            ]
            profiles_by_address = {
                profile["address"]: profile for profile in contract_profiles
            }
            enriched_logs = _enrich_decoded_logs_with_profiles(
                decoded_logs, profiles_by_address
            )
            if mode_normalized == "tx_investigate":
                investigation_summary = _summarize_transaction_investigation(
                    transaction,
                    enriched_logs,
                )
                return _format_structured_output(
                    "Transaction investigation",
                    {
                        "chain": chain_name,
                        "transaction_hash": tx_hash,
                        "status": receipt.get("status"),
                        "block_number": receipt.get("blockNumber"),
                        "from": transaction.get("from"),
                        "to": transaction.get("to"),
                        "gas_used": receipt.get("gasUsed"),
                        "log_count": len(logs),
                        "events_decoded": len(enriched_logs),
                        "events_shown": min(len(enriched_logs), event_limit),
                        "investigation": investigation_summary,
                        "events": enriched_logs[:max_results],
                        "contracts_profiled": contract_profiles,
                        "profiled_at_block": summary_block_identifier,
                    },
                )
            return _format_structured_output(
                "Transaction summary",
                {
                    "chain": chain_name,
                    "transaction_hash": tx_hash,
                    "status": receipt.get("status"),
                    "block_number": receipt.get("blockNumber"),
                    "from": transaction.get("from"),
                    "to": transaction.get("to"),
                    "gas_used": receipt.get("gasUsed"),
                    "log_count": len(logs),
                    "events_shown": min(len(enriched_logs), max_results),
                    "events": enriched_logs,
                    "contracts_profiled": contract_profiles,
                    "profiled_at_block": summary_block_identifier,
                },
            )

        if mode_normalized == "logs":
            parsed_topics = _json_loads_or_default(topics_json, [])
            if not isinstance(parsed_topics, list):
                return "topics_json must be a JSON array."
            latest_block = await asyncio.to_thread(
                lambda: web3_instance.eth.block_number
            )
            parsed_from_block = _parse_block_identifier_with_latest(
                from_block,
                latest_block=latest_block,
            )
            parsed_to_block = _parse_block_identifier_with_latest(
                to_block,
                latest_block=latest_block,
            )
            filter_params: dict[str, Any] = {
                "fromBlock": (
                    parsed_from_block
                    if parsed_from_block is not None
                    else "latest"
                ),
                "toBlock": (
                    parsed_to_block if parsed_to_block is not None else "latest"
                ),
            }
            if parsed_topics:
                filter_params["topics"] = parsed_topics
            if address:
                filter_params["address"] = _checksum_address(address, label="address")
            logs = await asyncio.to_thread(web3_instance.eth.get_logs, filter_params)
            event_abis = _parse_event_abis(event_abis_json)
            decoded_logs = _decode_logs_with_abis(
                web3_instance, list(logs)[:max_results], event_abis
            )
            if not decoded_logs:
                decoded_logs = [
                    {
                        "address": log["address"],
                        "log_index": log.get("logIndex"),
                        "transaction_hash": log["transactionHash"].hex(),
                        "topics": [topic.hex() for topic in log["topics"]],
                        "data": log["data"],
                    }
                    for log in list(logs)[:max_results]
                ]
            return _format_structured_output(
                "Log query result",
                {
                    "chain": chain_name,
                    "filter": filter_params,
                    "log_count": len(logs),
                    "logs_shown": min(len(logs), max_results),
                    "logs": decoded_logs,
                },
            )

        return "Unsupported mode. Use one of: call, receipt, logs, tx_summary, tx_investigate."
    except Exception as exc:
        logging.error("[CoreTool:inspect_onchain] Error: %s", exc, exc_info=True)
        return f"Error inspecting onchain data: {exc}"


# --- Helpers ---
