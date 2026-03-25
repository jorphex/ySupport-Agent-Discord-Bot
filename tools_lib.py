# tools_lib.py
import asyncio
import logging
import json
import requests
import re
import aiohttp
from datetime import datetime, timezone
from typing import Dict, Optional, Union, Any

from web3 import Web3
from web3._utils.events import get_event_data

import config
import docs_repo_tools

WEB3_INSTANCES = {}

# Loop through URLs defined in config.py
for name, url in config.RPC_URLS.items():
    try:
        # Create the connection
        WEB3_INSTANCES[name] = Web3(Web3.HTTPProvider(url))
        
        # Optional: Check connection immediately (good for debugging logs)
        if not WEB3_INSTANCES[name].is_connected():
             print(f"Warning: Failed to connect to Web3 for {name}")
    except Exception as e:
        print(f"Error initializing Web3 for {name}: {e}")

# Load v1 Vaults List (Consider making this dynamic or configurable)
try:
    with open("v1_vaults.json", "r") as f:
        V1_VAULTS = json.load(f)
except Exception as e:
    logging.warning(f"Could not load v1_vaults.json: {e}. V1 deposit checks will fail.")
    V1_VAULTS = []



def _json_loads_or_default(raw_value: Optional[str], default: Any) -> Any:
    if raw_value in (None, ""):
        return default
    return json.loads(raw_value)


def _normalize_chain(chain: str) -> str:
    return (chain or "").strip().lower()


def _parse_block_identifier(value: Optional[str]) -> Optional[Union[str, int]]:
    if value in (None, ""):
        return None
    normalized = value.strip().lower()
    if normalized in {"latest", "earliest", "pending", "safe", "finalized"}:
        return normalized
    if normalized.startswith("0x"):
        return int(normalized, 16)
    return int(normalized)


def _parse_function_signature(signature: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*", signature or "")
    if not match:
        raise ValueError("Function signature must look like 'allowance(address,address)'.")
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
            raise ValueError("function_abi_json must be a JSON object describing a single function ABI.")
        return abi

    if not function_signature:
        raise ValueError("Either function_abi_json or function_signature must be provided for mode='call'.")

    output_types = _json_loads_or_default(output_types_json, [])
    if not isinstance(output_types, list):
        raise ValueError("output_types_json must be a JSON array of solidity output types.")

    function_name, input_types = _parse_function_signature(function_signature)
    return {
        "type": "function",
        "name": function_name,
        "stateMutability": "view",
        "inputs": [{"name": f"arg_{idx}", "type": solidity_type} for idx, solidity_type in enumerate(input_types)],
        "outputs": [{"name": f"out_{idx}", "type": solidity_type} for idx, solidity_type in enumerate(output_types)],
    }


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
        raise ValueError("event_abis_json must be a JSON object or array of event ABI objects.")
    event_abis: list[dict[str, Any]] = []
    for event_abi in parsed:
        if not isinstance(event_abi, dict) or event_abi.get("type") != "event":
            raise ValueError("event_abis_json entries must be event ABI objects.")
        event_abis.append(event_abi)
    return event_abis


def _decode_logs_with_abis(web3_instance: Web3, logs: list[Any], event_abis: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


async def _call_optional_contract_function(
    web3_instance: Web3,
    contract_address: str,
    function_abi: dict[str, Any],
    *,
    block_identifier: Any = None,
) -> Any:
    contract = web3_instance.eth.contract(address=contract_address, abi=[function_abi])
    contract_call = getattr(contract.functions, function_abi["name"])()
    if block_identifier is None:
        return await asyncio.to_thread(contract_call.call)
    return await asyncio.to_thread(contract_call.call, block_identifier=block_identifier)


async def _inspect_contract_profile(
    web3_instance: Web3,
    contract_address: str,
    *,
    block_identifier: Any = None,
) -> dict[str, Any]:
    checksum_address = Web3.to_checksum_address(contract_address)
    profile: dict[str, Any] = {
        "address": checksum_address,
        "symbol": None,
        "name": None,
        "decimals": None,
        "asset": None,
        "asset_symbol": None,
        "asset_decimals": None,
        "kind": "unclassified",
    }
    code = await asyncio.to_thread(web3_instance.eth.get_code, checksum_address)
    profile["has_code"] = bool(code and code != b"")
    if not profile["has_code"]:
        profile["kind"] = "eoa_or_missing_code"
        return profile

    symbol_abi = {
        "type": "function",
        "name": "symbol",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
    }
    name_abi = {
        "type": "function",
        "name": "name",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "string"}],
    }
    decimals_abi = {
        "type": "function",
        "name": "decimals",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint8"}],
    }
    asset_abi = {
        "type": "function",
        "name": "asset",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    }
    for field_name, abi in (
        ("symbol", symbol_abi),
        ("name", name_abi),
        ("decimals", decimals_abi),
    ):
        try:
            profile[field_name] = await _call_optional_contract_function(
                web3_instance,
                checksum_address,
                abi,
                block_identifier=block_identifier,
            )
        except Exception:
            continue

    try:
        asset_address = await _call_optional_contract_function(
            web3_instance,
            checksum_address,
            asset_abi,
            block_identifier=block_identifier,
        )
        if Web3.is_address(asset_address):
            checksum_asset = Web3.to_checksum_address(asset_address)
            profile["asset"] = checksum_asset
            profile["kind"] = "erc4626_like"
            for asset_field, abi in (
                ("asset_symbol", symbol_abi),
                ("asset_decimals", decimals_abi),
            ):
                try:
                    profile[asset_field] = await _call_optional_contract_function(
                        web3_instance,
                        checksum_asset,
                        abi,
                        block_identifier=block_identifier,
                    )
                except Exception:
                    continue
    except Exception:
        if profile["symbol"] is not None or profile["decimals"] is not None:
            profile["kind"] = "erc20_like"

    return profile


def _format_token_amount(raw_value: Any, decimals: Any) -> str | None:
    if not isinstance(raw_value, int) or not isinstance(decimals, int):
        return None
    if decimals < 0 or decimals > 36:
        return None
    scaled_value = raw_value / (10 ** decimals)
    return f"{scaled_value:.18f}".rstrip("0").rstrip(".")


def _enrich_decoded_logs_with_profiles(
    decoded_logs: list[dict[str, Any]],
    profiles_by_address: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched_logs: list[dict[str, Any]] = []
    for entry in decoded_logs:
        enriched = dict(entry)
        address = entry.get("address")
        profile = profiles_by_address.get(Web3.to_checksum_address(address)) if address and Web3.is_address(address) else None
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
                    if value_field == "shares" and profile and isinstance(profile.get("decimals"), int):
                        decimals = profile["decimals"]
                    elif value_field == "assets" and profile and isinstance(profile.get("asset_decimals"), int):
                        decimals = profile["asset_decimals"]
                    elif value_field == "value" and profile and isinstance(profile.get("decimals"), int):
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



def _extract_repo_artifact_refs(text: str) -> list[str]:
    return docs_repo_tools._extract_repo_artifact_refs(text)


async def close_shared_openai_clients() -> None:
    await docs_repo_tools.close_shared_openai_clients()


async def core_answer_from_docs(user_query: str) -> str:
    return await docs_repo_tools.core_answer_from_docs(user_query)


async def core_search_repo_context(
    query: str,
    limit: Optional[int] = None,
    include_legacy: bool = False,
    include_ui: bool = False,
) -> str:
    return await docs_repo_tools.core_search_repo_context(
        query,
        limit=limit,
        include_legacy=include_legacy,
        include_ui=include_ui,
    )


async def core_fetch_repo_artifacts(artifact_refs_text: str) -> str:
    return await docs_repo_tools.core_fetch_repo_artifacts(artifact_refs_text)


async def core_pretriage_repo_claim(
    claim_text: str,
    *,
    include_docs: bool = True,
    limit: Optional[int] = None,
    include_legacy: bool = False,
    include_ui: bool = False,
) -> str:
    logging.info(
        "[CoreTool:pretriage_repo_claim] claim='%s' include_docs=%s limit=%s include_legacy=%s include_ui=%s",
        claim_text,
        include_docs,
        limit,
        include_legacy,
        include_ui,
    )
    sections: list[str] = []

    repo_search = await core_search_repo_context(
        claim_text,
        limit=limit,
        include_legacy=include_legacy,
        include_ui=include_ui,
    )
    sections.append(f"Repo search:\n{repo_search}")

    artifact_refs = _extract_repo_artifact_refs(repo_search)[:2]
    if artifact_refs:
        artifact_text = await core_fetch_repo_artifacts(", ".join(artifact_refs))
        sections.append(f"Fetched repo artifacts:\n{artifact_text}")

    if include_docs:
        docs_answer = await core_answer_from_docs(claim_text)
        sections.append(f"Docs context:\n{docs_answer}")

    return "\n\n".join(section for section in sections if section.strip())


async def core_fetch_report_artifact(report_url: str, max_chars: int = 12000) -> str:
    return await docs_repo_tools.core_fetch_report_artifact(report_url, max_chars=max_chars)


async def core_repo_context_status() -> str:
    return await docs_repo_tools.core_repo_context_status()

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
    if chain_name not in WEB3_INSTANCES:
        return f"Unsupported chain '{chain}'. Supported chains: {', '.join(sorted(WEB3_INSTANCES))}."

    web3_instance = WEB3_INSTANCES[chain_name]
    mode_normalized = (mode or "").strip().lower()
    max_results = max(1, min(max_results, 25))

    try:
        if mode_normalized == "call":
            checksum_to = _checksum_address(to_address, label="to_address")
            args = _json_loads_or_default(args_json, [])
            if not isinstance(args, list):
                return "args_json must be a JSON array."
            function_abi = _build_function_abi(
                function_abi_json=function_abi_json,
                function_signature=function_signature,
                output_types_json=output_types_json,
            )
            contract = web3_instance.eth.contract(address=checksum_to, abi=[function_abi])
            function_name = function_abi["name"]
            contract_call = getattr(contract.functions, function_name)(*args)
            parsed_block_identifier = _parse_block_identifier(block_identifier)
            if parsed_block_identifier is None:
                result = await asyncio.to_thread(contract_call.call)
            else:
                result = await asyncio.to_thread(contract_call.call, block_identifier=parsed_block_identifier)
            return _format_structured_output(
                "Onchain call result",
                {
                    "chain": chain_name,
                    "contract": checksum_to,
                    "function": function_signature or function_name,
                    "args": args,
                    "result": result,
                    "block_identifier": parsed_block_identifier or "latest",
                },
            )

        if mode_normalized == "receipt":
            if not tx_hash:
                return "tx_hash is required for mode='receipt'."
            receipt = await asyncio.to_thread(web3_instance.eth.get_transaction_receipt, tx_hash)
            transaction = await asyncio.to_thread(web3_instance.eth.get_transaction, tx_hash)
            logs = list(receipt["logs"])
            event_abis = _parse_event_abis(event_abis_json)
            decoded_logs = _decode_logs_with_abis(web3_instance, logs[:max_results], event_abis)
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
            receipt = await asyncio.to_thread(web3_instance.eth.get_transaction_receipt, tx_hash)
            transaction = await asyncio.to_thread(web3_instance.eth.get_transaction, tx_hash)
            logs = list(receipt["logs"])
            summary_block_identifier = _parse_block_identifier(block_identifier)
            if summary_block_identifier is None:
                summary_block_identifier = receipt.get("blockNumber")
            event_limit = len(logs) if mode_normalized == "tx_investigate" else max_results
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
            profile_limit = min(len(unique_addresses), 12 if mode_normalized == "tx_investigate" else 8)
            contract_profiles = [
                await _inspect_contract_profile(
                    web3_instance,
                    address,
                    block_identifier=summary_block_identifier,
                )
                for address in unique_addresses[:profile_limit]
            ]
            profiles_by_address = {
                profile["address"]: profile
                for profile in contract_profiles
            }
            enriched_logs = _enrich_decoded_logs_with_profiles(decoded_logs, profiles_by_address)
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
            filter_params: dict[str, Any] = {
                "fromBlock": _parse_block_identifier(from_block) or "latest",
                "toBlock": _parse_block_identifier(to_block) or "latest",
            }
            if parsed_topics:
                filter_params["topics"] = parsed_topics
            if address:
                filter_params["address"] = _checksum_address(address, label="address")
            logs = await asyncio.to_thread(web3_instance.eth.get_logs, filter_params)
            event_abis = _parse_event_abis(event_abis_json)
            decoded_logs = _decode_logs_with_abis(web3_instance, list(logs)[:max_results], event_abis)
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
def extract_address_or_ens(text: str) -> Optional[str]:
    """Extracts the first 0x address or .eth ENS name from text."""
    # Regex for Ethereum address
    addr_match = re.search(r'(0x[a-fA-F0-9]{40})', text)
    if addr_match:
        return addr_match.group(1)
    # Regex for ENS name
    ens_match = re.search(r'\b([a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.eth)\b', text, re.IGNORECASE)
    if ens_match:
        return ens_match.group(1)
    return None

def resolve_ens(name: str) -> Optional[str]:
    """
    Resolves an ENS name or a SAFE-style prefixed address (e.g., 'eth:0x...') to a standard checksummed address.
    Returns the checksummed address if valid, otherwise None.
    """
    if not isinstance(name, str):
        return None
        
    address_to_check = name.strip()

    # --- Handle SAFE-style prefixes (e.g., 'eth:') ---
    if ':' in address_to_check:
        parts = address_to_check.split(':', 1)
        if len(parts) == 2:
            prefix, potential_address = parts
            logging.info(f"Detected prefixed address. Prefix: '{prefix}', Address part: '{potential_address}'")
            address_to_check = potential_address.strip()

    # Basic address check (on the potentially stripped address)
    if Web3.is_address(address_to_check):
        try:
            return Web3.to_checksum_address(address_to_check)
        except ValueError:
            # This can happen if the address has mixed-case but an invalid checksum
            logging.warning(f"Address '{address_to_check}' has invalid checksum.")
            return None

    # ENS check (only if it ends with .eth and we have an Ethereum instance)
    if address_to_check.endswith(".eth") and "ethereum" in WEB3_INSTANCES:
        try:
            logging.info(f"Attempting to resolve ENS name: '{address_to_check}'")
            resolved = WEB3_INSTANCES["ethereum"].ens.address(address_to_check)
            if resolved and Web3.is_address(resolved):
                logging.info(f"Successfully resolved ENS '{address_to_check}' to '{resolved}'")
                return Web3.to_checksum_address(resolved)
            else:
                 logging.warning(f"ENS name '{address_to_check}' did not resolve to a valid address.")
                 return None
        except Exception as e:
            logging.error(f"Error resolving ENS '{address_to_check}': {e}")
            return None
            
    # If it's not a valid address format or a resolvable ENS name
    logging.warning(f"Input '{name}' could not be resolved to a valid address.")
    return None

async def _fetch_vault_and_gauge_balances(
    vault_info: Dict,
    web3_instance: Web3,
    user_checksum_addr: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """
    Fetches a user's balance from a vault AND its associated staking gauge concurrently.
    Returns a dictionary with vault info and balances.
    """
    async with semaphore:
        vault_addr_str = vault_info.get("address")
        if not vault_addr_str:
            return {"vault_info": vault_info, "error": "Missing vault address"}

        try:
            vault_checksum_addr = Web3.to_checksum_address(vault_addr_str)
            vault_contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)

            # --- Create all web3 call coroutines first ---
            wallet_balance_coro = asyncio.to_thread(
                vault_contract.functions.balanceOf(user_checksum_addr).call
            )

            gauge_balance_coro = None
            staking_info = vault_info.get("staking")
            if staking_info and staking_info.get("available") and Web3.is_address(staking_info.get("address")):
                gauge_addr_str = staking_info.get("address")
                gauge_checksum_addr = Web3.to_checksum_address(gauge_addr_str)
                gauge_contract = web3_instance.eth.contract(address=gauge_checksum_addr, abi=config.GAUGE_ABI)
                gauge_balance_coro = asyncio.to_thread(
                    gauge_contract.functions.balanceOf(user_checksum_addr).call
                )

            # --- Run wallet and gauge balance checks in parallel ---
            if gauge_balance_coro:
                results = await asyncio.gather(wallet_balance_coro, gauge_balance_coro, return_exceptions=True)
                wallet_balance = results[0] if not isinstance(results[0], Exception) else 0
                gauge_token_balance = results[1] if not isinstance(results[1], Exception) else 0
                if isinstance(results[0], Exception):
                    logging.warning(f"Error fetching wallet balance for {vault_addr_str}: {results[0]}")
                if isinstance(results[1], Exception):
                    logging.warning(f"Error fetching gauge balance for {gauge_addr_str}: {results[1]}")
            else:
                # Only run the wallet balance check if no gauge
                wallet_balance = await wallet_balance_coro
                gauge_token_balance = 0

            logging.debug(f"Vault {vault_addr_str}: Wallet balance={wallet_balance}, Gauge token balance={gauge_token_balance}")

            # --- If staked, perform the final conversion call ---
            staked_balance_in_yvtoken = 0
            if gauge_token_balance > 0:
                # This is a final, conditional call. It's acceptable to await it here
                # as the two main calls have already completed.
                staked_balance_in_yvtoken = await asyncio.to_thread(
                    gauge_contract.functions.convertToAssets(gauge_token_balance).call
                )
                logging.debug(f"Vault {vault_addr_str}: Staked balance (in yvToken) is {staked_balance_in_yvtoken}")

            return {
                "vault_info": vault_info,
                "wallet_balance": wallet_balance,
                "staked_balance": staked_balance_in_yvtoken,
                "error": None
            }

        except Exception as e:
            logging.error(f"Critical error in balance fetching logic for vault {vault_addr_str}: {e}", exc_info=True)
            return {
                "vault_info": vault_info,
                "wallet_balance": 0,
                "staked_balance": 0,
                "error": str(e)
            }
        
async def query_active_deposits_logic(resolved_address: str, chain: Optional[str] = None, token_symbol: Optional[str] = None) -> str:
    logging.info(f"[Logic:query_active_deposits] Checking Active for {resolved_address}, Chain: {chain}, Token: {token_symbol}")

    chains_to_check = []
    if chain:
        chain_lower = chain.lower()
        if chain_lower in WEB3_INSTANCES:
            chains_to_check.append(chain_lower)
        else:
            return f"Unsupported chain: {chain}."
    else:
        chains_to_check = [c for c in WEB3_INSTANCES.keys() if c != 'berachain']

    all_vaults_data = []

    url = "https://ydaemon.yearn.fi/vaults/detected?hideAlways=false&strategiesDetails=withDetails&strategiesCondition=all&limit=2000"

    try:
        response = await asyncio.to_thread(requests.get, url, timeout=30)
        response.raise_for_status()
        all_vaults_data = response.json()
        if not isinstance(all_vaults_data, list):
             logging.error("[Logic:query_active_deposits] Unexpected yDaemon response format.")
             return "Error: Received unexpected data format from vault API."
        logging.info(f"[Logic:query_active_deposits] Successfully fetched {len(all_vaults_data)} vault definitions from yDaemon.")
    except Exception as e:
        logging.error(f"[Logic:query_active_deposits] Failed to fetch vault list from yDaemon: {e}")
        return f"Error: Could not fetch the list of active vaults: {e}"

    user_checksum_addr = Web3.to_checksum_address(resolved_address)

    all_results = []
    total_deposits_found = 0
    for chain_name in chains_to_check:
        web3_instance = WEB3_INSTANCES.get(chain_name)
        if not web3_instance:
            continue

        chain_id = config.CHAIN_NAME_TO_ID.get(chain_name)
        if not chain_id:
            continue

        chain_vaults = [v for v in all_vaults_data if v.get("chainID") == chain_id]
        if token_symbol:
            token_lower = token_symbol.lower()
            chain_vaults = [v for v in chain_vaults if token_lower in v.get("token", {}).get("symbol", "").lower()]

        if not chain_vaults:
            logging.info(f"No vaults to check for chain '{chain_name}' after filtering.")
            continue

        logging.info(f"Creating {len(chain_vaults)} balance check tasks for chain '{chain_name}'.")
        semaphore = asyncio.Semaphore(25)
        tasks = [_fetch_vault_and_gauge_balances(v, web3_instance, user_checksum_addr, semaphore) for v in chain_vaults]
        
        balance_results = await asyncio.gather(*tasks, return_exceptions=True)
        logging.info(f"Completed balance checks for chain '{chain_name}'.")

        chain_deposits = []
        for result in balance_results:
            if isinstance(result, Exception) or result.get("error"):
                continue

            total_balance = result.get("wallet_balance", 0) + result.get("staked_balance", 0)

            if total_balance > 0:
                try:
                    vault_info = result["vault_info"]
                    decimals = int(vault_info.get("decimals", 18))
                    total_display_balance = total_balance / (10 ** decimals)
                    vault_address = Web3.to_checksum_address(vault_info.get("address"))
                    vault_url = f"https://yearn.fi/vaults/{chain_id}/{vault_address}"
                    vault_name = vault_info.get('name', 'Unknown Vault')
                    vault_symbol = vault_info.get('symbol', 'N/A')
                    deposit_lines = [
                        f"**Vault:** [{vault_name}]({vault_url}) (Symbol: {vault_symbol})",
                        f"  Address: `{vault_address}`",
                        f"  Total Position: **{total_display_balance:,.6f} {vault_symbol}**"
                    ]
                    staked_balance = result.get("staked_balance", 0)
                    if staked_balance > 0:
                        wallet_balance = result.get("wallet_balance", 0)
                        wallet_display = wallet_balance / (10 ** decimals)
                        staked_display = staked_balance / (10 ** decimals)
                        if wallet_balance > 0:
                            breakdown = f"(Breakdown: {wallet_display:,.6f} liquid + {staked_display:,.6f} staked)"
                        else:
                            breakdown = "(Staked in gauge)"
                        deposit_lines.append(f"    {breakdown}")
                    chain_deposits.append("\n".join(deposit_lines))
                    total_deposits_found += 1
                except Exception as e:
                    logging.error(f"Error processing deposit for vault {vault_info.get('address')} on {chain_name}: {e}")

        if chain_deposits:
            all_results.append(f"**{chain_name.capitalize()} Active Deposits:**\n" + "\n\n".join(chain_deposits))
        elif chain:
             all_results.append(f"No active deposits found on {chain.capitalize()} for this address" + (f" matching token '{token_symbol}'." if token_symbol else "."))

    if total_deposits_found > 0:
        return "\n\n---\n\n".join(all_results)
    elif not chain:
        return "No active vault deposits found for that address on any supported Yearn chain" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    else:
        return "".join(all_results) if all_results else "No active vault deposits found."

async def query_v1_deposits_logic(resolved_address: str, token_symbol: Optional[str] = None) -> str:
    logging.info(f"[Logic:query_v1_deposits] Checking V1 for {resolved_address}, Token: {token_symbol}")
    if not V1_VAULTS:
        return "V1 vault data is not loaded."
    if "ethereum" not in WEB3_INSTANCES:
        return "Ethereum connection unavailable."

    web3_eth = WEB3_INSTANCES["ethereum"]
    try:
        user_checksum_addr = Web3.to_checksum_address(resolved_address)
    except ValueError:
         return f"Invalid Ethereum address format: {resolved_address}"

    found_deposits = []
    for vault in V1_VAULTS:
        if token_symbol and token_symbol.lower() not in vault.get("symbol", "").lower():
            continue
        try:
            vault_checksum_addr = Web3.to_checksum_address(vault.get("address"))
            contract = web3_eth.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)
            balance = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)

            if balance > 0:
                decimals = int(vault.get("decimals", 18))
                display_balance = balance / (10 ** decimals)
                etherscan_link = f"https://etherscan.io/address/{vault_checksum_addr}"
                vault_name = vault.get('name', 'Unknown V1 Vault')
                vault_symbol = vault.get('symbol', 'N/A')
                instruction_verb = "'withdrawAll'" if vault.get("withdraw_all", False) else "'withdraw' (entering your balance)"
                instr = (
                    f"**Vault:** [{vault_name}]({etherscan_link}) (Symbol: {vault_symbol})\n"
                    f"  Deposit Found: {display_balance:.6f} tokens.\n"
                    f"  *Withdrawal:* Go to Etherscan link -> 'Contract' tab -> 'Write Contract' -> Connect Wallet -> Use the {instruction_verb} function."
                )
                found_deposits.append(instr)
        except Exception as e:
            logging.error(f"[Logic:query_v1_deposits] Error checking V1 vault {vault.get('address')} for {user_checksum_addr}: {e}")

    if found_deposits:
        return "**Deprecated V1 Vault Deposits Found (Ethereum Only):**\n\n" + "\n\n".join(found_deposits)
    else:
        return "No deposits found in deprecated V1 vaults for this address."
    
async def core_check_all_deposits(user_address_or_ens: str, token_symbol: Optional[str] = None) -> str:
    """
    Checks for user deposits in BOTH active (v2/v3) and deprecated (v1) Yearn vaults across all supported chains.
    Provide the user's wallet address or ENS name. Optionally filter by token symbol (e.g., 'USDC').
    Returns a combined summary of any deposits found in either type of vault.
    """
    logging.info(f"[CoreTool:check_deposits] User: '{user_address_or_ens}'")
    resolved_address = resolve_ens(user_address_or_ens)
    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum address."

    # Run checks concurrently
    v1_task = asyncio.create_task(query_v1_deposits_logic(resolved_address, token_symbol))
    active_task = asyncio.create_task(query_active_deposits_logic(resolved_address, chain=None, token_symbol=token_symbol))

    v1_results, active_results = await asyncio.gather(v1_task, active_task)

    # Combine results
    final_output = []
    v1_found = "No deposits found in deprecated V1 vaults" not in v1_results
    active_found = "No active vault deposits found" not in active_results

    if v1_found:
        final_output.append(v1_results)
    if active_found:
        final_output.append(active_results)

    if not v1_found and not active_found:
        # If neither found anything, return a single message
        return f"No deposits found in any active or deprecated Yearn vaults for address {resolved_address}" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    elif not v1_found:
        # Only active found, maybe add a note about V1
        final_output.append("(No deposits found in deprecated V1 vaults)")
    elif not active_found:
         # Only V1 found, maybe add a note about active
         final_output.append("(No deposits found in active V2/V3 vaults)")

    combined_result = "\n\n---\n\n".join(final_output)
    logging.info(f"[CoreTool:check_deposits] Combined Result for {resolved_address}:\n{combined_result}")
    return combined_result

async def core_get_withdrawal_instructions(user_address_or_ens: Optional[str], vault_address: str, chain: str) -> str:
    """
    Generates step-by-step instructions for withdrawing from a specific Yearn vault (v1, v2, or v3) using a block explorer.
    Also provides the direct link to the vault on the Yearn website for reference (for v2/v3).
    Provide the vault's address, the chain name (e.g., 'ethereum', 'optimism', 'arbitrum'), and optionally the user's address/ENS.
    Use this when a user asks how to withdraw or reports issues using the Yearn website for a specific vault.
    """
    logging.info(f"[Tool:get_withdrawal_instructions] Args: User={user_address_or_ens}, Vault={vault_address}, Chain={chain}")

    # --- Step 1: Input Validation & Setup ---
    resolved_user_address = None
    user_checksum_addr = None
    if user_address_or_ens:
        resolved_user_address = resolve_ens(user_address_or_ens)
        if not resolved_user_address:
            logging.warning(f"Could not resolve provided user address/ENS: '{user_address_or_ens}'. Proceeding without it.")
        else:
             try:
                 user_checksum_addr = Web3.to_checksum_address(resolved_user_address)
             except ValueError:
                  logging.warning(f"Invalid format for resolved user address: '{resolved_user_address}'. Proceeding without it.")
                  user_checksum_addr = None

    chain_lower = chain.lower()
    web3_instance = WEB3_INSTANCES.get(chain_lower)
    chain_id = config.CHAIN_NAME_TO_ID.get(chain_lower)
    explorer_base_url = config.BLOCK_EXPLORER_URLS.get(chain_lower)

    if not web3_instance or not chain_id or not explorer_base_url:
        return f"Unsupported or invalid chain: '{chain}'. Cannot generate instructions."

    try:
        vault_checksum_addr = Web3.to_checksum_address(vault_address)
    except ValueError as e:
        return f"Invalid vault address format provided: {e}. Please provide a valid vault address."

    explorer_vault_url = f"{explorer_base_url}/address/{vault_checksum_addr}"

    # --- Step 2: Check V1 List (Ethereum Only) ---
    if chain_lower == 'ethereum':
        v1_vault_info = next((v for v in V1_VAULTS if v.get("address", "").lower() == vault_checksum_addr.lower()), None)

        if v1_vault_info:
            logging.info(f"Vault {vault_checksum_addr} identified as V1. Generating V1 instructions.")
            vault_name = v1_vault_info.get('name', vault_checksum_addr)

            # --- REVERTED V1 INSTRUCTION WORDING ---
            instructions = [
                f"This vault (**{vault_name}** `{vault_checksum_addr}`) is a **deprecated Yearn V1 vault**.",
                "Withdrawals must be done directly via the block explorer:",
                f"1. Go to the vault contract page: {explorer_vault_url}",
                "2. Click the **'Contract'** tab.",
                "3. Click the **'Write Contract'** tab.",
                f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}.",
                "5. Look for a suitable withdrawal function (often named like 'withdraw', 'withdrawAll', or similar). Prioritize functions that take no arguments if available.", # Generic guidance
                "6. Click the **'Write'** button next to the chosen function.",
                "7. Confirm the transaction in your wallet.",
                "\nOnce the transaction confirms, your funds should be back in your wallet."
            ]
            final_instructions = "\n".join(instructions)
            logging.info(f"Generated V1 instructions for {vault_checksum_addr}:\n{final_instructions}")
            return final_instructions
        else:
            logging.info(f"Vault {vault_checksum_addr} not found in V1 list. Proceeding to check V2/V3.")

    # --- Step 3: Fetch V2/V3 Details via yDaemon ---
    vault_details_json: Optional[Dict] = None
    api_url = "https://ydaemon.yearn.fi/vaults/detected?limit=2000" # Or a more targeted one if available
    async with aiohttp.ClientSession() as session:
        try:
            logging.info(f"[Tool:get_withdrawal_instructions] Fetching bulk data to find {vault_checksum_addr}")
            async with session.get(api_url, timeout=25) as response:
                response.raise_for_status()
                all_vaults = await response.json()
                if isinstance(all_vaults, list):
                    for v_data in all_vaults:
                        if v_data.get("address", "").lower() == vault_checksum_addr.lower() and v_data.get("chainID") == chain_id:
                            vault_details_json = v_data
                            break
                    if vault_details_json:
                        logging.info(f"Found details for {vault_checksum_addr} in bulk fetch.")
                    else:
                        logging.warning(f"Vault {vault_checksum_addr} not found in bulk fetch from {api_url}")
                else:
                    logging.error("Unexpected format from bulk vault fetch.")
        except Exception as e:
            logging.error(f"Error fetching bulk vault data for withdrawal tool: {e}")

    if not vault_details_json:
        logging.warning(f"Failed to fetch V2/V3 vault details for {vault_checksum_addr} on chain {chain_id} via yDaemon.")
        return (f"Could not fetch vault details from the Yearn API for `{vault_checksum_addr}` on {chain.capitalize()} "
                f"to determine the correct withdrawal method for V2/V3 vaults. \n"
                f"Please double-check the vault address and chain name. \n"
                f"You can view the contract directly here: {explorer_vault_url}")

    # --- Step 4: Determine V2/V3 Version and Format Instructions ---
    api_version_str = vault_details_json.get("version", "")
    vault_name = vault_details_json.get("name", vault_checksum_addr)
    yearn_ui_link = f"https://yearn.fi/vaults/{chain_id}/{vault_checksum_addr}"

    intro_message = (
        f"Okay, here are instructions for withdrawing from the **{vault_name}** vault (`{vault_checksum_addr}`) on **{chain.capitalize()}** using the block explorer.\n\n"
        f"You can also try the Yearn website here: {yearn_ui_link}\n\n"
        f"If using the block explorer:"
    )
    instructions = [intro_message]
    instructions.append(f"1. Go to the vault contract page: {explorer_vault_url}")
    instructions.append("2. Click the **'Contract'** tab.")
    write_tab_options = "'Write Contract as Proxy' (if available, otherwise 'Write Contract')"
    instructions.append(f"3. Click the **{write_tab_options}** tab.")
    instructions.append(f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}.")

    is_v3 = api_version_str.startswith("3.")
    is_v2 = api_version_str.startswith("0.")

    if is_v3:
        logging.info(f"Vault {vault_checksum_addr} identified as V3 (version: {api_version_str}). Generating 'redeem' instructions.")
        user_balance: Optional[int] = None
        balance_input_value = "**(Enter your full share balance manually)**"

        if user_checksum_addr:
            try:
                contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)
                balance_raw = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)
                user_balance = int(balance_raw)
                if user_balance == 0:
                    return (f"Confirmed user `{user_checksum_addr}` has zero balance in V3 vault `{vault_checksum_addr}`.\n"
                            f"Vault link for reference: {yearn_ui_link}\n"
                            f"No withdrawal needed or possible via explorer.")
                balance_input_value = f"`{user_balance}` (This should be your full share balance)"
            except Exception as e:
                logging.error(f"Failed to fetch user balance for V3 vault {vault_checksum_addr}: {e}")
                logging.warning("Generating V3 instructions without fetched balance.")
        else:
            logging.warning("User address not provided; cannot fetch V3 balance. Instructions require manual input.")

        instructions.extend([
            "5. Find the **'redeem'** function.",
            "6. Enter the following values:",
            f"   - `shares (uint256)`: {balance_input_value}",
            f"   - `receiver (address)`: {f'`{user_checksum_addr}` (Your wallet address)' if user_checksum_addr else '**(Your wallet address)**'}",
            f"   - `owner (address)`: {f'`{user_checksum_addr}` (Your wallet address again)' if user_checksum_addr else '**(Your wallet address again)**'}",
            "7. Click the **'Write'** button.",
            "8. Confirm the transaction in your wallet."
        ])

    elif is_v2:
        logging.info(f"Vault {vault_checksum_addr} identified as V2 (version: {api_version_str}). Generating 'withdraw' (no args) instructions.")
        instructions.extend([
            "5. Find the **'withdraw()'** function that takes **no arguments**.",
            "   *Important: Do NOT use a 'withdraw' function that asks for '_shares' or an amount.*",
            "6. Click the **'Write'** button next to that specific 'withdraw()' function.",
            "7. Confirm the transaction in your wallet."
        ])

    else:
        logging.warning(f"Vault {vault_checksum_addr} has unknown or missing V2/V3 version: '{api_version_str}'. Providing generic instructions.")
        balance_info = "(Cannot check balance without your address)"
        if user_checksum_addr:
            try:
                contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)
                balance_raw = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)
                user_balance = int(balance_raw)
                if user_balance == 0:
                    return (f"Confirmed user `{user_checksum_addr}` has zero balance in vault `{vault_checksum_addr}` (Version Unknown).\n"
                            f"Vault link for reference: {yearn_ui_link}\n"
                            f"No withdrawal needed or possible via explorer.")
                balance_info = f"`{user_balance}`"
            except Exception:
                logging.warning("Could not fetch balance for unknown version vault.")
                balance_info = "(Could not fetch balance)"

        instructions.extend([
            f"5. **Unknown Vault Version (API Version: {api_version_str or 'Not Found'})**: Look for a function named **'withdraw'** or **'redeem'**.",
            "   - Try **'withdraw()'** (with no input fields) first.",
            f"   - If that's not present or doesn't work, try **'redeem'**. If it asks for `shares`, `receiver`, and `owner`, enter your balance {balance_info} for `shares` and your address ({f'`{user_checksum_addr}`' if user_checksum_addr else 'Your Address'}) for `receiver` and `owner`.",
            "   - If unsure, please ask for human help again, mentioning the vault version is unclear.",
            "6. Click the **'Write'** button for the chosen function.",
            "7. Confirm the transaction in your wallet."
        ])

    instructions.append("\nOnce the transaction confirms on the blockchain, your funds should be back in your wallet.")
    final_instructions = "\n".join(instructions)
    logging.info(f"Generated V2/V3/Fallback instructions for {vault_checksum_addr}:\n{final_instructions}")
    return final_instructions

def format_timestamp_to_readable(timestamp: Optional[Union[int, float, str]]) -> str:
    if timestamp is None:
        return "N/A"
    try:
        dt_object = datetime.fromtimestamp(int(timestamp), timezone.utc)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, TypeError):
        return str(timestamp)

def format_single_vault_data_for_llm(data: Dict, chain_id_for_url: int) -> str:
    """
    Formats a single vault's JSON data into a readable string for the LLM.
    """
    output_lines = []
    name = data.get('name', 'N/A')
    symbol = data.get('symbol', 'N/A')
    address = data.get('address', 'N/A')
    api_version_str = data.get('version', 'Unknown')
    simplified_version = "Unknown"
    yearn_ui_link = "N/A"

    if api_version_str.startswith("3."):
        simplified_version = f"V3 (API: {api_version_str})"
        yearn_ui_link = f"https://yearn.fi/vaults/{chain_id_for_url}/{address}"
    elif api_version_str.startswith("0."):
        simplified_version = f"V2 (API: {api_version_str})"
        yearn_ui_link = f"https://yearn.fi/vaults/{chain_id_for_url}/{address}"

    output_lines.append(f"Vault: {name} ({symbol})")
    output_lines.append(f"Address: `{address}`")
    if yearn_ui_link != "N/A":
        output_lines.append(f"Yearn UI Link: {yearn_ui_link}")
    output_lines.append(f"Version: {simplified_version}")
    output_lines.append(f"Kind: {data.get('kind', 'N/A')}")
    description = data.get('description', 'No description available.')
    if description and len(description) > 250:
        description = description[:247] + "..."
    output_lines.append(f"Description: {description}")

    token_info = data.get('token', {})
    underlying_name = token_info.get('name', 'N/A')
    underlying_symbol = token_info.get('symbol', 'N/A')
    underlying_address = token_info.get('address', 'N/A')
    tvl_data = data.get('tvl', {})
    underlying_price = tvl_data.get('price', 0.0)
    output_lines.append(f"Underlying Token: {underlying_name} ({underlying_symbol}) - `{underlying_address}` - Price: ${underlying_price:,.4f}")
    output_lines.append("")

    output_lines.append("TVL & Share Price:")
    output_lines.append(f"  TVL (USD): ${tvl_data.get('tvl', 0.0):,.2f}")
    raw_pps = data.get('pricePerShare', '0')
    try:
        vault_decimals = int(data.get('decimals', 18))
        scaled_pps = float(raw_pps) / (10**vault_decimals)
        output_lines.append(f"  Vault Token Price Per Share (in underlying): {scaled_pps:.6f} (Raw: {raw_pps})")
    except (ValueError, TypeError):
        output_lines.append(f"  Vault Token Price Per Share (Raw): {raw_pps}")
    output_lines.append("")

    apr_data = data.get('apr', {})
    output_lines.append("APY Information:")
    net_apy = apr_data.get('netAPR', 0.0) * 100
    output_lines.append(f"  Current Net APY (compounded): {net_apy:.2f}% (Type: {apr_data.get('type', 'N/A')})")
    
    forward_apr_data = apr_data.get('forwardAPR', {})
    if forward_apr_data and forward_apr_data.get('netAPR') is not None:
        forward_net_apy = forward_apr_data.get('netAPR', 0.0) * 100
        output_lines.append(f"  Estimated Forward APY (projection): {forward_net_apy:.2f}% (Type: {forward_apr_data.get('type', 'N/A')})")
    
    fees = apr_data.get('fees', {})
    perf_fee = fees.get('performance', 0.0) * 100
    mgmt_fee = fees.get('management', 0.0) * 100
    output_lines.append(f"  Vault Fees: Performance={perf_fee:.2f}%, Management={mgmt_fee:.2f}%")
    
    points = apr_data.get('points', {})
    week_ago_apy = points.get('weekAgo', 0.0) * 100
    month_ago_apy = points.get('monthAgo', 0.0) * 100
    inception_apy = points.get('inception', 0.0) * 100
    output_lines.append(f"  Historical Net APY: Week Ago={week_ago_apy:.2f}%, Month Ago={month_ago_apy:.2f}%, Inception={inception_apy:.2f}%")
    output_lines.append("")

    output_lines.append("Other Info:")
    output_lines.append(f"  Featuring Score: {data.get('featuringScore', 'N/A')}")
    info_obj = data.get('info', {})
    output_lines.append(f"  Risk Level: {info_obj.get('riskLevel', 'N/A')}")
    output_lines.append(f"  Status Flags: Retired={info_obj.get('isRetired', False)}, Boosted={info_obj.get('isBoosted', False)}, Highlighted={info_obj.get('isHighlighted', False)}")
    migration_data = data.get('migration', {})
    output_lines.append(f"  Migration Available: {migration_data.get('available', False)}")
    if migration_data.get('available', False):
        output_lines.append(f"    Migration Target Address: `{migration_data.get('address', 'N/A')}`")
    output_lines.append("")

    strategies = data.get('strategies', [])
    if strategies:
        output_lines.append(f"Strategies ({len(strategies)}):")
        for i, strat in enumerate(strategies):
            strat_name = strat.get('name', 'Unnamed Strategy')
            strat_addr = strat.get('address', 'N/A')
            strat_status = strat.get('status', 'N/A')
            strat_apy = strat.get('netAPR', 0.0) * 100
            strat_details = strat.get('details', {})
            debt_ratio_raw = strat_details.get('debtRatio')
            debt_ratio_percent = "N/A"
            if debt_ratio_raw is not None:
                try:
                    debt_ratio_percent = f"{float(debt_ratio_raw) / 100:.2f}%"
                except (ValueError, TypeError):
                    pass
            last_report = format_timestamp_to_readable(strat_details.get('lastReport'))
            output_lines.append(f"  {i+1}. Name: {strat_name} (`{strat_addr}`)")
            output_lines.append(f"     Status: {strat_status}")
            output_lines.append(f"     Individual APY: {strat_apy:.2f}%")
            output_lines.append(f"     Allocation (Debt Ratio): {debt_ratio_percent}")
            output_lines.append(f"     Last Report: {last_report}")
    else:
        output_lines.append("Strategies: None listed.")
    output_lines.append("")

    staking_info = data.get('staking')
    if staking_info and staking_info.get('available'):
        output_lines.append("Staking Opportunity: Yes")
        output_lines.append(f"  Source: {staking_info.get('source', 'N/A')}")
        output_lines.append(f"  Staking Contract: `{staking_info.get('address', 'N/A')}`")
        rewards_list = staking_info.get('rewards', [])
        if rewards_list:
            output_lines.append(f"  Rewards ({len(rewards_list)}):")
            for rew_idx, reward in enumerate(rewards_list):
                rew_name = reward.get('name', 'N/A')
                rew_sym = reward.get('symbol', 'N/A')
                rew_addr = reward.get('address', 'N/A')
                rew_apy = reward.get('apr', 0.0) * 100
                rew_finished = reward.get('isFinished', False)
                rew_ends = format_timestamp_to_readable(reward.get('finishedAt'))
                output_lines.append(f"    - Token: {rew_name} ({rew_sym}) `{rew_addr}`")
                output_lines.append(f"      APY: {rew_apy:.2f}%")
                output_lines.append(f"      Status: {'Finished' if rew_finished else 'Ongoing'} (Ends: {rew_ends})")
        else:
            output_lines.append("  Rewards: None listed.")
    else:
        output_lines.append("Staking Opportunity: No")
    
    return "\n".join(output_lines)

# --- Core Search Function ---

async def core_search_vaults(
    query: str,
    chain: Optional[str] = None,
    sort_by: Optional[str] = None,
    recommended_only: bool = False,
) -> str:
    """
    Core logic to search for Yearn vaults.
    """
    logging.info(
        "[CoreTool:search_vaults] Query: '%s', Chain: '%s', Sort By: '%s', Recommended Only: %s",
        query,
        chain,
        sort_by,
        recommended_only,
    )
    api_url = "https://ydaemon.yearn.fi/vaults/detected?limit=2000"
    
    # Use config for max results if defined, else default
    MAX_RESULTS = getattr(config, 'MAX_RESULTS_TO_SHOW', 5)

    async with aiohttp.ClientSession() as session:
        try:
            logging.info(f"[Tool:search_vaults] Fetching data from {api_url}")
            async with session.get(api_url, timeout=25) as response:
                response.raise_for_status()
                all_vaults_data_list = await response.json()
        except Exception as e:
            logging.error(f"[Tool:search_vaults] Error during yDaemon fetch: {e}", exc_info=True)
            return f"Error: An unexpected error occurred while fetching vault data: {e}."

        # --- Filtering ---
        filtered_vaults = all_vaults_data_list
        
        if chain:
            chain_lower = chain.lower()
            query_chain_id = config.CHAIN_NAME_TO_ID.get(chain_lower)
            if query_chain_id:
                filtered_vaults = [v for v in filtered_vaults if v.get("chainID") == query_chain_id]

        query_lower = query.lower().strip()
        matched_vaults = []
        recommendation_fallback_vaults = []
        is_address_query = Web3.is_address(query_lower)
        match_all_vaults = query_lower in {"all", "*"}

        def _is_recommendable_vault(v_data: dict) -> bool:
            symbol = (v_data.get("symbol") or "").lower()
            kind = (v_data.get("kind") or "").lower()
            strategies = v_data.get("strategies") or []
            if symbol.startswith("ys"):
                return False
            if "single strategy" in kind:
                return False
            if len(strategies) <= 1:
                return False
            return True

        def _recommendation_sort_key(v_data: dict) -> tuple[float, float, float, float]:
            info_obj = v_data.get("info", {})
            featuring_score = float(v_data.get("featuringScore") or 0.0)
            risk_level = info_obj.get("riskLevel")
            try:
                risk_score = -float(risk_level)
            except (TypeError, ValueError):
                risk_score = float("-inf")
            return (
                featuring_score,
                risk_score,
                v_data.get("_computedTVL_USD", 0.0),
                v_data.get("_computedAPY", 0.0),
            )

        for v_data in filtered_vaults:
            vault_address = v_data.get("address", "").lower()
            name = v_data.get("name", "").lower()
            symbol = v_data.get("symbol", "").lower()
            token_info = v_data.get("token", {})
            token_name = token_info.get("name", "").lower() if token_info else ""
            token_symbol = token_info.get("symbol", "").lower() if token_info else ""
            underlying_address = token_info.get("address", "").lower() if token_info else ""

            match = False
            if match_all_vaults:
                match = True
            elif is_address_query:
                if query_lower == vault_address or query_lower == underlying_address:
                    match = True
            elif query_lower == symbol or query_lower == token_symbol:
                match = True
            elif query_lower in name or query_lower in token_name:
                match = True
            
            if match:
                # Pre-calc sort keys
                apr_data = v_data.get("apr", {})
                primary_apr = apr_data.get("netAPR")
                forward_apr_data = apr_data.get("forwardAPR", {})
                fallback_apr = forward_apr_data.get("netAPR") if forward_apr_data else None
                apr_value = 0.0
                if primary_apr is not None:
                    apr_value = float(primary_apr)
                elif fallback_apr is not None:
                    apr_value = float(fallback_apr)
                v_data["_computedAPY"] = apr_value * 100
                try:
                    v_data["_computedTVL_USD"] = float(v_data.get('tvl', {}).get('tvl', 0))
                except (ValueError, TypeError):
                    v_data["_computedTVL_USD"] = 0.0
                recommendation_fallback_vaults.append(v_data)
                if recommended_only and not _is_recommendable_vault(v_data):
                    continue
                matched_vaults.append(v_data)

        if recommended_only and not matched_vaults:
            matched_vaults = recommendation_fallback_vaults

        if not matched_vaults:
            return "No active Yearn vaults found matching your criteria."

        # --- Sorting ---
        if recommended_only and sort_by not in {"highest_apr", "lowest_apr"}:
            matched_vaults.sort(key=_recommendation_sort_key, reverse=True)
        elif sort_by == "highest_apr":
            matched_vaults.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=True)
        elif sort_by == "lowest_apr":
            matched_vaults.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=False)
        else: # Default sort by TVL descending
            matched_vaults.sort(key=lambda v: v.get("_computedTVL_USD", 0.0), reverse=True)

        top_vaults = matched_vaults[:MAX_RESULTS]

        # --- Format ---
        formatted_strings = []
        for vault_data in top_vaults:
            chain_id = vault_data.get("chainID")
            if chain_id:
                formatted_text = format_single_vault_data_for_llm(vault_data, chain_id)
                formatted_strings.append(formatted_text)
            else:
                formatted_strings.append(f"Partial info for Vault: {vault_data.get('name', 'N/A')}")

        if not formatted_strings:
             return "Found matching vault(s), but could not format their details."

        # --- Assemble ---
        num_total = len(matched_vaults)
        num_shown = len(formatted_strings)
        sort_desc = sort_by if sort_by else "TVL (Descending)"
        header = f"Found {num_total} Yearn vault(s) matching '{query}'."
        if num_total > num_shown:
             header += f" Showing top {num_shown} (sorted by {sort_desc}) with details:"
        
        return header + "\n\n---\n\n" + "\n\n---\n\n".join(formatted_strings)
