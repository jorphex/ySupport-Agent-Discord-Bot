import asyncio
import logging
from typing import Any, Optional

from web3 import Web3

import config


LOGGER = logging.getLogger(__name__)

WEB3_INSTANCES: dict[str, Web3] = {}
_WEB3_BOOTSTRAPPED = False


def _normalize_chain(chain: str) -> str:
    return (chain or "").strip().lower()


def ensure_web3_instances() -> dict[str, Web3]:
    global _WEB3_BOOTSTRAPPED
    if _WEB3_BOOTSTRAPPED and WEB3_INSTANCES:
        return WEB3_INSTANCES

    if not _WEB3_BOOTSTRAPPED and WEB3_INSTANCES:
        _WEB3_BOOTSTRAPPED = True
        return WEB3_INSTANCES

    if not _WEB3_BOOTSTRAPPED:
        for name, url in config.RPC_URLS.items():
            try:
                web3_instance = Web3(Web3.HTTPProvider(url))
                WEB3_INSTANCES[name] = web3_instance
                if not web3_instance.is_connected():
                    LOGGER.warning("Failed to connect to Web3 for %s", name)
            except Exception as exc:
                LOGGER.warning("Error initializing Web3 for %s: %s", name, exc)
        _WEB3_BOOTSTRAPPED = True

    return WEB3_INSTANCES


def get_web3_instances() -> dict[str, Web3]:
    return ensure_web3_instances()


def get_web3_instance(chain: str | None) -> Web3 | None:
    normalized_chain = _normalize_chain(chain or "")
    if not normalized_chain:
        return None
    return ensure_web3_instances().get(normalized_chain)


def resolve_ens_name(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None

    address_to_check = name.strip()
    if ":" in address_to_check:
        parts = address_to_check.split(":", 1)
        if len(parts) == 2:
            prefix, potential_address = parts
            LOGGER.info(
                "Detected prefixed address. Prefix: '%s', Address part: '%s'",
                prefix,
                potential_address,
            )
            address_to_check = potential_address.strip()

    if Web3.is_address(address_to_check):
        try:
            return Web3.to_checksum_address(address_to_check)
        except ValueError:
            LOGGER.warning("Address '%s' has invalid checksum.", address_to_check)
            return None

    ethereum_web3 = get_web3_instance("ethereum")
    if address_to_check.endswith(".eth") and ethereum_web3 is not None:
        try:
            LOGGER.info("Attempting to resolve ENS name: '%s'", address_to_check)
            resolved = ethereum_web3.ens.address(address_to_check)
            if resolved and Web3.is_address(resolved):
                LOGGER.info(
                    "Successfully resolved ENS '%s' to '%s'",
                    address_to_check,
                    resolved,
                )
                return Web3.to_checksum_address(resolved)
            LOGGER.warning(
                "ENS name '%s' did not resolve to a valid address.",
                address_to_check,
            )
            return None
        except Exception as exc:
            LOGGER.error("Error resolving ENS '%s': %s", address_to_check, exc)
            return None

    LOGGER.warning("Input '%s' could not be resolved to a valid address.", name)
    return None


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


async def inspect_contract_profile(
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
