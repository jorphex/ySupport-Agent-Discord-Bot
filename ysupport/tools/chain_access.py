import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import json
import threading
import time
from typing import Any, Optional

from web3 import Web3

from ysupport import config


LOGGER = logging.getLogger(__name__)

WEB3_INSTANCES: dict[str, Web3] = {}
_WEB3_BOOTSTRAPPED = False


class RpcReadBudget:
    """A turn's RPC allowance, shared with its worker threads."""

    READ_METHODS = frozenset({
        "web3_clientVersion", "net_version", "eth_chainId", "eth_blockNumber",
        "eth_call", "eth_estimateGas", "eth_getBalance", "eth_getCode",
        "eth_getTransactionCount", "eth_getTransactionReceipt",
        "eth_getTransactionByHash", "eth_getBlockByNumber", "eth_getBlockByHash",
        "eth_getLogs",
        "eth_getStorageAt", "eth_gasPrice", "eth_maxPriorityFeePerGas", "eth_feeHistory",
    })
    DIRECT_READ_METHODS = READ_METHODS - {
        "web3_clientVersion", "net_version", "eth_call", "eth_estimateGas",
        "eth_getTransactionReceipt", "eth_getTransactionByHash", "eth_getLogs",
    }

    def __init__(self, *, max_calls: int, timeout_seconds: float):
        self.max_calls = max_calls
        self.deadline = time.monotonic() + timeout_seconds
        self.used = 0
        self.closed = False
        self.exhausted = False
        self._lock = threading.Lock()

    def consume(self, method, params):
        with self._lock:
            if method not in self.READ_METHODS:
                raise ValueError("Internal RPC permits read methods only.")
            if method in {"eth_getBlockByNumber", "eth_getBlockByHash"} and len(params) > 1 and params[1] is not False:
                raise ValueError("Internal block reads return transaction hashes, not full transactions.")
            if method == "eth_feeHistory":
                count = int(params[0], 16) if isinstance(params[0], str) else int(params[0])
                if not 1 <= count <= 100 or (len(params) > 2 and len(params[2]) > 10):
                    raise ValueError("Internal fee history is limited to 100 blocks and ten percentiles.")
            if method == "eth_getLogs":
                query = params[0]
                start, end = query.get("fromBlock", "latest"), query.get("toBlock", "latest")
                if not query.get("address"):
                    raise ValueError("Internal log queries require a contract address.")
                if not (start == end == "latest"):
                    try:
                        start = int(start, 16) if isinstance(start, str) else int(start)
                        end = int(end, 16) if isinstance(end, str) else int(end)
                    except (TypeError, ValueError):
                        raise ValueError("Internal log queries require a bounded numeric range.") from None
                    if start < 0 or end < start or end - start >= 5000:
                        raise ValueError("Internal log range must contain at most 5000 blocks.")
            if self.closed or time.monotonic() >= self.deadline or self.used >= self.max_calls:
                self.exhausted = True
                raise RuntimeError("Internal RPC budget exhausted; observation is incomplete.")
            self.used += 1

    def close(self):
        with self._lock:
            self.closed = True


_RPC_READ_BUDGET = ContextVar("ysupport_rpc_read_budget", default=None)


@contextmanager
def rpc_read_scope(budget):
    token = _RPC_READ_BUDGET.set(budget)
    try:
        yield
    finally:
        _RPC_READ_BUDGET.reset(token)


class InternalRpcHttpProvider(Web3.HTTPProvider):
    def make_request(self, method, params):
        budget = _RPC_READ_BUDGET.get()
        if budget is not None:
            budget.consume(method, params)
        return super().make_request(method, params)

    def make_batch_request(self, batch_requests):
        budget = _RPC_READ_BUDGET.get()
        if budget is not None:
            for method, params in batch_requests:
                budget.consume(method, params)
        return super().make_batch_request(batch_requests)


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
                web3_instance = Web3(InternalRpcHttpProvider(
                    url, request_kwargs={"timeout": 15},
                    exception_retry_configuration=None,
                ))
                WEB3_INSTANCES[name] = web3_instance
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


async def core_read_chain_state(*, chain: str, method: str, params_json: str) -> str:
    if method not in RpcReadBudget.DIRECT_READ_METHODS:
        raise ValueError("Unsupported basic chain-state read; use specialised helpers for transactions, contract calls and logs.")
    params = json.loads(params_json)
    if not isinstance(params, list):
        raise ValueError("Chain-state params_json must contain a JSON array.")
    web3_instance = get_web3_instance(chain)
    if web3_instance is None:
        raise ValueError("Unsupported chain for internal state reads.")
    result = await asyncio.to_thread(web3_instance.manager.request_blocking, method, params)
    return Web3.to_json({"chain": chain, "method": method, "result": result})


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
    code = await asyncio.to_thread(
        web3_instance.eth.get_code, checksum_address,
        **({"block_identifier": block_identifier} if block_identifier is not None else {}),
    )
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
