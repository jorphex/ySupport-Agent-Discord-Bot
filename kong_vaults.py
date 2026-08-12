from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

import aiohttp


KONG_BASE_URL = "https://kong.yearn.fi"
_CACHE_TTL_SECONDS = 300.0
_VAULT_CATALOG_CACHE: dict[str, Any] = {"fetched_at": 0.0, "value": None}
_ADDRESS_CATALOG_CACHE: dict[str, Any] = {"fetched_at": 0.0, "value": None}

_ADDRESS_CATALOG_QUERY = """
query YSupportAddressCatalog {
  vaults(yearn: true) {
    chainId
    address
    name
    symbol
    strategies
    staking { address available source }
  }
  strategies {
    chainId
    address
    name
  }
}
"""


def _cached_value(cache: dict[str, Any]) -> Any | None:
    age = monotonic() - float(cache["fetched_at"] or 0.0)
    return cache.get("value") if age < _CACHE_TTL_SECONDS else None


def _store_cached_value(cache: dict[str, Any], value: Any) -> Any:
    cache["value"] = value
    cache["fetched_at"] = monotonic()
    return value


def is_yearn_vault(vault: dict[str, Any]) -> bool:
    """Use Kong's canonical origin marker rather than name/symbol heuristics."""
    return vault.get("origin") == "yearn"


async def fetch_kong_vault_catalog() -> list[dict[str, Any]]:
    cached = _cached_value(_VAULT_CATALOG_CACHE)
    if isinstance(cached, list):
        return cached

    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{KONG_BASE_URL}/api/rest/list/vaults") as response:
            response.raise_for_status()
            payload = await response.json()
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("Unexpected Kong vault catalog response.")

    vaults = [row for row in payload if is_yearn_vault(row)]
    return _store_cached_value(_VAULT_CATALOG_CACHE, vaults)


async def _fetch_kong_vault_snapshot(
    session: aiohttp.ClientSession,
    chain_id: int,
    address: str,
) -> dict[str, Any] | None:
    url = f"{KONG_BASE_URL}/api/rest/snapshot/{chain_id}/{address}"
    async with session.get(url) as response:
        if response.status == 404:
            return None
        response.raise_for_status()
        payload = await response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected Kong vault snapshot response.")
    return payload


async def fetch_kong_vault_snapshot(
    chain_id: int,
    address: str,
) -> dict[str, Any] | None:
    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await _fetch_kong_vault_snapshot(session, chain_id, address)


async def fetch_kong_vault_snapshots(
    targets: list[tuple[int, str]],
) -> list[dict[str, Any] | None | BaseException]:
    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(
            *(
                _fetch_kong_vault_snapshot(session, chain_id, address)
                for chain_id, address in targets
            ),
            return_exceptions=True,
        )


async def fetch_kong_address_catalog() -> dict[str, list[dict[str, Any]]]:
    cached = _cached_value(_ADDRESS_CATALOG_CACHE)
    if isinstance(cached, dict):
        return cached

    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{KONG_BASE_URL}/api/gql",
            json={"query": _ADDRESS_CATALOG_QUERY},
        ) as response:
            response.raise_for_status()
            payload = await response.json()

    if not isinstance(payload, dict) or payload.get("errors"):
        raise ValueError("Kong address catalog query failed.")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Unexpected Kong address catalog response.")
    vaults = data.get("vaults")
    strategies = data.get("strategies")
    if (
        not isinstance(vaults, list)
        or not isinstance(strategies, list)
        or any(not isinstance(row, dict) for row in vaults)
        or any(not isinstance(row, dict) for row in strategies)
    ):
        raise ValueError("Unexpected Kong address catalog response.")

    return _store_cached_value(
        _ADDRESS_CATALOG_CACHE,
        {"vaults": vaults, "strategies": strategies},
    )
