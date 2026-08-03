from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote, urlencode

import aiohttp

import config


def _dashboard_url(path: str, params: dict[str, Any]) -> str:
    base = config.SUPPORT_DASHBOARD_BASE_URL.rstrip("/")
    if not base:
        raise RuntimeError(
            "SUPPORT_DASHBOARD_BASE_URL is not configured for support dashboard tools."
        )
    query = urlencode(
        {
            key: value
            for key, value in params.items()
            if value is not None and value != ""
        }
    )
    if not query:
        return f"{base}{path}"
    return f"{base}{path}?{query}"


async def _fetch_dashboard_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = _dashboard_url(path, params)
    timeout = aiohttp.ClientTimeout(total=config.SUPPORT_DASHBOARD_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(ssl=config.SUPPORT_DASHBOARD_VERIFY_SSL)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            payload = await response.json()
            if not isinstance(payload, dict):
                raise TypeError(f"Dashboard endpoint {path} returned non-object JSON.")
            return payload


def _json_block(title: str, payload: dict[str, Any]) -> str:
    return f"{title}\n{json.dumps(payload, indent=2, sort_keys=True)}"


async def core_support_dashboard_discover(
    *,
    chain_id: int | None = None,
    market: str = "all",
    universe: str = "core",
    sort_by: str = "tvl",
    limit: int = 10,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/discover",
        {
            "chain_id": chain_id,
            "market": market,
            "universe": universe,
            "sort_by": sort_by,
            "limit": limit,
        },
    )
    rows = payload.get("rows") or []
    compact_rows = [
        {
            "vault_address": row.get("vault_address"),
            "chain_id": row.get("chain_id"),
            "symbol": row.get("symbol"),
            "tvl_usd": row.get("tvl_usd"),
            "est_apy": row.get("est_apy"),
            "realized_apy_30d": row.get("realized_apy_30d"),
            "momentum_7d_30d": row.get("momentum_7d_30d"),
            "market": row.get("market"),
        }
        for row in rows[:limit]
    ]
    return _json_block(
        "Support dashboard discover result",
        {
            "source": "/api/discover",
            "filters": payload.get("filters"),
            "realized_apy_policy": payload.get("realized_apy_policy"),
            "pagination": payload.get("pagination"),
            "summary": payload.get("summary"),
            "coverage": payload.get("coverage"),
            "rows": compact_rows,
        },
    )


async def core_support_dashboard_harvests(
    *,
    days: int = 30,
    chain_id: int | None = None,
    vault_address: str | None = None,
    limit: int = 20,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/reports",
        {
            "days": days,
            "chain_id": chain_id,
            "vault_address": vault_address,
            "limit": limit,
        },
    )
    recent = payload.get("recent") or []
    compact_recent = [
        {
            "block_time": row.get("block_time"),
            "tx_hash": row.get("tx_hash"),
            "vault_address": row.get("vault_address"),
            "vault_symbol": row.get("vault_symbol"),
            "token_symbol": row.get("token_symbol"),
            "strategy_address": row.get("strategy_address"),
            "gain": row.get("gain"),
            "loss": row.get("loss"),
            "debt_after": row.get("debt_after"),
            "fee_assets": row.get("fee_assets"),
            "refund_assets": row.get("refund_assets"),
        }
        for row in recent[:limit]
    ]
    return _json_block(
        "Support dashboard harvest history",
        {
            "source": "/api/reports",
            "filters": {
                "days": days,
                "chain_id": chain_id,
                "vault_address": vault_address,
                "limit": limit,
            },
            "event": payload.get("event"),
            "trailing_24h": payload.get("trailing_24h"),
            "available_chains": payload.get("available_chains"),
            "recent": compact_recent,
        },
    )


async def core_support_dashboard_changes(
    *,
    window: str = "7d",
    universe: str = "core",
    limit: int = 10,
    stale_threshold: str = "auto",
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/changes",
        {
            "window": window,
            "universe": universe,
            "limit": limit,
            "stale_threshold": stale_threshold,
        },
    )
    movers = payload.get("movers") or {}
    return _json_block(
        "Support dashboard recent changes",
        {
            "source": "/api/changes",
            "window": payload.get("window"),
            "realized_apy_policy": payload.get("realized_apy_policy"),
            "summary": payload.get("summary"),
            "freshness": payload.get("freshness"),
            "risers": (movers.get("risers") or [])[:limit],
            "fallers": (movers.get("fallers") or [])[:limit],
        },
    )


async def core_support_dashboard_token_venues(
    *,
    token_symbol: str,
    universe: str = "core",
) -> str:
    normalized_symbol = token_symbol.strip()
    if not normalized_symbol:
        raise ValueError("A token symbol is required for dashboard venue lookup.")
    symbol_path = quote(normalized_symbol, safe="")
    source_path = f"/api/assets/{symbol_path}/vaults"
    payload = await _fetch_dashboard_json(
        source_path,
        {
            "universe": universe,
            "limit": 25,
        },
    )
    rows = payload.get("rows") or []
    compact_rows = [
        {
            "vault_address": row.get("vault_address"),
            "chain_id": row.get("chain_id"),
            "symbol": row.get("symbol"),
            "tvl_usd": row.get("tvl_usd"),
            "est_apy": row.get("est_apy"),
            "realized_apy_30d": row.get("realized_apy_30d"),
            "momentum_7d_30d": row.get("momentum_7d_30d"),
        }
        for row in rows
    ]
    return _json_block(
        f"Support dashboard venues for {normalized_symbol}",
        {
            "source": source_path,
            "token_symbol": payload.get("token_symbol"),
            "identity": payload.get("identity"),
            "filters": payload.get("filters"),
            "realized_apy_policy": payload.get("realized_apy_policy"),
            "summary": payload.get("summary"),
            "rows": compact_rows,
        },
    )


async def core_support_dashboard_styfi(
    *,
    days: int = 30,
    epoch_limit: int = 12,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/styfi",
        {
            "days": days,
            "epoch_limit": epoch_limit,
        },
    )
    series = payload.get("series") or {}
    snapshots = series.get("snapshots") or []
    return _json_block(
        "Support dashboard stYFI status",
        {
            "source": "/api/styfi",
            "filters": payload.get("filters"),
            "summary": payload.get("summary"),
            "reward_token": payload.get("reward_token"),
            "current_reward_state": payload.get("current_reward_state"),
            "latest_snapshots": snapshots[-5:],
        },
    )
