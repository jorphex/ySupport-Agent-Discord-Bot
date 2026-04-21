from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

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
    category: str | None = None,
    token_symbol: str | None = None,
    universe: str = "core",
    sort_by: str = "tvl",
    limit: int = 10,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/discover",
        {
            "chain_id": chain_id,
            "category": category,
            "token_symbol": token_symbol,
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
            "name": row.get("name"),
            "symbol": row.get("symbol"),
            "token_symbol": row.get("token_symbol"),
            "category": row.get("category"),
            "tvl_usd": row.get("tvl_usd"),
            "est_apy": row.get("est_apy"),
            "safe_apy_30d": row.get("safe_apy_30d"),
            "realized_apy_30d": row.get("realized_apy_30d"),
            "strategies_count": row.get("strategies_count"),
            "last_point_time": row.get("last_point_time"),
            "regime": row.get("regime"),
            "is_retired": row.get("is_retired"),
            "migration_available": row.get("migration_available"),
        }
        for row in rows[:limit]
    ]
    return _json_block(
        "Support dashboard discover result",
        {
            "source": "/api/discover",
            "filters": payload.get("filters"),
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
        "/api/harvests",
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
            "source": "/api/harvests",
            "generated_at_utc": payload.get("generated_at_utc"),
            "scope": payload.get("scope"),
            "filters": payload.get("filters"),
            "trailing_24h": payload.get("trailing_24h"),
            "chain_rollups": payload.get("chain_rollups"),
            "recent": compact_recent,
            "last_run": payload.get("last_run"),
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
            "filters": payload.get("filters"),
            "summary": payload.get("summary"),
            "freshness": payload.get("freshness"),
            "risers": (movers.get("risers") or [])[:limit],
            "fallers": (movers.get("fallers") or [])[:limit],
            "largest_abs_delta": (movers.get("largest_abs_delta") or [])[:limit],
            "stale": (payload.get("stale") or [])[:limit],
        },
    )


async def core_support_dashboard_token_venues(
    *,
    token_symbol: str,
    universe: str = "core",
) -> str:
    payload = await _fetch_dashboard_json(
        f"/api/assets/{token_symbol}/venues",
        {
            "universe": universe,
        },
    )
    rows = payload.get("rows") or []
    compact_rows = [
        {
            "vault_address": row.get("vault_address"),
            "chain_id": row.get("chain_id"),
            "name": row.get("name"),
            "symbol": row.get("symbol"),
            "category": row.get("category"),
            "tvl_usd": row.get("tvl_usd"),
            "est_apy": row.get("est_apy"),
            "safe_apy_30d": row.get("safe_apy_30d"),
            "realized_apy_30d": row.get("realized_apy_30d"),
            "regime": row.get("regime"),
            "last_point_time": row.get("last_point_time"),
        }
        for row in rows
    ]
    return _json_block(
        f"Support dashboard venues for {token_symbol}",
        {
            "source": f"/api/assets/{token_symbol}/venues",
            "filters": payload.get("filters"),
            "summary": payload.get("summary"),
            "rows": compact_rows,
        },
    )


async def core_support_dashboard_styfi(
    *,
    days: int = 30,
    epoch_limit: int = 12,
    chain_id: int = 1,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/styfi",
        {
            "days": days,
            "epoch_limit": epoch_limit,
            "chain_id": chain_id,
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
