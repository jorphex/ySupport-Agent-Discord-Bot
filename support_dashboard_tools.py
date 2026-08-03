from __future__ import annotations

import asyncio
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
        for attempt in range(2):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    payload = await response.json()
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"Dashboard endpoint {path} returned non-object JSON."
                        )
                    return payload
            except aiohttp.ClientResponseError as exc:
                if attempt or exc.status < 500:
                    raise
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt:
                    raise
    raise RuntimeError(f"Dashboard endpoint {path} failed without an error.")


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
            "direction": "desc",
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


async def core_support_dashboard_reports(
    *,
    chain_id: int,
    vault_address: str,
    days: int = 30,
    limit: int = 50,
) -> str:
    normalized_address = vault_address.strip()
    if not normalized_address:
        raise ValueError("A vault address is required for dashboard report lookup.")
    payload = await _fetch_dashboard_json(
        "/api/reports",
        {
            "chain_id": chain_id,
            "vault_address": normalized_address,
            "days": days,
            "limit": limit,
            "meaningful_only": False,
        },
    )
    recent = payload.get("recent") or []
    compact_recent = [
        {
            "block_time": row.get("block_time"),
            "tx_hash": row.get("tx_hash"),
            "log_index": row.get("log_index"),
            "chain_id": row.get("chain_id"),
            "vault_address": row.get("vault_address"),
            "vault_symbol": row.get("vault_symbol"),
            "token_symbol": row.get("token_symbol"),
            "strategy_address": row.get("strategy_address"),
            "strategy_name": row.get("strategy_name"),
            "report_type": row.get("report_type"),
            "gain": row.get("gain"),
            "loss": row.get("loss"),
            "fee_assets": row.get("fee_assets"),
            "refund_assets": row.get("refund_assets"),
            "token_decimals": row.get("token_decimals"),
            "vault_version": row.get("vault_version"),
            "debt_after": row.get("debt_after"),
        }
        for row in recent[:limit]
    ]
    return _json_block(
        "Support dashboard vault reports",
        {
            "source": "/api/reports",
            "filters": {
                "chain_id": chain_id,
                "vault_address": normalized_address,
                "days": days,
                "limit": limit,
                "meaningful_only": False,
            },
            "interpretation": {
                "event_semantics": (
                    "Each row proves an on-chain StrategyReported event, not an "
                    "off-chain harvest job or realized profit."
                ),
                "amount_units": (
                    "gain, loss, fee_assets, refund_assets, and debt_after are raw "
                    "token-unit integer strings; scale by token_decimals. Null means "
                    "unavailable, not zero."
                ),
                "history_limit": (
                    "Only the newest matching rows are returned and the endpoint has no "
                    "continuation token. Absence of an older report is inconclusive."
                ),
            },
            "event": payload.get("event"),
            "trailing_24h": payload.get("trailing_24h"),
            "recent": compact_recent,
        },
    )


async def core_support_dashboard_freshness(
    *,
    threshold: str = "24h",
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/meta/freshness",
        {"threshold": threshold},
    )
    alerts = payload.get("alerts")
    if not isinstance(alerts, dict):
        alerts = {}
    compact_alerts = {
        name: {
            "status": alert.get("status"),
            "is_firing": alert.get("is_firing"),
            "last_success_at": alert.get("last_success_at"),
            "current_age_seconds": alert.get("current_age_seconds"),
            "threshold_seconds": alert.get("threshold_seconds"),
        }
        for name, alert in alerts.items()
        if isinstance(alert, dict)
    }
    return _json_block(
        "Support dashboard system freshness",
        {
            "source": "/api/meta/freshness",
            "scope": (
                "System and cohort health only. This cannot prove that one specific "
                "vault's PPS data is current."
            ),
            "as_of_utc": payload.get("as_of_utc"),
            "threshold": payload.get("threshold"),
            "stale_threshold_seconds": payload.get("stale_threshold_seconds"),
            "latest_pps_at": payload.get("latest_pps_at"),
            "latest_pps_age_seconds": payload.get("latest_pps_age_seconds"),
            "metrics_newest_point_at": payload.get("metrics_newest_point_at"),
            "metrics_newest_age_seconds": payload.get("metrics_newest_age_seconds"),
            "metrics_rows": payload.get("metrics_rows"),
            "pps_vaults_total": payload.get("pps_vaults_total"),
            "pps_vaults_stale": payload.get("pps_vaults_stale"),
            "pps_stale_ratio": payload.get("pps_stale_ratio"),
            "stale_by_chain": payload.get("stale_by_chain"),
            "stale_by_category": payload.get("stale_by_category"),
            "ingestion_jobs": payload.get("ingestion_jobs"),
            "alerts": compact_alerts,
        },
    )


async def core_support_dashboard_styfi(
    *,
    days: int = 7,
    epoch_limit: int = 3,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/styfi",
        {
            "days": days,
            "epoch_limit": epoch_limit,
        },
    )
    ingestion = payload.get("ingestion")
    if not isinstance(ingestion, dict):
        ingestion = {}
    return _json_block(
        "Support dashboard stYFI status",
        {
            "source": "/api/styfi",
            "summary": payload.get("summary"),
            "current_reward_state": payload.get("current_reward_state"),
            "freshness": payload.get("freshness"),
            "ingestion_last_run": ingestion.get("last_run"),
            "recent_activity": payload.get("recent_activity"),
        },
    )
