from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal
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
    token_symbol: str | None = None,
    market: str = "all",
    universe: str = "core",
    min_tvl_usd: float | None = None,
    min_points: int | None = None,
    sort_by: str = "tvl",
    direction: str = "desc",
    limit: int = 10,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/discover",
        {
            "chain_id": chain_id,
            "token_symbol": token_symbol,
            "market": market,
            "universe": universe,
            "min_tvl_usd": min_tvl_usd,
            "min_points": min_points,
            "sort_by": sort_by,
            "direction": direction,
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
            "token_symbol": row.get("token_symbol"),
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
    meaningful_only: bool = True,
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
            "meaningful_only": meaningful_only,
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
                "meaningful_only": meaningful_only,
            },
            "interpretation": {
                "event_semantics": (
                    "Each row proves an on-chain StrategyReported event, not an "
                    "off-chain harvest job or realized profit."
                ),
                "filter_semantics": (
                    "meaningful_only=true returns reports with nonzero gain, loss, "
                    "fees, or refunds. Use false when accounting-only updates are "
                    "relevant to the support question."
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


def _compact_alerts(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    alerts = payload.get("alerts")
    if not isinstance(alerts, dict):
        alerts = {}
    return {
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


def _compact_freshness(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "as_of_utc": payload.get("as_of_utc"),
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
        "ingestion_jobs": payload.get("ingestion_jobs"),
        "alerts": _compact_alerts(payload),
    }


def _compact_coverage(payload: object) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "as_of_utc": payload.get("as_of_utc"),
        "filters": payload.get("filters"),
        "global": payload.get("global"),
    }


async def core_support_dashboard_status() -> str:
    payload = await _fetch_dashboard_json("/api/meta/status", {})
    protocol_context = payload.get("protocol_context")
    if not isinstance(protocol_context, dict):
        protocol_context = {}
    protocol = protocol_context.get("protocol")
    if not isinstance(protocol, dict):
        protocol = {}
    return _json_block(
        "Support dashboard operational status",
        {
            "source": "/api/meta/status",
            "scope": (
                "System, source, and cohort health only. This cannot prove that one "
                "specific vault's PPS data is current."
            ),
            "status": payload.get("status"),
            "generated_at_utc": payload.get("generated_at_utc"),
            "data_policy": payload.get("data_policy"),
            "protocol_source": {
                "status": protocol_context.get("status"),
                "source": protocol_context.get("source"),
                "as_of_utc": protocol_context.get("as_of_utc"),
                "protocol_tvl_usd": protocol.get("tvl_usd"),
                "fetched_at": protocol.get("fetched_at"),
                "age_seconds": protocol.get("age_seconds"),
                "freshness_status": protocol.get("freshness_status"),
            },
            "tracked_scope": payload.get("tracked_scope"),
            "freshness": _compact_freshness(payload.get("freshness")),
            "coverage": _compact_coverage(payload.get("coverage")),
        },
    )


async def core_support_dashboard_changes(
    *,
    window: str = "7d",
    stale_threshold: str = "auto",
    universe: str = "core",
    market: str = "all",
    min_tvl_usd: float | None = None,
    min_points: int | None = None,
    limit: int = 10,
) -> str:
    payload = await _fetch_dashboard_json(
        "/api/changes",
        {
            "window": window,
            "stale_threshold": stale_threshold,
            "universe": universe,
            "market": market,
            "min_tvl_usd": min_tvl_usd,
            "min_points": min_points,
            "limit": limit,
        },
    )
    movers = payload.get("movers")
    if not isinstance(movers, dict):
        movers = {}

    def compact_rows(name: str) -> list[dict[str, Any]]:
        rows = movers.get(name)
        if not isinstance(rows, list):
            return []
        return [
            {
                "vault_address": row.get("vault_address"),
                "chain_id": row.get("chain_id"),
                "symbol": row.get("symbol"),
                "token_symbol": row.get("token_symbol"),
                "tvl_usd": row.get("tvl_usd"),
                "realized_apy_window": row.get("realized_apy_window"),
                "realized_apy_prev_window": row.get("realized_apy_prev_window"),
                "delta_apy": row.get("delta_apy"),
                "age_seconds": row.get("age_seconds"),
            }
            for row in rows[:limit]
            if isinstance(row, dict)
        ]

    return _json_block(
        "Support dashboard realized APY changes",
        {
            "source": "/api/changes",
            "scope": (
                "Market context only, not evidence of a deposit, withdrawal, report, "
                "or specific-vault failure. delta_apy is a decimal APY difference."
            ),
            "window": payload.get("window"),
            "realized_apy_policy": payload.get("realized_apy_policy"),
            "summary": payload.get("summary"),
            "freshness": payload.get("freshness"),
            "movers": {
                "risers": compact_rows("risers"),
                "fallers": compact_rows("fallers"),
            },
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


FlexView = Literal[
    "protocol", "markets", "market", "redemption_priority", "trove_health",
    "history", "activity", "performance",
]
_FLEX_VIEWS = {
    "protocol": ("/protocol", ("summary",), set()),
    "markets": ("/markets", ("filters", "summary", "rows"), {"status"}),
    "market": ("/markets/{market}", ("market", "risk", "oracle"), {"market_address"}),
    "redemption_priority": (
        "/markets/{market}/redemption-priority",
        ("market_address", "borrow_token", "rate_scale", "total_debt_raw", "total_debt",
         "idle_liquidity_raw", "idle_liquidity", "points", "source_url"), {"market_address"},
    ),
    "trove_health": (
        "/markets/{market}/trove-health",
        ("market_address", "collateral_token", "borrow_token", "metrics"), {"market_address"},
    ),
    "history": (
        "/markets/{market}/history", ("market_address", "filters", "coverage", "points"),
        {"market_address", "days", "interval"},
    ),
    "activity": ("/activity", ("filters", "pagination", "rows"), {"market_address", "event", "cursor"}),
    "performance": ("/performance", ("period", "borrowers", "lenders"), {"wallet_address", "days"}),
}


async def core_support_dashboard_flex(
    *,
    view: FlexView,
    market_address: str = "",
    wallet_address: str = "",
    status: str | None = None,
    days: int | None = None,
    interval: str | None = None,
    event: str = "",
    cursor: str = "",
    limit: int = 10,
) -> str:
    if not isinstance(view, str) or view not in _FLEX_VIEWS:
        raise ValueError("Unknown Flex data view.")
    suffix, fields, allowed = _FLEX_VIEWS[view]
    for name, address in (("market_address", market_address), ("wallet_address", wallet_address)):
        if not isinstance(address, str) or (address and not re.fullmatch(r"0x[0-9a-fA-F]{40}", address)):
            raise ValueError(f"{name} must be an Ethereum address.")
    if "{market}" in suffix and not market_address:
        raise ValueError("This Flex view requires market_address; discover it with the markets view.")
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Flex output limit must be between 1 and 50.")
    if status is not None and status not in {"all", "active", "deprecated", "unendorsed"}:
        raise ValueError("Unknown Flex market status.")
    if days is not None and days not in {7, 30, 90}:
        raise ValueError("Flex history/performance days must be 7, 30 or 90.")
    if interval is not None and interval not in {"hour", "day"}:
        raise ValueError("Flex history interval must be hour or day.")
    if not isinstance(event, str) or len(event) > 64:
        raise ValueError("Flex event must contain at most 64 characters.")
    if not isinstance(cursor, str) or (cursor and not re.fullmatch(r"\d{1,20}:\d{1,20}", cursor)):
        raise ValueError("Flex activity cursor must be block_number:log_index.")
    supplied = {"market_address": market_address or None, "wallet_address": wallet_address or None,
                "status": status, "days": days, "interval": interval,
                "event": event or None, "cursor": cursor or None}
    if any(value is not None and name not in allowed for name, value in supplied.items()):
        raise ValueError("Arguments do not apply to this Flex view.")
    if view in {"protocol", "market", "trove_health"} and limit != 10:
        raise ValueError("limit applies only to Flex views returning lists.")
    params: dict[str, Any] = {}
    if view == "markets":
        params = {"status": status or "active"}
    elif view == "history":
        params = {"days": days or 7, "interval": interval or "day"}
    elif view == "activity":
        params = {"market_address": market_address or None, "event": event or None,
                  "cursor": cursor or None, "limit": limit}
    elif view == "performance":
        params = {"address": wallet_address or None, "days": days or 7}
    path = "/api/flex" + suffix.format(market=market_address.lower())
    payload = await _fetch_dashboard_json(path, params)
    if not isinstance(payload.get("scope"), dict) or payload["scope"].get("chain_id") != 1:
        raise ValueError("Flex response has missing or unexpected Ethereum scope.")
    if not isinstance(payload.get("freshness"), dict) or any(key not in payload for key in fields):
        raise ValueError("Flex response is missing required data or freshness.")
    if view in {"markets", "history", "activity"}:
        filters = payload.get("filters")
        if not isinstance(filters, dict):
            raise ValueError("Flex response is missing its request filters.")
        for name, expected in params.items():
            if name in {"cursor", "limit"}:
                continue
            observed = filters.get(name)
            if name == "market_address":
                expected = (expected or "").lower()
                if observed is None:
                    observed = ""
                if isinstance(observed, str):
                    observed = observed.lower()
            if observed != expected:
                raise ValueError("Flex response filters do not match the request.")
    if view == "activity" and payload.get("pagination", {}).get("limit") != limit:
        raise ValueError("Flex activity response has a mismatched page limit.")
    if "{market}" in suffix:
        identity = (payload.get("market", {}).get("addresses", {}).get("market")
                    if view == "market" else payload.get("market_address"))
        if not isinstance(identity, str) or identity.lower() != market_address.lower():
            raise ValueError("Flex response does not match the requested market.")
    if view == "performance":
        identity = payload["scope"].get("address") or ""
        if not isinstance(identity, str) or identity.lower() != wallet_address.lower():
            raise ValueError("Flex performance response does not match the requested wallet scope.")
        if payload.get("period", {}).get("days") != params["days"]:
            raise ValueError("Flex performance response has a mismatched time window.")
    result = {"view": view, "source": path, "scope": payload["scope"],
              "freshness": payload["freshness"], **{key: payload[key] for key in fields}}
    output_limits = {}

    def bounded_rows(rows, key, *, latest=False):
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Flex response contains missing or malformed rows.")
        selected = rows[-limit:] if latest else rows[:limit]
        output_limits[key] = {"available": len(rows), "returned": len(selected),
                              "truncated": len(rows) > limit}
        return selected

    for key in ("rows", "points"):
        if key in result:
            if view == "history" and isinstance(result[key], list) and result[key]:
                result["window_endpoints"] = {"first": result[key][0], "last": result[key][-1]}
            result[key] = bounded_rows(result[key], key, latest=view == "history")
    if view == "performance":
        for side in ("borrowers", "lenders"):
            if not isinstance(result[side], dict):
                raise ValueError("Flex performance response contains a missing side.")
            result[side] = {**result[side], "rows": bounded_rows(result[side].get("rows"), side + ".rows")}
    if output_limits:
        result["output_limits"] = output_limits
        if view == "history":
            result["point_selection"] = "Most recent available points; coverage describes the full upstream window."
    text = _json_block("Flex lending protocol data", result)
    if len(text) > 60000:
        raise ValueError("Flex response is too large; request fewer rows.")
    return text
