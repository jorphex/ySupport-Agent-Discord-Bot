from __future__ import annotations

import logging
import math
from time import monotonic
from typing import Any, Literal, Optional

from web3 import Web3

from ysupport import config
from ysupport.tools import kong_vaults
from ysupport.tools.kong_vaults import fetch_kong_vault_catalog, fetch_kong_vault_snapshots


VaultSort = Literal["", "tvl", "highest_apy", "lowest_apy", "highest_apr", "lowest_apr"]
_YBOLD_CHAIN_ID = 1
_YBOLD_ADDRESS = "0x9f4330700a36b29952869fac9b33f45eedd8a3d8"
_STAKED_YBOLD_ADDRESS = "0x23346b04a7f55b8760e5860aa5a77383d63491cd"
_YBOLD_PRODUCT_ADDRESSES = {_YBOLD_ADDRESS, _STAKED_YBOLD_ADDRESS}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value) if not isinstance(value, bool) else math.nan
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _metadata(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("meta")
    return meta if isinstance(meta, dict) else data


def _tvl_usd(data: dict[str, Any]) -> float | None:
    tvl = data.get("tvl")
    return _safe_float(tvl.get("close") if isinstance(tvl, dict) else tvl)


def _is_ybold_product(data: dict[str, Any]) -> bool:
    return (
        data.get("chainId") == _YBOLD_CHAIN_ID
        and str(data.get("address") or "").lower() in _YBOLD_PRODUCT_ADDRESSES
    )


def _staked_ybold(vaults: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            vault
            for vault in vaults
            if vault.get("chainId") == _YBOLD_CHAIN_ID
            and str(vault.get("address") or "").lower() == _STAKED_YBOLD_ADDRESS
        ),
        None,
    )


def _ybold_display_apy(
    data: dict[str, Any],
    staked_ybold: dict[str, Any] | None,
) -> tuple[float, float | None, float | None] | None:
    if not _is_ybold_product(data):
        return None
    address = str(data.get("address") or "").lower()
    if staked_ybold is None and address != _STAKED_YBOLD_ADDRESS:
        return None
    source = staked_ybold or data
    performance = source.get("performance") or {}
    oracle = performance.get("oracle") or {}
    historical = performance.get("historical") or {}
    oracle_apy = _safe_float(oracle.get("netAPY"), None)
    weekly_apy = _safe_float(historical.get("weeklyNet"), None)
    available = [value for value in (oracle_apy, weekly_apy) if value is not None]
    if not available:
        return None
    return max(available), oracle_apy, weekly_apy


def _estimated_apy(data: dict[str, Any]) -> tuple[float | None, str | None]:
    performance = data.get("performance") or {}
    # yBOLD uses its separate oracle/PPS display rule, including the base-only fallback.
    estimated = {} if _is_ybold_product(data) else performance.get("estimated") or {}
    oracle = performance.get("oracle") or {}
    for source, value in (
        ("performance.estimated.apy", estimated.get("apy")),
        ("performance.oracle.netAPY", oracle.get("netAPY")),
    ):
        numeric_value = _safe_float(value, None)
        if numeric_value is not None:
            return numeric_value, source
    return None, None


def _apy_fields(data, staked_ybold):
    estimated, source = _estimated_apy(data)
    fields = {"estimated_apy": estimated, "estimated_apy_source": source}
    ybold = _ybold_display_apy(data, staked_ybold)
    if ybold is not None:
        fields.update(estimated_apy=ybold[0], estimated_apy_source="yearn_ui.ybold_max_staked_oracle_weekly",
                      apy_inputs={"source_vault_address": _STAKED_YBOLD_ADDRESS,
                                  "oracle_net_apy": ybold[1], "weekly_net_apy": ybold[2]})
    return fields


def _vault_row(data, *, detailed, detail_status, staked_ybold, apy_input_source):
    meta = _metadata(data)
    performance = data.get("performance") or {}
    historical = performance.get("historical") or {}
    asset = data.get("asset") or meta.get("token") or {}
    chain_id, address = data["chainId"], data["address"]
    composition = data.get("composition")
    row = {
        "vault_address": address, "chain_id": chain_id,
        "name": data.get("name") or meta.get("name"),
        "symbol": data.get("symbol") or meta.get("displaySymbol"),
        "yearn_url": f"https://yearn.fi/vaults/{chain_id}/{address}",
        "api_version": data.get("apiVersion"), "kind": meta.get("kind") or data.get("kind"),
        "asset": {key: asset.get(key) for key in ("address", "name", "symbol", "decimals")},
        "tvl_usd": _tvl_usd(data), **_apy_fields(data, staked_ybold),
        "realized_apy": _safe_float(historical.get("net")),
        "realized_apy_7d": _safe_float(historical.get("weeklyNet")),
        "realized_apy_30d": _safe_float(historical.get("monthlyNet")),
        "realized_apy_inception": _safe_float(historical.get("inceptionNet")),
        "risk_level": (data.get("risk") or {}).get("riskLevel", data.get("riskLevel")),
        "is_retired": meta.get("isRetired", data.get("isRetired")),
        "is_hidden": meta.get("isHidden", data.get("isHidden")),
        "is_highlighted": meta.get("isHighlighted", data.get("isHighlighted")),
        "is_boosted": meta.get("isBoosted", data.get("isBoosted")),
        "strategy_count": len(composition) if detailed and composition is not None else data.get("strategiesCount"),
        "detail_status": detail_status,
        "data_source": "snapshot" if detailed else "catalog",
        "block_time": data.get("blockTime"), "block_number": data.get("blockNumber"),
    }
    if "apy_inputs" in row:
        row["apy_inputs"].update(apy_input_source)
    if detailed:
        fees = data.get("fees") or {}
        row["details"] = {
            "description": str(meta.get("description") or "")[:250],
            "share_decimals": data.get("decimals"),
            "price_per_share_raw": str(data["pricePerShare"]) if data.get("pricePerShare") is not None else None,
            "fees": {"performance_fee_bps": _safe_float(fees.get("performanceFee")),
                     "management_fee_bps": _safe_float(fees.get("managementFee"))},
            "migration": meta.get("migration", data.get("migration")),
            "staking": data.get("staking"),
            "strategies": [{
                "name": strategy.get("name"), "address": strategy.get("address"),
                "status": strategy.get("status"),
                "realized_apy": _safe_float(((strategy.get("performance") or {}).get("historical") or {}).get("net")),
                "current_debt_usd": _safe_float(strategy.get("currentDebtUsd")),
                "last_report": strategy.get("lastReport"),
            } for strategy in composition] if composition is not None else None,
        }
    return row


def _is_recommendable_vault(vault: dict[str, Any]) -> bool:
    inclusion = vault.get("inclusion") or {}
    return (
        inclusion.get("isYearn") is True
        and not vault.get("isRetired") and not vault.get("isHidden")
        and not str(vault.get("symbol") or "").lower().startswith("ys")
        and "single strategy" not in str(vault.get("kind") or "").lower()
        and (_safe_float(vault.get("strategiesCount")) or 0) > 1
    )


def _matches_query(vault: dict[str, Any], query: str) -> bool:
    asset = vault.get("asset") or {}
    if query in {"all", "*"}:
        return True
    if Web3.is_address(query):
        return query in {str(vault.get("address") or "").lower(), str(asset.get("address") or "").lower()}
    return (
        query == str(vault.get("symbol") or "").lower()
        or query == str(asset.get("symbol") or "").lower()
        or query in str(vault.get("name") or "").lower()
        or query in str(asset.get("name") or "").lower()
    )


async def core_search_vaults(
    query: str,
    chain: Optional[str] = None,
    sort_by: Optional[str] = None,
    recommended_only: bool = False,
    *,
    limit: int | None = None,
    offset: int = 0,
    include_details: bool = False,
) -> dict[str, Any]:
    """Page the Kong catalog; enrich exact vaults or explicitly requested page rows."""
    if not isinstance(query, str) or not query.strip() or len(query) > 256:
        raise ValueError("Vault query must contain 1 to 256 characters.")
    query = query.strip().lower()
    limit = config.MAX_RESULTS_TO_SHOW if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= 25 or type(offset) is not int or offset < 0:
        raise ValueError("Vault limit must be 1 to 25 and offset must be non-negative.")
    sort_by = {"highest_apr": "highest_apy", "lowest_apr": "lowest_apy"}.get(sort_by, sort_by) or "tvl"
    if sort_by not in {"tvl", "highest_apy", "lowest_apy"}:
        raise ValueError("Unknown vault sort; use tvl, highest_apy or lowest_apy.")
    chain_id = config.CHAIN_NAME_TO_ID.get(chain.strip().lower()) if chain else None
    if chain and chain_id is None:
        raise ValueError(f"Unsupported chain: '{chain}'.")
    catalog = await fetch_kong_vault_catalog()
    catalog_source = kong_vaults.vault_catalog_freshness()
    source_captured_at = monotonic()
    matches = [v for v in catalog if (chain_id is None or v.get("chainId") == chain_id)
               and _matches_query(v, query) and (not recommended_only or _is_recommendable_vault(v))]
    companion = _staked_ybold(catalog)

    def order(vault):
        exact = str(vault["address"]).lower() == query
        active = not vault.get("isRetired") and not vault.get("isHidden")
        if sort_by == "tvl" and recommended_only:
            metric = (not vault.get("isHighlighted"), _safe_float(vault.get("riskLevel"), 999), -(_tvl_usd(vault) or 0))
        else:
            value = _tvl_usd(vault) if sort_by == "tvl" else _apy_fields(vault, companion)["estimated_apy"]
            metric = (value is None, (value or 0) * (1 if sort_by == "lowest_apy" else -1))
        return (not exact, not active, *metric, vault["chainId"], str(vault["address"]).lower())

    matches.sort(key=order)
    page = matches[offset:offset+limit]
    targets = [(int(v["chainId"]), str(v["address"])) for v in page
               if include_details or str(v["address"]).lower() == query]
    snapshots = await fetch_kong_vault_snapshots(targets) if targets else []
    enriched = dict(zip(targets, snapshots, strict=True))
    valid = {}
    for (target_chain, address), snapshot in enriched.items():
        if isinstance(snapshot, dict) and snapshot.get("chainId") == target_chain and str(snapshot.get("address") or "").lower() == address.lower():
            valid[(target_chain, address)] = snapshot
        else:
            logging.warning("Kong details unavailable or mismatched for %s/%s", target_chain, address)
    snapshot_companion = _staked_ybold(list(valid.values()))
    companion = snapshot_companion or companion
    apy_input_source = {
        "data_source": "snapshot" if snapshot_companion is not None else "catalog",
        "block_time": companion.get("blockTime") if companion else None,
        "catalog_retrieved_at": None if snapshot_companion is not None else catalog_source["retrieved_at"],
    }
    rows = []
    for vault in page:
        target = (int(vault["chainId"]), str(vault["address"]))
        snapshot = valid.get(target)
        status = "available" if snapshot is not None else "unavailable" if target in enriched else "not_requested"
        if snapshot is not None and any(snapshot.get(key) is None for key in (
            "performance", "tvl", "fees", "pricePerShare", "composition",
        )):
            status = "partial"
        data = snapshot if snapshot is not None else vault
        rows.append(_vault_row(data, detailed=snapshot is not None, detail_status=status,
                               staked_ybold=companion, apy_input_source=apy_input_source))
    next_offset = offset + len(rows) if offset + len(rows) < len(matches) else None
    if catalog_source["cache_age_seconds"] is not None:
        catalog_source["cache_age_seconds"] = round(
            catalog_source["cache_age_seconds"] + monotonic() - source_captured_at, 3,
        )
    return {
        "source": {"provider": "kong", "catalog_url": kong_vaults.KONG_BASE_URL + "/api/rest/list/vaults",
                   **catalog_source},
        "filters": {"query": query, "chain_id": chain_id, "sort_by": sort_by, "recommended_only": recommended_only},
        "units": {"apy": "decimal_fraction", "tvl_usd": "USD", "fees": "basis_points", "raw_amounts": "base_units"},
        "ui_apy_observation": {
            "verified": False, "displayed_metric": None, "zero_explanation": None,
            "scope": "API estimates and realised rates do not observe the current UI label or explain its displayed zero.",
        },
        "sort_source": "cached_catalog; detail snapshots may be newer; missing metrics sort last among active peers",
        "pagination": {"total": len(matches), "offset": offset, "limit": limit, "returned": len(rows), "next_offset": next_offset},
        "rows": rows,
    }
