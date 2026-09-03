from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Optional

from web3 import Web3

import config
from kong_vaults import fetch_kong_vault_catalog, fetch_kong_vault_snapshots


_YBOLD_CHAIN_ID = 1
_YBOLD_ADDRESS = "0x9f4330700a36b29952869fac9b33f45eedd8a3d8"
_STAKED_YBOLD_ADDRESS = "0x23346b04a7f55b8760e5860aa5a77383d63491cd"
_YBOLD_PRODUCT_ADDRESSES = {_YBOLD_ADDRESS, _STAKED_YBOLD_ADDRESS}


def format_timestamp_to_readable(timestamp: int | float | str | None) -> str:
    if timestamp is None:
        return "N/A"
    try:
        dt_object = datetime.fromtimestamp(int(timestamp), timezone.utc)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return str(timestamp)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_percent_field(value: Any) -> str:
    numeric_value = _safe_float(value, None)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value * 100:.2f}%"


def _format_basis_points(value: Any) -> str:
    numeric_value = _safe_float(value, None)
    if numeric_value is None:
        return "N/A"
    return f"{numeric_value / 100:.2f}%"


def _metadata(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("meta")
    return meta if isinstance(meta, dict) else data


def _tvl_usd(data: dict[str, Any]) -> float:
    tvl = data.get("tvl")
    if isinstance(tvl, dict):
        return _safe_float(tvl.get("close"), 0.0) or 0.0
    return _safe_float(tvl, 0.0) or 0.0


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


def _net_apy(
    data: dict[str, Any],
    staked_ybold: dict[str, Any] | None = None,
) -> float:
    ybold_apy = _ybold_display_apy(data, staked_ybold)
    if ybold_apy is not None:
        return ybold_apy[0]
    performance = data.get("performance") or {}
    oracle = performance.get("oracle") or {}
    historical = performance.get("historical") or {}
    value = _safe_float(oracle.get("netAPY"), None)
    if value is None:
        value = _safe_float(historical.get("net"), 0.0)
    return value or 0.0


def format_single_vault_data_for_llm(
    data: dict[str, Any],
    chain_id_for_url: int,
    *,
    staked_ybold: dict[str, Any] | None = None,
) -> str:
    """Format a Kong vault list row or exact snapshot for model consumption."""
    meta = _metadata(data)
    name = data.get("name") or meta.get("name") or "N/A"
    symbol = data.get("symbol") or meta.get("displaySymbol") or "N/A"
    address = data.get("address", "N/A")
    api_version = str(data.get("apiVersion") or "Unknown")
    kind = meta.get("kind") or data.get("kind") or "N/A"
    asset = data.get("asset") or meta.get("token") or {}
    risk = data.get("risk") or {}
    risk_level = risk.get("riskLevel", data.get("riskLevel", "N/A"))
    is_retired = bool(meta.get("isRetired", data.get("isRetired", False)))
    is_hidden = bool(meta.get("isHidden", data.get("isHidden", False)))
    is_boosted = bool(meta.get("isBoosted", data.get("isBoosted", False)))
    is_highlighted = bool(
        meta.get("isHighlighted", data.get("isHighlighted", False))
    )

    output_lines = [
        f"Vault: {name} ({symbol})",
        f"Address: `{address}`",
        f"Yearn UI Link: https://yearn.fi/vaults/{chain_id_for_url}/{address}",
        f"Version: {api_version}",
        f"Kind: {kind}",
    ]
    description = str(meta.get("description") or "No description available.")
    if len(description) > 250:
        description = description[:247] + "..."
    output_lines.append(f"Description: {description}")
    output_lines.append(
        "Underlying Token: "
        f"{asset.get('name', 'N/A')} ({asset.get('symbol', 'N/A')}) - "
        f"`{asset.get('address', 'N/A')}`"
    )

    output_lines.extend(
        [
            "",
            "TVL & Share Price:",
            f"  TVL (USD): ${_tvl_usd(data):,.2f}",
        ]
    )
    raw_pps = data.get("pricePerShare")
    try:
        decimals = int(data.get("decimals", 18))
        scaled_pps = int(raw_pps) / (10**decimals)
        output_lines.append(
            "  Vault Token Price Per Share (in underlying): "
            f"{scaled_pps:.6f} (Raw: {raw_pps})"
        )
    except (TypeError, ValueError):
        output_lines.append(f"  Vault Token Price Per Share (Raw): {raw_pps or 'N/A'}")

    performance = data.get("performance") or {}
    oracle = performance.get("oracle") or {}
    historical = performance.get("historical") or {}
    ybold_apy = _ybold_display_apy(data, staked_ybold)
    displayed_net_apy = ybold_apy[0] if ybold_apy is not None else oracle.get("netAPY")
    output_lines.extend(
        [
            "",
            "APY Information:",
            f"  Current Estimated Net APY: {_format_percent_field(displayed_net_apy)}",
            f"  Realized Net APY: {_format_percent_field(historical.get('net'))}",
            "  Historical Realized Net APY: "
            f"Week={_format_percent_field(historical.get('weeklyNet'))}, "
            f"Month={_format_percent_field(historical.get('monthlyNet'))}, "
            f"Inception={_format_percent_field(historical.get('inceptionNet'))}",
        ]
    )
    if ybold_apy is not None:
        output_lines.append(
            "  yBOLD Display Inputs (staked product): "
            f"Oracle={_format_percent_field(ybold_apy[1])}, "
            f"7-day PPS={_format_percent_field(ybold_apy[2])}"
        )
    fees = data.get("fees") or {}
    output_lines.append(
        "  Vault Fees: "
        f"Performance={_format_basis_points(fees.get('performanceFee'))}, "
        f"Management={_format_basis_points(fees.get('managementFee'))}"
    )

    output_lines.extend(
        [
            "",
            "Other Info:",
            f"  Risk Level: {risk_level}",
            "  Status Flags: "
            f"Retired={is_retired}, Hidden={is_hidden}, "
            f"Boosted={is_boosted}, Highlighted={is_highlighted}",
        ]
    )
    migration = meta.get("migration", data.get("migration", False))
    if isinstance(migration, dict):
        output_lines.append(
            f"  Migration Available: {bool(migration.get('available'))}"
        )
        if migration.get("available"):
            output_lines.append(
                f"    Migration Target Address: `{migration.get('target', 'N/A')}`"
            )
    else:
        output_lines.append(f"  Migration Available: {bool(migration)}")

    composition = data.get("composition") or []
    if composition:
        output_lines.extend(["", f"Strategies ({len(composition)}):"])
        for index, strategy in enumerate(composition, 1):
            strategy_performance = strategy.get("performance") or {}
            strategy_historical = strategy_performance.get("historical") or {}
            output_lines.extend(
                [
                    f"  {index}. {strategy.get('name', 'Unnamed Strategy')} "
                    f"(`{strategy.get('address', 'N/A')}`)",
                    f"     Status: {strategy.get('status', 'N/A')}",
                    "     Realized APY: "
                    f"{_format_percent_field(strategy_historical.get('net'))}",
                    "     Current Debt (USD): "
                    f"${(_safe_float(strategy.get('currentDebtUsd'), 0.0) or 0.0):,.2f}",
                    "     Last Report: "
                    f"{format_timestamp_to_readable(strategy.get('lastReport'))}",
                ]
            )
    else:
        strategy_count = data.get("strategiesCount")
        output_lines.extend(
            ["", f"Strategies: {strategy_count if strategy_count is not None else 'Not included in summary.'}"]
        )

    staking = data.get("staking") or {}
    output_lines.append("")
    if staking.get("available"):
        output_lines.extend(
            [
                "Staking Opportunity: Yes",
                f"  Source: {staking.get('source', 'N/A')}",
                f"  Staking Contract: `{staking.get('address', 'N/A')}`",
            ]
        )
    else:
        output_lines.append("Staking Opportunity: No")

    block_time = data.get("blockTime")
    if block_time is not None:
        output_lines.append(
            f"Snapshot Time: {format_timestamp_to_readable(block_time)}"
        )
    return "\n".join(output_lines)


def _is_recommendable_vault(vault: dict[str, Any]) -> bool:
    inclusion = vault.get("inclusion") or {}
    return (
        inclusion.get("isYearn") is True
        and not vault.get("isRetired")
        and not vault.get("isHidden")
        and not str(vault.get("symbol") or "").lower().startswith("ys")
        and "single strategy" not in str(vault.get("kind") or "").lower()
        and int(vault.get("strategiesCount") or 0) > 1
    )


def _matches_query(vault: dict[str, Any], query: str) -> bool:
    query_lower = query.lower().strip()
    if query_lower in {"all", "*"}:
        return True
    asset = vault.get("asset") or {}
    if Web3.is_address(query_lower):
        return query_lower in {
            str(vault.get("address") or "").lower(),
            str(asset.get("address") or "").lower(),
        }
    return (
        query_lower == str(vault.get("symbol") or "").lower()
        or query_lower == str(asset.get("symbol") or "").lower()
        or query_lower in str(vault.get("name") or "").lower()
        or query_lower in str(asset.get("name") or "").lower()
    )


async def core_search_vaults(
    query: str,
    chain: Optional[str] = None,
    sort_by: Optional[str] = None,
    recommended_only: bool = False,
) -> str:
    """Search Kong's canonical Yearn vault catalog and enrich the top matches."""
    logging.info(
        "[CoreTool:search_vaults] Query: '%s', Chain: '%s', Sort By: '%s', Recommended Only: %s",
        query,
        chain,
        sort_by,
        recommended_only,
    )
    max_results = getattr(config, "MAX_RESULTS_TO_SHOW", 5)
    query_chain_id = None
    if chain:
        query_chain_id = config.CHAIN_NAME_TO_ID.get(chain.strip().lower())
        if query_chain_id is None:
            supported_chains = ", ".join(config.CHAIN_NAME_TO_ID)
            return f"Unsupported chain: '{chain}'. Supported chains: {supported_chains}."

    try:
        catalog = await fetch_kong_vault_catalog()
    except Exception as exc:
        logging.error("[Tool:search_vaults] Kong catalog fetch failed: %s", exc)
        return f"Error: An unexpected error occurred while fetching vault data: {exc}."

    matches = [
        vault
        for vault in catalog
        if (query_chain_id is None or vault.get("chainId") == query_chain_id)
        and _matches_query(vault, query)
        and (not recommended_only or _is_recommendable_vault(vault))
    ]
    if not matches:
        if recommended_only:
            return "No recommendation-grade active Yearn vaults found matching your criteria."
        return "No Yearn vaults found matching your criteria."

    staked_ybold_catalog = _staked_ybold(catalog)

    def active(vault: dict[str, Any]) -> bool:
        return not vault.get("isRetired") and not vault.get("isHidden")

    def display_net_apy(vault: dict[str, Any]) -> float:
        return _net_apy(vault, staked_ybold_catalog)

    if sort_by == "highest_apr":
        matches.sort(
            key=lambda vault: (active(vault), display_net_apy(vault)),
            reverse=True,
        )
    elif sort_by == "lowest_apr":
        matches.sort(
            key=lambda vault: (not active(vault), display_net_apy(vault))
        )
    elif recommended_only:
        matches.sort(
            key=lambda vault: (
                bool(vault.get("isHighlighted")),
                -(_safe_float(vault.get("riskLevel"), 999.0) or 999.0),
                _tvl_usd(vault),
                display_net_apy(vault),
            ),
            reverse=True,
        )
    else:
        matches.sort(key=lambda vault: (active(vault), _tvl_usd(vault)), reverse=True)

    top_vaults = matches[:max_results]

    snapshots = await fetch_kong_vault_snapshots(
        [
            (int(vault["chainId"]), str(vault["address"]))
            for vault in top_vaults
        ]
    )
    details = []
    for vault, snapshot in zip(top_vaults, snapshots, strict=True):
        if isinstance(snapshot, BaseException):
            logging.warning(
                "Kong snapshot fetch failed for %s/%s: %s",
                vault.get("chainId"),
                vault.get("address"),
                snapshot,
            )
            details.append(vault)
        else:
            details.append(snapshot or vault)

    staked_ybold_details = _staked_ybold(details) or staked_ybold_catalog
    formatted = [
        format_single_vault_data_for_llm(
            data,
            int(vault["chainId"]),
            staked_ybold=staked_ybold_details,
        )
        for vault, data in zip(top_vaults, details, strict=True)
    ]
    num_total = len(matches)
    header = f"Found {num_total} Yearn vault(s) matching '{query}'."
    if num_total > len(formatted):
        sort_description = sort_by or "TVL (Descending)"
        header += (
            f" Showing top {len(formatted)} (sorted by {sort_description}) with details:"
        )
    return header + "\n\n---\n\n" + "\n\n---\n\n".join(formatted)
