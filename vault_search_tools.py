from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Union

import aiohttp
from web3 import Web3

import config


def format_timestamp_to_readable(timestamp: Optional[Union[int, float, str]]) -> str:
    if timestamp is None:
        return "N/A"
    try:
        dt_object = datetime.fromtimestamp(int(timestamp), timezone.utc)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return str(timestamp)


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
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


def format_single_vault_data_for_llm(data: Dict, chain_id_for_url: int) -> str:
    """
    Formats a single vault's JSON data into a readable string for the LLM.
    """
    output_lines = []
    name = data.get("name", "N/A")
    symbol = data.get("symbol", "N/A")
    address = data.get("address", "N/A")
    api_version_str = data.get("version", "Unknown")
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
    description = data.get("description", "No description available.")
    if description and len(description) > 250:
        description = description[:247] + "..."
    output_lines.append(f"Description: {description}")

    token_info = data.get("token", {})
    underlying_name = token_info.get("name", "N/A")
    underlying_symbol = token_info.get("symbol", "N/A")
    underlying_address = token_info.get("address", "N/A")
    tvl_data = data.get("tvl", {})
    underlying_price = _safe_float(tvl_data.get("price"), 0.0) or 0.0
    output_lines.append(
        f"Underlying Token: {underlying_name} ({underlying_symbol}) - `{underlying_address}` - Price: ${underlying_price:,.4f}"
    )
    output_lines.append("")

    output_lines.append("TVL & Share Price:")
    tvl_usd = _safe_float(tvl_data.get("tvl"), 0.0) or 0.0
    output_lines.append(f"  TVL (USD): ${tvl_usd:,.2f}")
    raw_pps = data.get("pricePerShare", "0")
    try:
        vault_decimals = int(data.get("decimals", 18))
        scaled_pps = float(raw_pps) / (10**vault_decimals)
        output_lines.append(
            f"  Vault Token Price Per Share (in underlying): {scaled_pps:.6f} (Raw: {raw_pps})"
        )
    except (ValueError, TypeError):
        output_lines.append(f"  Vault Token Price Per Share (Raw): {raw_pps}")
    output_lines.append("")

    apr_data = data.get("apr", {})
    output_lines.append("APY Information:")
    net_apy_text = _format_percent_field(apr_data.get("netAPR"))
    output_lines.append(
        f"  Current Net APY (compounded): {net_apy_text} (Type: {apr_data.get('type', 'N/A')})"
    )

    forward_apr_data = apr_data.get("forwardAPR", {})
    if forward_apr_data and forward_apr_data.get("netAPR") is not None:
        forward_net_apy_text = _format_percent_field(forward_apr_data.get("netAPR"))
        output_lines.append(
            f"  Estimated Forward APY (projection): {forward_net_apy_text} (Type: {forward_apr_data.get('type', 'N/A')})"
        )

    fees = apr_data.get("fees", {})
    perf_fee = _format_percent_field(fees.get("performance"))
    mgmt_fee = _format_percent_field(fees.get("management"))
    output_lines.append(f"  Vault Fees: Performance={perf_fee}, Management={mgmt_fee}")

    points = apr_data.get("points", {})
    week_ago_apy = _format_percent_field(points.get("weekAgo"))
    month_ago_apy = _format_percent_field(points.get("monthAgo"))
    inception_apy = _format_percent_field(points.get("inception"))
    output_lines.append(
        f"  Historical Net APY: Week Ago={week_ago_apy}, Month Ago={month_ago_apy}, Inception={inception_apy}"
    )
    output_lines.append("")

    output_lines.append("Other Info:")
    output_lines.append(f"  Featuring Score: {data.get('featuringScore', 'N/A')}")
    info_obj = data.get("info", {})
    output_lines.append(f"  Risk Level: {info_obj.get('riskLevel', 'N/A')}")
    output_lines.append(
        f"  Status Flags: Retired={info_obj.get('isRetired', False)}, Boosted={info_obj.get('isBoosted', False)}, Highlighted={info_obj.get('isHighlighted', False)}"
    )
    migration_data = data.get("migration", {})
    output_lines.append(
        f"  Migration Available: {migration_data.get('available', False)}"
    )
    if migration_data.get("available", False):
        output_lines.append(
            f"    Migration Target Address: `{migration_data.get('address', 'N/A')}`"
        )
    output_lines.append("")

    strategies = data.get("strategies", [])
    if strategies:
        output_lines.append(f"Strategies ({len(strategies)}):")
        for i, strat in enumerate(strategies):
            strat_name = strat.get("name", "Unnamed Strategy")
            strat_addr = strat.get("address", "N/A")
            strat_status = strat.get("status", "N/A")
            strat_apy = _format_percent_field(strat.get("netAPR"))
            strat_details = strat.get("details", {})
            debt_ratio_raw = strat_details.get("debtRatio")
            debt_ratio_percent = "N/A"
            if debt_ratio_raw is not None:
                try:
                    debt_ratio_percent = f"{float(debt_ratio_raw) / 100:.2f}%"
                except (ValueError, TypeError):
                    pass
            last_report = format_timestamp_to_readable(strat_details.get("lastReport"))
            output_lines.append(f"  {i + 1}. Name: {strat_name} (`{strat_addr}`)")
            output_lines.append(f"     Status: {strat_status}")
            output_lines.append(f"     Individual APY: {strat_apy}")
            output_lines.append(f"     Allocation (Debt Ratio): {debt_ratio_percent}")
            output_lines.append(f"     Last Report: {last_report}")
    else:
        output_lines.append("Strategies: None listed.")
    output_lines.append("")

    staking_info = data.get("staking")
    if staking_info and staking_info.get("available"):
        output_lines.append("Staking Opportunity: Yes")
        output_lines.append(f"  Source: {staking_info.get('source', 'N/A')}")
        output_lines.append(
            f"  Staking Contract: `{staking_info.get('address', 'N/A')}`"
        )
        rewards_list = staking_info.get("rewards", [])
        if rewards_list:
            output_lines.append(f"  Rewards ({len(rewards_list)}):")
            for rew_idx, reward in enumerate(rewards_list):
                rew_name = reward.get("name", "N/A")
                rew_sym = reward.get("symbol", "N/A")
                rew_addr = reward.get("address", "N/A")
                rew_apy = _format_percent_field(reward.get("apr"))
                rew_finished = reward.get("isFinished", False)
                rew_ends = format_timestamp_to_readable(reward.get("finishedAt"))
                output_lines.append(f"    - Token: {rew_name} ({rew_sym}) `{rew_addr}`")
                output_lines.append(f"      APY: {rew_apy}")
                output_lines.append(
                    f"      Status: {'Finished' if rew_finished else 'Ongoing'} (Ends: {rew_ends})"
                )
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
    MAX_RESULTS = getattr(config, "MAX_RESULTS_TO_SHOW", 5)
    query_chain_id = None
    if chain:
        chain_lower = chain.strip().lower()
        query_chain_id = config.CHAIN_NAME_TO_ID.get(chain_lower)
        if query_chain_id is None:
            supported_chains = ", ".join(config.CHAIN_NAME_TO_ID)
            return (
                f"Unsupported chain: '{chain}'. Supported chains: "
                f"{supported_chains}."
            )

    async with aiohttp.ClientSession() as session:
        try:
            logging.info(f"[Tool:search_vaults] Fetching data from {api_url}")
            async with session.get(api_url, timeout=25) as response:
                response.raise_for_status()
                all_vaults_data_list = await response.json()
        except Exception as e:
            logging.error(
                f"[Tool:search_vaults] Error during yDaemon fetch: {e}", exc_info=True
            )
            return (
                f"Error: An unexpected error occurred while fetching vault data: {e}."
            )

        if not isinstance(all_vaults_data_list, list):
            logging.error("[Tool:search_vaults] Unexpected yDaemon response format.")
            return "Error: Received unexpected data format from vault API."

        # --- Filtering ---
        filtered_vaults = all_vaults_data_list

        if query_chain_id is not None:
            filtered_vaults = [
                v for v in filtered_vaults if v.get("chainID") == query_chain_id
            ]

        query_lower = query.lower().strip()
        matched_vaults = []
        is_address_query = Web3.is_address(query_lower)
        match_all_vaults = query_lower in {"all", "*"}

        def _is_recommendable_vault(v_data: dict) -> bool:
            symbol = (v_data.get("symbol") or "").lower()
            kind = (v_data.get("kind") or "").lower()
            strategies = v_data.get("strategies") or []
            if (v_data.get("info") or {}).get("isRetired"):
                return False
            if symbol.startswith("ys"):
                return False
            if "single strategy" in kind:
                return False
            if len(strategies) <= 1:
                return False
            return True

        def _recommendation_sort_key(v_data: dict) -> tuple[float, float, float, float]:
            info_obj = v_data.get("info", {})
            featuring_score = _safe_float(v_data.get("featuringScore"), 0.0) or 0.0
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
            underlying_address = (
                token_info.get("address", "").lower() if token_info else ""
            )

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
                fallback_apr = (
                    forward_apr_data.get("netAPR") if forward_apr_data else None
                )
                apr_value = _safe_float(primary_apr, None)
                if apr_value is None:
                    apr_value = _safe_float(fallback_apr, 0.0) or 0.0
                v_data["_computedAPY"] = apr_value * 100
                try:
                    v_data["_computedTVL_USD"] = float(
                        v_data.get("tvl", {}).get("tvl", 0)
                    )
                except (ValueError, TypeError):
                    v_data["_computedTVL_USD"] = 0.0
                if recommended_only and not _is_recommendable_vault(v_data):
                    continue
                matched_vaults.append(v_data)

        if not matched_vaults:
            if recommended_only:
                return "No recommendation-grade active Yearn vaults found matching your criteria."
            return "No active Yearn vaults found matching your criteria."

        # --- Sorting ---
        if recommended_only and sort_by not in {"highest_apr", "lowest_apr"}:
            matched_vaults.sort(key=_recommendation_sort_key, reverse=True)
        elif sort_by == "highest_apr":
            matched_vaults.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=True)
        elif sort_by == "lowest_apr":
            matched_vaults.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=False)
        else:  # Default sort by TVL descending
            matched_vaults.sort(
                key=lambda v: v.get("_computedTVL_USD", 0.0), reverse=True
            )

        top_vaults = matched_vaults[:MAX_RESULTS]

        # --- Format ---
        formatted_strings = []
        for vault_data in top_vaults:
            chain_id = vault_data.get("chainID")
            if chain_id:
                formatted_text = format_single_vault_data_for_llm(vault_data, chain_id)
                formatted_strings.append(formatted_text)
            else:
                formatted_strings.append(
                    f"Partial info for Vault: {vault_data.get('name', 'N/A')}"
                )

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
