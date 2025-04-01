import asyncio
import aiohttp
import traceback
import discord
import sys
import re
import json
import requests
import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from openai import OpenAI, AsyncOpenAI
from pinecone import Pinecone
from web3 import Web3

from agents import (
    Agent, Runner, function_tool, RunContextWrapper, Model,
    OpenAIResponsesModel, ModelSettings, Tool, Handoff, handoff,
    RunResult, AgentHooks, RunHooks, TResponseInputItem,
    set_default_openai_key, enable_verbose_stdout_logging,
    InputGuardrail, OutputGuardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, MaxTurnsExceeded, AgentsException,
    RunConfig
)
from agents.models.interface import ModelProvider
from agents.items import ItemHelpers
from agents import input_guardrail

sys.stdout = sys.stderr

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "key") 
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "key") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "key") 
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY", "key") 

PINECONE_INDEX_NAME = "INDEX_NAME"
PINECONE_NAMESPACE = "NAMESPACE" 

SUPPORT_USER_ID = "id" 
HUMAN_HANDOFF_TAG_PLACEHOLDER = "<@{SUPPORT_USER_ID}>" 
TICKET_CATEGORY_ID = id 
PUBLIC_TRIGGER_CHAR = "q" 
PR_MARKETING_CHANNEL_ID = id 

COOLDOWN_SECONDS = 5 
MAX_TICKET_CONVERSATION_TURNS = 15 
MAX_RESULTS_TO_SHOW = 5 
STRATEGY_FETCH_CONCURRENCY = 10 

class BDRedirectCheckOutput(BaseModel):
    is_bd_pr_request: bool = Field(..., description="True if the input message appears to be a business development, partnership, marketing, or listing proposal.")
    reasoning: str = Field(..., description="Brief explanation of why the message is classified as BD/PR or not.")

set_default_openai_key(OPENAI_API_KEY)

openai_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

openai_sync_client = OpenAI(api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_list_response = pc.list_indexes()

index_names = [index_info['name'] for index_info in index_list_response.get('indexes', [])]

if PINECONE_INDEX_NAME not in index_names:
    print(f"Error: Pinecone index '{PINECONE_INDEX_NAME}' not found in project.")
    print(f"Available indexes: {index_names}")
    sys.exit(1)

pinecone_index = pc.Index(PINECONE_INDEX_NAME)
print(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

try:
    print(f"Index stats: {pinecone_index.describe_index_stats()}")
except Exception as e:
    print(f"Warning: Could not fetch index stats for '{PINECONE_INDEX_NAME}': {e}")

WEB3_INSTANCES = {}
RPC_URLS = {
    "ethereum": f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "base": f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "polygon": f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "arbitrum": f"https://arb-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "op": f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "fantom": f"https://fantom-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
}
for name, url in RPC_URLS.items():
    try:
        WEB3_INSTANCES[name] = Web3(Web3.HTTPProvider(url))
        if not WEB3_INSTANCES[name].is_connected():
             print(f"Warning: Failed to connect to Web3 for {name}")
    except Exception as e:
        print(f"Error initializing Web3 for {name}: {e}")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

CHAIN_NAME_TO_ID = {
    "ethereum": 1, "base": 8453, "polygon": 137,
    "arbitrum": 42161, "op": 10, "fantom": 250,
}
ID_TO_CHAIN_NAME = {v: k.capitalize() for k, v in CHAIN_NAME_TO_ID.items()}

ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"}
]

try:
    with open("v1_vaults.json", "r") as f:
        V1_VAULTS = json.load(f)
except Exception as e:
    logging.warning(f"Could not load v1_vaults.json: {e}. V1 deposit checks will fail.")
    V1_VAULTS = []

def resolve_ens(name: str) -> Optional[str]:
    name = name.strip()

    if Web3.is_address(name):
        return Web3.to_checksum_address(name)

    if name.endswith(".eth") and "ethereum" in WEB3_INSTANCES:
        try:
            resolved = WEB3_INSTANCES["ethereum"].ens.address(name)
            if resolved and Web3.is_address(resolved):
                return Web3.to_checksum_address(resolved)
            else:
                 logging.warning(f"ENS name '{name}' did not resolve to a valid address.")
                 return None
        except Exception as e:
            logging.error(f"Error resolving ENS '{name}': {e}")
            return None
    return None 

def extract_address_or_ens(text: str) -> Optional[str]:
    addr_match = re.search(r'(0x[a-fA-F0-9]{40})', text)
    if addr_match:
        return addr_match.group(1)

    ens_match = re.search(r'\b([a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.eth)\b', text, re.IGNORECASE)
    if ens_match:
        return ens_match.group(1)
    return None

async def _fetch_vault_details(
    session: aiohttp.ClientSession,
    chain_id: int,
    vault_address: str,
    semaphore: asyncio.Semaphore
) -> Optional[Dict]:
    """Fetches full details for a single vault address, including its strategies list."""
    if not chain_id or not vault_address:
        logging.error(f"_fetch_vault_details called with invalid chain_id ({chain_id}) or address ({vault_address})")
        return None
    url = f"https://ydaemon.yearn.fi/{chain_id}/vaults/{vault_address}?strategiesDetails=withDetails&strategiesCondition=inQueue"
    async with semaphore:
        try:

            logging.info(f"Fetching details for vault {vault_address} on chain {chain_id}")
            async with session.get(url, timeout=10) as response:
                status = response.status
                response.raise_for_status()
                data = await response.json()
                logging.info(f"Successfully fetched details for vault {vault_address} (Status: {status})")
                return data
        except aiohttp.ClientResponseError as e:
            logging.warning(f"HTTP Error {e.status} fetching details for vault {vault_address} on chain {chain_id}: {e.message}")
            return None
        except aiohttp.ClientError as e:
            logging.warning(f"Client Error fetching details for vault {vault_address} on chain {chain_id}: {e}")
            return None
        except asyncio.TimeoutError:
             logging.warning(f"Timeout fetching details for vault {vault_address} on chain {chain_id}")
             return None
        except Exception as e:
            logging.error(f"Unexpected error processing details for vault {vault_address} on chain {chain_id}: {e}", exc_info=True) 
            return None

async def _fetch_strategy_description(
    session: aiohttp.ClientSession,
    chain_id: int,
    strategy_address: str,
    semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Fetches the description for a single strategy address."""
    if not chain_id or not strategy_address or not isinstance(strategy_address, str) or not strategy_address.startswith('0x'):
        logging.warning(f"Invalid input for fetching strategy description: chain={chain_id}, addr={strategy_address}")
        return None

    url = f"https://ydaemon.yearn.fi/{chain_id}/vaults/{strategy_address}?strategiesDetails=withDetails&strategiesCondition=inQueue"
    async with semaphore:
        try:
            logging.info(f"Fetching description for strategy {strategy_address} on chain {chain_id}")
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                description = data.get("description")
                logging.info(f"Fetched description for strategy {strategy_address}: {'Found' if description else 'Not Found'}")
                return description if isinstance(description, str) and description.strip() else None
        except aiohttp.ClientResponseError as e:
            logging.warning(f"HTTP Error {e.status} fetching description for strategy {strategy_address} on chain {chain_id}: {e.message}")
            return None
        except aiohttp.ClientError as e:
             logging.warning(f"Client Error fetching description for strategy {strategy_address} on chain {chain_id}: {e}")
             return None
        except asyncio.TimeoutError:
             logging.warning(f"Timeout fetching description for strategy {strategy_address} on chain {chain_id}")
             return None
        except Exception as e:
            logging.warning(f"Error processing description for strategy {strategy_address} on chain {chain_id}: {e}") 
            return None

@function_tool
async def search_vaults_tool(
    query: str,
    chain: Optional[str] = None,
    sort_by: Optional[str] = None 
) -> str:
    """
    Search for active Yearn vaults (v2/v3) using yDaemon API.
    Provide a search query (like token symbol 'USDC', vault name fragment 'staked eth', or vault address).
    Optionally filter by 'chain' (e.g., 'ethereum', 'base').
    Optionally sort by 'highest_apr' or 'lowest_apr' to get the top results based on net APR.
    If no sort is specified, results are sorted by TVL (descending).
    Returns detailed information for the top matching vaults, including strategy names and their descriptions.
    """
    logging.info(f"[Tool:search_vaults] Query: '{query}', Chain: '{chain}', Sort By: '{sort_by}'")
    initial_url = "https://ydaemon.yearn.fi/vaults/detected?limit=1000"
    VAULT_DETAIL_CONCURRENCY = 5
    STRATEGY_FETCH_CONCURRENCY = 10 
    MAX_RESULTS_TO_SHOW = 5

    async with aiohttp.ClientSession() as session:
        try:

            async with session.get(initial_url, timeout=15) as response:
                response.raise_for_status()
                vaults_data = await response.json()
                if not isinstance(vaults_data, list):
                    logging.error(f"[Tool:search_vaults] Unexpected yDaemon response format: {type(vaults_data)}")
                    return "Error: Received unexpected data format from vault API."
                logging.info(f"[Tool:search_vaults] Retrieved {len(vaults_data)} initial vaults from yDaemon.")
        except Exception as e:
            logging.error(f"[Tool:search_vaults] Error during yDaemon initial fetch: {e}")
            return f"Error: An unexpected error occurred while fetching initial vault data: {e}."

        filtered_vaults = vaults_data
        query_chain_id = None
        if chain:
            chain_lower = chain.lower()
            query_chain_id = CHAIN_NAME_TO_ID.get(chain_lower)
            if query_chain_id:
                filtered_vaults = [v for v in filtered_vaults if v.get("chainID") == query_chain_id]
            else:
                 logging.warning(f"[Tool:search_vaults] Chain '{chain}' not recognized. No chain filter applied.")

        query_lower = query.lower()
        matched_vaults_basic_info = []
        is_address_query = query_lower.startswith("0x") and len(query_lower) == 42

        for v in filtered_vaults:

            vault_address = v.get("address", "").lower()
            name = v.get("name", "").lower()
            symbol = v.get("symbol", "").lower()
            token_info = v.get("token", {})
            token_name = token_info.get("name", "").lower() if token_info else ""
            token_symbol = token_info.get("symbol", "").lower() if token_info else ""

            match = False
            if is_address_query:
                if query_lower == vault_address: match = True
            elif query_lower == symbol or query_lower == token_symbol: match = True
            elif query_lower in name or query_lower in symbol or query_lower in token_name or query_lower in token_symbol: match = True

            if match:
                apr_data = v.get("apr", {})
                primary_apr = apr_data.get("netAPR")
                fallback_apr = apr_data.get("forwardAPR", {}).get("netAPR")
                apr_value = 0.0
                if primary_apr is not None and isinstance(primary_apr, (int, float)): apr_value = float(primary_apr)
                elif fallback_apr is not None and isinstance(fallback_apr, (int, float)): apr_value = float(fallback_apr)
                v["_computedAPR"] = apr_value

                try: v["_computedTVL"] = float(v.get('tvl', {}).get('tvl', 0))
                except (ValueError, TypeError): v["_computedTVL"] = 0.0

                matched_vaults_basic_info.append(v)

        logging.info(f"[Tool:search_vaults] Found {len(matched_vaults_basic_info)} vaults matching query '{query}' after initial filter.")

        if not matched_vaults_basic_info:
            return "No active vaults found matching your criteria."

        final_list_basic_info = []
        sort_applied = "None"
        if sort_by in ["highest_apr", "lowest_apr"]:
            sort_applied = sort_by
            vaults_with_apr = [v for v in matched_vaults_basic_info if v.get("_computedAPR", 0.0) != 0.0]
            if not vaults_with_apr:
                logging.warning(f"[Tool:search_vaults] No vaults with non-zero APR found for sorting by {sort_by}. Falling back to TVL sort.")
                sort_by = None
            else:
                reverse_sort = (sort_by == "highest_apr")
                sorted_vaults = sorted(vaults_with_apr, key=lambda v: v.get("_computedAPR", 0.0), reverse=reverse_sort)
                final_list_basic_info = sorted_vaults[:MAX_RESULTS_TO_SHOW]

        if sort_by not in ["highest_apr", "lowest_apr"]:
            sort_applied = "TVL (Descending)"
            logging.info("[Tool:search_vaults] Sorting by TVL (desc).")
            sorted_by_tvl = sorted(matched_vaults_basic_info, key=lambda v: v.get("_computedTVL", 0.0), reverse=True)
            final_list_basic_info = sorted_by_tvl[:MAX_RESULTS_TO_SHOW]

        logging.info(f"[Tool:search_vaults] Selected top {len(final_list_basic_info)} vaults based on {sort_applied}.")

        detail_fetch_tasks = []
        detail_semaphore = asyncio.Semaphore(VAULT_DETAIL_CONCURRENCY)
        vaults_to_fetch_details_input = []
        logging.info("Starting preparation loop for detail fetching.")

        try:
            for idx, v_basic in enumerate(final_list_basic_info):
                chain_id = v_basic.get("chainID")
                vault_address = v_basic.get("address")
                vault_name_debug = v_basic.get('name', 'N/A')
                logging.info(f"Prep #{idx}: Processing vault '{vault_name_debug}'") 

                is_valid_for_fetch = isinstance(chain_id, int) and isinstance(vault_address, str) and vault_address.startswith('0x') and len(vault_address) == 42
                if is_valid_for_fetch:
                    vaults_to_fetch_details_input.append((chain_id, vault_address))
                    logging.info(f"Prep #{idx}: Added vault '{vault_name_debug}' ({vault_address}) to fetch list.") 
                else:
                    logging.warning(f"Prep #{idx}: Skipping detail fetch for vault '{vault_name_debug}' due to missing/invalid chainID or address format.")
            logging.info(f"Finished preparation loop. Vaults prepared for fetch: {len(vaults_to_fetch_details_input)}")
        except Exception as e_prep:
            logging.error(f"Unexpected error during detail fetch preparation loop: {e_prep}", exc_info=True)
            return f"An internal error occurred while preparing vault details: {e_prep}"

        if not vaults_to_fetch_details_input:
             logging.warning("Preparation resulted in no vaults to fetch details for.")
             return "Found matching vault(s), but couldn't verify their details (missing/invalid ID or address in initial data)."

        logging.info(f"Attempting to fetch full details for {len(vaults_to_fetch_details_input)} prepared vaults.")
        for chain_id, vault_address in vaults_to_fetch_details_input:
             task = asyncio.create_task(
                 _fetch_vault_details(session, chain_id, vault_address, detail_semaphore)
             )
             detail_fetch_tasks.append(task)

        detailed_vault_results_raw = await asyncio.gather(*detail_fetch_tasks, return_exceptions=True)
        logging.info(f"Finished fetching full vault details. Received {len(detailed_vault_results_raw)} results/exceptions.")

        final_vault_details = []
        vault_detail_map = {} 
        for i, result_or_exc in enumerate(detailed_vault_results_raw):
            original_chain_id, original_address = vaults_to_fetch_details_input[i]
            if isinstance(result_or_exc, dict):
                final_vault_details.append(result_or_exc)
                vault_detail_map[original_address.lower()] = result_or_exc 
            elif isinstance(result_or_exc, Exception):
                logging.error(f"Exception fetching details for vault {original_address} on chain {original_chain_id}: {result_or_exc}")
            else:
                 logging.warning(f"Detail fetch for vault {original_address} on chain {original_chain_id} returned None or unexpected type ({type(result_or_exc)}).")

        if not final_vault_details:
             logging.warning("Failed to fetch details successfully for any selected vaults.")
             return "Found matching vault(s), but could not retrieve their full details due to errors during fetch."

        strategy_fetch_tasks = []
        strategy_results_map = {} 
        strategy_semaphore = asyncio.Semaphore(STRATEGY_FETCH_CONCURRENCY)
        strategies_to_fetch = [] 

        logging.info("Preparing to fetch strategy descriptions.")
        for vault_detail in final_vault_details:
            vault_chain_id = vault_detail.get("chainID")
            strategies = vault_detail.get("strategies", [])
            if not vault_chain_id or not strategies:
                continue
            for strategy in strategies:
                strat_addr = strategy.get("address")
                if strat_addr and isinstance(strat_addr, str) and strat_addr.startswith('0x'):

                    if strat_addr.lower() not in strategy_results_map:
                         strategies_to_fetch.append((vault_chain_id, strat_addr))
                         strategy_results_map[strat_addr.lower()] = None 
                else:
                     logging.warning(f"Invalid strategy address found in vault {vault_detail.get('address')}: {strat_addr}")

        if strategies_to_fetch:
            logging.info(f"Attempting to fetch descriptions for {len(strategies_to_fetch)} unique strategies.")
            for chain_id, strat_addr in strategies_to_fetch:
                task = asyncio.create_task(
                    _fetch_strategy_description(session, chain_id, strat_addr, strategy_semaphore)
                )

                strategy_fetch_tasks.append({"address": strat_addr, "task": task})

            await asyncio.gather(*[t["task"] for t in strategy_fetch_tasks], return_exceptions=True)
            logging.info("Finished fetching strategy descriptions.")

            for item in strategy_fetch_tasks:
                task = item["task"]
                strat_addr_lower = item["address"].lower()
                if task.done() and not task.cancelled() and task.exception() is None:
                    strategy_results_map[strat_addr_lower] = task.result() 
                elif task.exception():
                     logging.error(f"Exception fetching description for strategy {item['address']}: {task.exception()}")
                     strategy_results_map[strat_addr_lower] = None 
                else:

                      strategy_results_map[strat_addr_lower] = None

        summaries = []
        num_total_matches = len(matched_vaults_basic_info)
        num_shown = len(final_vault_details)

        header = f"Found {num_total_matches} vault(s) matching '{query}'."
        if num_total_matches > len(final_list_basic_info):
             header += f" Showing top {num_shown} (of {len(final_list_basic_info)} selected by {sort_applied}) with details:"
        elif num_shown > 0 :
             header += f" Details for {num_shown} vault(s) (sorted by {sort_applied}):"
        else:
             header += " Could not retrieve details."
        summaries.append(header)
        summaries.append("---")

        for i, v_detail in enumerate(final_vault_details):

            vault_addr = v_detail.get("address", "N/A")
            vault_chain_id = v_detail.get("chainID", "N/A")
            vault_url = f"https://yearn.fi/v3/{vault_chain_id}/{vault_addr}" if vault_addr != "N/A" and vault_chain_id != "N/A" else "N/A"
            vault_name = v_detail.get('name', 'Unknown Vault')
            vault_name_link = f"[{vault_name}]({vault_url})" if vault_url != "N/A" else vault_name
            apr_data = v_detail.get("apr", {})
            primary_apr = apr_data.get("netAPR")
            fallback_apr = apr_data.get("forwardAPR", {}).get("netAPR")
            apr_value = 0.0
            if primary_apr is not None and isinstance(primary_apr, (int, float)): apr_value = float(primary_apr)
            elif fallback_apr is not None and isinstance(fallback_apr, (int, float)): apr_value = float(fallback_apr)
            display_apr = apr_value * 100
            tvl_usd = v_detail.get('tvl', {}).get('tvl', 0)
            try: tvl_display = f"${float(tvl_usd):,.2f}"
            except (ValueError, TypeError): tvl_display = "$N/A"
            token_info = v_detail.get("token", {})
            token_name = token_info.get("name", "N/A") if token_info else "N/A"
            token_symbol = token_info.get("symbol", "N/A") if token_info else "N/A"
            desc = v_detail.get("description", "No description available.")
            if desc and len(desc) > 200: desc = desc[:197] + "..."
            chain_name_display = ID_TO_CHAIN_NAME.get(vault_chain_id, f"Unknown Chain ({vault_chain_id})")
            risk_level = v_detail.get('info', {}).get('riskLevel', 'N/A')

            summary_lines = [
                f"**{i+1}. Vault:** {vault_name_link}",
                f"   - Symbol: {v_detail.get('symbol', 'N/A')}",
                f"   - Address: `{vault_addr}`",
                f"   - Chain: {chain_name_display}",
                f"   - Token: {token_name} ({token_symbol})",
                f"   - Net APR: {display_apr:.2f}%",
                f"   - TVL: {tvl_display}",
                f"   - Risk Level: {risk_level}",
                f"   - Description: {desc}"
            ]

            strategy_details_list = []
            strategies = v_detail.get("strategies", [])
            if strategies:
                summary_lines.append(f"   - Strategies ({len(strategies)}):")
                for strategy in strategies:
                    strat_name = strategy.get("name", "Unnamed Strategy")
                    strat_addr = strategy.get("address")
                    description = None
                    if strat_addr and isinstance(strat_addr, str):

                         description = strategy_results_map.get(strat_addr.lower())

                    if description:

                        strategy_details_list.append(f"     - **{strat_name}:** {description}")
                    else:
                        strategy_details_list.append(f"     - **{strat_name}:** (Description unavailable)")
                summary_lines.extend(strategy_details_list)
            else:
                 summary_lines.append("   - Strategies: None listed in details.")

            summaries.append("\n".join(summary_lines))

        result_text = "\n\n".join(summaries)
        logging.info(f"[Tool:search_vaults] Formatted Result:\n{result_text}")
        return result_text

@function_tool
async def query_v1_deposits_tool(user_address_or_ens: str, token_symbol: Optional[str] = None) -> str:
    """
    Checks for deposits in deprecated Yearn V1 vaults (Ethereum only) for a given user address or ENS name.
    Optionally filter by token symbol (e.g., 'USDC', 'yCRV').
    Returns withdrawal instructions if deposits are found.
    """
    logging.info(f"[Tool:query_v1_deposits] Checking V1 for {user_address_or_ens}, Token: {token_symbol}")
    if not V1_VAULTS:
        return "V1 vault data is not loaded. Cannot check V1 deposits."
    if "ethereum" not in WEB3_INSTANCES:
        return "Ethereum connection is not available. Cannot check V1 deposits."

    web3_eth = WEB3_INSTANCES["ethereum"]
    resolved_address = resolve_ens(user_address_or_ens)

    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum address."

    try:
        user_checksum_addr = Web3.to_checksum_address(resolved_address)
    except ValueError:
         return f"Invalid Ethereum address format after resolution: {resolved_address}"

    found_deposits = []
    for vault in V1_VAULTS:

        if token_symbol and token_symbol.lower() not in vault.get("symbol", "").lower():
            continue

        try:
            vault_checksum_addr = Web3.to_checksum_address(vault.get("address"))
            contract = web3_eth.eth.contract(address=vault_checksum_addr, abi=ERC20_ABI)

            balance = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)

            if balance > 0:
                decimals = int(vault.get("decimals", 18)) 
                display_balance = balance / (10 ** decimals)
                etherscan_link = f"https://etherscan.io/address/{vault_checksum_addr}"
                vault_name = vault.get('name', 'Unknown V1 Vault')
                vault_symbol = vault.get('symbol', 'N/A')

                instruction_verb = "'withdrawAll'" if vault.get("withdraw_all", False) else "'withdraw' (entering your balance)"
                instr = (
                    f"**Vault:** [{vault_name}]({etherscan_link}) (Symbol: {vault_symbol})\n"
                    f"  Deposit Found: {display_balance:.6f} tokens.\n"
                    f"  *Withdrawal:* Go to Etherscan link -> 'Contract' tab -> 'Write Contract' -> Connect Wallet -> Use the {instruction_verb} function."
                )
                found_deposits.append(instr)
        except Exception as e:
            logging.error(f"[Tool:query_v1_deposits] Error checking V1 vault {vault.get('address')} for {user_checksum_addr}: {e}")

    if found_deposits:
        result = "**Deprecated V1 Vault Deposits Found (Ethereum Only):**\n\n" + "\n\n".join(found_deposits)
    else:
        result = "No deposits found in deprecated V1 vaults for this address."

    logging.info(f"[Tool:query_v1_deposits] Result for {user_checksum_addr}: {result}")
    return result

async def _fetch_active_balance(vault_info: Dict, web3_instance: Web3, user_checksum_addr: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            vault_addr_str = vault_info.get("address")
            if not vault_addr_str: return (vault_info, None)

            vault_checksum_addr = Web3.to_checksum_address(vault_addr_str)
            contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=ERC20_ABI)

            balance = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)
            return (vault_info, balance)
        except Exception as e:
            logging.warning(f"Error fetching balance for vault {vault_addr_str} for user {user_checksum_addr}: {e}")
            return (vault_info, None)

@function_tool
async def query_active_deposits_tool(user_address_or_ens: str, chain: Optional[str] = None, token_symbol: Optional[str] = None) -> str:
    """
    Checks for deposits in active Yearn vaults (v2/v3) across specified or all supported chains for a user address or ENS name.
    Optionally filter by 'chain' (e.g., 'ethereum', 'base'). If no chain is specified, checks all supported chains.
    Optionally filter by 'token_symbol' (e.g., 'USDC').
    Returns details of found deposits including vault name, link, symbol, and balance.
    """
    logging.info(f"[Tool:query_active_deposits] Checking Active for {user_address_or_ens}, Chain: {chain}, Token: {token_symbol}")

    resolved_address = resolve_ens(user_address_or_ens)
    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum address."

    chains_to_check = []
    if chain:
        chain_lower = chain.lower()
        if chain_lower in WEB3_INSTANCES:
            chains_to_check.append(chain_lower)
        else:
            return f"Unsupported or unrecognized chain: {chain}. Supported chains are: {', '.join(WEB3_INSTANCES.keys())}."
    else:
        chains_to_check = list(WEB3_INSTANCES.keys())

    all_results = []
    total_deposits_found = 0

    url = "https://ydaemon.yearn.fi/vaults/detected?limit=1000"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15)
        response.raise_for_status()
        all_vaults_data = response.json()
        if not isinstance(all_vaults_data, list):
             logging.error("[Tool:query_active_deposits] Unexpected yDaemon response format.")
             return "Error: Received unexpected data format from vault API."
    except Exception as e:
        logging.error(f"[Tool:query_active_deposits] Failed to fetch vault list from yDaemon: {e}")
        return f"Error: Could not fetch the list of active vaults: {e}"

    for chain_name in chains_to_check:
        logging.info(f"[Tool:query_active_deposits] Checking chain: {chain_name}")
        web3_instance = WEB3_INSTANCES[chain_name]
        try:
            user_checksum_addr = Web3.to_checksum_address(resolved_address) 
        except ValueError:

             all_results.append(f"**{chain_name.capitalize()}**: Invalid address format for {resolved_address}")
             continue

        chain_id = CHAIN_NAME_TO_ID.get(chain_name)
        if not chain_id: continue 

        chain_vaults = [v for v in all_vaults_data if v.get("chainID") == chain_id]
        if token_symbol:
            token_lower = token_symbol.lower()
            chain_vaults = [
                v for v in chain_vaults
                if token_lower in v.get("token", {}).get("symbol", "").lower()
            ]

        if not chain_vaults:
            logging.info(f"[Tool:query_active_deposits] No vaults matching criteria on {chain_name}.")
            continue 

        semaphore = asyncio.Semaphore(20) 
        tasks = [_fetch_active_balance(v, web3_instance, user_checksum_addr, semaphore) for v in chain_vaults]
        balance_results = await asyncio.gather(*tasks)

        chain_deposits = []
        for vault_info, balance in balance_results:
            if balance is not None and balance > 0:
                try:
                    decimals = int(vault_info.get("decimals", 18))
                    display_balance = balance / (10 ** decimals)
                    vault_address = Web3.to_checksum_address(vault_info.get("address"))
                    vault_url = f"https://yearn.fi/v3/{chain_id}/{vault_address}"
                    vault_name = vault_info.get('name', 'Unknown Vault')
                    vault_symbol = vault_info.get('symbol', 'N/A')

                    deposit_info = (
                        f"**Vault:** [{vault_name}]({vault_url}) (Symbol: {vault_symbol})\n"
                        f"  Address: `{vault_address}`\n"
                        f"  Deposit: {display_balance:.6f} tokens"
                    )
                    chain_deposits.append(deposit_info)
                    total_deposits_found += 1
                except Exception as e:
                    logging.error(f"Error processing deposit for vault {vault_info.get('address')} on {chain_name}: {e}")

        if chain_deposits:
            all_results.append(f"**{chain_name.capitalize()} Deposits:**\n" + "\n\n".join(chain_deposits))
        else:
            logging.info(f"[Tool:query_active_deposits] No active deposits found on {chain_name} for {user_checksum_addr}.")

            if len(chains_to_check) == 1:
                 all_results.append(f"No active deposits found on {chain_name.capitalize()} for this address" + (f" matching token '{token_symbol}'." if token_symbol else "."))

    if total_deposits_found > 0:
        final_result = "\n\n---\n\n".join(all_results)
    elif len(chains_to_check) > 1 : 
        final_result = "No active vault deposits found for that address on any supported chain" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    elif not all_results: 
         final_result = f"No active vault deposits found on {chains_to_check[0].capitalize()} for this address" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    else: 
         final_result = "\n\n".join(all_results) 

    logging.info(f"[Tool:query_active_deposits] Final Result for {resolved_address}: {final_result}")
    return final_result

@function_tool
async def answer_from_docs_tool(user_query: str) -> str:
    """
    Answers questions based on Yearn's documentation using a vector search (Pinecone).
    Use this for general questions about how Yearn works, concepts, strategies, risks, or specific documentation details.
    Do not use for real-time data like APRs, TVL, or user balances.
    """
    logging.info(f"[Tool:answer_from_docs] --- Tool Invoked ---")
    logging.info(f"[Tool:answer_from_docs] Received query: '{user_query}'")
    top_k = 15

    try:
        response = await asyncio.to_thread(
            openai_sync_client.embeddings.create,
            model="text-embedding-3-small",
            input=[user_query],
            encoding_format="float"
        )
        query_embedding = response.data[0].embedding
        logging.info(f"[Tool:answer_from_docs] Successfully generated embedding for query.")
    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error generating query embedding: {e}")
        return "Sorry, I couldn't process your question to search the documentation."

    try:
        search_results = await asyncio.to_thread(
            pinecone_index.query,
            namespace=PINECONE_NAMESPACE,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = search_results.get("matches", [])
        logging.info(f"[Tool:answer_from_docs] Pinecone query returned {len(matches)} matches.")
        if matches:

             first_match_metadata = matches[0].metadata if matches[0].metadata else {}
             logging.info(f"[Tool:answer_from_docs] Metadata keys of first match: {list(first_match_metadata.keys())}")
             top_match_info = {
                 "id": matches[0].id,
                 "score": matches[0].score,

                 "filename": first_match_metadata.get('filename', 'N/A'),
                 "chunk_id": first_match_metadata.get('chunk_id', 'N/A'),
                 "text_preview": (first_match_metadata.get('text_preview', '')[:100] + "...") if first_match_metadata.get('text_preview') else "N/A"
             }
             logging.info(f"[Tool:answer_from_docs] Top match details (using standalone keys): {top_match_info}")

    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error querying Pinecone: {e}")
        return "Sorry, I encountered an error while searching the documentation."

    context_pieces = []
    if matches:
        for match in matches:
            metadata = match.get("metadata", {})

            text_chunk = metadata.get("text_preview", "") 
            filename = metadata.get("filename", "Unknown Source") 
            chunk_id = metadata.get("chunk_id", "N/A") 

            source_info = f"{filename} (Chunk: {chunk_id})"

            if text_chunk: 
                context_pieces.append(f"Source: {source_info}\nContent:\n{text_chunk}")
            else:
                 logging.warning(f"Match ID {match.id} had empty 'text_preview' in metadata.")

        context_text = "\n\n---\n\n".join(context_pieces)
        logging.info(f"[Tool:answer_from_docs] Built context string (length: {len(context_text)}). Preview:\n---\n{context_text[:500]}...\n---")
    else:
        logging.info("[Tool:answer_from_docs] No relevant documents found in Pinecone.")
        logging.info(f"[Tool:answer_from_docs] --- Tool Returning Early (No Matches) ---")
        return "I couldn't find any relevant information in the documentation to answer that question."

    if not context_text:
         logging.warning("[Tool:answer_from_docs] Context string is empty even though matches were found. Check metadata keys ('text_preview', 'filename', 'chunk_id').")
         logging.info(f"[Tool:answer_from_docs] --- Tool Returning Early (Empty Context Built) ---")
         return "I found potential matches in the documentation, but couldn't extract the content correctly."

    system_prompt = (
        "You are an assistant answering questions based *only* on the provided Yearn documentation context. "
        "Use the information given below to answer the user's question accurately and concisely. "
        "If the answer is not present in the context, state that clearly ('I couldn't find information about X in the provided context.'). Do not add external knowledge. "
        "Cite the source if possible (included in the context)."
    )
    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Documentation Context:\n{context_text}\n\nUser Question: {user_query}"}
    ]
    logging.info(f"[Tool:answer_from_docs] Sending final prompt to LLM (gpt-4o-mini). User content length: {len(messages_for_llm[1]['content'])}")

    try:
        response = await openai_async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_llm,
            temperature=0.2

        )
        final_answer = response.choices[0].message.content.strip()
        logging.info(f"[Tool:answer_from_docs] Received final answer from LLM: '{final_answer}'")
        logging.info(f"[Tool:answer_from_docs] --- Tool Execution Complete ---")
        return final_answer
    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error generating final answer: {e}")
        logging.info(f"[Tool:answer_from_docs] --- Tool Returning Error (LLM Call Failed) ---")
        return "Sorry, I found relevant documentation but encountered an error while formulating the final answer."

TContext = Any

yearn_data_agent = Agent[TContext](
    name="Yearn Data Specialist",

    instructions=(
        "You are a specialist in retrieving specific data about Yearn Finance vaults and user deposits using provided tools.\n"
        "Use the available tools to answer questions about:\n"
        "- Finding active vaults based on criteria (use 'search_vaults_tool'). This includes details like APR, TVL, risk, description, and **strategies**.\n"
        "- Checking user deposits in active vaults (use 'query_active_deposits_tool').\n"
        "- Checking user deposits in old V1 vaults (use 'query_v1_deposits_tool').\n"
        "Provide the information clearly based on the tool's output. "

        "If a user provides an address or ENS, proactively check *both* active and V1 deposits unless asked otherwise.\n"
        "Always resolve ENS names to addresses before using deposit checking tools.\n"
        "When displaying APR, ensure it's presented as a percentage (multiply raw value by 100 if the tool hasn't already)." 
    ),
    tools=[
        search_vaults_tool,
        query_v1_deposits_tool,
        query_active_deposits_tool
    ],
    model="o3-mini" 

)

docs_qa_agent = Agent[TContext](
    name="Yearn Docs QA Specialist",
    instructions=(
        "You are a specialist in answering questions based on Yearn's documentation. "
        "Use the 'answer_from_docs_tool' to find relevant information and formulate an answer. "
        "Answer *only* based on the information provided by the tool. If the tool indicates no information was found, state that."
        "Do not answer questions about real-time data (APR, TVL, balances) - state that these require the Data Specialist."
    ),
    tools=[answer_from_docs_tool],
    model="gpt-4o-mini", 
    model_settings=ModelSettings(temperature=0.2)
)

bd_redirect_guardrail_agent = Agent[TContext](
    name="BD/PR Guardrail Check",
    instructions=(
        "Analyze the user's message. Determine if it is primarily a business development proposal, partnership request, "
        "marketing collaboration offer, token listing request, or similar commercial inquiry targeting Yearn. "
        "Focus on the intent behind the message. Ignore simple questions about Yearn itself, even if from a project."
        "Output 'is_bd_pr_request' as true ONLY if it's clearly one of these proposal types."
    ),
    output_type=BDRedirectCheckOutput,
    model="gpt-4o-mini", 
    model_settings=ModelSettings(temperature=0.1) 
)

@input_guardrail(name="BD/PR Redirect Guardrail")
async def bd_pr_redirect_guardrail(
    ctx: RunContextWrapper[TContext],
    agent: Agent, 
    input_data: Union[str, List[TResponseInputItem]]
) -> GuardrailFunctionOutput:
    """
    Checks if the initial user input is a BD/PR request.
    If it is, triggers a tripwire to stop the main agent flow.
    """

    if isinstance(input_data, str):
        text_input = input_data
    elif isinstance(input_data, list):

        text_input = ""
        for item in reversed(input_data):
             if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
                 text_input = item["content"]
                 break
    else:
        text_input = "" 

    if not text_input:

        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    logging.info(f"[Guardrail:BD/PR] Analyzing input: '{text_input[:100]}...'")

    try:

        guardrail_runner = Runner() 
        result = await guardrail_runner.run(
            starting_agent=bd_redirect_guardrail_agent,
            input=text_input,

            run_config=RunConfig(workflow_name="BD/PR Guardrail Check", tracing_disabled=True) 
        )
        check_output = result.final_output_as(BDRedirectCheckOutput)

        logging.info(f"[Guardrail:BD/PR] Check result: is_bd_pr={check_output.is_bd_pr_request}, Reasoning: {check_output.reasoning}")

        return GuardrailFunctionOutput(
            output_info=check_output, 
            tripwire_triggered=check_output.is_bd_pr_request
        )
    except Exception as e:
        logging.error(f"[Guardrail:BD/PR] Error during check: {e}", exc_info=True)

        return GuardrailFunctionOutput(output_info={"error": str(e)}, tripwire_triggered=False)

triage_agent = Agent[TContext](
    name="Support Triage Agent",
    instructions=(
        "You are the first point of contact for Yearn support in Discord tickets. Your goal is to understand the user's request and either handle it directly, delegate it to a specialist agent, or request human assistance.\n\n"
        "**Workflow:**\n"
        "1.  **Analyze Request:** Determine the user's core need.\n"
        "2.  **BD/PR/Marketing:** (Handled by Guardrail - You won't see these if detected early). If somehow a BD/PR message gets past the guardrail, state you cannot handle it and redirect them to the #pr-marketing channel, mentioning <@{BD_CONTACT_USER_ID}>.\n"
        "3.  **Specific Data Request:** If the user asks for specific, real-time data (vault details, APRs, TVL, their deposits/balances), handoff to the 'Yearn Data Specialist'. If they provide an address/ENS, assume they want deposit info unless asked otherwise.\n"
        "4.  **General/Docs Question:** If the user asks a general question about how Yearn works, concepts, risks, strategy types, or documentation details, handoff to the 'Yearn Docs QA Specialist'.\n"
        "5.  **UI Errors/Bugs/Complex Issues:** If the user reports a UI error, a website issue, a transaction failure you cannot explain with tools (e.g., not just insufficient balance), mentions a GitHub issue, or asks a question requiring deep investigation or sensitive account review, state clearly that you cannot resolve this and that human support is needed. **Crucially, include the text '<@{SUPPORT_USER_ID}>' in your response** to ping the human support member.\n"
        "6.  **Ambiguity:** If unsure about the user's need (Data vs. Docs vs. Bug), ask for clarification before handing off or requesting human help.\n"
        "7.  **Greetings/Chit-chat:** Respond briefly and politely to simple greetings or off-topic messages without handing off or requesting human help.\n"
        "8.  **Farewells:** If the user says thank you, goodbye, or indicates the issue is resolved, respond politely and conclude the interaction (e.g., 'Glad I could help! Let us know if you need anything else.')."
    ),
    handoffs=[
        handoff(yearn_data_agent, tool_description_override="Handoff to specialist for specific Yearn data (vaults, deposits, APR, TVL)."),
        handoff(docs_qa_agent, tool_description_override="Handoff to specialist for questions based on Yearn documentation (concepts, how-to, risks).")
    ],

    input_guardrails=[bd_pr_redirect_guardrail],
    model="gpt-4o-mini", 
    model_settings=ModelSettings(temperature=0.4) 
)

conversation_threads: Dict[int, List[TResponseInputItem]] = {} 
stopped_channels: set[int] = set() 
pending_messages: Dict[int, str] = {} 
pending_tasks: Dict[int, asyncio.Task] = {} 

class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self.runner = Runner 

    async def on_ready(self):
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logging.info(f"Monitoring ticket category ID: {TICKET_CATEGORY_ID}")
        logging.info(f"Support User ID for triggers: {SUPPORT_USER_ID}")
        logging.info(f"Public trigger character: '{PUBLIC_TRIGGER_CHAR}'")
        print("------")

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):

        if isinstance(channel, discord.TextChannel) and channel.category and channel.category.id == TICKET_CATEGORY_ID:
            logging.info(f"New ticket channel created: {channel.name} (ID: {channel.id}). Initializing state.")
            conversation_threads[channel.id] = []
            stopped_channels.discard(channel.id) 
            pending_messages.pop(channel.id, None)
            if channel.id in pending_tasks:
                pending_tasks.pop(channel.id).cancel()

    async def on_message(self, message: discord.Message):

        if message.author.bot or message.author.id == self.user.id:
            return

        is_reply = message.reference is not None
        is_support_trigger = str(message.author.id) == SUPPORT_USER_ID and message.content.strip() == PUBLIC_TRIGGER_CHAR

        if is_reply and is_support_trigger:

            try:
                original_message = await message.channel.fetch_message(message.reference.message_id)
                if original_message and not original_message.author.bot:
                    logging.info(f"Public trigger '{PUBLIC_TRIGGER_CHAR}' detected by {message.author.name} in reply to user {original_message.author.name} in channel {message.channel.id}")

                    try:
                        await message.delete()
                        logging.info(f"Deleted trigger message {message.id}")
                    except discord.Forbidden:
                        logging.warning(f"Missing permissions to delete trigger message {message.id} in {message.channel.id}")
                    except discord.NotFound:
                        logging.warning(f"Trigger message {message.id} already deleted.")

                    original_content = original_message.content
                    if not original_content:
                        logging.info("Original message has no text content to process.")
                        return

                    async with message.channel.typing():
                        try:

                            run_config = RunConfig(workflow_name="Public Channel Query")
                            result: RunResult = await self.runner.run(
                                starting_agent=triage_agent, 
                                input=original_content,
                                max_turns=5, 
                                run_config=run_config

                            )

                            raw_reply_content = result.final_output if result.final_output else "I couldn't determine a response."
                            actual_mention = f"<@{SUPPORT_USER_ID}>" 
                            reply_content = raw_reply_content.replace(HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                            await original_message.reply(reply_content, suppress_embeds=True)
                            logging.info(f"Sent public reply to {original_message.id} in {message.channel.id}")

                        except InputGuardrailTripwireTriggered as e:
                            logging.warning(f"BD/PR Input Guardrail triggered for public query (Original msg ID: {original_message.id}). Guardrail Output: {e.guardrail_result.output.output_info}")
                            reply_content = ( 
                                f"Thank you for your interest! For partnership, marketing, listing, or business development proposals, "
                                f"please post your message in the <#{PR_MARKETING_CHANNEL_ID}> channel and tag **name**. "
                                f"They handle these inquiries."
                            )
                            await original_message.reply(reply_content) 

                        except MaxTurnsExceeded as e:
                             logging.warning(f"Max turns exceeded for public query (Original msg ID: {original_message.id}): {e}")
                             reply_content = f"Sorry, the request took too long to process. Please try simplifying or ask <@{SUPPORT_USER_ID}> for help." 
                             await original_message.reply(reply_content) 

                        except AgentsException as e:
                             logging.error(f"Agent SDK error during public query (Original msg ID: {original_message.id}): {e}")
                             reply_content = f"Sorry, an error occurred while processing your request ({type(e).__name__})." 
                             await original_message.reply(reply_content) 
                        except Exception as e:
                             logging.error(f"Unexpected error during public query processing (Original msg ID: {original_message.id}): {e}", exc_info=True)

                             await original_message.reply(reply_content) 

                    return 

            except discord.NotFound:
                logging.warning(f"Original message for reply {message.id} not found.")
            except discord.Forbidden:
                 logging.warning(f"Missing permissions to fetch original message for reply {message.id}.")
            except Exception as e:
                 logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)

        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category or message.channel.category.id != TICKET_CATEGORY_ID:
            return

        channel_id = message.channel.id

        if channel_id in stopped_channels:
            logging.info(f"Ignoring message in stopped channel {channel_id}")
            return

        if message.content.strip().lower() == "/stop":
            stopped_channels.add(channel_id)
            conversation_threads.pop(channel_id, None) 
            pending_messages.pop(channel_id, None)
            if channel_id in pending_tasks:
                pending_tasks.pop(channel_id).cancel()
            await message.reply("Support bot stopped for this channel. Conversation history cleared.")
            logging.info(f"Bot stopped in channel {channel_id} by command.")
            return

        logging.info(f"Processing ticket message in {channel_id} from {message.author.name}")

        if channel_id not in pending_messages:
            pending_messages[channel_id] = message.content
        else:
            pending_messages[channel_id] += "\n" + message.content 

        if channel_id in pending_tasks:
            pending_tasks[channel_id].cancel()
            logging.debug(f"Cancelled pending task for channel {channel_id}")

        pending_tasks[channel_id] = asyncio.create_task(self.process_ticket_message(channel_id))
        logging.debug(f"Scheduled processing task for channel {channel_id} in {COOLDOWN_SECONDS}s")

    async def process_ticket_message(self, channel_id: int):
        """Processes aggregated messages for a ticket channel after cooldown."""
        try:
            await asyncio.sleep(COOLDOWN_SECONDS)
        except asyncio.CancelledError:
            logging.debug(f"Processing task for channel {channel_id} cancelled (new message arrived).")
            return 

        aggregated_text = pending_messages.pop(channel_id, None)
        pending_tasks.pop(channel_id, None)

        if not aggregated_text: return
        channel = self.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel): return
        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": aggregated_text}]

        logging.info(f"Processing aggregated text for ticket {channel_id}: '{aggregated_text[:100]}...'")

        async with channel.typing():
            final_reply = "An unexpected error occurred." 
            should_stop_processing = False 

            try:

                run_config = RunConfig(
                    workflow_name=f"Ticket Channel {channel_id}",
                    group_id=str(channel_id) 
                )
                result: RunResult = await self.runner.run(
                    starting_agent=triage_agent, 
                    input=input_list,
                    max_turns=MAX_TICKET_CONVERSATION_TURNS,
                    run_config=run_config

                )

                conversation_threads[channel_id] = result.to_input_list()

                raw_final_reply = result.final_output if result.final_output else "I'm not sure how to respond to that."

                actual_mention = f"<@{SUPPORT_USER_ID}>" 
                final_reply = raw_final_reply.replace(HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                if actual_mention in final_reply:
                    logging.info(f"Human handoff tag detected in response for channel {channel_id}.")
                    should_stop_processing = True 

                farewell_keywords = ["thank", "bye", "goodbye", "stop talking", "that's all", "resolved", "worked", "fixed"]
                user_said_farewell = any(kw in aggregated_text.lower() for kw in farewell_keywords)
                agent_said_farewell = "glad i could help" in final_reply.lower() or "concluding interaction" in final_reply.lower() 

                if user_said_farewell or agent_said_farewell:
                    if not should_stop_processing: 
                         final_reply += "\n\n*(Conversation ended. Use `/stop` to clear history or just start a new query.)*"
                    should_stop_processing = True
                    logging.info(f"Conversation ended naturally or by farewell in channel {channel_id}.")

            except InputGuardrailTripwireTriggered as e:

                 logging.warning(f"BD/PR Input Guardrail triggered in channel {channel_id}. Guardrail Output: {e.guardrail_result.output.output_info}")

                 final_reply = (
                     f"Thank you for your interest! For partnership, marketing, or business development proposals, "
                     f"please post your message in the <#{PR_MARKETING_CHANNEL_ID}> channel and tag **name**. "
                     f"They handle these inquiries."
                 )
                 should_stop_processing = True 

                 conversation_threads.pop(channel_id, None) 

            except MaxTurnsExceeded:
                 logging.warning(f"Max turns ({MAX_TICKET_CONVERSATION_TURNS}) exceeded in channel {channel_id}.")
                 final_reply = f"This conversation has reached its maximum length. <@{SUPPORT_USER_ID}> may need to intervene."
                 should_stop_processing = True
                 conversation_threads.pop(channel_id, None) 

            except AgentsException as e:
                 logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                 final_reply = f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again or notify <@{SUPPORT_USER_ID}>."

            except Exception as e:
                 logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                 final_reply = f"An unexpected error occurred. Please notify <@{SUPPORT_USER_ID}>."
                 should_stop_processing = True 

            try:
                await channel.send(final_reply)
                logging.info(f"Sent ticket reply in channel {channel_id}. Stop processing: {should_stop_processing}")
                if should_stop_processing:
                    stopped_channels.add(channel_id)
                    logging.info(f"Added channel {channel_id} to stopped channels.")

            except discord.Forbidden:
                 logging.error(f"Missing permissions to send message in channel {channel_id}")
                 stopped_channels.add(channel_id) 
            except Exception as e:
                 logging.error(f"Failed to send final reply to channel {channel_id}: {e}")

            finally:

                 pending_messages.pop(channel_id, None)
                 pending_tasks.pop(channel_id, None)

if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True 
    intents.guilds = True          
    intents.messages = True        

    client = TicketBot(intents=intents)
    client.run(DISCORD_BOT_TOKEN)
