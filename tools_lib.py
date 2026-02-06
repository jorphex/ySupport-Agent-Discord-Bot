# tools_lib.py
import asyncio
import logging
import json
import requests
import re
import aiohttp
from datetime import datetime, timezone
from typing import Dict, Optional, List, Union, Literal

from web3 import Web3
from pinecone import Pinecone
from openai import OpenAI, AsyncOpenAI

import config

# Init Clients
openai_async_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
openai_sync_client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)
pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)

WEB3_INSTANCES = {}

# Loop through URLs defined in config.py
for name, url in config.RPC_URLS.items():
    try:
        # Create the connection
        WEB3_INSTANCES[name] = Web3(Web3.HTTPProvider(url))
        
        # Optional: Check connection immediately (good for debugging logs)
        if not WEB3_INSTANCES[name].is_connected():
             print(f"Warning: Failed to connect to Web3 for {name}")
    except Exception as e:
        print(f"Error initializing Web3 for {name}: {e}")

# Load v1 Vaults List (Consider making this dynamic or configurable)
try:
    with open("v1_vaults.json", "r") as f:
        V1_VAULTS = json.load(f)
except Exception as e:
    logging.warning(f"Could not load v1_vaults.json: {e}. V1 deposit checks will fail.")
    V1_VAULTS = []


async def core_answer_from_docs(user_query: str) -> str:
    """
    Core logic for answering questions from docs.
    """
    logging.info(f"[CoreTool:answer_from_docs] Query: '{user_query}'")
    
    initial_retrieval_k = 15
    rerank_top_n = 8
    namespaces_to_query = ["yearn-docs", "yearn-yips"]

    # --- STEP 1: HyDE ---
    try:
        hyde_prompt = (
            f"You are a Yearn documentation expert. A user has asked: '{user_query}'.\n"
            "Generate a concise, hypothetical answer..."
        )
        hyde_response = await openai_async_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "system", "content": hyde_prompt}],
            temperature=0.0,
        )
        hypothetical_answer = hyde_response.choices[0].message.content.strip()
        embedding_text = f"{user_query}\n\n{hypothetical_answer}"
    except Exception as e:
        logging.error(f"HyDE error: {e}")
        embedding_text = user_query

    # --- STEP 2: Embedding ---
    try:
        response = await asyncio.to_thread(
            openai_sync_client.embeddings.create,
            model="text-embedding-3-large",
            input=[embedding_text],
            encoding_format="float"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return "Error generating embedding."

    # --- STEP 3: Search ---
    all_matches = []
    try:
        query_tasks = [
            asyncio.to_thread(
                pinecone_index.query,
                namespace=ns,
                vector=query_embedding,
                top_k=initial_retrieval_k,
                include_metadata=True
            ) for ns in namespaces_to_query
        ]
        results_list = await asyncio.gather(*query_tasks)
        for res in results_list:
            all_matches.extend(res.get("matches", []))
    except Exception as e:
        logging.error(f"Pinecone error: {e}")
        return "Error searching docs."

    if not all_matches:
        return "No information found in documentation."

    # --- STEP 4: Rerank ---
    try:
        unique_matches = {match.id: match for match in all_matches}.values()
        docs_to_rerank = [match.get("metadata", {}).get("text", "") for match in unique_matches]
        
        rerank_response = await asyncio.to_thread(
            pc.inference.rerank,
            model="bge-reranker-v2-m3",
            query=user_query,
            documents=docs_to_rerank,
            top_n=rerank_top_n,
            return_documents=False
        )
        unique_matches_list = list(unique_matches)
        reranked_matches = [unique_matches_list[result.index] for result in rerank_response.data]
    except Exception as e:
        logging.error(f"Rerank error: {e}")
        all_matches.sort(key=lambda x: x.score, reverse=True)
        reranked_matches = all_matches[:rerank_top_n]

    # --- STEP 5: Context ---
    context_pieces = []
    for match in reranked_matches:
        meta = match.get("metadata", {})
        text = meta.get("text")
        if text:
            source = f"{meta.get('doc_title', 'Unk')} ({meta.get('source_path', '')})"
            context_pieces.append(f"Source: {source}\nContent:\n{text}")
    
    context_text = "\n\n---\n\n".join(context_pieces)

    # --- STEP 6: LLM Generation ---
    system_prompt = (
        "You are an expert Yearn assistant. Answer based SOLELY on the context.\n"
        "Answer the user's question directly and authoritatively using only this knowledge.\n"
        "1. LINKS ARE MANDATORY.\n"
        "2. YIP STATUS IS MANDATORY.\n"
        "3. NO META-COMMENTARY.\n"
    )

    try:
        response = await openai_async_client.chat.completions.create(
            model="gpt-4o", # Use your preferred model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Gen error: {e}")
        return "Error generating final answer."


# --- Helpers ---
def extract_address_or_ens(text: str) -> Optional[str]:
    """Extracts the first 0x address or .eth ENS name from text."""
    # Regex for Ethereum address
    addr_match = re.search(r'(0x[a-fA-F0-9]{40})', text)
    if addr_match:
        return addr_match.group(1)
    # Regex for ENS name
    ens_match = re.search(r'\b([a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.eth)\b', text, re.IGNORECASE)
    if ens_match:
        return ens_match.group(1)
    return None

def resolve_ens(name: str) -> Optional[str]:
    """
    Resolves an ENS name or a SAFE-style prefixed address (e.g., 'eth:0x...') to a standard checksummed address.
    Returns the checksummed address if valid, otherwise None.
    """
    if not isinstance(name, str):
        return None
        
    address_to_check = name.strip()

    # --- Handle SAFE-style prefixes (e.g., 'eth:') ---
    if ':' in address_to_check:
        parts = address_to_check.split(':', 1)
        if len(parts) == 2:
            prefix, potential_address = parts
            logging.info(f"Detected prefixed address. Prefix: '{prefix}', Address part: '{potential_address}'")
            address_to_check = potential_address.strip()

    # Basic address check (on the potentially stripped address)
    if Web3.is_address(address_to_check):
        try:
            return Web3.to_checksum_address(address_to_check)
        except ValueError:
            # This can happen if the address has mixed-case but an invalid checksum
            logging.warning(f"Address '{address_to_check}' has invalid checksum.")
            return None

    # ENS check (only if it ends with .eth and we have an Ethereum instance)
    if address_to_check.endswith(".eth") and "ethereum" in WEB3_INSTANCES:
        try:
            logging.info(f"Attempting to resolve ENS name: '{address_to_check}'")
            resolved = WEB3_INSTANCES["ethereum"].ens.address(address_to_check)
            if resolved and Web3.is_address(resolved):
                logging.info(f"Successfully resolved ENS '{address_to_check}' to '{resolved}'")
                return Web3.to_checksum_address(resolved)
            else:
                 logging.warning(f"ENS name '{address_to_check}' did not resolve to a valid address.")
                 return None
        except Exception as e:
            logging.error(f"Error resolving ENS '{address_to_check}': {e}")
            return None
            
    # If it's not a valid address format or a resolvable ENS name
    logging.warning(f"Input '{name}' could not be resolved to a valid address.")
    return None

async def _fetch_vault_and_gauge_balances(
    vault_info: Dict,
    web3_instance: Web3,
    user_checksum_addr: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """
    Fetches a user's balance from a vault AND its associated staking gauge concurrently.
    Returns a dictionary with vault info and balances.
    """
    async with semaphore:
        vault_addr_str = vault_info.get("address")
        if not vault_addr_str:
            return {"vault_info": vault_info, "error": "Missing vault address"}

        try:
            vault_checksum_addr = Web3.to_checksum_address(vault_addr_str)
            vault_contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)

            # --- Create all web3 call coroutines first ---
            wallet_balance_coro = asyncio.to_thread(
                vault_contract.functions.balanceOf(user_checksum_addr).call
            )

            gauge_balance_coro = None
            staking_info = vault_info.get("staking")
            if staking_info and staking_info.get("available") and Web3.is_address(staking_info.get("address")):
                gauge_addr_str = staking_info.get("address")
                gauge_checksum_addr = Web3.to_checksum_address(gauge_addr_str)
                gauge_contract = web3_instance.eth.contract(address=gauge_checksum_addr, abi=config.GAUGE_ABI)
                gauge_balance_coro = asyncio.to_thread(
                    gauge_contract.functions.balanceOf(user_checksum_addr).call
                )

            # --- Run wallet and gauge balance checks in parallel ---
            if gauge_balance_coro:
                results = await asyncio.gather(wallet_balance_coro, gauge_balance_coro, return_exceptions=True)
                wallet_balance = results[0] if not isinstance(results[0], Exception) else 0
                gauge_token_balance = results[1] if not isinstance(results[1], Exception) else 0
                if isinstance(results[0], Exception): logging.warning(f"Error fetching wallet balance for {vault_addr_str}: {results[0]}")
                if isinstance(results[1], Exception): logging.warning(f"Error fetching gauge balance for {gauge_addr_str}: {results[1]}")
            else:
                # Only run the wallet balance check if no gauge
                wallet_balance = await wallet_balance_coro
                gauge_token_balance = 0

            logging.debug(f"Vault {vault_addr_str}: Wallet balance={wallet_balance}, Gauge token balance={gauge_token_balance}")

            # --- If staked, perform the final conversion call ---
            staked_balance_in_yvtoken = 0
            if gauge_token_balance > 0:
                # This is a final, conditional call. It's acceptable to await it here
                # as the two main calls have already completed.
                staked_balance_in_yvtoken = await asyncio.to_thread(
                    gauge_contract.functions.convertToAssets(gauge_token_balance).call
                )
                logging.debug(f"Vault {vault_addr_str}: Staked balance (in yvToken) is {staked_balance_in_yvtoken}")

            return {
                "vault_info": vault_info,
                "wallet_balance": wallet_balance,
                "staked_balance": staked_balance_in_yvtoken,
                "error": None
            }

        except Exception as e:
            logging.error(f"Critical error in balance fetching logic for vault {vault_addr_str}: {e}", exc_info=True)
            return {
                "vault_info": vault_info,
                "wallet_balance": 0,
                "staked_balance": 0,
                "error": str(e)
            }
        
async def query_active_deposits_logic(resolved_address: str, chain: Optional[str] = None, token_symbol: Optional[str] = None) -> str:
    logging.info(f"[Logic:query_active_deposits] Checking Active for {resolved_address}, Chain: {chain}, Token: {token_symbol}")

    chains_to_check = []
    if chain:
        chain_lower = chain.lower()
        if chain_lower in WEB3_INSTANCES:
            chains_to_check.append(chain_lower)
        else:
            return f"Unsupported chain: {chain}."
    else:
        chains_to_check = [c for c in WEB3_INSTANCES.keys() if c != 'berachain']

    all_vaults_data = []

    url = "https://ydaemon.yearn.fi/vaults/detected?hideAlways=false&strategiesDetails=withDetails&strategiesCondition=all&limit=2000"

    try:
        response = await asyncio.to_thread(requests.get, url, timeout=30)
        response.raise_for_status()
        all_vaults_data = response.json()
        if not isinstance(all_vaults_data, list):
             logging.error("[Logic:query_active_deposits] Unexpected yDaemon response format.")
             return "Error: Received unexpected data format from vault API."
        logging.info(f"[Logic:query_active_deposits] Successfully fetched {len(all_vaults_data)} vault definitions from yDaemon.")
    except Exception as e:
        logging.error(f"[Logic:query_active_deposits] Failed to fetch vault list from yDaemon: {e}")
        return f"Error: Could not fetch the list of active vaults: {e}"

    user_checksum_addr = Web3.to_checksum_address(resolved_address)

    all_results = []
    total_deposits_found = 0
    for chain_name in chains_to_check:
        web3_instance = WEB3_INSTANCES.get(chain_name)
        if not web3_instance: continue

        chain_id = config.CHAIN_NAME_TO_ID.get(chain_name)
        if not chain_id: continue

        chain_vaults = [v for v in all_vaults_data if v.get("chainID") == chain_id]
        if token_symbol:
            token_lower = token_symbol.lower()
            chain_vaults = [v for v in chain_vaults if token_lower in v.get("token", {}).get("symbol", "").lower()]

        if not chain_vaults:
            logging.info(f"No vaults to check for chain '{chain_name}' after filtering.")
            continue

        logging.info(f"Creating {len(chain_vaults)} balance check tasks for chain '{chain_name}'.")
        semaphore = asyncio.Semaphore(25)
        tasks = [_fetch_vault_and_gauge_balances(v, web3_instance, user_checksum_addr, semaphore) for v in chain_vaults]
        
        balance_results = await asyncio.gather(*tasks, return_exceptions=True)
        logging.info(f"Completed balance checks for chain '{chain_name}'.")

        chain_deposits = []
        for result in balance_results:
            if isinstance(result, Exception) or result.get("error"):
                continue

            total_balance = result.get("wallet_balance", 0) + result.get("staked_balance", 0)

            if total_balance > 0:
                try:
                    vault_info = result["vault_info"]
                    decimals = int(vault_info.get("decimals", 18))
                    total_display_balance = total_balance / (10 ** decimals)
                    vault_address = Web3.to_checksum_address(vault_info.get("address"))
                    api_version_str = vault_info.get('version', '')
                    if api_version_str.startswith("3."):
                        vault_url = f"https://yearn.fi/v3/{chain_id}/{vault_address}"
                    else:
                        vault_url = f"https://yearn.fi/vaults/{chain_id}/{vault_address}"
                    vault_name = vault_info.get('name', 'Unknown Vault')
                    vault_symbol = vault_info.get('symbol', 'N/A')
                    deposit_lines = [
                        f"**Vault:** [{vault_name}]({vault_url}) (Symbol: {vault_symbol})",
                        f"  Address: `{vault_address}`",
                        f"  Total Position: **{total_display_balance:,.6f} {vault_symbol}**"
                    ]
                    staked_balance = result.get("staked_balance", 0)
                    if staked_balance > 0:
                        wallet_balance = result.get("wallet_balance", 0)
                        wallet_display = wallet_balance / (10 ** decimals)
                        staked_display = staked_balance / (10 ** decimals)
                        if wallet_balance > 0:
                            breakdown = f"(Breakdown: {wallet_display:,.6f} liquid + {staked_display:,.6f} staked)"
                        else:
                            breakdown = "(Staked in gauge)"
                        deposit_lines.append(f"    {breakdown}")
                    chain_deposits.append("\n".join(deposit_lines))
                    total_deposits_found += 1
                except Exception as e:
                    logging.error(f"Error processing deposit for vault {vault_info.get('address')} on {chain_name}: {e}")

        if chain_deposits:
            all_results.append(f"**{chain_name.capitalize()} Active Deposits:**\n" + "\n\n".join(chain_deposits))
        elif chain:
             all_results.append(f"No active deposits found on {chain.capitalize()} for this address" + (f" matching token '{token_symbol}'." if token_symbol else "."))

    if total_deposits_found > 0:
        return "\n\n---\n\n".join(all_results)
    elif not chain:
        return "No active vault deposits found for that address on any supported Yearn chain" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    else:
        return "".join(all_results) if all_results else "No active vault deposits found."

async def query_v1_deposits_logic(resolved_address: str, token_symbol: Optional[str] = None) -> str:
    logging.info(f"[Logic:query_v1_deposits] Checking V1 for {resolved_address}, Token: {token_symbol}")
    if not V1_VAULTS: return "V1 vault data is not loaded."
    if "ethereum" not in WEB3_INSTANCES: return "Ethereum connection unavailable."

    web3_eth = WEB3_INSTANCES["ethereum"]
    try:
        user_checksum_addr = Web3.to_checksum_address(resolved_address)
    except ValueError:
         return f"Invalid Ethereum address format: {resolved_address}"

    found_deposits = []
    for vault in V1_VAULTS:
        if token_symbol and token_symbol.lower() not in vault.get("symbol", "").lower():
            continue
        try:
            vault_checksum_addr = Web3.to_checksum_address(vault.get("address"))
            contract = web3_eth.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)
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
            logging.error(f"[Logic:query_v1_deposits] Error checking V1 vault {vault.get('address')} for {user_checksum_addr}: {e}")

    if found_deposits:
        return "**Deprecated V1 Vault Deposits Found (Ethereum Only):**\n\n" + "\n\n".join(found_deposits)
    else:
        return "No deposits found in deprecated V1 vaults for this address."
    
async def core_check_all_deposits(user_address_or_ens: str, token_symbol: Optional[str] = None) -> str:
    """
    Checks for user deposits in BOTH active (v2/v3) and deprecated (v1) Yearn vaults across all supported chains.
    Provide the user's wallet address or ENS name. Optionally filter by token symbol (e.g., 'USDC').
    Returns a combined summary of any deposits found in either type of vault.
    """
    logging.info(f"[CoreTool:check_deposits] User: '{user_address_or_ens}'")
    resolved_address = resolve_ens(user_address_or_ens)
    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum address."

    # Run checks concurrently
    v1_task = asyncio.create_task(query_v1_deposits_logic(resolved_address, token_symbol))
    active_task = asyncio.create_task(query_active_deposits_logic(resolved_address, chain=None, token_symbol=token_symbol))

    v1_results, active_results = await asyncio.gather(v1_task, active_task)

    # Combine results
    final_output = []
    v1_found = "No deposits found in deprecated V1 vaults" not in v1_results
    active_found = "No active vault deposits found" not in active_results

    if v1_found:
        final_output.append(v1_results)
    if active_found:
        final_output.append(active_results)

    if not v1_found and not active_found:
        # If neither found anything, return a single message
        return f"No deposits found in any active or deprecated Yearn vaults for address {resolved_address}" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    elif not v1_found:
        # Only active found, maybe add a note about V1
        final_output.append("(No deposits found in deprecated V1 vaults)")
    elif not active_found:
         # Only V1 found, maybe add a note about active
         final_output.append("(No deposits found in active V2/V3 vaults)")

    combined_result = "\n\n---\n\n".join(final_output)
    logging.info(f"[CoreTool:check_deposits] Combined Result for {resolved_address}:\n{combined_result}")
    return combined_result

async def core_get_withdrawal_instructions(user_address_or_ens: Optional[str], vault_address: str, chain: str) -> str:
    """
    Generates step-by-step instructions for withdrawing from a specific Yearn vault (v1, v2, or v3) using a block explorer.
    Also provides the direct link to the vault on the Yearn website for reference (for v2/v3).
    Provide the vault's address, the chain name (e.g., 'ethereum', 'optimism', 'arbitrum'), and optionally the user's address/ENS.
    Use this when a user asks how to withdraw or reports issues using the Yearn website for a specific vault.
    """
    logging.info(f"[Tool:get_withdrawal_instructions] Args: User={user_address_or_ens}, Vault={vault_address}, Chain={chain}")

    # --- Step 1: Input Validation & Setup ---
    resolved_user_address = None
    user_checksum_addr = None
    if user_address_or_ens:
        resolved_user_address = resolve_ens(user_address_or_ens)
        if not resolved_user_address:
            logging.warning(f"Could not resolve provided user address/ENS: '{user_address_or_ens}'. Proceeding without it.")
        else:
             try:
                 user_checksum_addr = Web3.to_checksum_address(resolved_user_address)
             except ValueError:
                  logging.warning(f"Invalid format for resolved user address: '{resolved_user_address}'. Proceeding without it.")
                  user_checksum_addr = None

    chain_lower = chain.lower()
    web3_instance = WEB3_INSTANCES.get(chain_lower)
    chain_id = config.CHAIN_NAME_TO_ID.get(chain_lower)
    explorer_base_url = config.BLOCK_EXPLORER_URLS.get(chain_lower)

    if not web3_instance or not chain_id or not explorer_base_url:
        return f"Unsupported or invalid chain: '{chain}'. Cannot generate instructions."

    try:
        vault_checksum_addr = Web3.to_checksum_address(vault_address)
    except ValueError as e:
        return f"Invalid vault address format provided: {e}. Please provide a valid vault address."

    explorer_vault_url = f"{explorer_base_url}/address/{vault_checksum_addr}"

    # --- Step 2: Check V1 List (Ethereum Only) ---
    if chain_lower == 'ethereum':
        v1_vault_info = next((v for v in V1_VAULTS if v.get("address", "").lower() == vault_checksum_addr.lower()), None)

        if v1_vault_info:
            logging.info(f"Vault {vault_checksum_addr} identified as V1. Generating V1 instructions.")
            vault_name = v1_vault_info.get('name', vault_checksum_addr)

            # --- REVERTED V1 INSTRUCTION WORDING ---
            instructions = [
                f"This vault (**{vault_name}** `{vault_checksum_addr}`) is a **deprecated Yearn V1 vault**.",
                "Withdrawals must be done directly via the block explorer:",
                f"1. Go to the vault contract page: {explorer_vault_url}",
                "2. Click the **'Contract'** tab.",
                "3. Click the **'Write Contract'** tab.",
                f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}.",
                f"5. Look for a suitable withdrawal function (often named like 'withdraw', 'withdrawAll', or similar). Prioritize functions that take no arguments if available.", # Generic guidance
                "6. Click the **'Write'** button next to the chosen function.",
                "7. Confirm the transaction in your wallet.",
                "\nOnce the transaction confirms, your funds should be back in your wallet."
            ]
            final_instructions = "\n".join(instructions)
            logging.info(f"Generated V1 instructions for {vault_checksum_addr}:\n{final_instructions}")
            return final_instructions
        else:
            logging.info(f"Vault {vault_checksum_addr} not found in V1 list. Proceeding to check V2/V3.")

    # --- Step 3: Fetch V2/V3 Details via yDaemon ---
    vault_details_json: Optional[Dict] = None
    api_url = "https://ydaemon.yearn.fi/vaults/detected?limit=2000" # Or a more targeted one if available
    async with aiohttp.ClientSession() as session:
        try:
            logging.info(f"[Tool:get_withdrawal_instructions] Fetching bulk data to find {vault_checksum_addr}")
            async with session.get(api_url, timeout=25) as response:
                response.raise_for_status()
                all_vaults = await response.json()
                if isinstance(all_vaults, list):
                    for v_data in all_vaults:
                        if v_data.get("address", "").lower() == vault_checksum_addr.lower() and v_data.get("chainID") == chain_id:
                            vault_details_json = v_data
                            break
                    if vault_details_json:
                        logging.info(f"Found details for {vault_checksum_addr} in bulk fetch.")
                    else:
                        logging.warning(f"Vault {vault_checksum_addr} not found in bulk fetch from {api_url}")
                else:
                    logging.error("Unexpected format from bulk vault fetch.")
        except Exception as e:
            logging.error(f"Error fetching bulk vault data for withdrawal tool: {e}")

    if not vault_details_json:
        logging.warning(f"Failed to fetch V2/V3 vault details for {vault_checksum_addr} on chain {chain_id} via yDaemon.")
        return (f"Could not fetch vault details from the Yearn API for `{vault_checksum_addr}` on {chain.capitalize()} "
                f"to determine the correct withdrawal method for V2/V3 vaults. \n"
                f"Please double-check the vault address and chain name. \n"
                f"You can view the contract directly here: {explorer_vault_url}")

    # --- Step 4: Determine V2/V3 Version and Format Instructions ---
    api_version_str = vault_details_json.get("version", "")
    vault_name = vault_details_json.get("name", vault_checksum_addr)
    yearn_ui_link = f"https://yearn.fi/v3/{chain_id}/{vault_checksum_addr}"

    intro_message = (
        f"Okay, here are instructions for withdrawing from the **{vault_name}** vault (`{vault_checksum_addr}`) on **{chain.capitalize()}** using the block explorer.\n\n"
        f"You can also try the Yearn website here: {yearn_ui_link}\n\n"
        f"If using the block explorer:"
    )
    instructions = [intro_message]
    instructions.append(f"1. Go to the vault contract page: {explorer_vault_url}")
    instructions.append("2. Click the **'Contract'** tab.")
    write_tab_options = "'Write Contract as Proxy' (if available, otherwise 'Write Contract')"
    instructions.append(f"3. Click the **{write_tab_options}** tab.")
    instructions.append(f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}.")

    is_v3 = api_version_str.startswith("3.")
    is_v2 = api_version_str.startswith("0.")

    if is_v3:
        logging.info(f"Vault {vault_checksum_addr} identified as V3 (version: {api_version_str}). Generating 'redeem' instructions.")
        user_balance: Optional[int] = None
        balance_input_value = "**(Enter your full share balance manually)**"

        if user_checksum_addr:
            try:
                contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)
                balance_raw = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)
                user_balance = int(balance_raw)
                if user_balance == 0:
                    return (f"Confirmed user `{user_checksum_addr}` has zero balance in V3 vault `{vault_checksum_addr}`.\n"
                            f"Vault link for reference: {yearn_ui_link}\n"
                            f"No withdrawal needed or possible via explorer.")
                balance_input_value = f"`{user_balance}` (This should be your full share balance)"
            except Exception as e:
                logging.error(f"Failed to fetch user balance for V3 vault {vault_checksum_addr}: {e}")
                logging.warning("Generating V3 instructions without fetched balance.")
        else:
            logging.warning("User address not provided; cannot fetch V3 balance. Instructions require manual input.")

        instructions.extend([
            f"5. Find the **'redeem'** function.",
            f"6. Enter the following values:",
            f"   - `shares (uint256)`: {balance_input_value}",
            f"   - `receiver (address)`: {f'`{user_checksum_addr}` (Your wallet address)' if user_checksum_addr else '**(Your wallet address)**'}",
            f"   - `owner (address)`: {f'`{user_checksum_addr}` (Your wallet address again)' if user_checksum_addr else '**(Your wallet address again)**'}",
            "7. Click the **'Write'** button.",
            "8. Confirm the transaction in your wallet."
        ])

    elif is_v2:
        logging.info(f"Vault {vault_checksum_addr} identified as V2 (version: {api_version_str}). Generating 'withdraw' (no args) instructions.")
        instructions.extend([
            f"5. Find the **'withdraw()'** function that takes **no arguments**.",
            f"   *Important: Do NOT use a 'withdraw' function that asks for '_shares' or an amount.*",
            "6. Click the **'Write'** button next to that specific 'withdraw()' function.",
            "7. Confirm the transaction in your wallet."
        ])

    else:
        logging.warning(f"Vault {vault_checksum_addr} has unknown or missing V2/V3 version: '{api_version_str}'. Providing generic instructions.")
        balance_info = "(Cannot check balance without your address)"
        if user_checksum_addr:
            try:
                contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=config.ERC20_ABI)
                balance_raw = await asyncio.to_thread(contract.functions.balanceOf(user_checksum_addr).call)
                user_balance = int(balance_raw)
                if user_balance == 0:
                    return (f"Confirmed user `{user_checksum_addr}` has zero balance in vault `{vault_checksum_addr}` (Version Unknown).\n"
                            f"Vault link for reference: {yearn_ui_link}\n"
                            f"No withdrawal needed or possible via explorer.")
                balance_info = f"`{user_balance}`"
            except Exception:
                logging.warning("Could not fetch balance for unknown version vault.")
                balance_info = "(Could not fetch balance)"

        instructions.extend([
            f"5. **Unknown Vault Version (API Version: {api_version_str or 'Not Found'})**: Look for a function named **'withdraw'** or **'redeem'**.",
            f"   - Try **'withdraw()'** (with no input fields) first.",
            f"   - If that's not present or doesn't work, try **'redeem'**. If it asks for `shares`, `receiver`, and `owner`, enter your balance {balance_info} for `shares` and your address ({f'`{user_checksum_addr}`' if user_checksum_addr else 'Your Address'}) for `receiver` and `owner`.",
            f"   - If unsure, please ask for human help again, mentioning the vault version is unclear.",
            "6. Click the **'Write'** button for the chosen function.",
            "7. Confirm the transaction in your wallet."
        ])

    instructions.append("\nOnce the transaction confirms on the blockchain, your funds should be back in your wallet.")
    final_instructions = "\n".join(instructions)
    logging.info(f"Generated V2/V3/Fallback instructions for {vault_checksum_addr}:\n{final_instructions}")
    return final_instructions

def format_timestamp_to_readable(timestamp: Optional[Union[int, float, str]]) -> str:
    if timestamp is None:
        return "N/A"
    try:
        dt_object = datetime.fromtimestamp(int(timestamp), timezone.utc)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, TypeError):
        return str(timestamp)

def format_single_vault_data_for_llm(data: Dict, chain_id_for_url: int) -> str:
    """
    Formats a single vault's JSON data into a readable string for the LLM.
    """
    output_lines = []
    name = data.get('name', 'N/A')
    symbol = data.get('symbol', 'N/A')
    address = data.get('address', 'N/A')
    api_version_str = data.get('version', 'Unknown')
    simplified_version = "Unknown"
    yearn_ui_link = "N/A"

    if api_version_str.startswith("3."):
        simplified_version = f"V3 (API: {api_version_str})"
        yearn_ui_link = f"https://yearn.fi/v3/{chain_id_for_url}/{address}"
    elif api_version_str.startswith("0."):
        simplified_version = f"V2 (API: {api_version_str})"
        yearn_ui_link = f"https://yearn.fi/vaults/{chain_id_for_url}/{address}"

    output_lines.append(f"Vault: {name} ({symbol})")
    output_lines.append(f"Address: `{address}`")
    if yearn_ui_link != "N/A":
        output_lines.append(f"Yearn UI Link: {yearn_ui_link}")
    output_lines.append(f"Version: {simplified_version}")
    output_lines.append(f"Kind: {data.get('kind', 'N/A')}")
    description = data.get('description', 'No description available.')
    if description and len(description) > 250: description = description[:247] + "..."
    output_lines.append(f"Description: {description}")

    token_info = data.get('token', {})
    underlying_name = token_info.get('name', 'N/A')
    underlying_symbol = token_info.get('symbol', 'N/A')
    underlying_address = token_info.get('address', 'N/A')
    tvl_data = data.get('tvl', {})
    underlying_price = tvl_data.get('price', 0.0)
    output_lines.append(f"Underlying Token: {underlying_name} ({underlying_symbol}) - `{underlying_address}` - Price: ${underlying_price:,.4f}")
    output_lines.append("")

    output_lines.append("TVL & Share Price:")
    output_lines.append(f"  TVL (USD): ${tvl_data.get('tvl', 0.0):,.2f}")
    raw_pps = data.get('pricePerShare', '0')
    try:
        vault_decimals = int(data.get('decimals', 18))
        scaled_pps = float(raw_pps) / (10**vault_decimals)
        output_lines.append(f"  Vault Token Price Per Share (in underlying): {scaled_pps:.6f} (Raw: {raw_pps})")
    except (ValueError, TypeError):
        output_lines.append(f"  Vault Token Price Per Share (Raw): {raw_pps}")
    output_lines.append("")

    apr_data = data.get('apr', {})
    output_lines.append("APY Information:")
    net_apy = apr_data.get('netAPR', 0.0) * 100
    output_lines.append(f"  Current Net APY (compounded): {net_apy:.2f}% (Type: {apr_data.get('type', 'N/A')})")
    
    forward_apr_data = apr_data.get('forwardAPR', {})
    if forward_apr_data and forward_apr_data.get('netAPR') is not None:
        forward_net_apy = forward_apr_data.get('netAPR', 0.0) * 100
        output_lines.append(f"  Estimated Forward APY (projection): {forward_net_apy:.2f}% (Type: {forward_apr_data.get('type', 'N/A')})")
    
    fees = apr_data.get('fees', {})
    perf_fee = fees.get('performance', 0.0) * 100
    mgmt_fee = fees.get('management', 0.0) * 100
    output_lines.append(f"  Vault Fees: Performance={perf_fee:.2f}%, Management={mgmt_fee:.2f}%")
    
    points = apr_data.get('points', {})
    week_ago_apy = points.get('weekAgo', 0.0) * 100
    month_ago_apy = points.get('monthAgo', 0.0) * 100
    inception_apy = points.get('inception', 0.0) * 100
    output_lines.append(f"  Historical Net APY: Week Ago={week_ago_apy:.2f}%, Month Ago={month_ago_apy:.2f}%, Inception={inception_apy:.2f}%")
    output_lines.append("")

    output_lines.append("Other Info:")
    output_lines.append(f"  Featuring Score: {data.get('featuringScore', 'N/A')}")
    info_obj = data.get('info', {})
    output_lines.append(f"  Risk Level: {info_obj.get('riskLevel', 'N/A')}")
    output_lines.append(f"  Status Flags: Retired={info_obj.get('isRetired', False)}, Boosted={info_obj.get('isBoosted', False)}, Highlighted={info_obj.get('isHighlighted', False)}")
    migration_data = data.get('migration', {})
    output_lines.append(f"  Migration Available: {migration_data.get('available', False)}")
    if migration_data.get('available', False):
        output_lines.append(f"    Migration Target Address: `{migration_data.get('address', 'N/A')}`")
    output_lines.append("")

    strategies = data.get('strategies', [])
    if strategies:
        output_lines.append(f"Strategies ({len(strategies)}):")
        for i, strat in enumerate(strategies):
            strat_name = strat.get('name', 'Unnamed Strategy')
            strat_addr = strat.get('address', 'N/A')
            strat_status = strat.get('status', 'N/A')
            strat_apy = strat.get('netAPR', 0.0) * 100
            strat_details = strat.get('details', {})
            debt_ratio_raw = strat_details.get('debtRatio')
            debt_ratio_percent = "N/A"
            if debt_ratio_raw is not None:
                try: debt_ratio_percent = f"{float(debt_ratio_raw) / 100:.2f}%"
                except (ValueError, TypeError): pass
            last_report = format_timestamp_to_readable(strat_details.get('lastReport'))
            output_lines.append(f"  {i+1}. Name: {strat_name} (`{strat_addr}`)")
            output_lines.append(f"     Status: {strat_status}")
            output_lines.append(f"     Individual APY: {strat_apy:.2f}%")
            output_lines.append(f"     Allocation (Debt Ratio): {debt_ratio_percent}")
            output_lines.append(f"     Last Report: {last_report}")
    else:
        output_lines.append("Strategies: None listed.")
    output_lines.append("")

    staking_info = data.get('staking')
    if staking_info and staking_info.get('available'):
        output_lines.append("Staking Opportunity: Yes")
        output_lines.append(f"  Source: {staking_info.get('source', 'N/A')}")
        output_lines.append(f"  Staking Contract: `{staking_info.get('address', 'N/A')}`")
        rewards_list = staking_info.get('rewards', [])
        if rewards_list:
            output_lines.append(f"  Rewards ({len(rewards_list)}):")
            for rew_idx, reward in enumerate(rewards_list):
                rew_name = reward.get('name', 'N/A'); rew_sym = reward.get('symbol', 'N/A'); rew_addr = reward.get('address', 'N/A')
                rew_apy = reward.get('apr', 0.0) * 100; rew_finished = reward.get('isFinished', False)
                rew_ends = format_timestamp_to_readable(reward.get('finishedAt'))
                output_lines.append(f"    - Token: {rew_name} ({rew_sym}) `{rew_addr}`")
                output_lines.append(f"      APY: {rew_apy:.2f}%")
                output_lines.append(f"      Status: {'Finished' if rew_finished else 'Ongoing'} (Ends: {rew_ends})")
        else: output_lines.append("  Rewards: None listed.")
    else: output_lines.append("Staking Opportunity: No")
    
    return "\n".join(output_lines)

# --- Core Search Function ---

async def core_search_vaults(
    query: str,
    chain: Optional[str] = None,
    sort_by: Optional[str] = None 
) -> str:
    """
    Core logic to search for Yearn vaults.
    """
    logging.info(f"[CoreTool:search_vaults] Query: '{query}', Chain: '{chain}', Sort By: '{sort_by}'")
    api_url = "https://ydaemon.yearn.fi/vaults/detected?limit=2000"
    
    # Use config for max results if defined, else default
    MAX_RESULTS = getattr(config, 'MAX_RESULTS_TO_SHOW', 5)

    async with aiohttp.ClientSession() as session:
        try:
            logging.info(f"[Tool:search_vaults] Fetching data from {api_url}")
            async with session.get(api_url, timeout=25) as response:
                response.raise_for_status()
                all_vaults_data_list = await response.json()
        except Exception as e:
            logging.error(f"[Tool:search_vaults] Error during yDaemon fetch: {e}", exc_info=True)
            return f"Error: An unexpected error occurred while fetching vault data: {e}."

        # --- Filtering ---
        filtered_vaults = all_vaults_data_list
        
        if chain:
            chain_lower = chain.lower()
            query_chain_id = config.CHAIN_NAME_TO_ID.get(chain_lower)
            if query_chain_id:
                filtered_vaults = [v for v in filtered_vaults if v.get("chainID") == query_chain_id]

        query_lower = query.lower()
        matched_vaults = []
        is_address_query = Web3.is_address(query_lower)

        for v_data in filtered_vaults:
            vault_address = v_data.get("address", "").lower()
            name = v_data.get("name", "").lower()
            symbol = v_data.get("symbol", "").lower()
            token_info = v_data.get("token", {})
            token_name = token_info.get("name", "").lower() if token_info else ""
            token_symbol = token_info.get("symbol", "").lower() if token_info else ""
            underlying_address = token_info.get("address", "").lower() if token_info else ""

            match = False
            if is_address_query:
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
                fallback_apr = forward_apr_data.get("netAPR") if forward_apr_data else None
                apr_value = 0.0
                if primary_apr is not None: apr_value = float(primary_apr)
                elif fallback_apr is not None: apr_value = float(fallback_apr)
                v_data["_computedAPY"] = apr_value * 100
                try: v_data["_computedTVL_USD"] = float(v_data.get('tvl', {}).get('tvl', 0))
                except: v_data["_computedTVL_USD"] = 0.0
                matched_vaults.append(v_data)

        if not matched_vaults:
            return "No active Yearn vaults found matching your criteria."

        # --- Sorting ---
        if sort_by == "highest_apr":
            matched_vaults.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=True)
        elif sort_by == "lowest_apr":
            matched_vaults.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=False)
        else: # Default sort by TVL descending
            matched_vaults.sort(key=lambda v: v.get("_computedTVL_USD", 0.0), reverse=True)

        top_vaults = matched_vaults[:MAX_RESULTS]

        # --- Format ---
        formatted_strings = []
        for vault_data in top_vaults:
            chain_id = vault_data.get("chainID")
            if chain_id:
                formatted_text = format_single_vault_data_for_llm(vault_data, chain_id)
                formatted_strings.append(formatted_text)
            else:
                formatted_strings.append(f"Partial info for Vault: {vault_data.get('name', 'N/A')}")

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
