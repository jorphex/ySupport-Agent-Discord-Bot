from typing import Optional

from agents import function_tool, RunContextWrapper

import tools_lib
from state import BotRunContext


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
    Optionally sort by 'highest_apr' or 'lowest_apr'.
    """
    return await tools_lib.core_search_vaults(query, chain, sort_by)


@function_tool
async def get_withdrawal_instructions_tool(user_address_or_ens: Optional[str], vault_address: str, chain: str) -> str:
    """
    Generates step-by-step instructions for withdrawing from a specific Yearn vault (v1, v2, or v3) using a block explorer.
    Also provides the direct link to the vault on the Yearn website for reference (for v2/v3).
    Provide the vault's address, the chain name (e.g., 'ethereum', 'optimism', 'arbitrum'), and optionally the user's address/ENS.
    Use this when a user asks how to withdraw or reports issues using the Yearn website for a specific vault.
    """
    return await tools_lib.core_get_withdrawal_instructions(user_address_or_ens, vault_address, chain)


@function_tool
async def check_all_deposits_tool(user_address_or_ens: str, token_symbol: Optional[str] = None) -> str:
    """
    Checks for user deposits in Yearn vaults.
    """
    return await tools_lib.core_check_all_deposits(user_address_or_ens, token_symbol)


@function_tool
async def answer_from_docs_tool(
    wrapper: RunContextWrapper[BotRunContext],
    user_query: str
) -> str:
    """
    Answers questions based on documentation.
    """
    return await tools_lib.core_answer_from_docs(user_query)
