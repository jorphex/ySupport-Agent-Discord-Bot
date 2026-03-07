import logging
import re
from typing import Optional

from agents import function_tool, RunContextWrapper

import config
import tools_lib
from state import BotRunContext


def _extract_artifact_refs(text: str) -> list[str]:
    refs = re.findall(r"(?:segment|fact):\d+", text or "")
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
    return ordered


def _repo_search_block_message(run_context: BotRunContext) -> str:
    refs = run_context.repo_last_search_artifact_refs[:3]
    if refs:
        refs_text = ", ".join(refs)
        next_step = (
            f"Use fetch_repo_artifacts_tool with one or more of these refs before searching again: {refs_text}. "
            "After reviewing the fetched artifacts, either answer the user or escalate if the claim remains unresolved."
        )
    else:
        next_step = (
            "Do not continue refining repo searches in this run. Use answer_from_docs_tool if docs context may help, "
            "or answer/escalate based on the evidence already gathered."
        )
    return (
        "Repo search loop guard triggered. Too many consecutive repo searches were made without fetching artifacts. "
        + next_step
    )


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
    Answers questions based on Yearn documentation and YIPs.
    """
    return await tools_lib.core_answer_from_docs(user_query)


@function_tool
async def search_repo_context_tool(
    wrapper: RunContextWrapper[BotRunContext],
    query: str,
    limit: Optional[int] = None,
    include_legacy: bool = False,
    include_ui: bool = False,
) -> str:
    """
    Searches the local Yearn repo-context index for contract, spec, deployment, or security artifacts.
    Use this for stYFI, vault contracts, strategies, routers, periphery, migrations, and bug or protocol behavior claims.
    Returns artifact references such as 'segment:12' that can be passed to fetch_repo_artifacts_tool for exact excerpts.
    """
    run_context = wrapper.context

    if run_context.repo_search_calls >= config.REPO_CONTEXT_MAX_SEARCH_CALLS_PER_RUN:
        logging.warning(
            "[RepoTool:search] Search budget exceeded for channel %s after %s searches.",
            run_context.channel_id,
            run_context.repo_search_calls,
        )
        return _repo_search_block_message(run_context)

    if (
        run_context.repo_searches_without_fetch >= config.REPO_CONTEXT_MAX_SEARCHES_WITHOUT_FETCH
        and run_context.repo_last_search_artifact_refs
    ):
        logging.warning(
            "[RepoTool:search] Blocking repeated search without fetch for channel %s after %s consecutive searches.",
            run_context.channel_id,
            run_context.repo_searches_without_fetch,
        )
        return _repo_search_block_message(run_context)

    response = await tools_lib.core_search_repo_context(query, limit, include_legacy, include_ui)
    run_context.repo_search_calls += 1
    run_context.repo_searches_without_fetch += 1
    run_context.repo_last_search_query = query
    run_context.repo_last_search_artifact_refs = _extract_artifact_refs(response)
    return response


@function_tool
async def fetch_repo_artifacts_tool(
    wrapper: RunContextWrapper[BotRunContext],
    artifact_refs_text: str,
) -> str:
    """
    Fetches exact repo artifacts by reference from the local Yearn repo-context index.
    Provide one or more artifact references such as 'segment:12' or 'fact:34'.
    """
    response = await tools_lib.core_fetch_repo_artifacts(artifact_refs_text)
    run_context = wrapper.context
    requested_refs = _extract_artifact_refs(artifact_refs_text)
    if requested_refs and not response.startswith("No valid repo artifact references were provided."):
        run_context.repo_fetch_calls += 1
        run_context.repo_searches_without_fetch = 0
        run_context.repo_last_search_artifact_refs = requested_refs
    return response


@function_tool
async def repo_context_status_tool() -> str:
    """
    Returns repo-context runtime status, including readiness and freshness.
    Use this when repo results appear unavailable or stale.
    """
    return await tools_lib.core_repo_context_status()
