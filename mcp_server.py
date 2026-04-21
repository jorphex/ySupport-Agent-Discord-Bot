# mcp_server.py
from mcp.server.fastmcp import FastMCP
import os
import tools_lib
import config
import logging
import support_dashboard_tools
from typing import Annotated
from pydantic import Field

try:
    from mcp.server.dependencies import get_http_headers
except Exception:
    get_http_headers = None

try:
    mcp_port = int(os.getenv("MCP_PORT", "8000"))
except ValueError:
    mcp_port = 8000
mcp_host = os.getenv("MCP_HOST", "0.0.0.0")
mcp = FastMCP(
    "ySupport",
    host=mcp_host,
    port=mcp_port,
    sse_path="/mcp/sse",
    message_path="/mcp/messages/",
    streamable_http_path="/mcp",
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _require_api_key() -> None:
    expected_key = config.MCP_SERVER_API_KEY
    if not expected_key:
        raise RuntimeError("MCP_SERVER_API_KEY is not set.")
    if get_http_headers is None:
        logging.warning("HTTP header access is unavailable; skipping API key check in process.")
        return
    headers = get_http_headers()
    auth_header = headers.get("authorization") if hasattr(headers, "get") else None
    key = None
    if auth_header and auth_header.lower().startswith("bearer "):
        key = auth_header.split(" ", 1)[1].strip()
    if not key and hasattr(headers, "get"):
        key = headers.get("x-api-key")
    if key != expected_key:
        raise PermissionError("Unauthorized")

# --- Expose Tools ---
@mcp.tool()
async def search_documentation(
    query: Annotated[
        str,
        Field(
            description=(
                "A clear, specific Yearn question or topic. Include the subject and key terms. "
                "Examples: 'What is veYFI and how does it work?', 'veYFI contract address', "
                "'How does a Yearn V3 strategy report work?'"
            )
        ),
    ],
) -> str:
    """
    Search Yearn documentation (Docs & YIPs) to answer technical or general questions.

    Args:
        query: A clear, specific Yearn question or topic. Include the subject and key terms.
            Examples:
              - "What is veYFI and how does it work?"
              - "veYFI contract address"
              - "How does a Yearn V3 strategy report work?"

    Returns:
        A concise answer with links and, when relevant, YIP status.
    """
    try:
        _require_api_key()
        return await tools_lib.core_answer_from_docs(query)
    except Exception as e:
        logging.error(f"Error in search_documentation: {e}")
        return f"Error querying documentation: {str(e)}"

@mcp.tool()
async def search_vaults(
    query: Annotated[
        str,
        Field(
            description=(
                "Token symbol, vault name fragment, or vault/underlying address. "
                "Examples: 'USDC', 'staked eth', '0x...'"
            )
        ),
    ],
    chain: Annotated[
        str,
        Field(
            default="",
            description=(
                "Optional chain filter. Supported: ethereum, base, arbitrum, optimism, "
                "polygon, sonic, katana. Leave blank for all."
            ),
        ),
    ] = "",
    sort_by: Annotated[
        str,
        Field(
            default="",
            description="Optional sorting: 'highest_apr', 'lowest_apr', or leave blank for TVL (default).",
        ),
    ] = "",
) -> str:
    """
    Search for active Yearn vaults using the yDaemon API.

    Args:
        query: Token symbol, vault name fragment, or vault/underlying address.
            Examples: "USDC", "staked eth", "0x..."
        chain: Optional chain filter. Supported: ethereum, base, arbitrum, optimism, polygon, sonic, katana.
        sort_by: Optional sorting. Use "highest_apr", "lowest_apr", or omit for TVL (default).

    Returns:
        Detailed info including address, version, strategy details, TVL, and APY.
    """
    chain_arg = chain if chain else None
    sort_arg = sort_by if sort_by else None
    
    try:
        _require_api_key()
        return await tools_lib.core_search_vaults(query, chain_arg, sort_arg)
    except Exception as e:
        logging.error(f"Error in search_vaults: {e}")
        return f"Error searching vaults: {str(e)}"


@mcp.tool()
async def search_repo_context(
    query: Annotated[
        str,
        Field(
            description=(
                "A Yearn protocol, contract, router, periphery, migration, or bug-claim query. "
                "Examples: 'VaultV3 _redeem accounting', 'stYFI cooldown contract', "
                "'ERC4626 router withdraw flow', 'veYFI migration behavior'."
            )
        ),
    ],
    limit: Annotated[
        int,
        Field(
            default=config.REPO_CONTEXT_TOP_K,
            description="Maximum number of repo artifacts to return. Defaults to the configured repo-context top-k.",
            ge=1,
        ),
    ] = config.REPO_CONTEXT_TOP_K,
    include_legacy: Annotated[
        bool,
        Field(
            default=False,
            description="Include legacy repos such as veYFI and vaults-v1 when searching migration or stale-claim context.",
        ),
    ] = False,
    include_ui: Annotated[
        bool,
        Field(
            default=False,
            description="Include UI/frontend repo context when investigating navigation or website-flow issues.",
        ),
    ] = False,
) -> str:
    """
    Search the local Yearn repo-context index for contract, spec, deployment, or security artifacts.

    Args:
        query: Contract-aware Yearn query for protocol behavior, migrations, or bug triage.
        limit: Maximum number of search results to return.
        include_legacy: Whether to include legacy repos such as veYFI and vaults-v1.
        include_ui: Whether to include UI/frontend repo context.

    Returns:
        A ranked list of repo artifacts with references such as 'segment:12' that can be passed to fetch_repo_artifacts.
    """
    try:
        _require_api_key()
        return await tools_lib.core_search_repo_context(query, limit, include_legacy, include_ui)
    except Exception as e:
        logging.error(f"Error in search_repo_context: {e}")
        return f"Error searching repo context: {str(e)}"


@mcp.tool()
async def fetch_repo_artifacts(
    artifact_refs_text: Annotated[
        str,
        Field(
            description=(
                "One or more repo artifact references returned by search_repo_context, such as "
                "'segment:12', 'fact:34', or 'segment:12, segment:18'."
            )
        ),
    ],
) -> str:
    """
    Fetch exact repo artifacts from the local Yearn repo-context index by reference.

    Args:
        artifact_refs_text: One or more artifact references such as 'segment:12' or 'fact:34'.

    Returns:
        Exact repo excerpts with file and repo provenance.
    """
    try:
        _require_api_key()
        return await tools_lib.core_fetch_repo_artifacts(artifact_refs_text)
    except Exception as e:
        logging.error(f"Error in fetch_repo_artifacts: {e}")
        return f"Error fetching repo artifacts: {str(e)}"


@mcp.tool()
async def repo_context_status() -> str:
    """
    Return local repo-context runtime status, including readiness and freshness.

    Returns:
        Repo-context status summary.
    """
    try:
        _require_api_key()
        return await tools_lib.core_repo_context_status()
    except Exception as e:
        logging.error(f"Error in repo_context_status: {e}")
        return f"Error checking repo context status: {str(e)}"


@mcp.tool()
async def support_dashboard_discover(
    chain_id: Annotated[
        int,
        Field(
            default=1,
            description=(
                "Optional numeric chain filter for the support dashboard discover index. "
                "Use chain ids such as 1 for Ethereum or 8453 for Base."
            ),
            ge=1,
        ),
    ] = 1,
    category: Annotated[
        str,
        Field(
            default="",
            description=(
                "Optional Yearn vault category filter such as 'Stablecoin' or 'auto'. "
                "Leave blank to search all categories."
            ),
        ),
    ] = "",
    token_symbol: Annotated[
        str,
        Field(
            default="",
            description=(
                "Optional token symbol filter such as 'USDC', 'WETH', or 'cbBTC'. "
                "Use this for support questions about where a token can be deployed or which vaults match a token."
            ),
        ),
    ] = "",
    universe: Annotated[
        str,
        Field(
            default="core",
            description=(
                "Dashboard universe filter. Use 'core' for the default support-safe view unless you have a specific "
                "reason to ask for a broader or different universe."
            ),
        ),
    ] = "core",
    sort_by: Annotated[
        str,
        Field(
            default="tvl",
            description=(
                "Sorting field for matching vault rows. Common values are 'tvl' or other dashboard-supported sort keys."
            ),
        ),
    ] = "tvl",
    limit: Annotated[
        int,
        Field(
            default=10,
            description=(
                "Maximum number of matching vault rows to return. Keep this small for support answers so the payload "
                "stays focused on the most relevant venues."
            ),
            ge=1,
            le=25,
        ),
    ] = 10,
) -> str:
    """
    Look up Yearn vaults from the support dashboard's discover index.

    Use this when support needs a fast vault lookup surface that returns current vault rows with
    chain, token, TVL, APY, regime, strategy count, and freshness fields. It is especially useful for:
    - identifying a vault from a token or category query
    - showing the most relevant Yearn vaults for a token like USDC or WETH
    - quickly grounding a support answer with current dashboard-visible vault metadata

    Returns:
        A compact JSON-style summary of the discover response, including filters, summary stats,
        coverage, and the matching vault rows.
    """
    try:
        _require_api_key()
        return await support_dashboard_tools.core_support_dashboard_discover(
            chain_id=chain_id,
            category=category or None,
            token_symbol=token_symbol or None,
            universe=universe,
            sort_by=sort_by,
            limit=limit,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_discover: {e}")
        return f"Error querying support dashboard discover: {str(e)}"


@mcp.tool()
async def support_dashboard_harvests(
    days: Annotated[
        int,
        Field(
            default=30,
            description=(
                "Trailing lookback window in days for vault harvest/report history. "
                "Use smaller windows such as 7 or 14 for recent support questions and larger windows when investigating "
                "longer inactivity claims."
            ),
            ge=1,
            le=365,
        ),
    ] = 30,
    chain_id: Annotated[
        int,
        Field(
            default=1,
            description=(
                "Optional numeric chain filter. Use 1 for Ethereum unless the user is asking about another supported chain."
            ),
            ge=1,
        ),
    ] = 1,
    vault_address: Annotated[
        str,
        Field(
            default="",
            description=(
                "Optional vault address filter. Use this for stale PPS, stale update, or 'is this vault still harvesting' "
                "questions about a specific vault."
            ),
        ),
    ] = "",
    limit: Annotated[
        int,
        Field(
            default=20,
            description=(
                "Maximum number of recent harvest/report rows to return from the dashboard endpoint."
            ),
            ge=1,
            le=50,
        ),
    ] = 20,
) -> str:
    """
    Fetch recent vault harvest/report history from the support dashboard.

    This is the best support tool for stale-update, stale-PPS, and 'has this vault harvested recently?'
    questions. The payload includes trailing 24h counts, chain rollups, and recent vault report rows with
    timestamps, tx hashes, strategy addresses, gain/loss, debt, fee assets, and refund assets.

    Important scope note:
    - this endpoint is based on vault report logs
    - it is strong evidence for recent vault activity
    - it may not fully represent deeper raw strategy-level Reported chronology

    Returns:
        A compact JSON-style summary of the harvest history response, focused on support-relevant fields.
    """
    try:
        _require_api_key()
        return await support_dashboard_tools.core_support_dashboard_harvests(
            days=days,
            chain_id=chain_id,
            vault_address=vault_address or None,
            limit=limit,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_harvests: {e}")
        return f"Error querying support dashboard harvests: {str(e)}"


@mcp.tool()
async def support_dashboard_changes(
    window: Annotated[
        str,
        Field(
            default="7d",
            description=(
                "Change window for dashboard movers and freshness context. Common values are '24h', '7d', and '30d'."
            ),
        ),
    ] = "7d",
    universe: Annotated[
        str,
        Field(
            default="core",
            description=(
                "Dashboard universe filter. Use 'core' for the default support-safe change feed."
            ),
        ),
    ] = "core",
    limit: Annotated[
        int,
        Field(
            default=10,
            description=(
                "Maximum number of risers, fallers, and largest-delta rows to include."
            ),
            ge=1,
            le=25,
        ),
    ] = 10,
    stale_threshold: Annotated[
        str,
        Field(
            default="auto",
            description=(
                "Optional staleness threshold override accepted by the dashboard endpoint. "
                "Leave as 'auto' for the endpoint default."
            ),
        ),
    ] = "auto",
) -> str:
    """
    Fetch recent Yearn APY and freshness changes from the support dashboard.

    Use this for support questions such as:
    - what changed recently
    - why did a vault's yield move up or down
    - which vaults are recent risers or fallers
    - whether freshness or ingestion staleness might explain a dashboard discrepancy

    Returns:
        A compact JSON-style summary with dashboard summary/freshness context plus bounded mover lists.
    """
    try:
        _require_api_key()
        return await support_dashboard_tools.core_support_dashboard_changes(
            window=window,
            universe=universe,
            limit=limit,
            stale_threshold=stale_threshold,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_changes: {e}")
        return f"Error querying support dashboard changes: {str(e)}"


@mcp.tool()
async def support_dashboard_token_venues(
    token_symbol: Annotated[
        str,
        Field(
            description=(
                "Token symbol to compare across Yearn venues, such as 'USDC', 'WETH', or 'cbBTC'. "
                "This is the best endpoint for 'where can I deploy token X in Yearn?' questions."
            )
        ),
    ],
    universe: Annotated[
        str,
        Field(
            default="core",
            description=(
                "Dashboard universe filter. Use 'core' for the default support-safe venue comparison."
            ),
        ),
    ] = "core",
) -> str:
    """
    Compare Yearn venues for a single token symbol using the support dashboard.

    This tool is best for support questions about where a token can be deployed across Yearn.
    It returns venue-level rows with chain, symbol, TVL, APY, and freshness fields so the bot can
    compare Yearn options for tokens like USDC, WETH, or cbBTC without improvising.

    Returns:
        A compact JSON-style summary of the token venue response.
    """
    try:
        _require_api_key()
        return await support_dashboard_tools.core_support_dashboard_token_venues(
            token_symbol=token_symbol,
            universe=universe,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_token_venues: {e}")
        return f"Error querying support dashboard token venues: {str(e)}"


@mcp.tool()
async def support_dashboard_styfi(
    days: Annotated[
        int,
        Field(
            default=30,
            description=(
                "Trailing lookback window in days for stYFI dashboard context and snapshots."
            ),
            ge=1,
            le=365,
        ),
    ] = 30,
    epoch_limit: Annotated[
        int,
        Field(
            default=12,
            description=(
                "Maximum number of reward epochs to request from the stYFI dashboard endpoint."
            ),
            ge=1,
            le=50,
        ),
    ] = 12,
    chain_id: Annotated[
        int,
        Field(
            default=1,
            description="Numeric chain id for the stYFI dashboard endpoint. Use 1 for Ethereum.",
            ge=1,
        ),
    ] = 1,
) -> str:
    """
    Fetch stYFI reward and staking state from the support dashboard.

    This is the best support tool for stYFI questions because it provides:
    - current reward epoch and reward token context
    - current and projected reward/APR state
    - balances and combined staking supply context
    - recent snapshot history for support investigations

    Returns:
        A compact JSON-style summary of the stYFI dashboard response with current reward state and recent snapshots.
    """
    try:
        _require_api_key()
        return await support_dashboard_tools.core_support_dashboard_styfi(
            days=days,
            epoch_limit=epoch_limit,
            chain_id=chain_id,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_styfi: {e}")
        return f"Error querying support dashboard stYFI state: {str(e)}"

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "sse")
    try:
        mcp.run(transport=transport)
    except TypeError:
        mcp.run(transport=transport)
