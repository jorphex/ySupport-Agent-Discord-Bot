# mcp_server.py
from mcp.server.fastmcp import FastMCP
import os
import tools_lib
import config
import logging
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

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "sse")
    try:
        mcp.run(transport=transport)
    except TypeError:
        mcp.run(transport=transport)
