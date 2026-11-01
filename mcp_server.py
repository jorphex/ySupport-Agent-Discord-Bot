# mcp_server.py
import hmac
import logging
import os
from typing import Annotated, Any, Literal

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

import config
import docs_repo_tools
from repo_context import MAX_REPO_SEARCH_RESULTS
import support_dashboard_tools
import vault_search_tools

try:
    mcp_port = int(os.getenv("MCP_PORT", "8000"))
except ValueError:
    mcp_port = 8000
mcp_host = os.getenv("MCP_HOST", "0.0.0.0")
_MCP_AUTH_SCOPE = "ysupport"


class _StaticBearerTokenVerifier:
    def __init__(self, expected_token: str | None) -> None:
        self.expected_token = expected_token or ""

    async def verify_token(self, token: str) -> AccessToken | None:
        if not self.expected_token or not hmac.compare_digest(
            token,
            self.expected_token,
        ):
            return None
        return AccessToken(
            token=token,
            client_id="ysupport-codex",
            scopes=[_MCP_AUTH_SCOPE],
        )


def _build_mcp_server(
    *,
    host: str,
    port: int,
    api_key: str | None,
) -> FastMCP:
    return FastMCP(
        "ySupport",
        host=host,
        port=port,
        sse_path="/mcp/sse",
        message_path="/mcp/messages/",
        streamable_http_path="/mcp",
        token_verifier=_StaticBearerTokenVerifier(api_key),
        auth=AuthSettings(
            issuer_url=f"http://localhost:{port}",
            resource_server_url=None,
            required_scopes=[_MCP_AUTH_SCOPE],
        ),
    )


mcp = _build_mcp_server(
    host=mcp_host,
    port=mcp_port,
    api_key=config.MCP_SERVER_API_KEY,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
            ),
            min_length=1,
            max_length=2000,
        ),
    ],
) -> str:
    """
    Search Yearn documentation and YIPs for grounded source excerpts.

    Args:
        query: A clear, specific Yearn question or topic. Include the subject and key terms.
            Examples:
              - "What is veYFI and how does it work?"
              - "veYFI contract address"
              - "How does a Yearn V3 strategy report work?"

    Returns:
        Ranked excerpts with source links and, when relevant, YIP status metadata.
        Use these excerpts to answer the user's question; they are not a prewritten answer.
    """
    try:
        return await docs_repo_tools.core_search_docs_context(query)
    except Exception as e:
        logging.error(f"Error in search_documentation: {e}")
        return f"Error querying documentation: {str(e)}"

@mcp.tool()
async def search_vaults(
    query: Annotated[
        str,
        Field(
            min_length=1, max_length=256,
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
        vault_search_tools.VaultSort,
        Field(
            default="",
            description="Sort by highest_apy, lowest_apy, or tvl (default). highest_apr/lowest_apr remain compatibility aliases for APY sorting.",
        ),
    ] = "",
    limit: Annotated[int, Field(default=5, ge=1, le=25, description="Vaults per page; defaults to 5.")] = 5,
    offset: Annotated[int, Field(default=0, ge=0, description="Use next_offset from pagination; repeat the same query, chain and sort. Catalog updates can change page order.")] = 0,
    include_details: Annotated[bool, Field(default=False, description="Fetch snapshots and strategy details for this page. Exact vault-address matches get details automatically; token and underlying-address searches stay compact by default.")] = False,
) -> dict[str, Any]:
    """
    Resolve Yearn vault identity and exact details using Kong's canonical catalog.

    Prefer support_dashboard_discover for browsing or comparing current vault yields.
    Use search_vaults for names/addresses, exact contract details, and hidden/retired
    or other known vaults outside dashboard analytics coverage. A missing dashboard
    row does not establish that a vault does not exist.

    Returns structured numeric rows and pagination; follow next_offset with the same
    filters. Broad lookups use a five-minute cached catalog without snapshot calls.
    Details are automatic for exact vault addresses or explicit via include_details.
    APYs are decimal fractions, distinct from APR. Estimated and realized metrics
    have separate fields; yBOLD includes its display inputs. Retrieval/cache time
    is not metric time: inspect block_time where available. Ranking uses the catalog;
    detail snapshots can be newer. Raw snapshot fields never inherit catalog values;
    absent fields stay null and incomplete snapshots have detail_status=partial.
    yBOLD's derived display APY reports its input sources separately. Catalog age belongs to the generation
    used by this call. Null metrics and unavailable details are unresolved.
    """
    chain_arg = chain if chain else None
    sort_arg = sort_by if sort_by else None
    
    try:
        return await vault_search_tools.core_search_vaults(
            query, chain_arg, sort_arg, limit=limit, offset=offset, include_details=include_details,
        )
    except Exception as e:
        logging.error(f"Error in search_vaults: {e}")
        return {"error": f"Error searching vaults: {e}"}


@mcp.tool()
async def search_repo_context(
    query: Annotated[
        str,
        Field(
            description=(
                "A Yearn protocol, contract, router, periphery, migration, or bug-claim query. "
                "Examples: 'VaultV3 _redeem accounting', 'stYFI unstake stream', "
                "'ERC4626 router withdraw flow', 'veYFI migration behavior'."
            ),
            min_length=1,
            max_length=2000,
        ),
    ],
    limit: Annotated[
        int,
        Field(
            default=config.REPO_CONTEXT_TOP_K,
            description="Maximum number of repo artifacts to return. Defaults to the configured repo-context top-k.",
            ge=1,
            le=MAX_REPO_SEARCH_RESULTS,
        ),
    ] = config.REPO_CONTEXT_TOP_K,
    include_legacy: Annotated[
        bool,
        Field(
            default=False,
            description="Include legacy repos such as veYFI and vaults-v1 when searching migration or stale-claim context.",
        ),
    ] = False,
) -> str:
    """
    Search the local Yearn repo-context index for contract, spec, deployment, or security artifacts.

    Args:
        query: Contract-aware Yearn query for protocol behavior, migrations, or bug triage.
        limit: Maximum number of search results to return.
        include_legacy: Whether to include legacy repos such as veYFI and vaults-v1.

    Returns:
        A ranked list of repo artifacts with references such as 'segment:12' that can be passed to fetch_repo_artifacts.
    """
    try:
        return await docs_repo_tools.core_search_repo_context(query, limit, include_legacy)
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
            ),
            min_length=1,
            max_length=512,
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
        return await docs_repo_tools.core_fetch_repo_artifacts(artifact_refs_text)
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
        return await docs_repo_tools.core_repo_context_status()
    except Exception as e:
        logging.error(f"Error in repo_context_status: {e}")
        return f"Error checking repo context status: {str(e)}"


@mcp.tool()
async def support_dashboard_discover(
    chain_id: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Optional numeric chain filter for the support dashboard discover index. "
                "Use chain ids such as 1 for Ethereum or 8453 for Base."
            ),
            ge=1,
        ),
    ] = None,
    token_symbol: Annotated[
        str,
        Field(
            default="",
            description=(
                "Optional exact token-symbol filter, such as USDC. Matching is "
                "case-insensitive but does not treat wrappers or bridged variants as equivalent."
            ),
            max_length=64,
        ),
    ] = "",
    market: Annotated[
        Literal["all", "stablecoins", "eth", "bitcoin", "other"],
        Field(
            default="all",
            description=(
                "Vault market filter. Supported values: all, stablecoins, eth, bitcoin, or other."
            ),
        ),
    ] = "all",
    universe: Annotated[
        Literal["core", "extended", "raw"],
        Field(
            default="core",
            description=(
                "Dashboard universe filter. Use 'core' for the default support-safe view unless you have a specific "
                "reason to ask for a broader or different universe."
            ),
        ),
    ] = "core",
    min_tvl_usd: Annotated[
        float | None,
        Field(
            default=None,
            description=(
                "Optional minimum vault TVL in USD. Prefer the universe defaults unless "
                "the support question requires a specific threshold."
            ),
            ge=0,
        ),
    ] = None,
    min_points: Annotated[
        int | None,
        Field(
            default=None,
            description=(
                "Optional minimum number of PPS observations. Prefer the universe defaults "
                "unless the support question requires a specific threshold."
            ),
            ge=0,
        ),
    ] = None,
    sort_by: Annotated[
        Literal["tvl", "est_apy", "apy_30d", "momentum"],
        Field(
            default="tvl",
            description=(
                "Sorting field. Supported values: tvl, est_apy, apy_30d, or momentum."
            ),
        ),
    ] = "tvl",
    direction: Annotated[
        Literal["asc", "desc"],
        Field(
            default="desc",
            description="Sort direction. Use descending unless the question requires the lowest values.",
        ),
    ] = "desc",
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

    Use this for current vault and yield context. It returns current rows with chain,
    market, TVL, estimated APY, realized APY, and momentum fields. It is especially useful for:
    - browsing vaults by chain or market
    - showing the most relevant Yearn vaults in a market such as stablecoins or ETH
    - quickly grounding a support answer with current dashboard-visible vault metadata

    Prefer this tool when browsing/comparing current vault yields. Use search_vaults
    for exact identity/details or a known vault missing from this filtered analytics
    universe; absence here does not establish that a vault does not exist. Keep
    dashboard estimated/realized yields distinct from Kong metrics and source times.
    Rows contain analytics rather than UI links, strategy counts or lifecycle flags.
    Resolve selected vaults with search_vaults when those details are needed.

    Use the core universe for user-facing context. Extended is for explicitly requested
    smaller vaults. Raw can help attempt to locate a known vault but is not authoritative
    resolution and may still omit records without metrics. APYs are decimal fractions, and
    summary total TVL is filtered analytics coverage rather than Yearn protocol TVL. Treat
    the returned filters as the current universe policy instead of hard-coding thresholds.

    Returns:
        A compact JSON-style summary of the discover response, including filters, summary stats,
        coverage, and the matching vault rows.
    """
    try:
        return await support_dashboard_tools.core_support_dashboard_discover(
            chain_id=chain_id,
            token_symbol=token_symbol or None,
            market=market,
            universe=universe,
            min_tvl_usd=min_tvl_usd,
            min_points=min_points,
            sort_by=sort_by,
            direction=direction,
            limit=limit,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_discover: {e}")
        return f"Error querying support dashboard discover: {str(e)}"


@mcp.tool()
async def support_dashboard_reports(
    chain_id: Annotated[
        int,
        Field(
            description=(
                "Numeric chain ID for the exact vault, such as 1 for Ethereum or "
                "8453 for Base. This is required because the global report feed is "
                "not a safe support diagnostic."
            ),
            ge=1,
        ),
    ],
    vault_address: Annotated[
        str,
        Field(
            description=(
                "Exact vault contract address. Resolve the vault and chain before "
                "calling this tool; broad report-feed queries are intentionally unavailable."
            ),
            min_length=1,
            max_length=128,
        ),
    ],
    days: Annotated[
        int,
        Field(
            default=30,
            description="Trailing report lookback in days.",
            ge=7,
            le=365,
        ),
    ] = 30,
    limit: Annotated[
        int,
        Field(
            default=50,
            description=(
                "Maximum number of newest matching report rows. The endpoint has no "
                "pagination, so absence of an older row is inconclusive."
            ),
            ge=1,
            le=50,
        ),
    ] = 50,
    meaningful_only: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Return only reports with a nonzero gain, loss, fee, or refund. Keep true "
                "for normal support; use false when accounting-only reports matter."
            ),
        ),
    ] = True,
) -> str:
    """
    Fetch exact-vault StrategyReported events from the support dashboard.

    Use this only after resolving both the vault address and chain. Every row proves that the
    vault emitted an on-chain StrategyReported event. It does not by itself prove that a
    traditional off-chain harvest job ran or that profit was realized. `realized_result`
    means at least one reported economic field was nonzero; `accounting_update` means all
    available economic fields were zero. Normal support defaults to meaningful economic
    results, while accounting-only rows remain available by setting meaningful_only=false.

    Raw gain, loss, fee, refund, and debt values are integer strings in token units. Scale
    them by token_decimals, preserve null as unavailable, and account for vault_version when
    interpreting debt. Accounting-only reports require meaningful_only=false. Results are the
    newest bounded rows with no continuation token, so an absent older report is inconclusive.

    Returns:
        A compact JSON-style report summary with event identity, economics, scaling,
        version, and debt context.
    """
    try:
        return await support_dashboard_tools.core_support_dashboard_reports(
            chain_id=chain_id,
            vault_address=vault_address,
            days=days,
            limit=limit,
            meaningful_only=meaningful_only,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_reports: {e}")
        return f"Error querying support dashboard reports: {str(e)}"


@mcp.tool()
async def support_dashboard_status() -> str:
    """
    Check yHelper source health, freshness, coverage, and operational status.

    Use this before relying on dashboard results when diagnosing missing, delayed, or
    inconsistent data. It reports compact protocol-source status, global/cohort PPS ages,
    ingestion jobs, alerts, tracked scope, and coverage. It cannot prove whether one
    specific vault has fresh PPS data.

    Returns:
        A compact operational status summary with the per-vault scope limitation included.
    """
    try:
        return await support_dashboard_tools.core_support_dashboard_status()
    except Exception as e:
        logging.error(f"Error in support_dashboard_status: {e}")
        return f"Error querying support dashboard status: {str(e)}"


@mcp.tool()
async def support_dashboard_changes(
    window: Annotated[
        Literal["24h", "7d", "30d"],
        Field(default="7d", description="Realized APY comparison window."),
    ] = "7d",
    stale_threshold: Annotated[
        Literal["auto", "24h", "7d", "30d"],
        Field(
            default="auto",
            description="Maximum acceptable comparison age. Auto follows the selected window.",
        ),
    ] = "auto",
    universe: Annotated[
        Literal["core", "extended", "raw"],
        Field(default="core", description="Use core for normal support market context."),
    ] = "core",
    market: Annotated[
        Literal["all", "stablecoins", "eth", "bitcoin", "other"],
        Field(default="all", description="Optional vault-market filter."),
    ] = "all",
    min_tvl_usd: Annotated[
        float | None,
        Field(default=None, description="Optional minimum vault TVL in USD.", ge=0),
    ] = None,
    min_points: Annotated[
        int | None,
        Field(default=None, description="Optional minimum PPS observation count.", ge=0),
    ] = None,
    limit: Annotated[
        int,
        Field(
            default=10,
            description="Maximum risers and fallers to return per list.",
            ge=1,
            le=25,
        ),
    ] = 10,
) -> str:
    """
    Fetch bounded realized-APY risers and fallers for market context.

    Use this for questions about whether realized yield is broadly rising or falling. It is
    not address-filterable and is not diagnostic evidence for deposits, withdrawals,
    StrategyReported events, missing funds, or one specific vault failure. APYs and delta_apy
    are decimal fractions, so 0.005 is a change of 0.5 percentage points.

    Returns:
        Compact riser, faller, summary, and freshness context.
    """
    try:
        return await support_dashboard_tools.core_support_dashboard_changes(
            window=window,
            stale_threshold=stale_threshold,
            universe=universe,
            market=market,
            min_tvl_usd=min_tvl_usd,
            min_points=min_points,
            limit=limit,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_changes: {e}")
        return f"Error querying support dashboard changes: {str(e)}"


@mcp.tool()
async def support_dashboard_styfi(
    days: Annotated[
        int,
        Field(
            default=7,
            description=(
                "Trailing lookback window in days for stYFI dashboard context and snapshots."
            ),
            ge=7,
            le=122,
        ),
    ] = 7,
    epoch_limit: Annotated[
        int,
        Field(
            default=3,
            description=(
                "Maximum number of reward epochs to request from the stYFI dashboard endpoint."
            ),
            ge=3,
            le=24,
        ),
    ] = 3,
) -> str:
    """
    Fetch stYFI reward and staking state from the support dashboard.

    Use this only for stYFI-specific support because it provides:
    - current reward epoch and reward/APR context
    - current and projected reward/APR state
    - balances and combined staking supply context
    - freshness, ingestion, and bounded recent activity context

    Returns:
        A compact, tolerant subset of the unversioned stYFI response. Historical series
        and component-layout fields are intentionally excluded.
    """
    try:
        return await support_dashboard_tools.core_support_dashboard_styfi(
            days=days,
            epoch_limit=epoch_limit,
        )
    except Exception as e:
        logging.error(f"Error in support_dashboard_styfi: {e}")
        return f"Error querying support dashboard stYFI state: {str(e)}"

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
async def support_dashboard_flex(
    view: Annotated[
        support_dashboard_tools.FlexView,
        Field(description=(
            "Flex lending read: protocol totals; markets discovery; market detail/risk/oracle; "
            "redemption_priority lower-rate debt curve; trove_health market-wide aggregates; "
            "history sampled metrics; activity events; performance flow-adjusted PnL."
        )),
    ],
    market_address: Annotated[
        str, Field(default="", pattern=r"^(?:0x[0-9a-fA-F]{40})?$",
                   description="Required for market, redemption_priority, trove_health and history; optional activity filter. Use markets to discover the exact Flex market address."),
    ] = "",
    wallet_address: Annotated[
        str, Field(default="", pattern=r"^(?:0x[0-9a-fA-F]{40})?$",
                   description="Optional Ethereum wallet filter for performance only. Omit for protocol-wide performance."),
    ] = "",
    status: Annotated[
        Literal["all", "active", "deprecated", "unendorsed"] | None,
        Field(default=None, description="Markets view only; defaults to active. Inclusion is not a recommendation."),
    ] = None,
    days: Annotated[
        Literal[7, 30, 90] | None,
        Field(default=None, description="History or performance only; defaults to 7 days."),
    ] = None,
    interval: Annotated[
        Literal["hour", "day"] | None,
        Field(default=None, description="History only; defaults to daily samples."),
    ] = None,
    event: Annotated[
        str, Field(default="", max_length=64, description="Optional exact event-name filter for activity only."),
    ] = "",
    cursor: Annotated[
        str, Field(default="", pattern=r"^(?:\d{1,20}:\d{1,20})?$",
                   description="Activity only: next_cursor from the previous response, formatted block_number:log_index."),
    ] = "",
    limit: Annotated[
        int, Field(default=10, ge=1, le=50,
                   description="Maximum rows/curve points, or rows per performance side. History keeps the newest points. Truncation is explicit; output limits do not change upstream summary or coverage."),
    ] = 10,
) -> str:
    """Read current and historical Flex lending protocol data from yHelper.

    Use for Flex markets, borrowing, lender liquidity, redemption exposure and PnL.
    These are Ethereum Flex markets, distinct from Kong's Yearn vault catalog.
    Check freshness/source blocks on each response; different views can use different
    snapshots. Null/partial/stale values are unresolved, not zero. Rates and LTV are
    decimal ratios (0.05 = 5%); USD amounts are estimates. Raw token amounts require
    the token decimals from market detail. History coverage describes the source
    window; output_limits describes what this tool actually returned.

    Redemption-priority points contain debt at strictly lower chosen rates and idle
    liquidity, not a guaranteed personal redemption outcome. Trove health is market
    aggregates, not an individual's LTV or liquidation safety. Performance can be
    incomplete and covers active markets; it is not a current wallet-position lookup.
    Do not infer full-wallet PnL from partial rows or explain its cause without evidence.
    """
    try:
        return await support_dashboard_tools.core_support_dashboard_flex(
            view=view, market_address=market_address, wallet_address=wallet_address,
            status=status, days=days, interval=interval, event=event, cursor=cursor, limit=limit,
        )
    except Exception as exc:
        logging.error("Error in support_dashboard_flex: %s", exc)
        return f"Error querying Flex data: {str(exc)}"


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "sse")
    try:
        mcp.run(transport=transport)
    except TypeError:
        mcp.run(transport=transport)
