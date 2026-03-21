# config.py
import os
from pathlib import Path
import shlex

from dotenv import load_dotenv

load_dotenv()

_INVALID_ENV_ERRORS: dict[str, str] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_command_prefixes(name: str) -> list[list[str]]:
    raw_value = os.getenv(name, "")
    prefixes: list[list[str]] = []
    for chunk in raw_value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = shlex.split(chunk)
        if parts:
            prefixes.append(parts)
    return prefixes


def _env_csv(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def _env_int(name: str, default: int | None = None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        _INVALID_ENV_ERRORS[name] = f"{name} must be an integer"
        return default


BASE_DIR = Path(__file__).resolve().parent

# --- Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY")
MCP_SERVER_API_KEY = os.getenv("MCP_SERVER_API_KEY")

# --- LLM Models ---
LLM_GUARDRAIL_MODEL = os.getenv("LLM_GUARDRAIL_MODEL", "gpt-5-nano")
LLM_GUARDRAIL_REASONING_EFFORT = os.getenv("LLM_GUARDRAIL_REASONING_EFFORT", "minimal")
LLM_GUARDRAIL_VERBOSITY = os.getenv("LLM_GUARDRAIL_VERBOSITY", "low")

LLM_DATA_AGENT_MODEL = os.getenv("LLM_DATA_AGENT_MODEL", "gpt-5-mini")
LLM_DATA_AGENT_REASONING_EFFORT = os.getenv("LLM_DATA_AGENT_REASONING_EFFORT", "low")
LLM_DATA_AGENT_VERBOSITY = os.getenv("LLM_DATA_AGENT_VERBOSITY", "low")

LLM_DOCS_AGENT_MODEL = os.getenv("LLM_DOCS_AGENT_MODEL", "gpt-5.4")
LLM_DOCS_AGENT_REASONING_EFFORT = os.getenv("LLM_DOCS_AGENT_REASONING_EFFORT", "medium")
LLM_DOCS_AGENT_VERBOSITY = os.getenv("LLM_DOCS_AGENT_VERBOSITY", "medium")

LLM_BUG_AGENT_MODEL = os.getenv("LLM_BUG_AGENT_MODEL", "gpt-5.4")
LLM_BUG_AGENT_REASONING_EFFORT = os.getenv("LLM_BUG_AGENT_REASONING_EFFORT", "high")
LLM_BUG_AGENT_VERBOSITY = os.getenv("LLM_BUG_AGENT_VERBOSITY", "medium")

LLM_TRIAGE_AGENT_MODEL = os.getenv("LLM_TRIAGE_AGENT_MODEL", "gpt-5-mini")
LLM_TRIAGE_AGENT_REASONING_EFFORT = os.getenv("LLM_TRIAGE_AGENT_REASONING_EFFORT", "low")
LLM_TRIAGE_AGENT_VERBOSITY = os.getenv("LLM_TRIAGE_AGENT_VERBOSITY", "low")

LLM_DOCS_HYDE_MODEL = os.getenv("LLM_DOCS_HYDE_MODEL", "gpt-5-mini")
LLM_DOCS_HYDE_REASONING_EFFORT = os.getenv("LLM_DOCS_HYDE_REASONING_EFFORT", "minimal")
LLM_DOCS_HYDE_VERBOSITY = os.getenv("LLM_DOCS_HYDE_VERBOSITY", "low")

LLM_DOCS_SYNTH_MODEL = os.getenv("LLM_DOCS_SYNTH_MODEL", "gpt-5.4")
LLM_DOCS_SYNTH_REASONING_EFFORT = os.getenv("LLM_DOCS_SYNTH_REASONING_EFFORT", "medium")
LLM_DOCS_SYNTH_VERBOSITY = os.getenv("LLM_DOCS_SYNTH_VERBOSITY", "medium")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# --- Pinecone ---
# Allow per-env overrides without code changes.
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
YEARN_PINECONE_NAMESPACE = os.getenv("YEARN_PINECONE_NAMESPACE")

# --- Discord ---
# IDs can be overridden via env to keep repo clean of internal identifiers.
PUBLIC_TRIGGER_USER_IDS = set(
    v.strip()
    for v in os.getenv("PUBLIC_TRIGGER_USER_IDS", "").split(",")
    if v.strip()
)
YEARN_TICKET_CATEGORY_ID = _env_int("YEARN_TICKET_CATEGORY_ID")
YEARN_PUBLIC_TRIGGER_CHAR = os.getenv("YEARN_PUBLIC_TRIGGER_CHAR")
HUMAN_HANDOFF_TARGET_USER_ID = os.getenv("HUMAN_HANDOFF_TARGET_USER_ID")
HUMAN_HANDOFF_TAG_PLACEHOLDER = os.getenv("HUMAN_HANDOFF_TAG_PLACEHOLDER", "{HUMAN_HANDOFF_TAG_PLACEHOLDER}",
)

CATEGORY_CONTEXT_MAP = (
    {YEARN_TICKET_CATEGORY_ID: "yearn"}
    if YEARN_TICKET_CATEGORY_ID is not None
    else {}
)

TRIGGER_CONTEXT_MAP = (
    {YEARN_PUBLIC_TRIGGER_CHAR: "yearn"}
    if YEARN_PUBLIC_TRIGGER_CHAR
    else {}
)

PR_MARKETING_CHANNEL_ID = _env_int("PR_MARKETING_CHANNEL_ID")
MAX_DISCORD_MESSAGE_LENGTH = 1990

# --- Bot Behavior ---
COOLDOWN_SECONDS = 5
BUG_REPORT_COOLDOWN_SECONDS = int(os.getenv("BUG_REPORT_COOLDOWN_SECONDS", "60"))
MAX_TICKET_CONVERSATION_TURNS = 10
MAX_RESULTS_TO_SHOW = 5
STRATEGY_FETCH_CONCURRENCY = 10
PUBLIC_TRIGGER_TIMEOUT_MINUTES = 30
TICKET_EXECUTION_ENDPOINT = os.getenv("TICKET_EXECUTION_ENDPOINT", "local").strip().lower()
TICKET_EXECUTION_FALLBACK_ENDPOINT = os.getenv(
    "TICKET_EXECUTION_FALLBACK_ENDPOINT", ""
).strip().lower()
TICKET_EXECUTION_SUBPROCESS_COMMAND = shlex.split(
    os.getenv("TICKET_EXECUTION_SUBPROCESS_COMMAND", "")
)
TICKET_EXECUTION_CODEX_COMMAND = shlex.split(
    os.getenv("TICKET_EXECUTION_CODEX_COMMAND", "")
)
TICKET_EXECUTION_CODEX_MODEL = os.getenv("TICKET_EXECUTION_CODEX_MODEL", "").strip() or None
TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES = _env_command_prefixes(
    "TICKET_EXECUTION_ALLOWED_COMMAND_PREFIXES"
)
TICKET_EXECUTION_SUBPROCESS_CWD = os.getenv(
    "TICKET_EXECUTION_SUBPROCESS_CWD",
    str(BASE_DIR),
)
TICKET_EXECUTION_SUBPROCESS_ENV_KEYS = _env_csv(
    "TICKET_EXECUTION_SUBPROCESS_ENV_KEYS"
)
TICKET_EXECUTION_SUBPROCESS_ENV_PREFIXES = _env_csv(
    "TICKET_EXECUTION_SUBPROCESS_ENV_PREFIXES"
)
TICKET_EXECUTION_ARTIFACT_DIR = os.getenv("TICKET_EXECUTION_ARTIFACT_DIR", "").strip()
TICKET_EXECUTION_RUN_DIR_ROOT = os.getenv("TICKET_EXECUTION_RUN_DIR_ROOT", "").strip()


def ticket_execution_runtime_summary() -> str:
    parts = [f"primary={TICKET_EXECUTION_ENDPOINT}"]
    if TICKET_EXECUTION_FALLBACK_ENDPOINT:
        parts.append(f"fallback={TICKET_EXECUTION_FALLBACK_ENDPOINT}")
    if (
        TICKET_EXECUTION_ENDPOINT == "codex_exec"
        or TICKET_EXECUTION_FALLBACK_ENDPOINT == "codex_exec"
    ):
        parts.append(f"codex_model={TICKET_EXECUTION_CODEX_MODEL or 'default'}")
    if TICKET_EXECUTION_ARTIFACT_DIR:
        parts.append(f"artifact_dir={TICKET_EXECUTION_ARTIFACT_DIR}")
    elif TICKET_EXECUTION_RUN_DIR_ROOT:
        parts.append(f"run_dir_root={TICKET_EXECUTION_RUN_DIR_ROOT}")
    return ", ".join(parts)


def ticket_execution_runtime_warnings() -> list[str]:
    warnings: list[str] = []
    if TICKET_EXECUTION_ENDPOINT == "codex_exec" and not TICKET_EXECUTION_FALLBACK_ENDPOINT:
        warnings.append(
            "primary codex_exec is enabled without a fallback endpoint"
        )
    return warnings


def validate_ticket_execution_runtime_config() -> None:
    uses_codex = (
        TICKET_EXECUTION_ENDPOINT == "codex_exec"
        or TICKET_EXECUTION_FALLBACK_ENDPOINT == "codex_exec"
    )
    has_persistent_workspace = bool(
        TICKET_EXECUTION_ARTIFACT_DIR or TICKET_EXECUTION_RUN_DIR_ROOT
    )
    if uses_codex and not has_persistent_workspace:
        raise ValueError(
            "codex_exec requires TICKET_EXECUTION_ARTIFACT_DIR or "
            "TICKET_EXECUTION_RUN_DIR_ROOT so each run has persistent workspace state."
        )


def runtime_environment_validation_issues() -> list[str]:
    issues: list[str] = list(_INVALID_ENV_ERRORS.values())
    if not OPENAI_API_KEY:
        issues.append("OPENAI_API_KEY is required")
    if not DISCORD_BOT_TOKEN:
        issues.append("DISCORD_BOT_TOKEN is required")
    if YEARN_TICKET_CATEGORY_ID is None:
        issues.append("YEARN_TICKET_CATEGORY_ID is required")
    if not YEARN_PUBLIC_TRIGGER_CHAR:
        issues.append("YEARN_PUBLIC_TRIGGER_CHAR is required")
    if PR_MARKETING_CHANNEL_ID is None:
        issues.append("PR_MARKETING_CHANNEL_ID is required")
    if not HUMAN_HANDOFF_TARGET_USER_ID:
        issues.append("HUMAN_HANDOFF_TARGET_USER_ID is required")
    return issues


def validate_runtime_environment_config() -> None:
    issues = runtime_environment_validation_issues()
    if issues:
        raise ValueError("; ".join(issues))

# --- Repo Context ---
ENABLE_REPO_CONTEXT = _env_bool("ENABLE_REPO_CONTEXT", default=False)
REPO_CONTEXT_MANIFEST_PATH = Path(
    os.getenv("REPO_CONTEXT_MANIFEST_PATH", str(BASE_DIR / "yearn_rag" / "repo_sources.json"))
)
REPO_CONTEXT_CACHE_DIR = Path(
    os.getenv("REPO_CONTEXT_CACHE_DIR", str(BASE_DIR / ".cache" / "repo_context"))
)
REPO_CONTEXT_DB_PATH = Path(
    os.getenv("REPO_CONTEXT_DB_PATH", str(REPO_CONTEXT_CACHE_DIR / "repo_context.sqlite3"))
)
REPO_CONTEXT_TOP_K = int(os.getenv("REPO_CONTEXT_TOP_K", "6"))
REPO_CONTEXT_MAX_SNIPPET_CHARS = int(os.getenv("REPO_CONTEXT_MAX_SNIPPET_CHARS", "1800"))
REPO_CONTEXT_INCLUDE_UI = _env_bool("REPO_CONTEXT_INCLUDE_UI", default=False)
REPO_CONTEXT_MAX_AGE_HOURS = int(os.getenv("REPO_CONTEXT_MAX_AGE_HOURS", "168"))
REPO_CONTEXT_REQUIRE_FRESH = _env_bool("REPO_CONTEXT_REQUIRE_FRESH", default=True)
REPO_CONTEXT_MAX_SEARCH_CALLS_PER_RUN = int(os.getenv("REPO_CONTEXT_MAX_SEARCH_CALLS_PER_RUN", "8"))
REPO_CONTEXT_MAX_SEARCHES_WITHOUT_FETCH = int(os.getenv("REPO_CONTEXT_MAX_SEARCHES_WITHOUT_FETCH", "4"))

# --- Web3 & Chains ---
RPC_URLS = {
    "ethereum": f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "base": f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "polygon": f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "arbitrum": f"https://arb-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "optimism": f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "sonic": f"https://sonic-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "katana": "https://rpc.katana.network",
}

CHAIN_NAME_TO_ID = {
    "ethereum": 1, "base": 8453, "polygon": 137,
    "arbitrum": 42161, "optimism": 10,
    "sonic": 146, "katana": 747474
}

ID_TO_CHAIN_NAME = {v: k.capitalize() for k, v in CHAIN_NAME_TO_ID.items()}

BLOCK_EXPLORER_URLS = {
    "ethereum": "https://etherscan.io",
    "polygon": "https://polygonscan.com",
    "optimism": "https://optimistic.etherscan.io",
    "base": "https://basescan.org",
    "arbitrum": "https://arbiscan.io",
    "sonic": "https://sonicscan.org",
    "katana": "https://explorer.katanarpc.com",
}

# --- ABIs ---
ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"}
]

GAUGE_ABI = [
    {"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"_shares","type":"uint256"}],"name":"convertToAssets","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]
