# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY")
MCP_SERVER_API_KEY = os.getenv("MCP_SERVER_API_KEY")

# --- Pinecone ---
# Allow per-env overrides without code changes.
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
YEARN_PINECONE_NAMESPACE = os.getenv("YEARN_PINECONE_NAMESPACE")

# --- Discord ---
# IDs can be overridden via env to keep repo clean of internal identifiers.
PUBLIC_TRIGGER_USER_IDS = set(
    v.strip()
    for v in os.getenv("PUBLIC_TRIGGER_USER_IDS").split(",")
    if v.strip()
)
YEARN_TICKET_CATEGORY_ID = int(os.getenv("YEARN_TICKET_CATEGORY_ID"))
YEARN_PUBLIC_TRIGGER_CHAR = os.getenv("YEARN_PUBLIC_TRIGGER_CHAR")
HUMAN_HANDOFF_TARGET_USER_ID = os.getenv("HUMAN_HANDOFF_TARGET_USER_ID")
HUMAN_HANDOFF_TAG_PLACEHOLDER = os.getenv("HUMAN_HANDOFF_TAG_PLACEHOLDER", "{HUMAN_HANDOFF_TAG_PLACEHOLDER}",
)

CATEGORY_CONTEXT_MAP = {
    YEARN_TICKET_CATEGORY_ID: "yearn",
}

TRIGGER_CONTEXT_MAP = {
    YEARN_PUBLIC_TRIGGER_CHAR: "yearn",
}

PR_MARKETING_CHANNEL_ID = int(os.getenv("PR_MARKETING_CHANNEL_ID"))
MAX_DISCORD_MESSAGE_LENGTH = 1990

# --- Bot Behavior ---
COOLDOWN_SECONDS = 5
MAX_TICKET_CONVERSATION_TURNS = 10
MAX_RESULTS_TO_SHOW = 5
STRATEGY_FETCH_CONCURRENCY = 10
PUBLIC_TRIGGER_TIMEOUT_MINUTES = 30

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
