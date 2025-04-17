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
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
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
from agents.handoffs import HandoffInputData 

sys.stdout = sys.stderr

@dataclass
class BotRunContext:
    """Context passed to agents during a run."""
    channel_id: int
    category_id: Optional[int] = None
    is_public_trigger: bool = False

    project_context: Literal["yearn", "bearn", "unknown"] = "unknown"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "key") 
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "token") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "key") 
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY", "key") 

PINECONE_INDEX_NAME = "index" 
YEARN_PINECONE_NAMESPACE = "namespace" 
BEARN_PINECONE_NAMESPACE = "namespace" 

SUPPORT_USER_ID = "id" 
HUMAN_HANDOFF_TAG_PLACEHOLDER = "%%SUPPORT_TAG_ME%%" 
YEARN_TICKET_CATEGORY_ID = id
BEARN_TICKET_CATEGORY_ID = id
YEARN_PUBLIC_TRIGGER_CHAR = "y"
BEARN_PUBLIC_TRIGGER_CHAR = "b"

CATEGORY_CONTEXT_MAP = {
    YEARN_TICKET_CATEGORY_ID: "yearn",
    BEARN_TICKET_CATEGORY_ID: "bearn",
}

TRIGGER_CONTEXT_MAP = {
    YEARN_PUBLIC_TRIGGER_CHAR: "yearn",
    BEARN_PUBLIC_TRIGGER_CHAR: "bearn",
}

PR_MARKETING_CHANNEL_ID = id 
MAX_DISCORD_MESSAGE_LENGTH = 1990 

COOLDOWN_SECONDS = 5 
MAX_TICKET_CONVERSATION_TURNS = 15 
MAX_RESULTS_TO_SHOW = 5 
STRATEGY_FETCH_CONCURRENCY = 10 

BERACHAIN_CHAIN_ID = 80094
BEARN_FACTORY_ADDRESS = "0x70b14cd0Cf7BD442DABEf5Cb0247aA478B82fcbb"
BEARN_UI_CONTROL_ADDRESS = "0xD36e0A4Ae7258Dd1FfE0D7f9f851461369a1AA0E"

class BDPriorityCheckOutput(BaseModel): 
    request_type: Literal["listing", "partnership", "marketing", "other_bd", "not_bd_pr"] = Field(..., description="Classify the user's primary intent: 'listing' (requesting Yearn list their token), 'partnership' (proposing integration/collaboration), 'marketing' (joint marketing/promotion), 'other_bd' (other business development), or 'not_bd_pr' (standard support request or unrelated).")
    reasoning: str = Field(..., description="Brief explanation for the classification.")

class GuardrailResponseMessageException(AgentsException):
    def __init__(self, message: str, guardrail_output: Optional[BDPriorityCheckOutput] = None):
        super().__init__(message)
        self.message = message
        self.guardrail_output = guardrail_output 

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
    "fantom": f"https://fantom-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "berachain": f"https://berachain-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}" 
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
logging.getLogger("httpx").setLevel(logging.WARNING) 

CHAIN_NAME_TO_ID = {
    "ethereum": 1, "base": 8453, "polygon": 137,
    "arbitrum": 42161, "op": 10, "fantom": 250,
    "berachain": 80094, 
}
ID_TO_CHAIN_NAME = {v: k.capitalize() for k, v in CHAIN_NAME_TO_ID.items()}

BLOCK_EXPLORER_URLS = {
    "ethereum": "https://etherscan.io",
    "polygon": "https://polygonscan.com",
    "op": "https://optimistic.etherscan.io",
    "base": "https://basescan.org",
    "arbitrum": "https://arbiscan.io",
    "fantom": "https://ftmscan.com",
    "berachain": "https://berascan.com" 
}

ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"} 
]

BEARN_FACTORY_ABI = [{"inputs":[{"internalType":"address","name":"_authorizer","type":"address"},{"internalType":"address","name":"_beraVaultFactory","type":"address"},{"internalType":"address","name":"_yBGT","type":"address"},{"internalType":"address","name":"_keeper","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"AlreadyExists","type":"error"},{"inputs":[],"name":"NoBeraVault","type":"error"},{"inputs":[],"name":"NotInitialized","type":"error"},{"anonymous":False,"inputs":[{"indexed":False,"internalType":"address","name":"newAuctionFactory","type":"address"}],"name":"NewAuctionFactory","type":"event"},{"anonymous":False,"inputs":[{"indexed":False,"internalType":"address","name":"newVaultManager","type":"address"}],"name":"NewVaultManager","type":"event"},{"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"stakingToken","type":"address"},{"indexed":False,"internalType":"address","name":"compoundingVault","type":"address"},{"indexed":False,"internalType":"address","name":"yBGTVault","type":"address"}],"name":"NewVaults","type":"event"},{"inputs":[],"name":"AUTHORIZER","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"bearnAuctionFactory","outputs":[{"internalType":"contract IBearnAuctionFactory","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"bearnVaultManager","outputs":[{"internalType":"contract IBearnVaultManager","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"beraVaultFactory","outputs":[{"internalType":"contract IRewardVaultFactory","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"stakingToken","type":"address"}],"name":"createVaults","outputs":[{"internalType":"address","name":"compoundingVault","type":"address"},{"internalType":"address","name":"yBGTVault","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"getAllBgtEarnerVaults","outputs":[{"internalType":"address[]","name":"","type":"address[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAllBgtEarnerVaultsLength","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAllCompoundingVaults","outputs":[{"internalType":"address[]","name":"","type":"address[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAllCompoundingVaultsLength","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"getBgtEarnerVault","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"getCompoundingVault","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"bearnVaults","type":"address"}],"name":"isBearnVault","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"keeper","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"_newAuctionFactory","type":"address"}],"name":"setAuctionFactory","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"_newBearnVaultManager","type":"address"}],"name":"setVaultManager","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"stakingToken","type":"address"}],"name":"stakingToBGTEarnerVaults","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"stakingToken","type":"address"}],"name":"stakingToCompoundingVaults","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"yBGT","outputs":[{"internalType":"contract ERC20","name":"","type":"address"}],"stateMutability":"view","type":"function"}]

BEARN_UI_CONTROL_ABI = [{"inputs":[{"internalType":"address","name":"_authorizer","type":"address"},{"internalType":"address","name":"_styBGT","type":"address"},{"internalType":"address","name":"_bearnVaultFactory","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"UnequalLengths","type":"error"},{"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"stakingToken","type":"address"},{"indexed":False,"internalType":"bool","name":"state","type":"bool"}],"name":"WhitelistChanged","type":"event"},{"inputs":[],"name":"AUTHORIZER","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"stakingToken","type":"address"},{"internalType":"bool","name":"state","type":"bool"}],"name":"adjustWhitelist","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address[]","name":"stakingTokens","type":"address[]"},{"internalType":"bool[]","name":"states","type":"bool[]"}],"name":"adjustWhitelists","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"bearnAuctionFactory","outputs":[{"internalType":"contract IBearnAuctionFactory","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"bearnVaultFactory","outputs":[{"internalType":"contract IBearnVaultFactory","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"bexVault","outputs":[{"internalType":"contract IBexVault","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"burrVault","outputs":[{"internalType":"contract IBexVault","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAllWhitelistedStakes","outputs":[{"internalType":"address[]","name":"","type":"address[]"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getAllWhitelistedStakesLength","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"bearnVault","type":"address"}],"name":"getApr","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"bexPool","type":"address"}],"name":"getBexLpPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"burrPool","type":"address"}],"name":"getBurrBearLpPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"hypervisor","type":"address"}],"name":"getHypervisorPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"kodiakIsland","type":"address"}],"name":"getKodiakIslandPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"kodiakV2Pair","type":"address"}],"name":"getKodiakV2Price","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"}],"name":"getPythPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"stakeToken","type":"address"}],"name":"getStakePrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"user","type":"address"},{"internalType":"address[]","name":"vaults","type":"address[]"}],"name":"getUserScaledAssets","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"user","type":"address"},{"internalType":"address","name":"vault","type":"address"}],"name":"getUserScaledAssets","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"user","type":"address"},{"internalType":"address[]","name":"vaults","type":"address[]"}],"name":"getUserUpdatedEarneds","outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"index","type":"uint256"}],"name":"getWhitelistedStake","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"getYBGTPrice","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"honey","outputs":[{"internalType":"contract ERC20","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"kodiakFactory","outputs":[{"internalType":"contract IUniswapV3Factory","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"stake","type":"address"}],"name":"nameOverrides","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"pythOracle","outputs":[{"internalType":"contract IPythOracle","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"}],"name":"pythOracleIds","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"stakingToken","type":"address"},{"internalType":"string","name":"name","type":"string"}],"name":"setNameOverride","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address[]","name":"stakingTokens","type":"address[]"},{"internalType":"string[]","name":"names","type":"string[]"}],"name":"setNameOverrides","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"bytes32","name":"oracleId","type":"bytes32"}],"name":"setPythOracleId","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"stakingToken","type":"address"},{"internalType":"address","name":"destinationAddress","type":"address"}],"name":"setTokenAddressOverride","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"styBGT","outputs":[{"internalType":"contract IStakedBearnBGT","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"styBGTApr","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"}],"name":"tokenAddressOverrides","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"wbera","outputs":[{"internalType":"contract ERC20","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"yBGT","outputs":[{"internalType":"contract IBearnBGT","name":"","type":"address"}],"stateMutability":"view","type":"function"}]

BEARN_VAULT_ABI = [
    {"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"stakingAsset","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"internalType":"uint8","name":"","type":"uint8"}],"stateMutability":"view","type":"function"}, 

    {"inputs":[],"name":"exit","outputs":[],"stateMutability":"nonpayable","type":"function"} 
]


try:
    with open("v1_vaults.json", "r") as f:
        V1_VAULTS = json.load(f)
except Exception as e:
    logging.warning(f"Could not load v1_vaults.json: {e}. V1 deposit checks will fail.")
    V1_VAULTS = []

async def send_long_message(
    target: Union[discord.TextChannel, discord.Message],
    text: str
):
    """Sends a potentially long message, splitting it into chunks if necessary."""
    if not text: 
        return

    chunks = []
    if len(text) <= MAX_DISCORD_MESSAGE_LENGTH:
        chunks.append(text)
    else:
        current_chunk = ""
        lines = text.splitlines(keepends=True)
        for line in lines:
            if len(line) > MAX_DISCORD_MESSAGE_LENGTH:
                 for i in range(0, len(line), MAX_DISCORD_MESSAGE_LENGTH):
                     part = line[i:i + MAX_DISCORD_MESSAGE_LENGTH]
                     if current_chunk: chunks.append(current_chunk)
                     current_chunk = ""
                     chunks.append(part)
                 continue

            if len(current_chunk) + len(line) > MAX_DISCORD_MESSAGE_LENGTH:
                if current_chunk: chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += line
        if current_chunk: chunks.append(current_chunk)

        if len(chunks) == 1 and len(chunks[0]) > MAX_DISCORD_MESSAGE_LENGTH:
             logging.warning("Message splitting resulted in a single chunk still exceeding limit. Truncating.")
             chunks = [chunks[0][:MAX_DISCORD_MESSAGE_LENGTH - 3] + "..."]
        elif not chunks:
             logging.warning("Message splitting resulted in zero chunks for long message.")
             chunks.append(text[:MAX_DISCORD_MESSAGE_LENGTH - 3] + "...")

    first_message = True
    try:
        for chunk in chunks:
            if not chunk.strip(): continue
            if first_message:
                if isinstance(target, discord.Message):
                    await target.reply(chunk, suppress_embeds=True)
                elif isinstance(target, discord.TextChannel):
                    await target.send(chunk, suppress_embeds=True)
                first_message = False
            else:
                channel = target.channel if isinstance(target, discord.Message) else target
                await channel.send(chunk, suppress_embeds=True)

            await asyncio.sleep(0.3)
    except discord.HTTPException as e:
         logging.error(f"Discord API error sending message chunk: {e}")
         try:
             channel = target.channel if isinstance(target, discord.Message) else target
             await channel.send(f"Error: Could not send full message due to Discord API issue ({e.status}).", suppress_embeds=True)
         except Exception:
             pass
    except Exception as e:
         logging.error(f"Unexpected error in send_long_message: {e}", exc_info=True)

def resolve_ens(name: str) -> Optional[str]:
    """Resolves an ENS name using the Ethereum Web3 instance."""
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
    """Extracts the first 0x address or .eth ENS name from text."""

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

async def _get_token_symbol(web3_instance: Web3, token_address: str) -> str:
    """Safely fetches ERC20 symbol."""
    if not web3_instance or not Web3.is_address(token_address):
        return "N/A"
    try:
        checksum_addr = Web3.to_checksum_address(token_address)
        contract = web3_instance.eth.contract(address=checksum_addr, abi=ERC20_ABI)
        symbol = await asyncio.to_thread(contract.functions.symbol().call)
        return symbol
    except Exception as e:
        logging.warning(f"Could not fetch symbol for token {token_address}: {e}")
        return "N/A"

async def _get_token_decimals(web3_instance: Web3, token_address: str) -> int:
    """Safely fetches ERC20 decimals, defaulting to 18."""
    if not web3_instance or not Web3.is_address(token_address):
        return 18
    try:
        checksum_addr = Web3.to_checksum_address(token_address)
        contract = web3_instance.eth.contract(address=checksum_addr, abi=ERC20_ABI)
        decimals = await asyncio.to_thread(contract.functions.decimals().call)
        return int(decimals)
    except Exception as e:
        logging.warning(f"Could not fetch decimals for token {token_address}: {e}. Defaulting to 18.")
        return 18

async def _fetch_bearn_vault_details(
    web3_bera: Web3,
    ui_contract, 
    vault_address: str,
    vault_type: Literal["Compounding", "BGT Earner"]
) -> Optional[Dict]:
    """Fetches details for a single Bearn vault."""
    if not web3_bera or not ui_contract or not Web3.is_address(vault_address):
        return None

    try:
        vault_checksum = Web3.to_checksum_address(vault_address)
        vault_contract = web3_bera.eth.contract(address=vault_checksum, abi=BEARN_VAULT_ABI)

        symbol_task = asyncio.create_task(asyncio.to_thread(vault_contract.functions.symbol().call))
        asset_addr_task = asyncio.create_task(asyncio.to_thread(vault_contract.functions.stakingAsset().call))
        apr_task = asyncio.create_task(asyncio.to_thread(ui_contract.functions.getApr(vault_checksum).call))

        vault_symbol, underlying_asset_addr, apr_raw = await asyncio.gather(
            symbol_task, asset_addr_task, apr_task, return_exceptions=True
        )

        if isinstance(vault_symbol, Exception):
            logging.warning(f"Error fetching symbol for Bearn vault {vault_address}: {vault_symbol}")
            vault_symbol = "Error"
        if isinstance(underlying_asset_addr, Exception):
            logging.warning(f"Error fetching stakingAsset for Bearn vault {vault_address}: {underlying_asset_addr}")
            underlying_asset_addr = None
        if isinstance(apr_raw, Exception):
            logging.warning(f"Error fetching APR for Bearn vault {vault_address}: {apr_raw}")
            apr_raw = 0

        underlying_symbol = "N/A"
        underlying_price_usd = 0.0
        if underlying_asset_addr and Web3.is_address(underlying_asset_addr):
            underlying_symbol = await _get_token_symbol(web3_bera, underlying_asset_addr)
            try:
                price_raw = await asyncio.to_thread(ui_contract.functions.getStakePrice(underlying_asset_addr).call)

                underlying_price_usd = float(price_raw) / 1e18
            except Exception as e:
                logging.warning(f"Error fetching stake price for {underlying_asset_addr}: {e}")
                underlying_price_usd = 0.0
        else:
             underlying_asset_addr = "N/A" 

        apr_percent = (float(apr_raw) / 1e18) * 100 if apr_raw else 0.0

        return {
            "address": vault_checksum,
            "symbol": vault_symbol,
            "type": vault_type,
            "apr_percent": apr_percent,
            "underlying_address": underlying_asset_addr,
            "underlying_symbol": underlying_symbol,
            "underlying_price_usd": underlying_price_usd,
            "chain_id": BERACHAIN_CHAIN_ID,
        }

    except Exception as e:
        logging.error(f"Unexpected error fetching details for Bearn vault {vault_address}: {e}", exc_info=True)
        return None

@function_tool
async def search_bearn_tool(
    query: str,

    vault_type: Optional[Literal["compounding", "bgt_earner", "both"]] 
) -> str:
    """
    Search for active Bearn vaults on Berachain.
    Provide a search query:
    - 'all': Lists all vaults.
    - Underlying Token Address (0x...): Finds the Compounding and BGT Earner vaults for that specific token.
    - Token Pair (e.g., 'OOGA/WBERA', 'HONEY-USDC'): Attempts to find vaults by matching token symbols within the vault's symbol (best-effort).
    Optionally filter by 'vault_type' ('compounding', 'bgt_earner', or 'both'). If omitted, defaults to 'both'.
    Returns details for matching vaults: Address, Symbol, Type, Underlying Asset, Underlying Price (USD), APR (%), and Berascan link.
    """

    effective_vault_type = vault_type if vault_type is not None else "both"

    logging.info(f"[Tool:search_bearn] Query: '{query}', Type requested: '{vault_type}', Effective type: '{effective_vault_type}'")

    web3_bera = WEB3_INSTANCES.get("berachain")
    if not web3_bera or not web3_bera.is_connected():
        return "Error: Berachain connection is not available."

    try:
        factory_contract = web3_bera.eth.contract(address=BEARN_FACTORY_ADDRESS, abi=BEARN_FACTORY_ABI)
        ui_contract = web3_bera.eth.contract(address=BEARN_UI_CONTROL_ADDRESS, abi=BEARN_UI_CONTROL_ABI)
    except Exception as e:
        logging.error(f"Error instantiating Bearn contracts: {e}")
        return "Error: Could not set up Bearn contracts."

    vaults_to_check: Dict[str, Literal["Compounding", "BGT Earner"]] = {} 

    query_lower = query.lower().strip()
    search_mode = "all" 

    if Web3.is_address(query):
        search_mode = "address"
        logging.info(f"[Tool:search_bearn] Search mode: address ({query})")
    elif query_lower == "all":
        search_mode = "all"
        logging.info("[Tool:search_bearn] Search mode: all")
    elif "/" in query or "-" in query or " " in query: 
        search_mode = "pair_symbol"
        logging.info(f"[Tool:search_bearn] Search mode: pair_symbol ({query})")
        pair_symbols = [s.strip().lower() for s in re.split(r'[/ -]', query) if s.strip()]
        if len(pair_symbols) < 2:
            return f"Error: Could not reliably extract two token symbols from pair query '{query}'. Please use format like 'TOKEN1/TOKEN2' or 'TOKEN1-TOKEN2'."
        logging.info(f"[Tool:search_bearn] Extracted pair symbols: {pair_symbols}")
    else:
        search_mode = "single_symbol"
        logging.info(f"[Tool:search_bearn] Search mode: single_symbol ({query}) - will search all vault symbols.")

    try:
        comp_vaults = []
        bgt_vaults = []

        if search_mode == "address":
            underlying_addr = Web3.to_checksum_address(query)
            comp_task = asyncio.create_task(asyncio.to_thread(factory_contract.functions.stakingToCompoundingVaults(underlying_addr).call))
            bgt_task = asyncio.create_task(asyncio.to_thread(factory_contract.functions.stakingToBGTEarnerVaults(underlying_addr).call))
            comp_addr, bgt_addr = await asyncio.gather(comp_task, bgt_task)

            if comp_addr and not comp_addr.startswith("0x00"): comp_vaults.append(comp_addr)
            if bgt_addr and not bgt_addr.startswith("0x00"): bgt_vaults.append(bgt_addr)

        elif search_mode in ["all", "pair_symbol", "single_symbol"]:
            comp_task = asyncio.create_task(asyncio.to_thread(factory_contract.functions.getAllCompoundingVaults().call))
            bgt_task = asyncio.create_task(asyncio.to_thread(factory_contract.functions.getAllBgtEarnerVaults().call))
            comp_vaults, bgt_vaults = await asyncio.gather(comp_task, bgt_task)

        if effective_vault_type != "bgt_earner":
            for addr in comp_vaults: vaults_to_check[addr] = "Compounding"
        if effective_vault_type != "compounding":
            for addr in bgt_vaults: vaults_to_check[addr] = "BGT Earner"

        if not vaults_to_check:
            if search_mode == "address":
                 return f"No Bearn vaults found for underlying token address: {query}"
            else:
                 return "No Bearn vaults found matching the criteria."

        logging.info(f"[Tool:search_bearn] Initial vault list size (after type filter): {len(vaults_to_check)}")

    except Exception as e:
        logging.error(f"Error fetching vault list from factory: {e}", exc_info=True)
        return f"Error: Could not retrieve vault list from Bearn factory contract: {e}"

    tasks = []
    semaphore = asyncio.Semaphore(10)
    detailed_results_list = []
    vault_addresses_to_fetch = list(vaults_to_check.keys())

    if search_mode in ["pair_symbol", "single_symbol"]:
        logging.info(f"[Tool:search_bearn] Performing symbol filtering for mode '{search_mode}'...")
        symbol_tasks = {}
        for addr in vault_addresses_to_fetch:
            try:
                vault_contract = web3_bera.eth.contract(address=Web3.to_checksum_address(addr), abi=BEARN_VAULT_ABI)
                symbol_tasks[addr] = asyncio.create_task(asyncio.to_thread(vault_contract.functions.symbol().call))
            except Exception as e:
                 logging.warning(f"Could not create symbol task for {addr}: {e}")
                 symbol_tasks[addr] = None

        await asyncio.gather(*[task for task in symbol_tasks.values() if task is not None], return_exceptions=True)

        filtered_addresses = []
        for addr, task in symbol_tasks.items():
            if task and task.done() and not task.cancelled() and task.exception() is None:
                vault_symbol = task.result().lower()
                match = False
                if search_mode == "pair_symbol":
                    if all(p_sym in vault_symbol for p_sym in pair_symbols):
                        match = True
                elif search_mode == "single_symbol":
                     if query_lower in vault_symbol:
                         match = True
                if match:
                    filtered_addresses.append(addr)

            elif task and task.exception():
                 logging.warning(f"Error fetching symbol for filtering vault {addr}: {task.exception()}")

        logging.info(f"[Tool:search_bearn] Symbol filtering reduced list from {len(vault_addresses_to_fetch)} to {len(filtered_addresses)}")
        vault_addresses_to_fetch = filtered_addresses
        if not vault_addresses_to_fetch:
             return f"No Bearn vaults found whose symbols match '{query}'."

    logging.info(f"[Tool:search_bearn] Fetching full details for {len(vault_addresses_to_fetch)} vaults.")
    for address in vault_addresses_to_fetch:
        vault_type_val = vaults_to_check.get(address)
        if not vault_type_val: 
            logging.warning(f"Could not determine vault type for address {address} during detail fetch.")
            continue
        async with semaphore:
            task = asyncio.create_task(_fetch_bearn_vault_details(web3_bera, ui_contract, address, vault_type_val))
            tasks.append(task)

    results_raw = await asyncio.gather(*tasks)
    detailed_results_list = [res for res in results_raw if res is not None]

    if not detailed_results_list:
        if search_mode in ["pair_symbol", "single_symbol"] and not vault_addresses_to_fetch:
             return f"No Bearn vaults found whose symbols match '{query}'." 
        else:
             return "Found matching Bearn vault(s), but failed to retrieve their details."

    summaries = []
    num_shown = len(detailed_results_list)

    header = f"Found {num_shown} Bearn vault(s) matching your query '{query}' (Type filter: {effective_vault_type}):"
    summaries.append(header)
    summaries.append("---")

    detailed_results_list.sort(key=lambda x: x.get('apr_percent', 0.0), reverse=True)

    for i, v_detail in enumerate(detailed_results_list):
        vault_addr = v_detail.get("address", "N/A")
        chain_name_display = ID_TO_CHAIN_NAME.get(v_detail.get("chain_id"), "Unknown")
        explorer_link = f"{BLOCK_EXPLORER_URLS.get('berachain', '')}/address/{vault_addr}" if BLOCK_EXPLORER_URLS.get('berachain') else "N/A"
        vault_symbol = v_detail.get("symbol", "N/A")
        vault_type_display = v_detail.get("type", "N/A")
        apr_display = v_detail.get("apr_percent", 0.0)
        underlying_symbol = v_detail.get("underlying_symbol", "N/A")
        underlying_price = v_detail.get("underlying_price_usd", 0.0)
        underlying_addr = v_detail.get("underlying_address", "N/A")

        summary_lines = [
            f"**{i+1}. Vault Symbol:** {vault_symbol}",
            f"   - Address: `{vault_addr}`",
            f"   - Type: {vault_type_display}",
            f"   - Chain: {chain_name_display}",
            f"   - Underlying: {underlying_symbol} (`{underlying_addr}`)",
            f"   - Underlying Price: ${underlying_price:,.4f}",
            f"   - APR: {apr_display:.2f}%",
            f"   - Explorer: {explorer_link if explorer_link != 'N/A' else 'Link unavailable'}"
        ]
        summaries.append("\n".join(summary_lines))

    result_text = "\n\n".join(summaries)
    logging.info(f"[Tool:search_bearn] Formatted Result:\n{result_text}")
    return result_text

@function_tool
async def check_bearn_deposits_tool(user_address_or_ens: str) -> str:
    """
    Checks a user's deposits across ALL active Bearn vaults (Compounding and BGT Earner) on Berachain.
    Provide the user's wallet address or ENS name.
    Returns a summary of vaults where the user has a non-zero balance.
    """
    logging.info(f"[Tool:check_bearn_deposits] Checking for {user_address_or_ens}")
    resolved_address = resolve_ens(user_address_or_ens)
    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum-style address."

    web3_bera = WEB3_INSTANCES.get("berachain")
    if not web3_bera or not web3_bera.is_connected():
        return "Error: Berachain connection is not available."

    try:
        user_checksum_addr = Web3.to_checksum_address(resolved_address)
        factory_contract = web3_bera.eth.contract(address=BEARN_FACTORY_ADDRESS, abi=BEARN_FACTORY_ABI)
        ui_contract = web3_bera.eth.contract(address=BEARN_UI_CONTROL_ADDRESS, abi=BEARN_UI_CONTROL_ABI)
    except ValueError:
         return f"Invalid address format after resolving: {resolved_address}"
    except Exception as e:
        logging.error(f"Error instantiating Bearn contracts: {e}")
        return "Error: Could not set up Bearn contracts."

    all_vault_addresses = []
    try:
        comp_task = asyncio.create_task(asyncio.to_thread(factory_contract.functions.getAllCompoundingVaults().call))
        bgt_task = asyncio.create_task(asyncio.to_thread(factory_contract.functions.getAllBgtEarnerVaults().call))
        comp_vaults, bgt_vaults = await asyncio.gather(comp_task, bgt_task)
        all_vault_addresses.extend(comp_vaults)
        all_vault_addresses.extend(bgt_vaults)

        all_vault_addresses = [addr for addr in all_vault_addresses if Web3.is_address(addr) and not addr.startswith("0x00")]
        logging.info(f"[Tool:check_bearn_deposits] Found {len(all_vault_addresses)} total Bearn vaults.")
        if not all_vault_addresses:
            return "No active Bearn vaults found in the factory contract."
    except Exception as e:
        logging.error(f"Error fetching all vault lists from factory: {e}", exc_info=True)
        return f"Error: Could not retrieve vault lists from Bearn factory contract: {e}"

    balances_raw = []
    try:
        checksummed_vaults = [Web3.to_checksum_address(addr) for addr in all_vault_addresses]
        logging.info(f"[Tool:check_bearn_deposits] Calling getUserScaledAssets for {len(checksummed_vaults)} vaults and user {user_checksum_addr}.")
        balances_raw = await asyncio.to_thread(
            ui_contract.functions.getUserScaledAssets(user_checksum_addr, checksummed_vaults).call
        )
        logging.info(f"[Tool:check_bearn_deposits] Received {len(balances_raw)} balances.")
        if len(balances_raw) != len(all_vault_addresses):
             logging.warning("Mismatch between number of vaults queried and balances received!")
             return "Error: Received an inconsistent number of balances from the contract."

    except Exception as e:
        logging.error(f"Error calling getUserScaledAssets: {e}", exc_info=True)
        return f"Error: Failed to fetch user balances from Bearn UI contract: {e}"

    deposits_found = []
    vaults_with_balance_addr = []
    vault_balances_map = {} 
    vault_decimals_map = {} 

    for i, balance_int in enumerate(balances_raw):
        if balance_int > 0:
            vault_addr = all_vault_addresses[i]
            vaults_with_balance_addr.append(vault_addr)
            vault_balances_map[vault_addr] = balance_int

    if not vaults_with_balance_addr:
        return f"No deposits found in any active Bearn vaults for address {resolved_address}."

    logging.info(f"[Tool:check_bearn_deposits] Found {len(vaults_with_balance_addr)} vaults with non-zero balance. Fetching symbols/decimals...")

    symbol_tasks = {}
    decimal_tasks = {}
    for addr in vaults_with_balance_addr:
        try:
            vault_contract = web3_bera.eth.contract(address=Web3.to_checksum_address(addr), abi=BEARN_VAULT_ABI)
            symbol_tasks[addr] = asyncio.create_task(asyncio.to_thread(vault_contract.functions.symbol().call))
            decimal_tasks[addr] = asyncio.create_task(asyncio.to_thread(vault_contract.functions.decimals().call))
        except Exception as e:
            logging.warning(f"Could not create symbol/decimal task for {addr}: {e}")
            symbol_tasks[addr] = None
            decimal_tasks[addr] = None

    await asyncio.gather(*[task for task in symbol_tasks.values() if task], return_exceptions=True)
    await asyncio.gather(*[task for task in decimal_tasks.values() if task], return_exceptions=True)

    for addr in vaults_with_balance_addr:
        symbol = "N/A"
        decimals = 18 
        balance_int = vault_balances_map.get(addr, 0)

        sym_task = symbol_tasks.get(addr)
        if sym_task and sym_task.done() and not sym_task.cancelled() and sym_task.exception() is None:
            symbol = sym_task.result()
        elif sym_task and sym_task.exception():
             logging.warning(f"Error fetching symbol for deposit vault {addr}: {sym_task.exception()}")

        dec_task = decimal_tasks.get(addr)
        if dec_task and dec_task.done() and not dec_task.cancelled() and dec_task.exception() is None:
            try:
                decimals = int(dec_task.result())
            except ValueError:
                 logging.warning(f"Invalid decimal value for deposit vault {addr}: {dec_task.result()}")
        elif dec_task and dec_task.exception():
             logging.warning(f"Error fetching decimals for deposit vault {addr}: {dec_task.exception()}")

        try:
            display_balance = float(balance_int) / (10 ** decimals)
            explorer_link = f"{BLOCK_EXPLORER_URLS.get('berachain', '')}/address/{addr}" if BLOCK_EXPLORER_URLS.get('berachain') else "N/A"
            deposit_info = (
                f"**Vault:** {symbol} (`{addr}`)\n"
                f"  Balance: {display_balance:.6f} shares\n" 
                f"  Explorer: {explorer_link if explorer_link != 'N/A' else 'Link unavailable'}"
            )
            deposits_found.append(deposit_info)
        except Exception as e:
            logging.error(f"Error formatting deposit info for vault {addr}: {e}")

    if deposits_found:
        return f"**Active Bearn Deposits Found for {resolved_address}:**\n\n" + "\n\n".join(deposits_found)
    else:
        return f"Found vaults with balance for {resolved_address}, but encountered errors formatting the details."

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
            logging.error(f"[Logic:query_v1_deposits] Error checking V1 vault {vault.get('address')} for {user_checksum_addr}: {e}")

    if found_deposits:
        return "**Deprecated V1 Vault Deposits Found (Ethereum Only):**\n\n" + "\n\n".join(found_deposits)
    else:
        return "No deposits found in deprecated V1 vaults for this address."

async def query_active_deposits_logic(resolved_address: str, chain: Optional[str] = None, token_symbol: Optional[str] = None) -> str:
    logging.info(f"[Logic:query_active_deposits] Checking Active for {resolved_address}, Chain: {chain}, Token: {token_symbol}")

    chains_to_check = []
    if chain:
        chain_lower = chain.lower()
        if chain_lower in WEB3_INSTANCES: chains_to_check.append(chain_lower)
        else: return f"Unsupported chain: {chain}."
    else:
        chains_to_check = list(WEB3_INSTANCES.keys())

    all_results = []
    total_deposits_found = 0
    all_vaults_data = [] 

    url = "https://ydaemon.yearn.fi/vaults/detected?limit=1000"
    try:
        response = await asyncio.to_thread(requests.get, url, timeout=15)
        response.raise_for_status()
        all_vaults_data = response.json()
        if not isinstance(all_vaults_data, list):
             logging.error("[Logic:query_active_deposits] Unexpected yDaemon response format.")
             return "Error: Received unexpected data format from vault API."
    except Exception as e:
        logging.error(f"[Logic:query_active_deposits] Failed to fetch vault list from yDaemon: {e}")
        return f"Error: Could not fetch the list of active vaults: {e}"

    for chain_name in chains_to_check:
        web3_instance = WEB3_INSTANCES.get(chain_name)
        if not web3_instance: continue
        try:
            user_checksum_addr = Web3.to_checksum_address(resolved_address)
        except ValueError:
             all_results.append(f"**{chain_name.capitalize()}**: Invalid address format {resolved_address}")
             continue

        chain_id = CHAIN_NAME_TO_ID.get(chain_name)
        
        if not chain_id: continue
        chain_vaults = [v for v in all_vaults_data if v.get("chainID") == chain_id]
        
        if token_symbol:
            token_lower = token_symbol.lower()
            chain_vaults = [v for v in chain_vaults if token_lower in v.get("token", {}).get("symbol", "").lower()]

        if not chain_vaults: continue
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
            all_results.append(f"**{chain_name.capitalize()} Active Deposits:**\n" + "\n\n".join(chain_deposits))
        elif len(chains_to_check) == 1:
             all_results.append(f"No active deposits found on {chain_name.capitalize()} for this address" + (f" matching token '{token_symbol}'." if token_symbol else "."))

    if total_deposits_found > 0:
        return "\n\n---\n\n".join(all_results)
    elif len(chains_to_check) > 1 :
        return "No active vault deposits found for that address on any supported chain" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    elif not all_results:
         return f"No active vault deposits found on {chains_to_check[0].capitalize()} for this address" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    else:
         return "\n\n".join(all_results)

@function_tool
async def check_all_deposits_tool(user_address_or_ens: str, token_symbol: Optional[str] = None) -> str:
    """
    Checks for user deposits in BOTH active (v2/v3) and deprecated (v1) Yearn vaults across all supported chains.
    Provide the user's wallet address or ENS name. Optionally filter by token symbol (e.g., 'USDC').
    Returns a combined summary of any deposits found in either type of vault.
    """
    logging.info(f"[Tool:check_all_deposits] Checking ALL vaults for {user_address_or_ens}, Token: {token_symbol}")
    resolved_address = resolve_ens(user_address_or_ens)
    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum address."

    v1_task = asyncio.create_task(query_v1_deposits_logic(resolved_address, token_symbol))
    active_task = asyncio.create_task(query_active_deposits_logic(resolved_address, chain=None, token_symbol=token_symbol))
    v1_results, active_results = await asyncio.gather(v1_task, active_task)
    final_output = []
    v1_found = "No deposits found in deprecated V1 vaults" not in v1_results
    active_found = "No active vault deposits found" not in active_results

    if v1_found:
        final_output.append(v1_results)
    if active_found:
        final_output.append(active_results)

    if not v1_found and not active_found:
        return f"No deposits found in any active or deprecated Yearn vaults for address {resolved_address}" + (f" matching token '{token_symbol}'." if token_symbol else ".")
    elif not v1_found:
        final_output.append("(No deposits found in deprecated V1 vaults)")
    elif not active_found:
         final_output.append("(No deposits found in active V2/V3 vaults)")

    combined_result = "\n\n---\n\n".join(final_output)
    logging.info(f"[Tool:check_all_deposits] Combined Result for {resolved_address}:\n{combined_result}")
    return combined_result

@function_tool
async def get_withdrawal_instructions_tool(user_address_or_ens: Optional[str], vault_address: str, chain: str) -> str:
    """
    Generates step-by-step instructions for withdrawing from a specific Yearn vault (v1, v2, or v3) using a block explorer.
    Also provides the direct link to the vault on the Yearn website for reference (for v2/v3).
    Provide the vault's address, the chain name (e.g., 'ethereum', 'fantom', 'arbitrum'), and optionally the user's address/ENS.
    Use this when a user asks how to withdraw or reports issues using the Yearn website for a specific vault.
    """
    logging.info(f"[Tool:get_withdrawal_instructions] Args: User={user_address_or_ens}, Vault={vault_address}, Chain={chain}")

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
    chain_id = CHAIN_NAME_TO_ID.get(chain_lower)
    explorer_base_url = BLOCK_EXPLORER_URLS.get(chain_lower)

    if not web3_instance or not chain_id or not explorer_base_url:
        return f"Unsupported or invalid chain: '{chain}'. Cannot generate instructions."

    try:
        vault_checksum_addr = Web3.to_checksum_address(vault_address)
    except ValueError as e:
        return f"Invalid vault address format provided: {e}. Please provide a valid vault address."

    explorer_vault_url = f"{explorer_base_url}/address/{vault_checksum_addr}"

    if chain_lower == 'ethereum':
        v1_vault_info = next((v for v in V1_VAULTS if v.get("address", "").lower() == vault_checksum_addr.lower()), None)

        if v1_vault_info:
            logging.info(f"Vault {vault_checksum_addr} identified as V1. Generating V1 instructions.")
            vault_name = v1_vault_info.get('name', vault_checksum_addr)

            instructions = [
                f"This vault (**{vault_name}** `{vault_checksum_addr}`) is a **deprecated Yearn V1 vault**.",
                "Withdrawals must be done directly via the block explorer:",
                f"1. Go to the vault contract page: {explorer_vault_url}",
                "2. Click the **'Contract'** tab.",
                "3. Click the **'Write Contract'** tab.",
                f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}.",
                f"5. Look for a suitable withdrawal function (often named like 'withdraw', 'withdrawAll', or similar). Prioritize functions that take no arguments if available.", 
                "6. Click the **'Write'** button next to the chosen function.",
                "7. Confirm the transaction in your wallet.",
                "\nOnce the transaction confirms, your funds should be back in your wallet."
            ]

            final_instructions = "\n".join(instructions)
            logging.info(f"Generated V1 instructions for {vault_checksum_addr}:\n{final_instructions}")
            return final_instructions
        else:
            logging.info(f"Vault {vault_checksum_addr} not found in V1 list. Proceeding to check V2/V3.")

    vault_details: Optional[Dict] = None
    async with aiohttp.ClientSession() as session:
        detail_semaphore = asyncio.Semaphore(5)
        vault_details = await _fetch_vault_details(session, chain_id, vault_checksum_addr, detail_semaphore)

    if not vault_details:
        logging.warning(f"Failed to fetch V2/V3 vault details for {vault_checksum_addr} on chain {chain_id} via yDaemon.")
        return (f"Could not fetch vault details from the Yearn API for `{vault_checksum_addr}` on {chain.capitalize()} "
                f"to determine the correct withdrawal method for V2/V3 vaults. \n"
                f"Please double-check the vault address and chain name. \n"
                f"You can view the contract directly here: {explorer_vault_url}")

    api_version_str = vault_details.get("version", "")
    vault_name = vault_details.get("name", vault_checksum_addr)
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
                contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=ERC20_ABI)
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
                contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=ERC20_ABI)
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
async def answer_from_docs_tool(
    wrapper: RunContextWrapper[BotRunContext],
    user_query: str
) -> str:
    """
    Answers questions based on documentation using a vector search (Pinecone).
    Determines whether to search Yearn or Bearn docs based on the run context.
    Use this for general questions about how the specified project works, concepts, etc.
    """

    project_context = wrapper.context.project_context

    logging.info(f"[Tool:answer_from_docs] --- Tool Invoked ---")
    logging.info(f"[Tool:answer_from_docs] Received query: '{user_query}'")
    logging.info(f"[Tool:answer_from_docs] Project context from wrapper: '{project_context}'") 
    top_k = 15

    if project_context == "yearn":
        namespace_to_query = YEARN_PINECONE_NAMESPACE
    elif project_context == "bearn":
        namespace_to_query = BEARN_PINECONE_NAMESPACE
    else:
        logging.warning(f"[Tool:answer_from_docs] Received unexpected project_context '{project_context}'. Defaulting to Yearn.")
        namespace_to_query = YEARN_PINECONE_NAMESPACE
        project_context = "yearn" 
    logging.info(f"[Tool:answer_from_docs] Querying Pinecone namespace: '{namespace_to_query}'")

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
            namespace=namespace_to_query, 
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = search_results.get("matches", [])
        logging.info(f"[Tool:answer_from_docs] Pinecone query returned {len(matches)} matches from namespace '{namespace_to_query}'.")
    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error querying Pinecone: {e}")
        return f"Sorry, I encountered an error while searching the {project_context.capitalize()} documentation."

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
        f"You are an assistant specialized in answering questions based **ONLY** on the provided {project_context.capitalize()} documentation context below. "
        "Your goal is to synthesize a comprehensive and accurate answer using *only* the information present in the 'Documentation Context' section.\n"
        "Use the information given below to answer the user's question accurately and concisely. "
        "If the answer is not present in the context, state that clearly ('I couldn't find information about X in the provided context.'). Do not add external knowledge. "
        "Cite the source if possible (included in the context)."
    )
    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Documentation Context:\n{context_text}\n\nUser Question: {user_query}"}
    ]
    logging.info(f"[Tool:answer_from_docs] Sending final prompt to LLM. User content length: {len(messages_for_llm[1]['content'])}")
    
    try:
        response = await openai_async_client.chat.completions.create(
            model="gpt-4o",
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
        return f"Sorry, I found relevant {project_context.capitalize()} documentation but encountered an error while formulating the final answer."

def add_project_context_to_handoff_input(
    handoff_input_data: HandoffInputData,
    project_context: Literal["yearn", "bearn"]
) -> HandoffInputData:
    """Adds a system message specifying the project context to the handoff input."""
    context_message: TResponseInputItem = {
        "role": "system",
        "content": f"CONTEXT_NOTE: You are handling a request related to the **{project_context.upper()}** project."
    }

    all_items = list(handoff_input_data.input_history) + \
                list(handoff_input_data.pre_handoff_items) + \
                list(handoff_input_data.new_items)

    filtered_items = [item for item in all_items if not (
        isinstance(item, dict) and
        item.get("role") == "system" and
        item.get("content", "").startswith("CONTEXT_NOTE:")
    )]

    new_input_list = [context_message] + filtered_items
    modified_new_items = list(handoff_input_data.new_items)
    pass 




yearn_data_agent = Agent[BotRunContext](
    name="Yearn Data Specialist",
    instructions=(
        "You are activated when a user needs specific Yearn data or withdrawal help. Your primary goal is to use tools to fetch data or provide instructions.\n\n"
        "**Workflow:**\n"
        "1.  **Analyze History & Request:** Review the conversation history. Identify the core need: checking deposits, finding vaults, or getting withdrawal help for a *specific* vault.\n"
        "2.  **Deposit Checks:** If the request is about general deposits/balances for an address, **IMMEDIATELY use the `check_all_deposits_tool`** with the user's address. Present the combined results.\n"
        "3.  **Vault Search:** If the request is to find vaults based on criteria (token, name, APR sort), use the `search_vaults_tool`.\n"
        "4.  **Specific Withdrawal Help:** If the user asks how to withdraw or has trouble withdrawing, and provides:\n"
        "    a.  **A specific vault address AND chain:** Use the `get_withdrawal_instructions_tool` directly.\n"
        "    b.  **A vault identifier (name/symbol like 'st-ycrv') BUT NOT a specific address/chain:** First, use the `search_vaults_tool` with the identifier to find the exact vault address and chain. If exactly one vault is found, proceed to use `get_withdrawal_instructions_tool` with the found address/chain and the user's address. If multiple vaults match the identifier, list them briefly and ask the user to confirm which one they mean before generating instructions. If no vault is found, report that.\n"
        "5.  **Address Resolution:** Ensure any ENS name provided is resolved to an address before using deposit or withdrawal tools.\n"
        "6.  **Direct Answers:** Answer directly based *only* on the output from the tools. Do not add speculation.\n"
        "7.  **Missing Info:** If necessary information (like an address for deposit checks, or vault address/chain for withdrawal instructions) is missing, ask the user clearly for it.\n"
        "8.  **Tool Errors:** If a tool returns an error message, relay that error message to the user.\n"
        f"9.  **Escalation:** If you cannot resolve the issue using tools (e.g., tool error persists, complex problem), state that human help is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}' in your response.**"
        "\n\n**CRITICAL:** Prioritize using the most appropriate tool based on the available information. Use `check_all_deposits_tool` for general balance checks. Use `get_withdrawal_instructions_tool` ONLY when a specific vault address is provided for withdrawal help."
    ),
    tools=[
        search_vaults_tool,
        check_all_deposits_tool,
        get_withdrawal_instructions_tool,
    ],
    model="o3-mini", 
)

yearn_docs_qa_agent = Agent[BotRunContext]( 
    name="Yearn Docs QA Specialist",
    instructions=(
        "You answer questions based *only* on Yearn documentation using the 'answer_from_docs_tool'.\n"
        "1. **IMMEDIATELY** use the `answer_from_docs_tool` with the user's query\n" 
        "2. Relay the answer or 'not found' message from the tool directly.\n"
        "4. Do NOT answer questions about real-time data (APR, TVL, balances) or specific user issues - state these require other specialists or human help.\n"
        f"5. If the question is complex even for the docs or seems like a bug report, state that human help is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}' in your response.**"
    ),
    tools=[answer_from_docs_tool],
    model="gpt-4o-mini", 
    model_settings=ModelSettings(temperature=0.2)
)

bd_priority_guardrail_agent = Agent[BotRunContext]( 
    name="BD/PR/Listing Guardrail Check",
    instructions=(
        "Analyze the user's message to classify its primary intent regarding Yearn. Use the following categories for the 'request_type' field:\n"
        "- 'listing': The user is asking Yearn to list their token on an exchange OR asking Yearn to provide liquidity for their token.\n"
        "- 'partnership': The user is proposing a technical integration, collaboration, or joint venture with Yearn.\n"
        "- 'marketing': The user is proposing a joint marketing campaign, AMA, or promotional activity.\n"
        "- 'other_bd': Other business development inquiries not covered above (e.g., grants, general BD contact).\n"
        "- 'not_bd_pr': This is a standard user support request, a question about using Yearn, a bug report, or unrelated chat.\n\n"
        "Focus on the main goal. If a message mentions multiple things, classify based on the primary ask. Be precise."
    ),
    output_type=BDPriorityCheckOutput, 
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.1)
)

bearn_data_agent = Agent[BotRunContext]( 
    name="Bearn Data Specialist",
    instructions=(
        "You handle requests for specific data related to the Bearn sub-project on Berachain.\n"
        "**Workflow:**\n"
        "1.  **Analyze Request:** Determine if the user wants to find Bearn vaults or check their deposits.\n"
        "2.  **Vault Search:** If the user asks to find vaults (e.g., 'list bearn vaults', 'find HONEY/WBERA vault', 'show vaults for 0x... token'), use the `search_bearn_tool`. Pass the user's query ('all', address, or pair) and optionally the vault type ('compounding', 'bgt_earner').\n"
        "3.  **Deposit Check:** If the user asks about their Bearn balance or deposits, use the `check_bearn_deposits_tool`. You MUST have the user's address for this. If missing, ask for it clearly.\n"
        "4.  **Present Results:** Relay the information returned by the tool directly and clearly. If a tool search returns no results, state that.\n"
        "5.  **Vault Not on UI:** If `search_bearn_tool` finds a vault but the user says it's not on the website, confirm it exists on-chain, provide the Berascan link, and explain the UI might lag.\n"
        "6.  **Tool Errors:** If a tool returns an error message, relay that error to the user.\n"
        f"7.  **Escalation:** For complex issues beyond vault searching/deposit checking, or if tools consistently fail, state human help is needed and include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}'."
    ),
    tools=[search_bearn_tool, check_bearn_deposits_tool], 
    model="o3-mini", 

)

bearn_docs_qa_agent = Agent[BotRunContext]( 
    name="Bearn Docs QA Specialist",
    instructions=(
        "You answer questions based *only* on **Bearn** documentation using the 'answer_from_docs_tool'.\n"
        "1. **IMMEDIATELY** use the `answer_from_docs_tool` with the user's query.\n" 
        "2. Relay the answer or 'not found' message from the tool directly.\n"
        "3. Do NOT answer about real-time data or Yearn-specific topics.\n"
        f"4. Escalate complex issues, bugs, or tool failures by including the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}'."
    ),
    tools=[answer_from_docs_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.2)
)

LISTING_DENIAL_MESSAGE = ( 
    "Thank you for your interest. Yearn Finance ($YFI) is permissionlessly listable on exchanges. "
    "Yearn does not pay listing fees, nor does it provide liquidity for exchange listings. "
    "No proposal is necessary for listing.\n\n"
    "Conversation ended. No follow up inquiries or responses necessary."
)

STANDARD_REDIRECT_MESSAGE = ( 
     f"Thank you for your interest! For partnership, marketing, or other business development proposals, "
     f"Go to <#{PR_MARKETING_CHANNEL_ID}>, share your proposal in **5 sentences** describing how it benefits both parties. "
     f"And tag **corn**. They handle these inquiries.\n\n"
     f"Conversation ended. No follow up inquiries or responses necessary."
)

@input_guardrail(name="BD/PR/Listing Guardrail")
async def bd_priority_guardrail(
    ctx: RunContextWrapper[BotRunContext],
    agent: Agent,
    input_data: Union[str, List[TResponseInputItem]] 
) -> GuardrailFunctionOutput: 
    """
    Checks if the initial user input is BD/PR/Listing.
    If it is, returns GuardrailFunctionOutput with tripwire_triggered=True
    and the appropriate response message stored in output_info.
    Otherwise, returns with tripwire_triggered=False.
    """

    if isinstance(input_data, str):
        text_input = input_data
    elif isinstance(input_data, list):
        text_input = ""
        for item in reversed(input_data):
             if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
                 text_input = item["content"]
                 break
    else: text_input = ""
    if not text_input: return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    logging.info(f"[Guardrail:BD/Priority] Analyzing input: '{text_input[:100]}...'")

    try:
        guardrail_runner = Runner()
        result = await guardrail_runner.run(
            starting_agent=bd_priority_guardrail_agent,
            input=text_input,
            run_config=RunConfig(workflow_name="BD/Priority Guardrail Check", tracing_disabled=True)
        )
        check_output = result.final_output_as(BDPriorityCheckOutput)
        logging.info(f"[Guardrail:BD/Priority] Check result: type={check_output.request_type}, Reasoning: {check_output.reasoning}")
        message_to_send = None
        should_trigger = False

        if check_output.request_type == "listing":
            message_to_send = LISTING_DENIAL_MESSAGE
            should_trigger = True
        elif check_output.request_type in ["partnership", "marketing", "other_bd"]:
            message_to_send = STANDARD_REDIRECT_MESSAGE
            should_trigger = True

        output_info_dict = {
            "classification": check_output.model_dump() 
        }
        if should_trigger and message_to_send:
            output_info_dict["message"] = message_to_send 
            logging.info(f"[Guardrail:BD/Priority] Returning tripwire=True. Type: {check_output.request_type}.")
            return GuardrailFunctionOutput(
                output_info=output_info_dict,
                tripwire_triggered=True 
            )
        else:
            logging.info("[Guardrail:BD/Priority] Returning tripwire=False.")
            return GuardrailFunctionOutput(
                output_info=output_info_dict, 
                tripwire_triggered=False
            )

    except Exception as e:
        logging.error(f"[Guardrail:BD/Priority] Error during check: {e}", exc_info=True)
        return GuardrailFunctionOutput(output_info={"error": str(e)}, tripwire_triggered=False)

triage_agent = Agent[BotRunContext](
    name="Support Triage Agent",
    instructions=(
        "You are the primary Yearn & Bearn support agent. Your task is to determine the **project context** (Yearn or Bearn) and the **request type**, then take immediate action based *first* on the project context.\n\n"
        "**1. Determine Project Context:**\n"
        "   - Check the `project_context` provided (Yearn or Bearn). This is the most reliable indicator.\n"
        "   - If context is 'unknown', analyze the message for keywords ('bearn', 'bera', 'bgt', specific Yearn names) to infer context. Default to 'yearn' if still unsure.\n\n"

        "**2. Execute Workflow Based on Determined Context:**\n\n"
        "   **--- IF Project Context is YEARN: ---**\n"
        "   a. **BD/PR/Marketing/Listing:** (Handled by Guardrail - You won't see these).\n"
        "   b. **Initial Address Handling:** If user provides an address (0x...) without specifying type: ASK them to clarify (wallet or vault) before proceeding.\n"
        "   c. **Data or Specific Withdrawal Request:** If the request is about finding vaults, checking deposits/balances, or asking how to withdraw from a specific vault address, AND the user's wallet address is known/confirmed: **IMMEDIATELY use `transfer_to_yearn_data_specialist` handoff.**\n"
        "   d. **Address Needed:** If user address is needed for (c) but missing: Ask clearly for the user's wallet address/ENS. Do NOT hand off yet.\n"
        "   e. **Handling Address Refusal:** If you asked for user address (for c) after they confirmed providing a vault address, and they refuse: Respond ONCE: 'Okay, I understand. Without your wallet address, I can't check your specific deposit balance. However, I *can* provide general withdrawal instructions for the vault `[Vault Address Provided By User]`. Would you like those instructions?' If yes, **IMMEDIATELY use `transfer_to_yearn_data_specialist` handoff.**\n"
        "   f. **General/Docs Question:** **IMMEDIATELY use `transfer_to_yearn_docs_qa_specialist` handoff.**\n"
        "   g. **UI Errors/Bugs/Complex Issues:** Respond that human support is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}'.** Do NOT hand off.\n"
        "   h. **Ambiguity:** If request type (Data vs Docs vs Bug) is unclear: Ask ONE clarifying question.\n"
        "   i. **Greetings/Chit-chat:** Respond briefly.\n\n"

        "   **--- ELSE IF Project Context is BEARN: ---**\n"
        "   a. **BD/PR/Marketing/Listing:** (Handled by Guardrail - You won't see these).\n"
        "   b. **Initial Address Handling:** If user provides an address (0x...) without specifying type: ASK them to clarify (wallet or vault) before proceeding.\n"
        "   c. **Data Request:** If the request is about finding vaults or checking deposits/balances: **IMMEDIATELY use `transfer_to_bearn_data_specialist` handoff.** (Requires user address for deposit checks).\n"
        "   d. **Address Needed:** If user address is needed for (c) but missing: Ask clearly for the user's wallet address/ENS. Do NOT hand off yet.\n"

        "   e. **General/Docs Question:** **IMMEDIATELY use `transfer_to_bearn_docs_qa_specialist` handoff.**\n"
        "   f. **UI Errors/Bugs/Complex Issues:** Respond that human support is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}'.** Do NOT hand off.\n"
        "   g. **Ambiguity:** If request type (Data vs Docs vs Bug) is unclear: Ask ONE clarifying question.\n"
        "   h. **Greetings/Chit-chat:** Respond briefly.\n\n"

        "**CRITICAL:** Always determine context first (Step 1). Then strictly follow the workflow for THAT context (Step 2). Execute handoffs immediately when conditions within the context's workflow are met. Do not describe the handoff."
    ),
    handoffs=[
        handoff(yearn_data_agent, tool_name_override="transfer_to_yearn_data_specialist", tool_description_override="Handoff for specific YEARN data (vaults, deposits, APR, TVL, balances, withdrawal instructions)."),
        handoff(yearn_docs_qa_agent, tool_name_override="transfer_to_yearn_docs_qa_specialist", tool_description_override="Handoff for general questions about YEARN concepts, documentation, risks."),
        handoff(bearn_data_agent, tool_name_override="transfer_to_bearn_data_specialist", tool_description_override="Handoff for specific BEARN data (vault search, deposit checks)."),
        handoff(bearn_docs_qa_agent, tool_name_override="transfer_to_bearn_docs_qa_specialist", tool_description_override="Handoff for general questions about BEARN concepts or documentation.")
    ],
    input_guardrails=[bd_priority_guardrail],
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.1) 
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
        logging.info(f"Monitoring Yearn Ticket Category ID: {YEARN_TICKET_CATEGORY_ID}")
        logging.info(f"Monitoring Bearn Ticket Category ID: {BEARN_TICKET_CATEGORY_ID}")
        logging.info(f"Support User ID for triggers: {SUPPORT_USER_ID}")
        logging.info(f"Yearn Public Trigger: '{YEARN_PUBLIC_TRIGGER_CHAR}', Bearn Public Trigger: '{BEARN_PUBLIC_TRIGGER_CHAR}'")
        print("------")

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        if isinstance(channel, discord.TextChannel) and channel.category:
            if channel.category.id in CATEGORY_CONTEXT_MAP:
                project_context = CATEGORY_CONTEXT_MAP.get(channel.category.id, "unknown") 
                logging.info(f"New {project_context.capitalize()} ticket channel created: {channel.name} (ID: {channel.id}). Initializing state.")
                conversation_threads[channel.id] = []
                stopped_channels.discard(channel.id) 
                pending_messages.pop(channel.id, None)

                if channel.id in pending_tasks:
                    try:
                        pending_tasks.pop(channel.id).cancel()
                    except Exception as e:
                        logging.warning(f"Error cancelling task during channel create for {channel.id}: {e}")

    async def on_message(self, message: discord.Message):
        if message.author.bot or message.author.id == self.user.id:
            return

        run_context = BotRunContext(channel_id=message.channel.id) 
        is_reply = message.reference is not None
        trigger_char = message.content.strip()
        is_support_trigger = str(message.author.id) == SUPPORT_USER_ID and trigger_char in TRIGGER_CONTEXT_MAP

        if is_reply and is_support_trigger:
            run_context.is_public_trigger = True
            run_context.project_context = TRIGGER_CONTEXT_MAP.get(trigger_char, "unknown") 
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
                        reply_content = "An unexpected error occurred." 
                        target_message_for_reply = original_message 
                        try:

                            run_config = RunConfig(workflow_name=f"Public Channel Query ({run_context.project_context})") 
                            result: RunResult = await self.runner.run(
                                starting_agent=triage_agent, 
                                input=original_content,
                                max_turns=5, 
                                run_config=run_config,
                                context=run_context 
                            )

                            raw_reply_content = result.final_output if result.final_output else "I couldn't determine a response."
                            actual_mention = f"<@{SUPPORT_USER_ID}>"
                            reply_content = raw_reply_content.replace(HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                            await send_long_message(target_message_for_reply, reply_content)
                            logging.info(f"Sent public reply/replies to {original_message.id} in {message.channel.id}")

                        except InputGuardrailTripwireTriggered as e:
                            logging.warning(f"BD/PR Input Guardrail triggered for public query (Original msg ID: {original_message.id}). Guardrail Output: {e.guardrail_result.output.output_info}")
                            reply_content = (
                                f"Thank you for your interest! For partnership, marketing, or business development proposals, "
                                f"please post your message in the <#{PR_MARKETING_CHANNEL_ID}> channel and tag **corn**. "
                                f"They handle these inquiries."
                            )
                            await send_long_message(target_message_for_reply, reply_content) 

                        except MaxTurnsExceeded as e:
                             logging.warning(f"Max turns exceeded for public query (Original msg ID: {original_message.id}): {e}")
                             reply_content = f"Sorry, the request took too long to process. Please try simplifying or ask <@{SUPPORT_USER_ID}> for help."
                             await send_long_message(target_message_for_reply, reply_content) 

                        except AgentsException as e:
                             logging.error(f"Agent SDK error during public query (Original msg ID: {original_message.id}): {e}")
                             reply_content = f"Sorry, an error occurred while processing your request ({type(e).__name__})."
                             await send_long_message(target_message_for_reply, reply_content) 
                        except Exception as e:
                             logging.error(f"Unexpected error during public query processing (Original msg ID: {original_message.id}): {e}", exc_info=True)

                             await send_long_message(target_message_for_reply, reply_content) 

                    return 

            except discord.NotFound:
                logging.warning(f"Original message for reply {message.id} not found.")
            except discord.Forbidden:
                 logging.warning(f"Missing permissions to fetch original message for reply {message.id}.")
            except Exception as e:
                 logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)

        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category:
            return 

        channel_id = message.channel.id

        run_context.category_id = message.channel.category.id
        run_context.project_context = CATEGORY_CONTEXT_MAP.get(message.channel.category.id, "unknown")

        if run_context.project_context == "unknown":
             return

        if message.content.strip().lower() == "%stop":
            logging.info(f"%stop command received in channel {channel_id} from {message.author.name}")
            try:
                await message.delete()
                logging.info(f"Deleted %stop command message {message.id}")
            except discord.Forbidden:
                logging.warning(f"Missing permissions to delete %stop message {message.id} in {channel_id}. Proceeding with stop.")
            except discord.NotFound:
                logging.warning(f"%stop message {message.id} already deleted.")
            except Exception as e:
                logging.error(f"Error deleting %stop message {message.id}: {e}. Proceeding with stop.")

            stopped_channels.add(channel_id)
            conversation_threads.pop(channel_id, None) 
            pending_messages.pop(channel_id, None)     
            if channel_id in pending_tasks:
                try:
                    pending_tasks.pop(channel_id).cancel() 
                    logging.info(f"Cancelled pending task for channel {channel_id} due to %stop.")
                except KeyError:
                    pass 
                except Exception as e:
                     logging.error(f"Error cancelling pending task for {channel_id} during %stop: {e}")

            confirmation_message = "Support bot stopped for this channel. ySupport contributors are available for other further inquiries."
            try:
                await message.channel.send(confirmation_message)
                logging.info(f"Sent stop confirmation to channel {channel_id}")
            except discord.Forbidden:
                logging.error(f"Missing permissions to send stop confirmation in channel {channel_id}")
            except Exception as e:
                logging.error(f"Error sending stop confirmation to channel {channel_id}: {e}")

            return 

        if channel_id in stopped_channels:
            logging.info(f"Ignoring message in stopped channel {channel_id}")
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

        category_id = channel.category.id if channel.category else None
        project_ctx = CATEGORY_CONTEXT_MAP.get(category_id, "unknown")
        run_context = BotRunContext(
            channel_id=channel_id,
            category_id=category_id,
            project_context=project_ctx
        )

        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": aggregated_text}]

        logging.info(f"Processing aggregated text for ticket {channel_id} (Context: {project_ctx}): '{aggregated_text[:100]}...'")

        async with channel.typing():
            final_reply = "An unexpected error occurred." 
            should_stop_processing = False 
            try:
                run_config = RunConfig(
                    workflow_name=f"Ticket Channel {channel_id} ({project_ctx})", 
                    group_id=str(channel_id) 
                )
                result: RunResult = await self.runner.run(
                    starting_agent=triage_agent,
                    input=input_list,
                    max_turns=MAX_TICKET_CONVERSATION_TURNS,
                    run_config=run_config,
                    context=run_context 
                )

                conversation_threads[channel_id] = result.to_input_list()
                raw_final_reply = result.final_output if result.final_output else "I'm not sure how to respond to that."
                actual_mention = f"<@{SUPPORT_USER_ID}>" 
                final_reply = raw_final_reply.replace(HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                if actual_mention in final_reply:
                    logging.info(f"Human handoff tag detected and replaced/present in response for channel {channel_id}.")
                    should_stop_processing = True
                elif HUMAN_HANDOFF_TAG_PLACEHOLDER in raw_final_reply:
                     logging.warning(f"Handoff placeholder '{HUMAN_HANDOFF_TAG_PLACEHOLDER}' found in raw reply but not replaced in channel {channel_id}.")

                farewell_keywords = ["thank", "bye", "goodbye", "stop talking", "that's all", "resolved", "worked", "fixed"]
                user_said_farewell = any(kw in aggregated_text.lower() for kw in farewell_keywords)
                agent_said_farewell = "glad i could help" in final_reply.lower() or "concluding interaction" in final_reply.lower() 

                if user_said_farewell or agent_said_farewell:
                    if not should_stop_processing: 
                         final_reply += "\n\n*(Conversation ended. ySupport contributors will answer any follow up questions, or this ticket may be closed if no further assistance is needed..)*"
                    should_stop_processing = True
                    logging.info(f"Conversation ended naturally or by farewell in channel {channel_id}.")

            except InputGuardrailTripwireTriggered as e: 
                 logging.warning(f"Input Guardrail triggered in channel {channel_id}. Extracting message from output_info.")
                 guardrail_info = e.guardrail_result.output.output_info
                 if isinstance(guardrail_info, dict) and "message" in guardrail_info:
                     final_reply = guardrail_info["message"]

                     if "classification" in guardrail_info and isinstance(guardrail_info["classification"], dict):
                          logging.info(f"Guardrail classification: {guardrail_info['classification'].get('request_type', 'Unknown')}")
                 else:
                     logging.error("Guardrail triggered but message not found in output_info!")
                     final_reply = "Your request could not be processed due to input checks."

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
                await send_long_message(channel, final_reply) 
                logging.info(f"Sent ticket reply/replies in channel {channel_id}. Stop processing: {should_stop_processing}")
                if should_stop_processing:
                    stopped_channels.add(channel_id)
                    logging.info(f"Added channel {channel_id} to stopped channels.")

            except discord.Forbidden:
                 logging.error(f"Missing permissions to send message in channel {channel_id}")
                 stopped_channels.add(channel_id) 
            except Exception as e:
                 logging.error(f"Unexpected error occurred during or after calling send_long_message for channel {channel_id}: {e}", exc_info=True)

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
