import asyncio
import aiohttp
import traceback
import discord
from discord.ui import View, Button, button
import sys
import re
import json
import requests
import os
import logging
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
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
from dotenv import load_dotenv
load_dotenv()

sys.stdout = sys.stderr

@dataclass
class BotRunContext:
    channel_id: int
    category_id: Optional[int] = None
    is_public_trigger: bool = False
    project_context: Literal["yearn"] = "unknown"
    initial_button_intent: Optional[str] = None

# ----------------------------
# Configuration
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ALCHEMY_KEY = os.getenv("ALCHEMY_KEY")

# --- Pinecone ---
PINECONE_INDEX_NAME = "-"
YEARN_PINECONE_NAMESPACE = "-"

# --- Discord ---
PUBLIC_TRIGGER_USER_IDS = {
    "-",
    "-",
    "-"
}
YEARN_TICKET_CATEGORY_ID = -
YEARN_PUBLIC_TRIGGER_CHAR = "y"
HUMAN_HANDOFF_TARGET_USER_ID = "-"
HUMAN_HANDOFF_TAG_PLACEHOLDER = "{HUMAN_HANDOFF_TAG_PLACEHOLDER}" # The literal string in the instructions

# Map category IDs to project contexts
CATEGORY_CONTEXT_MAP = {
    YEARN_TICKET_CATEGORY_ID: "yearn",
}

# Map trigger chars to project contexts
TRIGGER_CONTEXT_MAP = {
    YEARN_PUBLIC_TRIGGER_CHAR: "yearn",
}

PR_MARKETING_CHANNEL_ID = -
MAX_DISCORD_MESSAGE_LENGTH = 1990 # Be slightly conservative

# --- Bot Behavior ---
COOLDOWN_SECONDS = 5 # Debounce time for user messages in tickets
MAX_TICKET_CONVERSATION_TURNS = 10 # Limit conversation length in tickets
MAX_RESULTS_TO_SHOW = 5 # Define how many results to show by default or when sorted
STRATEGY_FETCH_CONCURRENCY = 10 # Limit concurrent requests for strategy details
PUBLIC_TRIGGER_TIMEOUT_MINUTES = 30 # Timeout in minutes

# --- State to track channels awaiting button press ---
# This set will store channel IDs where the initial button message has been sent
# and we are waiting for the user to click a button.
channels_awaiting_initial_button_press: set[int] = set()
channel_intent_after_button: Dict[int, str] = {}

@dataclass
class PublicConversation:
    """Stores the state for a temporary public conversation."""
    history: List[TResponseInputItem]
    last_interaction_time: datetime

public_conversations: Dict[int, PublicConversation] = {}

class InitialInquiryView(View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    async def handle_button_click_and_prompt(self, interaction: discord.Interaction, button_custom_id: str, prompt_message: str, intent_category: str):
        await interaction.response.defer()

        try:
            view_to_disable = View.from_message(interaction.message)
            if view_to_disable:
                for child_button in view_to_disable.children:
                    if isinstance(child_button, Button):
                        child_button.disabled = True
                await interaction.edit_original_response(view=view_to_disable)
        except Exception as e:
            logging.warning(f"Could not disable initial inquiry buttons in {interaction.channel_id} on message {interaction.message.id if interaction.message else 'Unknown'}: {e}")

        channels_awaiting_initial_button_press.discard(interaction.channel.id)
        channel_intent_after_button[interaction.channel.id] = intent_category

        if interaction.channel:
            await interaction.channel.send(prompt_message)
        else:
            await interaction.followup.send(prompt_message, ephemeral=False)

        logging.info(f"Button '{button_custom_id}' clicked in {interaction.channel.id}. Intent set to '{intent_category}'. Sent follow-up prompt.")

    @button(label="ℹ️ Vault Info", style=discord.ButtonStyle.secondary, custom_id="initial_find_vaults", row=0)
    async def find_vaults_button(self, interaction: discord.Interaction, button: Button):
        prompt = "Okay, you want to find vaults. What token, vault, or criteria (e.g., 'current APY for yvUSDS', 'highest APY for USDC on Ethereum') are you looking for? Please be as specific as possible for the best results."
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "data_vault_search")

    @button(label="🔍 My Deposits/Where are my funds?", style=discord.ButtonStyle.secondary, custom_id="initial_check_deposits", row=0)
    async def check_deposits_button(self, interaction: discord.Interaction, button: Button):
        prompt = "Understood. To check your deposits, please provide your wallet address (it starts with 0x)."
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "data_deposit_check")

    @button(label="💸 Withdrawal Help/Issues", style=discord.ButtonStyle.secondary, custom_id="initial_withdrawal_help", row=1)
    async def withdrawal_help_button(self, interaction: discord.Interaction, button: Button):
        prompt = "Okay, I can help with withdrawal instructions. Please provide your wallet addres (0x...). I can then check your deposits and you can tell me which one you want to withdraw from."
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "data_withdrawal_flow_start")

    @button(label="📖 General Info/How-To", style=discord.ButtonStyle.secondary, custom_id="initial_general_info", row=1)
    async def general_info_button(self, interaction: discord.Interaction, button: Button):
        project_ctx_name = "Yearn"
        if interaction.channel.category and interaction.channel.category.id in CATEGORY_CONTEXT_MAP:
            project_ctx_name = CATEGORY_CONTEXT_MAP[interaction.channel.category.id].capitalize()
        prompt = f"Great! What specific information or product are you looking for, or what how-to question do you have about {project_ctx_name}?"
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "docs_qa")

    @button(label="🐞 Bug Report/UI Issue", style=discord.ButtonStyle.secondary, custom_id="initial_bug_report", row=2)
    async def bug_report_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        actual_mention = f"<@{HUMAN_HANDOFF_TARGET_USER_ID}>"
        response_message = (
            "Thank you for reporting this. To help us investigate, please describe the bug or UI problem in detail. "
            "Include steps to reproduce it if possible, and mention which device/browser you are using. "
            f"{actual_mention} will review your report."
        )
        if interaction.channel: await interaction.channel.send(response_message)
        else: await interaction.followup.send(response_message, ephemeral=False)
        stopped_channels.add(interaction.channel.id)
        logging.info(f"Bug report initiated in {interaction.channel.id}. Bot stopped.")

    @button(label="🤝 Business/Partnerships/Marketing", style=discord.ButtonStyle.secondary, custom_id="initial_bd_partner", row=2)
    async def bd_partner_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        if interaction.channel: await interaction.channel.send(STANDARD_REDIRECT_MESSAGE)
        else: await interaction.followup.send(STANDARD_REDIRECT_MESSAGE, ephemeral=False)
        stopped_channels.add(interaction.channel.id)
        logging.info(f"BD/Partner inquiry redirected in {interaction.channel.id}. Bot stopped.")

    @button(label="🛠️ Contribute/Work/Grants", style=discord.ButtonStyle.secondary, custom_id="initial_contribute_work", row=3)
    async def contribute_work_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        if interaction.channel: await interaction.channel.send(JOB_INQUIRY_REDIRECT_MESSAGE)
        else: await interaction.followup.send(JOB_INQUIRY_REDIRECT_MESSAGE, ephemeral=False)
        stopped_channels.add(interaction.channel.id)
        logging.info(f"Contribute/Work inquiry redirected in {interaction.channel.id}. Bot stopped.")

    @button(label="❓ Other/My Issue Isn't Listed", style=discord.ButtonStyle.secondary, custom_id="initial_other_issue", row=3)
    async def other_issue_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        prompt = "Okay, please describe your issue or question in detail below. I'll do my best to assist or find the right help for you."
        if interaction.channel: await interaction.channel.send(prompt)
        else: await interaction.followup.send(prompt, ephemeral=False)
        channel_intent_after_button[interaction.channel.id] = "other_free_form"
        logging.info(f"User selected 'Other' in {interaction.channel.id}. Awaiting free-form input.")

    async def handle_button_click(self, interaction: discord.Interaction, button_custom_id: str):
        if not interaction.response.is_done():
            await interaction.response.defer()
        try:
            view_to_disable = View.from_message(interaction.message)
            if view_to_disable:
                for child_button in view_to_disable.children:
                    if isinstance(child_button, Button):
                        child_button.disabled = True
                await interaction.edit_original_response(view=view_to_disable)
        except Exception as e:
            logging.warning(f"Could not disable initial inquiry buttons in {interaction.channel_id} on message {interaction.message.id if interaction.message else 'Unknown'}: {e}")
        channels_awaiting_initial_button_press.discard(interaction.channel.id)
        logging.info(f"Button '{button_custom_id}' clicked in {interaction.channel.id}. Channel removed from awaiting_initial_button_press.")


class BDPriorityCheckOutput(BaseModel):
    request_type: Literal["listing", "partnership", "marketing", "other_bd", "job_inquiry", "not_bd_pr"] = Field(..., description="Classify the user's primary intent: 'listing' (requesting Yearn list their token), 'partnership' (proposing integration/collaboration), 'marketing' (joint marketing/promotion), 'other_bd' (other business development), 'job_inquiry' (asking to work for/contribute to Yearn, grant requests), or 'not_bd_pr' (standard support request or unrelated).")
    reasoning: str = Field(..., description="Brief explanation for the classification.")

class GuardrailResponseMessageException(AgentsException):
    def __init__(self, message: str, guardrail_output: Optional[BDPriorityCheckOutput] = None):
        super().__init__(message)
        self.message = message
        self.guardrail_output = guardrail_output

class StopBotView(View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @button(label="Stop Bot", style=discord.ButtonStyle.secondary, custom_id="stop_bot_button")
    async def stop_button_callback(self, interaction: discord.Interaction, button: Button):
        channel_id = interaction.channel.id
        user_who_clicked = interaction.user

        logging.info(f"Stop Bot button clicked in channel {channel_id} by {user_who_clicked.name} ({user_who_clicked.id})")

        await interaction.response.defer()

        stopped_channels.add(channel_id)
        conversation_threads.pop(channel_id, None)
        pending_messages.pop(channel_id, None)
        if channel_id in pending_tasks:
            try:
                pending_tasks.pop(channel_id).cancel()
                logging.info(f"Cancelled pending task for channel {channel_id} due to Stop Bot button.")
            except KeyError:
                pass
            except Exception as e:
                 logging.error(f"Error cancelling pending task for {channel_id} during Stop Bot button: {e}")

        try:
            disabled_view = StopBotView()
            for item in disabled_view.children:
                if isinstance(item, Button) and item.custom_id == "stop_bot_button":
                    item.disabled = True
                    break
            await interaction.edit_original_response(view=disabled_view)
            logging.info(f"Disabled Stop Bot button on message {interaction.message.id}")
        except discord.HTTPException as e:
            logging.warning(f"Could not disable Stop Bot button on message {interaction.message.id}: {e.status} {e.text}")
        except Exception as e:
            logging.error(f"Unexpected error disabling Stop Bot button on message {interaction.message.id}: {e}", exc_info=True)


        confirmation_message = f"Support bot stopped for this channel. ySupport contributors are available for further inquiries."
        try:
            await interaction.followup.send(confirmation_message, ephemeral=False)
            logging.info(f"Sent stop confirmation to channel {channel_id}")
        except discord.HTTPException as e:
            logging.error(f"Failed to send stop confirmation followup in {channel_id}: {e.status} {e.text}")
        except Exception as e:
            logging.error(f"Unexpected error sending stop confirmation followup in {channel_id}: {e}", exc_info=True)


#  Clients
set_default_openai_key(OPENAI_API_KEY)

openai_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
openai_sync_client = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone
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


# Web3
WEB3_INSTANCES = {}
RPC_URLS = {
    "ethereum": f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "base": f"https://base-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "polygon": f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "arbitrum": f"https://arb-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "op": f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "fantom": f"https://fantom-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "fantom": f"https://fantom-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}",
    "katana": f"https://rpc.katana.network",
}
for name, url in RPC_URLS.items():
    try:
        WEB3_INSTANCES[name] = Web3(Web3.HTTPProvider(url))
        if not WEB3_INSTANCES[name].is_connected():
             print(f"Warning: Failed to connect to Web3 for {name}")
    except Exception as e:
        print(f"Error initializing Web3 for {name}: {e}")

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

# Constants & Mappings
CHAIN_NAME_TO_ID = {
    "ethereum": 1, "base": 8453, "polygon": 137,
    "arbitrum": 42161, "op": 10, "fantom": 250,
    "sonic": 146, "katana": 747474
}
ID_TO_CHAIN_NAME = {v: k.capitalize() for k, v in CHAIN_NAME_TO_ID.items()}

BLOCK_EXPLORER_URLS = {
    "ethereum": "https://etherscan.io",
    "polygon": "https://polygonscan.com",
    "op": "https://optimistic.etherscan.io",
    "base": "https://basescan.org",
    "arbitrum": "https://arbiscan.io",
    "fantom": "https://ftmscan.com",
    "sonic": "https://sonicscan.org",
    "katana": "https://explorer.katanarpc.com",
}

ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "stateMutability": "view", "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"} # Added symbol
]

GAUGE_ABI = [
    {"inputs":[{"internalType":"address","name":"account","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"_shares","type":"uint256"}],"name":"convertToAssets","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]


try:
    with open("v1_vaults.json", "r") as f:
        V1_VAULTS = json.load(f)
except Exception as e:
    logging.warning(f"Could not load v1_vaults.json: {e}. V1 deposit checks will fail.")
    V1_VAULTS = []


# Helpers
async def send_long_message(
    target: Union[discord.TextChannel, discord.Message],
    text: str,
    view: Optional[View] = None
):
    """Sends a potentially long message, splitting it into chunks if necessary."""
    if not text:
        return

    # --- Splitting ---
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
    sent_view = False
    try:
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue

            # --- Attach view ONLY to the LAST chunk ---
            current_view = view if (i == len(chunks) - 1 and view and not sent_view) else None

            if first_message:
                if isinstance(target, discord.Message):
                    await target.reply(chunk, suppress_embeds=True, view=current_view)
                elif isinstance(target, discord.TextChannel):
                    await target.send(chunk, suppress_embeds=True, view=current_view)
                first_message = False
            else:
                channel = target.channel if isinstance(target, discord.Message) else target
                await channel.send(chunk, suppress_embeds=True, view=current_view)

            if current_view:
                sent_view = True

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
    """
    Resolves an ENS name or a SAFE-style prefixed address (e.g., 'eth:0x...') to a standard checksummed address.
    Returns the checksummed address if valid, otherwise None.
    """
    if not isinstance(name, str):
        return None
        
    address_to_check = name.strip()

    # --- Handle SAFE-style prefixes (e.g., 'eth:', 'base:') ---
    if ':' in address_to_check:
        parts = address_to_check.split(':', 1)
        if len(parts) == 2:
            prefix, potential_address = parts
            logging.info(f"Detected prefixed address. Prefix: '{prefix}', Address part: '{potential_address}'")
            address_to_check = potential_address.strip()

    # Basic address check
    if Web3.is_address(address_to_check):
        try:
            return Web3.to_checksum_address(address_to_check)
        except ValueError:
            logging.warning(f"Address '{address_to_check}' has invalid checksum.")
            return None

    # ENS check
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
            
    logging.warning(f"Input '{name}' could not be resolved to a valid address.")
    return None


def is_message_primarily_address(text: str) -> bool:
    """
    Checks if a message string consists mainly of one or more Ethereum-style addresses.
    Handles optional prefixes like 'eth:'.
    """
    normalized_text = text.lower().strip()
    for word in ['vault', 'wallet', 'address', 'is', 'my', 'for', 'check', 'the']:
        normalized_text = normalized_text.replace(word, '')
    
    addresses_found = re.findall(r'(?:[a-z]+:)?(0x[a-f0-9]{40})', normalized_text)
    
    if not addresses_found:
        return False
        
    total_address_length = sum(len(addr) for addr in addresses_found)
    
    if total_address_length / len(normalized_text.replace(" ", "").replace(":", "")) > 0.7:
        return True
        
    return False

def extract_address_or_ens(text: str) -> Optional[str]:
    """Extracts the first 0x address or .eth ENS name from text."""
    addr_match = re.search(r'(0x[a-fA-F0-9]{40})', text)
    if addr_match:
        return addr_match.group(1)
    ens_match = re.search(r'\b([a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.eth)\b', text, re.IGNORECASE)
    if ens_match:
        return ens_match.group(1)
    return None

def format_timestamp_to_readable(timestamp: Optional[Union[int, float, str]]) -> str:
    if timestamp is None:
        return "N/A"
    try:
        dt_object = datetime.fromtimestamp(int(timestamp), timezone.utc)
        return dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')
    except (ValueError, TypeError):
        return str(timestamp)


# -------------
# Tool Function
# -------------
def format_single_vault_data_for_llm(data: Dict, chain_id_for_url: int) -> str:
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
                try: debt_ratio_percent = f"{float(debt_ratio_raw) / 100:.2f}%" # Assuming 10000 = 100%
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
    Optionally sort by 'highest_apr' or 'lowest_apr' to get the top results based on net APY.
    If no sort is specified, results are sorted by TVL (descending).
    Returns detailed information for the top matching vaults.
    """
    logging.info(f"[Tool:search_vaults] Query: '{query}', Chain: '{chain}', Sort By: '{sort_by}'")
    api_url = "https://ydaemon.yearn.fi/vaults/detected?limit=2000"
    MAX_RESULTS_TO_SHOW = 3

    async with aiohttp.ClientSession() as session:
        try:
            logging.info(f"[Tool:search_vaults] Fetching data from {api_url}")
            async with session.get(api_url, timeout=25) as response:
                response.raise_for_status()
                all_vaults_data_list = await response.json()
                if not isinstance(all_vaults_data_list, list):
                    logging.error(f"[Tool:search_vaults] Unexpected yDaemon response format: {type(all_vaults_data_list)}")
                    return "Error: Received unexpected data format from vault API."
                logging.info(f"[Tool:search_vaults] Retrieved {len(all_vaults_data_list)} full vault details from yDaemon.")
        except Exception as e:
            logging.error(f"[Tool:search_vaults] Error during yDaemon fetch: {e}", exc_info=True)
            return f"Error: An unexpected error occurred while fetching vault data: {e}."

        # --- Filtering ---
        filtered_vaults_full_data = all_vaults_data_list
        query_chain_id = None
        if chain:
            chain_lower = chain.lower()
            query_chain_id = CHAIN_NAME_TO_ID.get(chain_lower)
            if query_chain_id:
                filtered_vaults_full_data = [v for v in filtered_vaults_full_data if v.get("chainID") == query_chain_id]

        query_lower = query.lower()
        matched_vaults_full_data = []
        is_address_query = Web3.is_address(query_lower)

        for v_data in filtered_vaults_full_data:
            vault_address = v_data.get("address", "").lower()
            name = v_data.get("name", "").lower()
            symbol = v_data.get("symbol", "").lower()
            token_info = v_data.get("token", {})
            token_name = token_info.get("name", "").lower() if token_info else ""
            token_symbol = token_info.get("symbol", "").lower() if token_info else ""
            underlying_address = token_info.get("address", "").lower() if token_info else ""

            match = False
            if is_address_query:
                if query_lower == vault_address or query_lower == underlying_address: match = True
            elif query_lower == symbol or query_lower == token_symbol: match = True
            elif query_lower in name or query_lower in token_name: match = True
            
            if match:
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
                matched_vaults_full_data.append(v_data)


        logging.info(f"[Tool:search_vaults] Found {len(matched_vaults_full_data)} vaults matching query '{query}' after filter.")
        if not matched_vaults_full_data:
            return "No active Yearn vaults found matching your criteria."

        # --- Sorting ---
        if sort_by == "highest_apr":
            matched_vaults_full_data.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=True)
        elif sort_by == "lowest_apr":
            matched_vaults_full_data.sort(key=lambda v: v.get("_computedAPY", 0.0), reverse=False)
        else:
            matched_vaults_full_data.sort(key=lambda v: v.get("_computedTVL_USD", 0.0), reverse=True)

        top_vaults_to_format = matched_vaults_full_data[:MAX_RESULTS_TO_SHOW]
        logging.info(f"[Tool:search_vaults] Selected top {len(top_vaults_to_format)} vaults. Now formatting their details.")

        formatted_details_strings = []
        for vault_data_item in top_vaults_to_format:
            chain_id_for_url = vault_data_item.get("chainID")
            if chain_id_for_url is not None:
                formatted_text = format_single_vault_data_for_llm(vault_data_item, chain_id_for_url)
                formatted_details_strings.append(formatted_text)
            else:
                logging.warning(f"Vault {vault_data_item.get('address')} missing chainID, cannot format fully.")
                formatted_details_strings.append(f"Partial info for Vault: {vault_data_item.get('name', 'N/A')} (`{vault_data_item.get('address', 'N/A')}`) - Full details unavailable due to missing chain ID.")


        if not formatted_details_strings:
             return "Found matching vault(s), but could not format their details."

        # --- Assemble Final Output ---
        num_total_matches = len(matched_vaults_full_data)
        num_shown = len(formatted_details_strings)
        sort_description = sort_by if sort_by else "TVL (Descending)"
        header = f"Found {num_total_matches} Yearn vault(s) matching '{query}'."
        if num_total_matches > num_shown:
             header += f" Showing top {num_shown} (sorted by {sort_description}) with details:"
        final_result_text = header + "\n\n---\n\n" + "\n\n---\n\n".join(formatted_details_strings)
        logging.info(f"[Tool:search_vaults] Formatted Result (first 500 chars):\n{final_result_text[:500]}")
        return final_result_text

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

        chain_id = CHAIN_NAME_TO_ID.get(chain_name)
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
    logging.info(f"[Tool:check_all_deposits] Combined Result for {resolved_address}:\n{combined_result}")
    return combined_result


# --- Withdrawal Instruction Tool ---
@function_tool
async def get_withdrawal_instructions_tool(user_address_or_ens: Optional[str], vault_address: str, chain: str) -> str:
    """
    Generates step-by-step instructions for withdrawing from a specific Yearn vault (v1, v2, or v3) using a block explorer.
    Also provides the direct link to the vault on the Yearn website for reference (for v2/v3).
    Provide the vault's address, the chain name (e.g., 'ethereum', 'fantom', 'arbitrum'), and optionally the user's address/ENS.
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
    chain_id = CHAIN_NAME_TO_ID.get(chain_lower)
    explorer_base_url = BLOCK_EXPLORER_URLS.get(chain_lower)

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
    api_url = "https://ydaemon.yearn.fi/vaults/detected?limit=2000"
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

# --- Fetch balance for query_active_deposits_tool ---
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
            vault_contract = web3_instance.eth.contract(address=vault_checksum_addr, abi=ERC20_ABI)

            # --- Create all web3 call coroutines first ---
            wallet_balance_coro = asyncio.to_thread(
                vault_contract.functions.balanceOf(user_checksum_addr).call
            )

            gauge_balance_coro = None
            staking_info = vault_info.get("staking")
            if staking_info and staking_info.get("available") and Web3.is_address(staking_info.get("address")):
                gauge_addr_str = staking_info.get("address")
                gauge_checksum_addr = Web3.to_checksum_address(gauge_addr_str)
                gauge_contract = web3_instance.eth.contract(address=gauge_checksum_addr, abi=GAUGE_ABI)
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


@function_tool
async def answer_from_docs_tool(
    wrapper: RunContextWrapper[BotRunContext],
    user_query: str
) -> str:
    """
    Answers questions based on documentation using a two-stage process:
    1. Vector search (Pinecone) to find semantically similar documents.
    2. Reranking to find the most relevant documents for the specific query.
    """
    project_context = wrapper.context.project_context
    logging.info(f"[Tool:answer_from_docs] --- Tool Invoked ---")
    logging.info(f"[Tool:answer_from_docs] Received query: '{user_query}'")
    

    initial_retrieval_k = 20  # Fetch a wider net of initial candidates
    rerank_top_n = 8          # Rerank and return the top N most relevant results


    namespaces_to_query = ["yearn-docs", "yearn-yips"]

    # --- QUERY TRANSFORMATION (HyDE) ---
    # Use an LLM to generate a hypothetical, ideal answer. We will use the embedding of this
    # answer to search Pinecone, as it's likely to be more semantically rich than the raw query.
    try:
        hyde_prompt = (
            f"You are a Yearn documentation expert. A user has asked the following question: '{user_query}'.\n"
            "Please generate a concise, hypothetical answer to this question as if you had all the necessary documentation. "
            "Include placeholders for specific details if you don't know them. This answer will be used to find the real documents. "
            "For example, if the question is about a YIP, your hypothetical answer should mention the YIP number, its status, and refer to a link."
        )
        
        hyde_response = await openai_async_client.chat.completions.create(
            model="gpt-4.1-nano", # A fast model is good for this step
            messages=[{"role": "system", "content": hyde_prompt}],
            temperature=0.0,
        )
        hypothetical_answer = hyde_response.choices[0].message.content.strip()
        logging.info(f"[Tool:answer_from_docs] Generated hypothetical answer for embedding: '{hypothetical_answer}'")
        
        # The text we will embed is now a combination of the original query and the hypothetical answer
        # to get the best of both worlds: the user's specific keywords and the rich semantic context.
        embedding_text = f"{user_query}\n\n{hypothetical_answer}"

    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error during HyDE step: {e}. Falling back to original query.")
        embedding_text = user_query # Fallback to the old method if this step fails

    # 1. Get Query Embedding (Unchanged)
    try:
        response = await asyncio.to_thread(
            openai_sync_client.embeddings.create,
            model="text-embedding-3-large",
            input=[embedding_text],
            encoding_format="float"
        )
        query_embedding = response.data[0].embedding
        logging.info(f"[Tool:answer_from_docs] Successfully generated embedding for query.")
    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error generating query embedding: {e}")
        return "Sorry, I couldn't process your question to search the documentation."

    # 2. Search Pinecone for Initial Candidates
    all_matches = []
    try:
        # Create a list of tasks to run concurrently
        query_tasks = [
            asyncio.to_thread(
                pinecone_index.query,
                namespace=ns,
                vector=query_embedding,
                top_k=initial_retrieval_k,
                include_metadata=True
            ) for ns in namespaces_to_query
        ]
        
        # Run the queries in parallel
        search_results_list = await asyncio.gather(*query_tasks)
        
        # Combine the results
        for search_results in search_results_list:
            all_matches.extend(search_results.get("matches", []))
        
        logging.info(f"[Tool:answer_from_docs] Pinecone query returned a total of {len(all_matches)} initial candidates from {len(namespaces_to_query)} namespaces.")
    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error querying Pinecone: {e}")
        return f"Sorry, I encountered an error while searching the documentation."

    if not all_matches:
        logging.info("[Tool:answer_from_docs] No initial documents found in Pinecone.")
        return f"I couldn't find any information in the documentation to answer that question."

    # --- 2.5: Rerank for Relevance ---
    try:
        # Remove duplicate documents before reranking, just in case
        unique_matches = {match.id: match for match in all_matches}.values()
        docs_to_rerank = [match.get("metadata", {}).get("text", "") for match in unique_matches]
        
        logging.info(f"[Tool:answer_from_docs] Sending {len(docs_to_rerank)} unique candidates to the reranker...")
        
        rerank_response = await asyncio.to_thread(
            pc.inference.rerank,
            model="bge-reranker-v2-m3",
            query=user_query,
            documents=docs_to_rerank,
            top_n=rerank_top_n,
            return_documents=False
        )
        
        # Rebuild the final list from the unique matches based on the reranked order
        unique_matches_list = list(unique_matches)
        reranked_matches = [unique_matches_list[result.index] for result in rerank_response.data]
        logging.info(f"[Tool:answer_from_docs] Reranking complete. Using top {len(reranked_matches)} results.")

    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error during reranking: {e}. Falling back to original search results.")
        # Fallback: sort all matches by score and take the top N
        all_matches.sort(key=lambda x: x.score, reverse=True)
        reranked_matches = all_matches[:rerank_top_n]

    # 3. Build Context
    context_pieces = []
    if reranked_matches:
        for match in reranked_matches:
            metadata = match.get("metadata", {})
            text_chunk = metadata.get("text")
            doc_title = metadata.get("doc_title", "Unknown Document")
            section_heading = metadata.get("section_heading", "Unknown Section")
            source_path = metadata.get("source_path", "Unknown")

            if text_chunk:
                source_description = (
                    f"Source Document: {doc_title} ({source_path})\n"
                    f"Section: {section_heading}"
                )
                # Check for YIP status and add it if present
                yip_status = metadata.get("yip_status")
                if yip_status:
                    source_description += f"\nYIP Status: {yip_status}"
                
                context_pieces.append(f"{source_description}\nContent:\n{text_chunk}")
            else:
                 logging.warning(f"[Tool:answer_from_docs] Reranked match ID {match.get('id', 'N/A')} had empty 'text' in metadata.")

        context_text = "\n\n---\n\n".join(context_pieces)
        logging.info(f"[Tool:answer_from_docs] Built context string from reranked results (length: {len(context_text)}).")

    else:
        logging.info("[Tool:answer_from_docs] No relevant documents found in Pinecone.")
        logging.info(f"[Tool:answer_from_docs] --- Tool Returning Early (No Matches) ---")
        # Provide a more specific message based on the project context
        return f"I couldn't find any relevant information in the {project_context.capitalize()} documentation to answer that specific question."

    # Check if context is still empty after processing matches (e.g., all matches had empty text)
    if not context_text:
         logging.warning("[Tool:answer_from_docs] Context string is empty even though matches were found. All matched chunks might have had empty 'text' metadata.")
         logging.info(f"[Tool:answer_from_docs] --- Tool Returning Early (Empty Context Built) ---")
         return f"I found potential matches in the {project_context.capitalize()} documentation, but couldn't extract the content correctly."


    # 4. Generate Answer using LLM
    system_prompt = (
        f"You are an expert assistant for the Yearn project operating in the Yearn Discord server. Your knowledge base consists **SOLELY** of the provided 'Documentation Context'. "
        f"Answer the user's question directly and authoritatively using only this knowledge.\n"
        f"**RESPONSE REQUIREMENTS:**\n"
        f"1.  **LINKING IS MANDATORY:** Any mention of a specific resource, tool, website, or proposal (like a Yearn Improvement Proposal or YIP) that has a URL in the context **MUST** be a clickable Markdown link. There are no exceptions. Example: `[Powerglove](https://yearn-powerglove.vercel.app/)`.\n"
        f"2.  **YIP STATUS IS MANDATORY:** Any mention of a YIP **MUST** include its status (e.g., Proposed, Implemented, etc). Example: `The **Proposed** [YIP-54](URL) suggests...`\n"
        f"3.  **NO META-COMMENTARY:** Do NOT mention 'context', 'sources', 'information provided', 'sections', or the process of finding information. Avoid any phrases that refer to the origin or structure of your knowledge (e.g., 'According to...', 'Based on...', 'The section on...', 'The provided info shows...'). Speak directly from the knowledge you possess.\n"
        f"4.  **DO NOT MENTION THE SUPPORT BOT:** Your response **MUST NOT** mention 'ySupport Bot' or refer to yourself as a support bot. Focus only on answering the user's question.\n"
        f"5.  **STRICTLY CONTEXT-BASED:** Your answer **MUST** be derived solely from the 'Documentation Context'. If the answer is not found within the context, state that you don't have the information directly. For example: 'My knowledge base doesn't include details on X.' or 'I don't have information about X.' Do **NOT** use any external knowledge or make assumptions.\n\n"
        f"Review the user's question and the context, then provide your final answer, strictly adhering to all requirements above."
        f"Synthesize your answer clearly. If the context provides specific details, include them.\n"
    )

    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Answer the following question based *only* on current knowledge.\n\nDocumentation Context:\n\n{context_text}\n\nUser Question: {user_query}\n---"}
    ]
    logging.info(f"[Tool:answer_from_docs] Sending final prompt to LLM. User content length approx: {len(messages_for_llm[1]['content'])}")

    try:
        response = await openai_async_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages_for_llm,
            temperature=0.1
        )
        final_answer = response.choices[0].message.content.strip()
        logging.info(f"[Tool:answer_from_docs] Received final answer from LLM: '{final_answer}'")
        logging.info(f"[Tool:answer_from_docs] --- Tool Execution Complete ---")
        return final_answer
    except Exception as e:
        logging.error(f"[Tool:answer_from_docs] Error generating final answer: {e}")
        logging.info(f"[Tool:answer_from_docs] --- Tool Returning Error (LLM Call Failed) ---")
        return f"Sorry, I found relevant {project_context.capitalize()} documentation but encountered an error while formulating the final answer based on it."



# ----------------------------
# Agent Definitions
# ----------------------------

TContext = Any

yearn_data_agent = Agent[BotRunContext](
    name="Yearn Data Specialist",
    instructions=(
        "# Role and Objective\n"
        "You are a Yearn Data Specialist. Your primary goal is to use tools to fetch detailed Yearn vault information, check user deposits, or provide withdrawal instructions, and then answer user questions based *only* on the information provided by these tools.\n\n"

        "**A. INITIAL TASK DETERMINATION (PRIORITY):**\n"
        "First, check the `initial_button_intent` provided in the context. This intent comes from the user's initial button selection and dictates your immediate action for THIS turn:\n"
        "   - **IF `initial_button_intent` is 'data_deposit_check':**\n"
        "     - The user has indicated they want to check their deposits.\n"
        "     - Their current message IS their wallet address.\n"
        "     - **Your ONLY action is to call the `check_all_deposits_tool` using the user's entire current message content as the `user_address_or_ens` parameter for that tool.**\n"
        "     - Example: If user message is '0x123...', call `check_all_deposits_tool(user_address_or_ens='0x123...')`.\n"
        "     - Present ONLY the deposit information returned by the tool.\n"
        "     - Do NOT use `search_vaults_tool`. Do NOT ask to clarify the address type.\n"
        "   - **IF `initial_button_intent` is 'data_withdrawal_flow_start':**\n"
        "     - The user has indicated they need withdrawal help and their current message IS their wallet address.\n"
        "     - **Your FIRST action for THIS turn is to call `check_all_deposits_tool` using the user's entire current message content as the `user_address_or_ens` parameter.**\n"
        "     - After getting the deposit list: \n"
        "       - If deposits are found, present them clearly (Vault Name, Symbol, Address) and then ASK the user: 'Which of these vaults would you like withdrawal instructions for? Please provide the vault address or name/symbol from the list.'\n"
        "       - If no deposits are found, inform the user and ask if they have a specific vault address in mind they need help with, or if they want to try a different wallet address.\n"
        "     - Do NOT attempt to provide withdrawal instructions yet. Your goal in this step is to identify the target vault based on their deposits.\n"
        "   - **IF `initial_button_intent` is 'data_vault_search':**\n"
        "     - Your SOLE TASK for this turn is to use the `search_vaults_tool`.\n"
        "     - The user's current message is their search query (e.g., token name, vault address, 'all'). Use it with the tool.\n"
        "     - Present ONLY the vault search results.\n"
        "   - **If `initial_button_intent` is not present or not one of the above data-related intents (e.g., it's 'docs_qa' which is handled by another agent, or it's a follow-up message):** Proceed to section B (Free-Form Request Analysis / Follow-Up Actions) to determine the task from the user's current free-form message.\n\n"

        "**B. FREE-FORM REQUEST ANALYSIS / FOLLOW-UP ACTIONS:**\n"
        "If not guided by a specific `initial_button_intent` from section A for direct tool use in the current turn, OR if this is a follow-up message from the user:\n"
        "1.  **Follow-up to 'Withdrawal Flow Start':** If you previously listed deposits (because `initial_button_intent` was 'data_withdrawal_flow_start') and asked the user to specify a vault, their current message is likely that vault identifier (address, name, or symbol). You also have their wallet address from the previous turn (from conversation history). Use the vault identifier and the user's wallet address to call `get_withdrawal_instructions_tool`. Assume 'ethereum' chain if not specified or inferable.\n"
        "2.  **Direct Request for Withdrawal Instructions (Free-Form):** If the user's message explicitly asks for withdrawal instructions and provides BOTH a user wallet address AND a vault address (e.g., 'how to withdraw from vault 0xabc with wallet 0x123'), then use the `get_withdrawal_instructions_tool` with these details.\n"
        "3.  **Deposit Checks (Free-Form):** If the user's message clearly asks to check their deposits or balance for a given address (and it wasn't from the 'data_deposit_check' button flow), use the `check_all_deposits_tool` with that address.\n"
        "4.  **Vault Search/Info (Free-Form):** If the user's message asks to find vaults or get info on a specific vault (by name or address) (and it wasn't from the 'data_vault_search' button flow), use the `search_vaults_tool` with the vault name/address as the query.\n"
        "5.  **Address Resolution:** Ensure any ENS name is resolved before using tools requiring an address.\n"
        "6.  **Missing Info:** If a tool requires information the user hasn't provided (e.g., user needs to specify which vault after you listed their deposits, or they asked for withdrawal help but didn't give a vault address), ask clearly for the missing information.\n\n"

        "**C. INTERPRETING VAULT DATA (from `search_vaults_tool` output):**\n"
        "The `search_vaults_tool` will provide detailed information for each vault, structured as follows:\n"
        " - `Vault: [Name] ([Symbol])`\n"
        " - `Address: [0x...]`\n"
        " - `Yearn UI Link: [URL]`\n"
        " - `Version: [V2 or V3 (with API version)]` (e.g., V3 (API: 3.0.3))\n"
        " - `Kind: [e.g., Multi Strategy]`\n"
        " - `Description: [Text description]`\n"
        " - `Underlying Token: [Name] ([Symbol]) - [0xAddress] - Price: $[Price]`\n"
        " - `TVL & Share Price:`\n"
        "   - `TVL (USD): $[Amount]`\n"
        "   - `Vault Token Price Per Share (in underlying): [Value] (Raw: [RawValue])` (This means 1 vault token = X underlying tokens)\n"
        " - `APY Information:`\n"
        "   - `Current Net APY (compounded): [Percentage]% (Type: [e.g., v2:averaged])` (This is the main displayed APY)\n"
        "   - `Estimated Forward APY (projection): [Percentage]% (Type: [e.g., v3:onchainOracle])` (This is a future estimate)\n"
        "   - `Vault Fees: Performance=[X]%, Management=[Y]%`\n"
        "   - `Historical Net APY: Week Ago=[X]% Month Ago=[Y]% Inception=[Z]%` (Use these to understand the period for 'Current Net APY')\n"
        " - `Other Info:`\n"
        "   - `Featuring Score: [Value]` (Influences UI ranking)\n"
        "   - `Risk Level: [Number]`\n"
        "   - `Status Flags: Retired=[Bool], Boosted=[Bool], Highlighted=[Bool]`\n"
        "   - `Migration Available: [Bool]` (If True, a migration is available for this vault)\n"
        " - `Strategies ([Count]):` (Details for each strategy within the vault)\n"
        "   - `Name: [Strategy Name] ([0xAddress])`\n"
        "   - `Status: [e.g., active]`\n"
        "   - `Individual APY: [Percentage]%`\n"
        "   - `Allocation (Debt Ratio): [Percentage]%` (How much of the vault's assets are in this strategy)\n"
        "   - `Last Report: [Date Time UTC]`\n\n"
        " - `Staking Opportunity: [Yes/No]`\n"
        "   - `Source: [e.g., VeYFI]`\n"
        "   - `Staking Contract: [0xAddress]`\n"
        "   - `Rewards ([Count]):` (Details for each reward token if staking is active)\n"
        "     - `Token: [Name] ([Symbol]) [0xAddress]`\n"
        "     - `APY: [Percentage]%`\n"
        "     - `Status: [Ongoing/Finished] (Ends: [Date Time UTC])`\n\n"

        "**D. ANSWERING USER QUESTIONS (General Rules):**\n"
        "- **Strict Adherence to Task:** If an `initial_button_intent` (from section A) directed a specific task, focus *only* on fulfilling that task with the designated tool in your current turn. Do not perform unrelated actions like checking deposits if withdrawal instructions were requested via button, unless the tool itself fails and you need to ask for clarification related to that specific tool.\n"
        "- Answer **ONLY** based on the information provided by the tools. Do not add external knowledge or speculate.\n"
        "- If the user asks about a specific field (e.g., 'What are the fees for yvUSDC?'), locate that vault in the tool output and find the 'Vault Fees' section.\n"
        "- If the user asks a comparative question (e.g., 'Which USDC vault has higher APY?'), use the 'Current Net APY' from multiple vaults if available in the tool output.\n"
        "- If information is not present in the tool output for a specific query, state that (e.g., 'The details for that vault do not include X information.').\n"
        "- For APY, always clarify if you are referring to 'Current Net APY' or 'Estimated Forward APY'.\n"
        "- If a tool returns an error message, relay that error message to the user.\n"
        f"- If you cannot resolve the issue or answer the question with the tools (even after trying the appropriate tool based on button intent or free-form analysis), state that human help is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}' in your response.**\n\n"

        "**E. ANSWERING USER QUESTIONS ABOUT STRATEGIES (Specific to `search_vaults_tool` output):**\n"
        "- If the user asks for a list of strategies for a specific vault (e.g., 'what strategies are in yvUSDC?'): Identify the vault in the tool output. List the names of all strategies found under its 'Strategies' section.\n"
        "- If the user asks for details about a specific strategy by name within a vault: Find that strategy under the vault's 'Strategies' section and provide its listed details (APY, Allocation, Last Report, Status).\n"
        "- **If the user asks to compare strategies within a single vault (e.g., 'which strategy in yvUSDC has the highest APY?'):**\n"
        "  1. Locate the specified vault in the tool output.\n"
        "  2. Examine each strategy listed under its 'Strategies' section.\n"
        "  3. Note the 'Individual APY' for each strategy.\n"
        "  4. Identify the strategy with the highest 'Individual APY'.\n"
        "  5. Respond with the name of that strategy and its APY. For example: 'For yvUSDC, the strategy [Strategy Name] currently has the highest APY at [X.XX]%.'\n"
        "  6. If multiple strategies are listed but their APYs are not provided or are unclear in the tool output, state that you can list the strategies but cannot determine which has the highest APY from the available details.\n"
        "- If the tool output for a vault shows 'Strategies: None listed.', inform the user that no strategies are detailed for that vault in the provided information.\n\n"
    ),
    tools=[
        search_vaults_tool,
        check_all_deposits_tool,
        get_withdrawal_instructions_tool,
    ],
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.0)
)

yearn_docs_qa_agent = Agent[BotRunContext](
    name="Yearn Docs QA Specialist",
    instructions=(
        "# Role and Objective\n"
        "You answer questions based *only* on Yearn documentation and historical proposals using the 'answer_from_docs_tool'.\n"
        "# Workflow\n"
        "1. Confirm you understand the user's question is about Yearn.\n"
        "2. Use the `answer_from_docs_tool` with the user's exact query.\n"
        "3. Relay the answer or 'not found' message from the tool directly.\n"

        "# Rules\n"
        "- **Tool Exclusivity:** You MUST rely *exclusively* on the information returned by the `answer_from_docs_tool`. If the tool finds no answer, state that clearly ('I couldn't find information on that in the Yearn documentation.').\n"
        "- **Do NOT Supplement:** Do NOT add information from your own knowledge, guess, or speculate.\n"
        "- **Distinguish Fact from Proposal:** Pay close attention to the status of Yearn Improvement Proposals (YIPs). If information comes from a YIP, ensure you reflect its status (e.g., 'Implemented', 'Proposed') as instructed by the tool.\n"
        "- **Scope Limit:** Do NOT answer questions about real-time data (APR, TVL, balances), specific user account issues, or provide financial advice. State these are outside your scope.\n"

        f"# Escalation\n"
        "- If the question is complex even for the docs (e.g., requires interpretation beyond retrieval), seems like a bug report, or the tool fails, state that human help is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}' in your response.**\n"
    ),
    tools=[answer_from_docs_tool],
    model="gpt-4.1-mini",
    tool_use_behavior="stop_on_first_tool",
    model_settings=ModelSettings(temperature=0.2)
)

# --- Guardrail Agent for BD/PR Detection ---
bd_priority_guardrail_agent = Agent[BotRunContext](
    name="BD/PR/Listing Guardrail Check",
    instructions=(
        "Analyze the user's message to classify its primary intent. Your goal is to catch business, partnership, or job inquiries while letting legitimate support requests pass through.\n\n"
        "**CRITICAL CONTEXT:** You are analyzing a message within a support ticket. The user may be responding to a previous message from the support bot. Phrases like 'the first one', 'the second option', 'yes', 'no', or providing an address are very likely part of an ongoing support conversation and should almost always be classified as 'not_bd_pr'. Be very conservative in your classification.\n\n"
        "Use the following categories for the 'request_type' field:\n"
        "- 'listing': The user is clearly asking Yearn to list their token or provide liquidity for a listing.\n"
        "- 'partnership': The user is clearly proposing a technical integration, collaboration, or joint venture. The proposal should be explicit.\n"
        "- 'marketing': The user is clearly proposing a joint marketing campaign, AMA, or promotional activity.\n"
        "- 'job_inquiry': The user is clearly asking about working for Yearn, contributing, or applying for a grant.\n"
        "- 'other_bd': Other clear business development inquiries not covered above.\n"
        "- 'not_bd_pr': This is the default. Use this for **ALL** standard user support requests, questions about using Yearn, bug reports, follow-up replies, ambiguous messages, or anything that is not an EXPLICIT, INITIAL business proposal.\n\n"
        "**Decision Heuristics:**\n"
        "- If the message is a follow-up (e.g., 'the second one', 'yes, that one', 'let's do the first option'), it is **'not_bd_pr'**.\n"
        "- If the message is just an address, it is **'not_bd_pr'**.\n"
        "- If the message is a question about how to use the protocol, it is **'not_bd_pr'**.\n"
        "- Only classify as a business category if the message is a clear, unsolicited, initial proposal. If in doubt, classify as **'not_bd_pr'**."
    ),
    output_type=BDPriorityCheckOutput,
    model="gpt-4.1-nano",
    model_settings=ModelSettings(temperature=0.1)
)


# --- Guardrail  ---
LISTING_DENIAL_MESSAGE = (
    "Thank you for your interest! "
    "Yearn Finance ($YFI) is permissionlessly listable on exchanges. Yearn does not pay listing fees, nor does it provide liquidity for exchange listings. "
    "No proposal is necessary for listing.\n\n"
    "No follow up inquiries or responses necessary."
)

STANDARD_REDIRECT_MESSAGE = (
     f"Thank you for your interest! "
     f"For partnership, marketing, or other business development proposals, go to <#{PR_MARKETING_CHANNEL_ID}>, share your proposal in **5 sentences** describing how it benefits both parties, and tag **corn**.\n\n"
     f"No follow up inquiries or responses necessary."
)

JOB_INQUIRY_REDIRECT_MESSAGE = (
    "Thank you for your interest in contributing to or working with Yearn!\n\n"
    "Yearn operates with project-based grants. You can find full details about the process in the [Yearn Docs](https://docs.yearn.finance/contributing/operations/budget).\n"
    "You may also work on open issues, report bugs, suggest improvements, write documentation, and more by visiting our [GitHub repository](https://github.com/yearn), where anyone is welcome to contribute.\n\n"
    "No follow up inquiries or responses necessary."
)

@input_guardrail(name="BD/PR/Listing/Job Guardrail")
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
        run_config_guardrail = RunConfig(workflow_name="BD/Priority Guardrail Check", tracing_disabled=True)
        result = await guardrail_runner.run(
            starting_agent=bd_priority_guardrail_agent,
            input=text_input,
            run_config=run_config_guardrail
        )
        check_output = result.final_output_as(BDPriorityCheckOutput)

        logging.info(f"[Guardrail:BD/Priority] Check result: type={check_output.request_type}, Reasoning: {check_output.reasoning}")

        # --- Determine Action Based on Classification ---
        message_to_send = None
        should_trigger = False

        if check_output.request_type == "listing":
            message_to_send = LISTING_DENIAL_MESSAGE
            should_trigger = True
        elif check_output.request_type in ["partnership", "marketing", "other_bd"]:
            message_to_send = STANDARD_REDIRECT_MESSAGE
            should_trigger = True
        elif check_output.request_type == "job_inquiry":
            message_to_send = JOB_INQUIRY_REDIRECT_MESSAGE
            should_trigger = True

        # --- Return GuardrailFunctionOutput ---
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
        "# Role and Objective\n"
        "You are the primary Yearn & Bearn support agent. Your task is to determine the **project context** (Yearn or Bearn) and the **request type**, then take immediate action based *first* on the project context.\n\n"

        "**PRIORITY STEP: Process Initial Button Selection (from `initial_button_intent` in context):**\n"
        "   - **IF `initial_button_intent` is 'data_deposit_check':**\n"
        "     - The user wants to check their deposits. Their current message is expected to be their wallet address.\n"
        "     - **ASSUME the provided address IS the user's wallet address.**\n"
        "     - **DO NOT ask for clarification on whether it's a wallet or vault address.**\n"
        "     - Proceed to Step 1 (Determine Project Context) and then IMMEDIATELY to the relevant data specialist handoff (2c for Yearn, 3c for Bearn) for deposit checking using this address.\n"
        "   - IF `initial_button_intent` is 'data_withdrawal_flow_start':\n"
        "     - The user wants to start the withdrawal help process. Their current message should contain their wallet address.\n"
        "     - Proceed to Step 1 (Determine Project Context) and then IMMEDIATELY to the `transfer_to_yearn_data_specialist` handoff.\n"
        "   - IF `initial_button_intent` is 'data_withdrawal_flow_start': The user needs withdrawal help. Their current message should contain vault and/or user address. Proceed to Step 1, then 2c (Yearn) for withdrawal help. Clarify if addresses are missing or their type is unclear.\n"
        "   - IF `initial_button_intent` is 'docs_qa': The user has a general question. Their current message is the question. Proceed to Step 1, then 2f/3e for docs specialist handoff.\n"
        "   - IF `initial_button_intent` is 'other_free_form' or is not present/None: This is a free-form user message. Proceed to Step 1 (Determine Project Context) and continue normally with the free-form text analysis below.\n\n"

        "**Workflow:**\n"
        "   a. **BD/PR/Marketing/Listing:** (Handled by Guardrail - You won't see these).\n"
        "   b. **Initial Address Handling (FOR FREE-FORM INPUT or AMBIGUOUS BUTTON FOLLOW-UP ONLY):** If `initial_button_intent` was NOT 'data_deposit_check' AND the user provides an address (0x...) in their message without specifying its type: ASK them to clarify (wallet or vault) before proceeding.\n"
        "   c. **Data or Specific Withdrawal Request:** If the user's message (or a relevant button intent) is about finding vaults (e.g., 'find vaults for [address/token]'), checking deposits/balances (e.g., 'deposits for [wallet_address]'), or asking how to withdraw from a specific vault address, AND any required user wallet address is known/confirmed (or provided in the current message): **IMMEDIATELY use `transfer_to_yearn_data_specialist` handoff.** (For 'find vaults for [address]', assume the address is a token/vault unless specified otherwise by user).\n"
        "   d. **Address Needed:** If user wallet address is needed for (c) (e.g., for deposit checks or withdrawal from a specific vault where user address is also needed) but missing: Ask clearly for the user's wallet address/ENS. Do NOT hand off yet.\n"
        "   e. **Handling Address Refusal:** If you asked for a user address and they refuse, offer to provide general withdrawal instructions for the vault address if they provided one. If they agree, use the `transfer_to_yearn_data_specialist` handoff..\n"
        "   f. **General/Docs Question:** If the user's message (or button intent 'docs_qa' was identified) is a general question: **IMMEDIATELY use `transfer_to_yearn_docs_qa_specialist` handoff.**\n"
        "   g. **UI Errors/Bugs/Complex Issues:** If the query seems too complex for tools, describes a UI error, or a potential bug: Respond that human support is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}'.** Do NOT hand off.\n"
        "   h. **Ambiguity:** If request type (Data vs Docs vs Bug) is unclear (and no button intent guided this): Ask ONE clarifying question.\n"
        "   i. **Greetings/Chit-chat:** Respond briefly.\n\n"

        "# Rules\n"
        "**CRITICAL:** If an `initial_button_intent` is present in the context, use that as the primary guide for the request type. Then determine project context. Queries like 'find vaults for [address]', 'check balance for [address]', 'deposits for [address]' are considered **Data Requests**. Execute handoffs immediately when conditions are met. Do not describe the handoff.\n"
        "- You are a routing agent. Your goal is to classify the request and either respond simply, ask for clarification, or hand off to a specialist.\n"
        "- Complete your analysis and required action (response, question, or handoff) based on the workflow before concluding your turn. Do not leave the task unfinished.\n"
        "- Only use the specifically defined handoff tools (`transfer_to_...`). Do not attempt to answer questions that require specialist tools yourself.\n"
        "- Do not ask follow-up questions unless explicitly allowed by the workflow (e.g., address clarification, ambiguity resolution).\n"
        "- Do not provide any information about the handoff process or the specialist agents. Your role is to route the request, not to explain the routing.\n"
    ),
    handoffs=[
        handoff(yearn_data_agent, tool_name_override="transfer_to_yearn_data_specialist", tool_description_override="Handoff for specific YEARN data (vaults, deposits, APR, TVL, balances, withdrawal instructions)."),
        handoff(yearn_docs_qa_agent, tool_name_override="transfer_to_yearn_docs_qa_specialist", tool_description_override="Handoff for general questions about YEARN concepts, documentation, risks."),
    ],
    input_guardrails=[bd_priority_guardrail],
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.1)
)





# -----------
# Discord Bot 
# -----------

conversation_threads: Dict[int, List[TResponseInputItem]] = {}
stopped_channels: set[int] = set()
pending_messages: Dict[int, str] = {}
pending_tasks: Dict[int, asyncio.Task] = {}
monitored_new_channels: set[int] = set()

class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self.runner = Runner

    async def on_ready(self):
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logging.info(f"Monitoring Yearn Ticket Category ID: {YEARN_TICKET_CATEGORY_ID}")
        logging.info(f"Support User ID for triggers: {PUBLIC_TRIGGER_USER_IDS}")
        logging.info(f"Yearn Public Trigger: '{YEARN_PUBLIC_TRIGGER_CHAR}'")
        print("------")

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        # Initialize state for new channels in monitored ticket categories
        if isinstance(channel, discord.TextChannel) and channel.category:
            if channel.category.id in CATEGORY_CONTEXT_MAP:
                project_context = CATEGORY_CONTEXT_MAP.get(channel.category.id, "unknown")
                logging.info(f"New {project_context.capitalize()} ticket channel created: {channel.name} (ID: {channel.id}). Initializing state.")
                conversation_threads[channel.id] = []
                stopped_channels.discard(channel.id)
                pending_messages.pop(channel.id, None)
                if channel.id in pending_tasks:
                    try: pending_tasks.pop(channel.id).cancel()
                    except Exception: pass
                monitored_new_channels.add(channel.id)
                logging.info(f"Added channel {channel.id} to monitored_new_channels set.")

                # This gives Ticket Tool bot time to send its message.
                delay_seconds = 1.5
                logging.info(f"Delaying welcome message in {channel.id} by {delay_seconds} seconds.")
                await asyncio.sleep(delay_seconds)

                # --- SEND INITIAL MESSAGE WITH BUTTONS ---
                welcome_message = (
                    f"Welcome to {project_context.capitalize()} Support!\n\n"
                    "To help you more efficiently, please select a category below.\n"
                    "*Please press a button to get started. You can share more details after making a selection.*"
                )
                try:
                    await channel.send(welcome_message, view=InitialInquiryView())
                    channels_awaiting_initial_button_press.add(channel.id)
                    logging.info(f"Sent initial inquiry buttons to channel {channel.id}")
                except discord.Forbidden:
                    logging.error(f"Missing permissions to send initial message with buttons in {channel.id}")
                except Exception as e:
                    logging.error(f"Error sending initial message with buttons in {channel.id}: {e}", exc_info=True)

    async def process_synthetic_button_input(self, channel: discord.TextChannel, synthetic_text: str, intent_category: str):
        """
        Processes a synthetic input generated from an initial button press.
        This will queue up a task similar to process_ticket_message but with predefined input.
        """
        channel_id = channel.id
        logging.info(f"Processing synthetic button input for ticket {channel_id}. Intent: '{intent_category}', Text: '{synthetic_text}'")

        category_id = channel.category.id if channel.category else None
        project_ctx = CATEGORY_CONTEXT_MAP.get(category_id, "unknown")
        run_context = BotRunContext(
            channel_id=channel_id,
            category_id=category_id,
            project_context=project_ctx,
            initial_button_intent=intent_category
        )

        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": synthetic_text}]
        conversation_threads[channel_id] = input_list

        if channel_id in pending_tasks:
            pending_tasks[channel_id].cancel()

        await self.process_ticket_message(channel_id, run_context, is_button_trigger=True, synthetic_user_message_for_log=synthetic_text)

    async def on_message(self, message: discord.Message):
        # Ignore bots and self
        if message.author.bot or message.author.id == self.user.id:
            return

        # STATEFUL PUBLIC TRIGGER
        is_reply = message.reference is not None
        trigger_char_used = message.content.strip()
        is_valid_trigger_char = trigger_char_used in TRIGGER_CONTEXT_MAP
        is_trigger_user = str(message.author.id) in PUBLIC_TRIGGER_USER_IDS

        if is_reply and is_trigger_user and is_valid_trigger_char:
            logging.info(f"Stateful public trigger '{trigger_char_used}' detected by {message.author.name} in channel {message.channel.id}")
            
            try:
                await message.delete()
            except Exception as e:
                logging.warning(f"Failed to delete trigger message {message.id}: {e}")

            try:
                original_message = await message.channel.fetch_message(message.reference.message_id)
                if original_message and not original_message.author.bot and original_message.content:
                    
                    # Context Management
                    original_author_id = original_message.author.id
                    current_history: List[TResponseInputItem] = []
                    
                    conversation = public_conversations.get(original_author_id)
                    if conversation:
                        time_since_last = datetime.now(timezone.utc) - conversation.last_interaction_time
                        if time_since_last <= timedelta(minutes=PUBLIC_TRIGGER_TIMEOUT_MINUTES):
                            logging.info(f"Continuing public conversation for user {original_author_id} (last active {time_since_last.total_seconds():.1f}s ago).")
                            current_history = conversation.history
                        else:
                            logging.info(f"Public conversation for user {original_author_id} expired ({time_since_last.total_seconds():.1f}s ago). Starting new context.")
                            public_conversations.pop(original_author_id, None) # Clean up expired entry
                    
                    input_list = current_history + [{"role": "user", "content": original_message.content}]

                    # Create a specific context for this public run
                    public_run_context = BotRunContext(
                        channel_id=message.channel.id,
                        is_public_trigger=True,
                        project_context=TRIGGER_CONTEXT_MAP.get(trigger_char_used, "unknown")
                    )

                    async with message.channel.typing():
                        try:
                            run_config = RunConfig(
                                workflow_name=f"Public Stateful Trigger-{message.channel.id}",
                                group_id=str(original_author_id) # Group traces by the user being helped
                            )
                            result: RunResult = await self.runner.run(
                                starting_agent=triage_agent, # Your main triage agent
                                input=input_list,           # Use the stateful input_list
                                max_turns=5,                # Keep a turn limit for safety
                                run_config=run_config,
                                context=public_run_context
                            )

                            # Save Updated Context
                            new_history = result.to_input_list()
                            new_conversation = PublicConversation(
                                history=new_history,
                                last_interaction_time=datetime.now(timezone.utc)
                            )
                            public_conversations[original_author_id] = new_conversation
                            logging.info(f"Saved updated public conversation context for user {original_author_id}. History length: {len(new_history)} items.")


                            raw_reply = result.final_output if result.final_output else "I could not determine a response."
                            actual_mention = f"<@{HUMAN_HANDOFF_TARGET_USER_ID}>"
                            final_reply = raw_reply.replace(HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)
                            
                            await send_long_message(original_message, final_reply)
                        
                        except Exception as e:
                            logging.error(f"Error during public trigger agent run for user {original_author_id}: {e}", exc_info=True)
                            await original_message.reply(f"Sorry, an error occurred while processing that request. Please notify <@{HUMAN_HANDOFF_TARGET_USER_ID}>.", mention_author=False)

                    return # IMPORTANT: Stop further processing after handling the trigger

            except discord.NotFound:
                logging.warning(f"Original message for public trigger reply {message.id} not found.")
                return # Can't proceed if original message is gone
            except discord.Forbidden:
                 logging.warning(f"Missing permissions to fetch original message for public trigger reply {message.id}.")
                 return # Can't proceed
            except Exception as e:
                 logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)
                 return

        # --- Ticket Channel ---
        # Check if the message is in a monitored ticket channel
        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category: return
        channel_id = message.channel.id
        if channel_id not in monitored_new_channels: return

        if channel_id in channels_awaiting_initial_button_press:
            # This means they typed before clicking a button on the initial message
            try:
                await message.reply("Please select an option from the buttons on my previous message to get started.", delete_after=20, mention_author=False)
            except Exception: pass
            return

        # This intent was set when the user clicked a button that prompts for more info.
        # It will be None if the user clicked "Other" or if this is a later message in the convo.
        current_intent_from_map = channel_intent_after_button.pop(channel_id, None)
        if current_intent_from_map:
            logging.info(f"Message in {channel_id} is a follow-up to button intent: {current_intent_from_map}")

        # If we are here, either a button was pressed (and intent is stored),
        # or it's a follow-up to a bot's prompt after a button press.
        ticket_run_context = BotRunContext(
            channel_id=channel_id,
            category_id=message.channel.category.id,
            project_context=CATEGORY_CONTEXT_MAP.get(message.channel.category.id, "unknown"),
            initial_button_intent=current_intent_from_map
        )

        if ticket_run_context.project_context == "unknown": return
        if channel_id in stopped_channels: return

        logging.info(f"Processing ticket message in {channel_id} from {message.author.name} (Context: {ticket_run_context.project_context}, Intent: {current_intent_from_map})")

        if channel_id not in pending_messages:
            pending_messages[channel_id] = message.content
        else:
            pending_messages[channel_id] += "\n" + message.content

        if channel_id in pending_tasks:
            pending_tasks[channel_id].cancel()
        
        # Pass the context to process_ticket_message
        pending_tasks[channel_id] = asyncio.create_task(self.process_ticket_message(channel_id, ticket_run_context))
        logging.debug(f"Scheduled processing task for channel {channel_id} in {COOLDOWN_SECONDS}s")

    async def process_ticket_message(self, channel_id: int, run_context: BotRunContext):
        # --- Debouncing and message retrieval ---
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

        # Check if an acknowledgement is needed before starting the heavy processing.
        # The intent is a strong signal that a data-heavy operation is next.
        intent = run_context.initial_button_intent
        is_data_intent = intent in ['data_deposit_check', 'data_withdrawal_flow_start', 'data_vault_search']
        
        # Send an acknowledgement if the user's message was an address
        if is_data_intent and is_message_primarily_address(aggregated_text):
            # Extract address
            parsed_address = resolve_ens(aggregated_text)
            if parsed_address:
                ack_message = f"Thank you. I've received the address `{parsed_address}` and am looking up the information now. This may take a moment..."
                try:
                    await channel.send(ack_message)
                    logging.info(f"Sent pre-run acknowledgement message to channel {channel_id}")
                except Exception as e:
                    logging.warning(f"Failed to send pre-run acknowledgement message: {e}")


        # --- Agent ---
        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": aggregated_text}]
        
        logging.info(f"Processing for ticket {channel_id} (Context: {run_context.project_context}, Initial Button Intent: {run_context.initial_button_intent}): '{aggregated_text[:100]}...'")

        async with channel.typing():
            final_reply = "An unexpected error occurred."
            should_stop_processing = False

            try:
                run_config = RunConfig(
                    workflow_name=f"Ticket Channel {channel_id} ({run_context.project_context}, Button Intent: {run_context.initial_button_intent})", # Use for logging
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
                actual_mention = f"<@{HUMAN_HANDOFF_TARGET_USER_ID}>"
                final_reply = raw_final_reply.replace(HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                if actual_mention in final_reply or HUMAN_HANDOFF_TAG_PLACEHOLDER in raw_final_reply:
                    logging.info(f"Human handoff tag detected in response for channel {channel_id}.")
                    should_stop_processing = True
            except InputGuardrailTripwireTriggered as e:
                 logging.warning(f"Input Guardrail triggered in channel {channel_id}. Extracting message from output_info.")
                 guardrail_info = e.guardrail_result.output.output_info
                 if isinstance(guardrail_info, dict) and "message" in guardrail_info:
                     final_reply = guardrail_info["message"]
                 else:
                     final_reply = "Your request could not be processed due to input checks."
                 should_stop_processing = True
                 conversation_threads.pop(channel_id, None)
            except MaxTurnsExceeded:
                 logging.warning(f"Max turns ({MAX_TICKET_CONVERSATION_TURNS}) exceeded in channel {channel_id}.")
                 final_reply = f"This conversation has reached its maximum length. <@{HUMAN_HANDOFF_TARGET_USER_ID}> may need to intervene."
                 should_stop_processing = True
                 conversation_threads.pop(channel_id, None)
            except AgentsException as e:
                 logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                 final_reply = f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again or notify <@{HUMAN_HANDOFF_TARGET_USER_ID}>."
                 should_stop_processing = True
            except Exception as e:
                 logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                 final_reply = f"An unexpected error occurred. Please notify <@{HUMAN_HANDOFF_TARGET_USER_ID}>."
                 should_stop_processing = True

            try:
                reply_view = StopBotView() if not should_stop_processing else None
                await send_long_message(channel, final_reply, view=reply_view)
                logging.info(f"Sent ticket reply/replies in channel {channel_id}. Stop processing flag: {should_stop_processing}")
                if should_stop_processing and channel_id not in stopped_channels:
                    stopped_channels.add(channel_id)
                    logging.info(f"Added channel {channel_id} to stopped channels due to error/handoff tag.")
            except discord.Forbidden:
                 logging.error(f"Missing permissions to send message in channel {channel_id}")
                 stopped_channels.add(channel_id)
            except Exception as e:
                 logging.error(f"Unexpected error occurred during or after calling send_long_message for channel {channel_id}: {e}", exc_info=True)



# ----------------------------
# Run the Bot
# ----------------------------
if __name__ == "__main__":
    if not OPENAI_API_KEY or "YOUR_OPENAI_API_KEY" in OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set or is using the placeholder value.")
        sys.exit(1)
    if not DISCORD_BOT_TOKEN or "YOUR_BOT_TOKEN" in DISCORD_BOT_TOKEN:
        print("Error: DISCORD_BOT_TOKEN is not set or is using the placeholder value.")
        sys.exit(1)
    if not PINECONE_API_KEY or "YOUR_PINECONE_API_KEY" in PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY is not set or is using the placeholder value.")
        sys.exit(1)
    if not ALCHEMY_KEY or "YOUR_ALCHEMY_KEY" in ALCHEMY_KEY:
        print("Warning: ALCHEMY_KEY is not set or is using the placeholder value. Web3 features may fail.")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = TicketBot(intents=intents)
    client.run(DISCORD_BOT_TOKEN)
