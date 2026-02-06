import asyncio
import aiohttp
import traceback
import discord
from discord.ui import View, Button, button
import sys
import re
import json
import logging
import os
from web3 import Web3
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta

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

import config
import tools_lib

from dotenv import load_dotenv
load_dotenv()

set_default_openai_key(config.OPENAI_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

# Globals
stopped_channels: set[int] = set()
conversation_threads: Dict[int, List[TResponseInputItem]] = {}
pending_messages: Dict[int, str] = {}
pending_tasks: Dict[int, asyncio.Task] = {}
monitored_new_channels: set[int] = set()

# State & Context Definitions
@dataclass
class BotRunContext:
    channel_id: int
    category_id: Optional[int] = None
    is_public_trigger: bool = False
    project_context: Literal["yearn", "unknown"] = "unknown"
    initial_button_intent: Optional[str] = None

channels_awaiting_initial_button_press: set[int] = set()
channel_intent_after_button: Dict[int, str] = {}

# State Management for Public Conversations
@dataclass
class PublicConversation:
    """Stores the state for a temporary public conversation."""
    history: List[TResponseInputItem]
    last_interaction_time: datetime

public_conversations: Dict[int, PublicConversation] = {}

# Views & UI

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
        if interaction.channel.category and interaction.channel.category.id in config.CATEGORY_CONTEXT_MAP:
            project_ctx_name = config.CATEGORY_CONTEXT_MAP[interaction.channel.category.id].capitalize()
        prompt = f"Great! What specific information or product are you looking for, or what how-to question do you have about {project_ctx_name}?"
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "docs_qa")


    @button(label="🐞 Bug Report/UI Issue", style=discord.ButtonStyle.secondary, custom_id="initial_bug_report", row=2)
    async def bug_report_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id) # Generic handler to disable buttons
        actual_mention = f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}>" # Use the actual ID
        response_message = (
            "Thank you for reporting this. To help us investigate, please describe the bug or UI problem in detail. "
            "Include steps to reproduce it if possible, and mention which device/browser you are using. "
            f"{actual_mention} will review your report." # Use the formatted mention
        )
        if interaction.channel: await interaction.channel.send(response_message)
        else: await interaction.followup.send(response_message, ephemeral=False) # Fallback
        stopped_channels.add(interaction.channel.id)
        logging.info(f"Bug report initiated in {interaction.channel.id}. Bot stopped.")

    @button(label="🤝 Business/Partnerships/Marketing", style=discord.ButtonStyle.secondary, custom_id="initial_bd_partner", row=2)
    async def bd_partner_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        if interaction.channel: await interaction.channel.send(STANDARD_REDIRECT_MESSAGE)
        else: await interaction.followup.send(STANDARD_REDIRECT_MESSAGE, ephemeral=False) # Fallback
        stopped_channels.add(interaction.channel.id)
        logging.info(f"BD/Partner inquiry redirected in {interaction.channel.id}. Bot stopped.")

    @button(label="🛠️ Contribute/Work/Grants", style=discord.ButtonStyle.secondary, custom_id="initial_contribute_work", row=3)
    async def contribute_work_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        if interaction.channel: await interaction.channel.send(JOB_INQUIRY_REDIRECT_MESSAGE)
        else: await interaction.followup.send(JOB_INQUIRY_REDIRECT_MESSAGE, ephemeral=False) # Fallback
        stopped_channels.add(interaction.channel.id)
        logging.info(f"Contribute/Work inquiry redirected in {interaction.channel.id}. Bot stopped.")

    @button(label="❓ Other/My Issue Isn't Listed", style=discord.ButtonStyle.secondary, custom_id="initial_other_issue", row=3)
    async def other_issue_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        prompt = "Okay, please describe your issue or question in detail below. I'll do my best to assist or find the right help for you."
        if interaction.channel: await interaction.channel.send(prompt)
        else: await interaction.followup.send(prompt, ephemeral=False) # Fallback
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


class BDPriorityCheckOutput(BaseModel): # Renamed for clarity
    request_type: Literal["listing", "partnership", "marketing", "other_bd", "job_inquiry", "not_bd_pr"] = Field(..., description="Classify the user's primary intent: 'listing' (requesting Yearn list their token), 'partnership' (proposing integration/collaboration), 'marketing' (joint marketing/promotion), 'other_bd' (other business development), 'job_inquiry' (asking to work for/contribute to Yearn, grant requests), or 'not_bd_pr' (standard support request or unrelated).")
    reasoning: str = Field(..., description="Brief explanation for the classification.")

class GuardrailResponseMessageException(AgentsException):
    def __init__(self, message: str, guardrail_output: Optional[BDPriorityCheckOutput] = None):
        super().__init__(message)
        self.message = message
        self.guardrail_output = guardrail_output # Store the agent's reasoning if needed

class StopBotView(View):
    def __init__(self, *, timeout=None): # Set timeout=None so it doesn't expire
        super().__init__(timeout=timeout)

    @button(label="Stop Bot", style=discord.ButtonStyle.secondary, custom_id="stop_bot_button")
    async def stop_button_callback(self, interaction: discord.Interaction, button: Button):
        channel_id = interaction.channel.id
        user_who_clicked = interaction.user

        logging.info(f"Stop Bot button clicked in channel {channel_id} by {user_who_clicked.name} ({user_who_clicked.id})")

        await interaction.response.defer()

        stopped_channels.add(channel_id)
        conversation_threads.pop(channel_id, None) # Clear history
        pending_messages.pop(channel_id, None)     # Clear pending text
        if channel_id in pending_tasks:
            try:
                pending_tasks.pop(channel_id).cancel() # Cancel & remove pending task
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
            await interaction.edit_original_response(view=disabled_view) # Edit the original message with the disabled button view
            logging.info(f"Disabled Stop Bot button on message {interaction.message.id}")
        except discord.HTTPException as e:
            logging.warning(f"Could not disable Stop Bot button on message {interaction.message.id}: {e.status} {e.text}")
        except Exception as e:
            logging.error(f"Unexpected error disabling Stop Bot button on message {interaction.message.id}: {e}", exc_info=True)


        confirmation_message = f"Support bot stopped for this channel. ySupport contributors are available for further inquiries."
        try:
            await interaction.followup.send(confirmation_message, ephemeral=False) # Send as a new message in the channel
            logging.info(f"Sent stop confirmation to channel {channel_id}")
        except discord.HTTPException as e:
            logging.error(f"Failed to send stop confirmation followup in {channel_id}: {e.status} {e.text}")
        except Exception as e:
            logging.error(f"Unexpected error sending stop confirmation followup in {channel_id}: {e}", exc_info=True)



# Helper Functions
async def send_long_message(
    target: Union[discord.TextChannel, discord.Message],
    text: str,
    view: Optional[View] = None
):
    """Sends a potentially long message, splitting it into chunks if necessary."""
    if not text:
        return

    chunks = []
    if len(text) <= config.MAX_DISCORD_MESSAGE_LENGTH:
        chunks.append(text)
    else:
        current_chunk = ""
        lines = text.splitlines(keepends=True)
        for line in lines:
            if len(line) > config.MAX_DISCORD_MESSAGE_LENGTH:
                 for i in range(0, len(line), config.MAX_DISCORD_MESSAGE_LENGTH):
                     part = line[i:i + config.MAX_DISCORD_MESSAGE_LENGTH]
                     if current_chunk: chunks.append(current_chunk)
                     current_chunk = ""
                     chunks.append(part)
                 continue

            if len(current_chunk) + len(line) > config.MAX_DISCORD_MESSAGE_LENGTH:
                if current_chunk: chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += line
        if current_chunk: chunks.append(current_chunk)

        if len(chunks) == 1 and len(chunks[0]) > config.MAX_DISCORD_MESSAGE_LENGTH:
             logging.warning("Message splitting resulted in a single chunk still exceeding limit. Truncating.")
             chunks = [chunks[0][:config.MAX_DISCORD_MESSAGE_LENGTH - 3] + "..."]
        elif not chunks:
             logging.warning("Message splitting resulted in zero chunks for long message.")
             chunks.append(text[:config.MAX_DISCORD_MESSAGE_LENGTH - 3] + "...")

    first_message = True
    sent_view = False # Track if the view has been sent
    try:
        for i, chunk in enumerate(chunks):
            if not chunk.strip(): continue

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
                sent_view = True # Mark view as sent

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

# Tool Function Implementations
@function_tool
async def search_vaults_tool(
    query: str,
    chain: Optional[str] = None,
    sort_by: Optional[str] = None # "highest_apr" or "lowest_apr"
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

# Agent Definitions
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
        f"- If you cannot resolve the issue or answer the question with the tools (even after trying the appropriate tool based on button intent or free-form analysis), state that human help is needed and **include the tag '{config.HUMAN_HANDOFF_TAG_PLACEHOLDER}' in your response.**\n\n"

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
    model="gpt-4.1", # Use a capable model for data interpretation and tool use
    model_settings=ModelSettings(temperature=0.0) # Lower temp for factual data tasks
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
    model="gpt-4.1-mini", # Can use a smaller model as the tool does heavy lifting
    tool_use_behavior="stop_on_first_tool",
    model_settings=ModelSettings(temperature=0.2)
)

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


LISTING_DENIAL_MESSAGE = (
    "Thank you for your interest! "
    "Yearn Finance ($YFI) is permissionlessly listable on exchanges. Yearn does not pay listing fees, nor does it provide liquidity for exchange listings. "
    "No proposal is necessary for listing.\n\n"
    "No follow up inquiries or responses necessary."
)

STANDARD_REDIRECT_MESSAGE = (
     f"Thank you for your interest! "
     f"For partnership, marketing, or other business development proposals, go to <#{config.PR_MARKETING_CHANNEL_ID}>, share your proposal in **5 sentences** describing how it benefits both parties, and tag **corn**.\n\n"
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
        "You are the primary Yearn support agent. Your task is to analyze the user's request and take the appropriate action: handoff to a specialist, ask for clarification, or respond directly.\n\n"

        "**Step 1: Check for Initial Button Intent (PRIORITY)**\n"
        "First, check the `initial_button_intent` provided in the context. This is your primary guide.\n"
        "   - **IF `initial_button_intent` is 'data_deposit_check':** The user wants to check their deposits. Their current message is their wallet address. **ASSUME it is their wallet address** and **IMMEDIATELY use the `transfer_to_yearn_data_specialist` handoff.** Do NOT ask for address type clarification.\n"
        "   - **IF `initial_button_intent` is 'data_withdrawal_flow_start':** The user wants to start the withdrawal help process. Their current message is their wallet address. **IMMEDIATELY use the `transfer_to_yearn_data_specialist` handoff.**\n"
        "   - **IF `initial_button_intent` is 'data_vault_search':** The user wants to find vaults. Their current message is the search query. **IMMEDIATELY use the `transfer_to_yearn_data_specialist` handoff.**\n"
        "   - **IF `initial_button_intent` is 'docs_qa':** The user has a general question. Their current message is the question. **IMMEDIATELY use the `transfer_to_yearn_docs_qa_specialist` handoff.**\n"
        "   - **IF `initial_button_intent` is 'other_free_form' or is not present:** This is a free-form user message. Proceed to Step 2.\n\n"

        "**Step 2: Free-Form Request Workflow (if no button intent from Step 1)**\n"
        "   a. **BD/PR/Marketing/Listing/Contribute:** (These are handled by Guardrails or Buttons and should not reach you).\n"
        "   b. **Initial Address Handling:** If a user provides an address (0x...) in their first message without specifying its type: ASK them to clarify if it's their wallet or a vault address before proceeding.\n"
        "   c. **Data or Specific Withdrawal Request:** If the request is about finding vaults, checking deposits/balances, or asking how to withdraw from a specific vault, AND any required addresses are known/confirmed: **IMMEDIATELY use `transfer_to_yearn_data_specialist` handoff.**\n"
        "   d. **Address Needed:** If a user address is needed for a task in (c) but is missing, ask clearly for it. Do NOT hand off yet.\n"
        "   e. **Handling Address Refusal:** If you asked for a user address and they refuse, offer to provide general withdrawal instructions for the vault address if they provided one. If they agree, use the `transfer_to_yearn_data_specialist` handoff.\n"
        "   f. **General/Docs Question:** If the request is a general 'how-to' or conceptual question: **IMMEDIATELY use the `transfer_to_yearn_docs_qa_specialist` handoff.**\n"
        "   g. **UI Errors/Bugs/Complex Issues:** If the issue seems like a bug, a complex problem beyond data retrieval, or a UI error: Respond that human support is needed and **include the tag '{HUMAN_HANDOFF_TAG_PLACEHOLDER}'.** Do NOT hand off.\n"
        "   h. **Ambiguity:** If the request type (Data vs Docs vs Bug) is unclear: Ask ONE clarifying question.\n"
        "   i. **Greetings/Chit-chat:** Respond briefly and politely.\n\n"

        "# Rules\n"
        "**CRITICAL:** Prioritize the `initial_button_intent` from Step 1. If it directs a handoff, execute it immediately. For free-form messages, follow Step 2. Execute handoffs immediately when conditions are met. Do not describe the handoff process.\n"
        "- You are a routing agent. Your goal is to classify the request and either respond simply, ask for clarification, or hand off to a specialist.\n"
        "- Only use the specifically defined handoff tools (`transfer_to_yearn_data_specialist`, `transfer_to_yearn_docs_qa_specialist`). Do not attempt to answer questions that require specialist tools yourself.\n"
        "- Do not ask follow-up questions unless explicitly allowed by the workflow (e.g., address clarification, ambiguity resolution).\n"
        "- Do not provide any information about the handoff process or the specialist agents. Your role is to route the request, not to explain the routing.\n"
    ),
    handoffs=[
        handoff(yearn_data_agent, tool_name_override="transfer_to_yearn_data_specialist", tool_description_override="Handoff for specific YEARN data (vaults, deposits, APR, TVL, balances, withdrawal instructions)."),
        handoff(yearn_docs_qa_agent, tool_name_override="transfer_to_yearn_docs_qa_specialist", tool_description_override="Handoff for general questions about YEARN concepts, documentation, risks."),
    ],
    input_guardrails=[bd_priority_guardrail],
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.1) # Keep temp very low
)

# Discord Bot Implementation
class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self.runner = Runner # Store runner for easy access

    async def on_ready(self):
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logging.info(f"Monitoring Yearn Ticket Category ID: {config.YEARN_TICKET_CATEGORY_ID}")
        logging.info(f"Support User ID for triggers: {config.PUBLIC_TRIGGER_USER_IDS}")
        logging.info(f"Yearn Public Trigger: '{config.YEARN_PUBLIC_TRIGGER_CHAR}'")
        print("------")

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        if isinstance(channel, discord.TextChannel) and channel.category:
            if channel.category.id in config.CATEGORY_CONTEXT_MAP:
                project_context = config.CATEGORY_CONTEXT_MAP.get(channel.category.id, "unknown")
                logging.info(f"New {project_context.capitalize()} ticket channel created: {channel.name} (ID: {channel.id}). Initializing state.")
                conversation_threads[channel.id] = []
                stopped_channels.discard(channel.id)
                pending_messages.pop(channel.id, None)
                if channel.id in pending_tasks:
                    try: pending_tasks.pop(channel.id).cancel()
                    except Exception: pass
                monitored_new_channels.add(channel.id)
                logging.info(f"Added channel {channel.id} to monitored_new_channels set.")

                delay_seconds = 1.2
                logging.info(f"Delaying welcome message in {channel.id} by {delay_seconds} seconds.")
                await asyncio.sleep(delay_seconds)

                welcome_message = (
                    f"Welcome to {project_context.capitalize()} Support!\n\n\n"
                    "Press a category button below to get started.\n"
                    "You can share more details after making a selection.\n\n\n"
                    "To process your request accurately, please wait for my response after you see the *'ySupport is typing...'* indicator before sending another message.\n\n"
                    "---"
                )
                try:
                    await channel.send(welcome_message, view=InitialInquiryView())
                    channels_awaiting_initial_button_press.add(channel.id) # Mark as awaiting button
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
        project_ctx = config.CATEGORY_CONTEXT_MAP.get(category_id, "unknown")
        run_context = BotRunContext(
            channel_id=channel_id,
            category_id=category_id,
            project_context=project_ctx,
            initial_button_intent=intent_category
        )

        current_history = conversation_threads.get(channel_id, []) # Should be empty
        input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": synthetic_text}]
        conversation_threads[channel_id] = input_list # Store this initial "user" message


        if channel_id in pending_tasks: # Cancel any stray debouncing task
            pending_tasks[channel_id].cancel()

        await self.process_ticket_message(channel_id, run_context, is_button_trigger=True, synthetic_user_message_for_log=synthetic_text)

    async def on_message(self, message: discord.Message):
        if message.author.bot or message.author.id == self.user.id:
            return

        is_reply = message.reference is not None
        trigger_char_used = message.content.strip()
        is_valid_trigger_char = trigger_char_used in config.TRIGGER_CONTEXT_MAP
        is_trigger_user = str(message.author.id) in config.PUBLIC_TRIGGER_USER_IDS

        if is_reply and is_trigger_user and is_valid_trigger_char:
            logging.info(f"Stateful public trigger '{trigger_char_used}' detected by {message.author.name} in channel {message.channel.id}")
            
            try:
                await message.delete()
            except Exception as e:
                logging.warning(f"Failed to delete trigger message {message.id}: {e}")

            try:
                original_message = await message.channel.fetch_message(message.reference.message_id)
                if original_message and not original_message.author.bot and original_message.content:
                    
                    original_author_id = original_message.author.id
                    current_history: List[TResponseInputItem] = []
                    
                    conversation = public_conversations.get(original_author_id)
                    if conversation:
                        time_since_last = datetime.now(timezone.utc) - conversation.last_interaction_time
                        if time_since_last <= timedelta(minutes=config.PUBLIC_TRIGGER_TIMEOUT_MINUTES):
                            logging.info(f"Continuing public conversation for user {original_author_id} (last active {time_since_last.total_seconds():.1f}s ago).")
                            current_history = conversation.history
                        else:
                            logging.info(f"Public conversation for user {original_author_id} expired ({time_since_last.total_seconds():.1f}s ago). Starting new context.")
                            public_conversations.pop(original_author_id, None) # Clean up expired entry
                    
                    input_list = current_history + [{"role": "user", "content": original_message.content}]

                    public_run_context = BotRunContext(
                        channel_id=message.channel.id,
                        is_public_trigger=True,
                        project_context=config.TRIGGER_CONTEXT_MAP.get(trigger_char_used, "unknown")
                    )

                    async with message.channel.typing():
                        try:
                            run_config = RunConfig(
                                workflow_name=f"Public Stateful Trigger-{message.channel.id}",
                                group_id=str(original_author_id) # Group traces by the user being helped
                            )
                            result: RunResult = await self.runner.run(
                                starting_agent=triage_agent, # Main triage agent
                                input=input_list,           # Use the stateful input_list
                                max_turns=5,                # Keep a turn limit for safety
                                run_config=run_config,
                                context=public_run_context
                            )

                            new_history = result.to_input_list()
                            new_conversation = PublicConversation(
                                history=new_history,
                                last_interaction_time=datetime.now(timezone.utc)
                            )
                            public_conversations[original_author_id] = new_conversation
                            logging.info(f"Saved updated public conversation context for user {original_author_id}. History length: {len(new_history)} items.")

                            raw_reply = result.final_output if result.final_output else "I could not determine a response."
                            actual_mention = f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}>"
                            final_reply = raw_reply.replace(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)
                            
                            await send_long_message(original_message, final_reply)
                        
                        except Exception as e:
                            logging.error(f"Error during public trigger agent run for user {original_author_id}: {e}", exc_info=True)
                            await original_message.reply(f"Sorry, an error occurred while processing that request. Please notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>.", mention_author=False)

                    return # Stop further processing after handling the trigger

            except discord.NotFound:
                logging.warning(f"Original message for public trigger reply {message.id} not found.")
                return # Can't proceed if original message is gone
            except discord.Forbidden:
                 logging.warning(f"Missing permissions to fetch original message for public trigger reply {message.id}.")
                 return # Can't proceed
            except Exception as e:
                 logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)
                 return

        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category: return
        channel_id = message.channel.id
        if channel_id not in monitored_new_channels: return

        if channel_id in channels_awaiting_initial_button_press:
            try:
                await message.reply("Please select an option from the buttons on my previous message to get started.", delete_after=20, mention_author=False)
            except Exception: pass
            return

        current_intent_from_map = channel_intent_after_button.pop(channel_id, None)
        if current_intent_from_map:
            logging.info(f"Message in {channel_id} is a follow-up to button intent: {current_intent_from_map}")

        ticket_run_context = BotRunContext(
            channel_id=channel_id,
            category_id=message.channel.category.id, # Assuming category is present
            project_context=config.CATEGORY_CONTEXT_MAP.get(message.channel.category.id, "unknown"),
            initial_button_intent=current_intent_from_map # Populate it here
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
        
        pending_tasks[channel_id] = asyncio.create_task(self.process_ticket_message(channel_id, ticket_run_context))
        logging.debug(f"Scheduled processing task for channel {channel_id} in {config.COOLDOWN_SECONDS}s")

    async def process_ticket_message(self, channel_id: int, run_context: BotRunContext):
        try:
            await asyncio.sleep(config.COOLDOWN_SECONDS)
        except asyncio.CancelledError:
            logging.debug(f"Processing task for channel {channel_id} cancelled (new message arrived).")
            return

        aggregated_text = pending_messages.pop(channel_id, None)
        pending_tasks.pop(channel_id, None)
        if not aggregated_text: return

        channel = self.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel): return

        intent = run_context.initial_button_intent
        is_data_intent = intent in ['data_deposit_check', 'data_withdrawal_flow_start', 'data_vault_search']
        
        if is_data_intent and is_message_primarily_address(aggregated_text):
            parsed_address = tools_lib.resolve_ens(aggregated_text)
            if parsed_address:
                ack_message = f"Thank you. I've received the address `{parsed_address}` and am looking up the information now. This may take a moment..."
                try:
                    await channel.send(ack_message)
                    logging.info(f"Sent pre-run acknowledgement message to channel {channel_id}")
                except Exception as e:
                    logging.warning(f"Failed to send pre-run acknowledgement message: {e}")


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
                    max_turns=config.MAX_TICKET_CONVERSATION_TURNS,
                    run_config=run_config,
                    context=run_context # run_context now contains initial_button_intent
                )
                conversation_threads[channel_id] = result.to_input_list()

                raw_final_reply = result.final_output if result.final_output else "I'm not sure how to respond to that."
                actual_mention = f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}>"
                final_reply = raw_final_reply.replace(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                if actual_mention in final_reply or config.HUMAN_HANDOFF_TAG_PLACEHOLDER in raw_final_reply:
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
                 logging.warning(f"Max turns ({config.MAX_TICKET_CONVERSATION_TURNS}) exceeded in channel {channel_id}.")
                 final_reply = f"This conversation has reached its maximum length. <@{config.HUMAN_HANDOFF_TARGET_USER_ID}> may need to intervene."
                 should_stop_processing = True
                 conversation_threads.pop(channel_id, None)
            except AgentsException as e:
                 logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                 final_reply = f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again or notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>."
                 should_stop_processing = True
            except Exception as e:
                 logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                 final_reply = f"An unexpected error occurred. Please notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>."
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

# Run the Bot
if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = TicketBot(intents=intents)
    client.run(config.DISCORD_BOT_TOKEN)
