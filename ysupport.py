import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List

import discord
from dotenv import load_dotenv

from agents import (
    Runner, RunResult,
    TResponseInputItem,
    InputGuardrailTripwireTriggered, MaxTurnsExceeded, AgentsException,
    RunConfig,
    set_default_openai_key,
)

import config
import tools_lib
from router import (
    select_starting_agent,
    is_message_primarily_address,
    is_bug_report_query,
    is_migration_issue_query,
    is_probable_wallet_address,
    is_wallet_confirmation,
    is_wallet_rejection,
)
from support_agents import yearn_bug_triage_agent, yearn_data_agent, yearn_docs_qa_agent, triage_agent
from state import (
    BotRunContext,
    PublicConversation,
    stopped_channels,
    conversation_threads,
    pending_messages,
    pending_tasks,
    monitored_new_channels,
    channels_awaiting_initial_button_press,
    channel_intent_after_button,
    bug_report_debounce_channels,
    public_conversations,
    last_specialist_by_channel,
    last_wallet_by_channel,
    pending_wallet_confirmation_by_channel,
    last_bot_reply_ts_by_channel,
)
from views import InitialInquiryView, StopBotView
from utils import send_long_message


load_dotenv()
set_default_openai_key(config.OPENAI_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)


def _resolve_agent(agent_key: str):
    if agent_key == "data":
        return yearn_data_agent
    if agent_key == "docs":
        return yearn_docs_qa_agent
    if agent_key == "bug":
        return yearn_bug_triage_agent
    return triage_agent


def _select_ticket_starting_agent(
    channel_id: int,
    aggregated_text: str,
    run_context: BotRunContext,
    current_history: List[TResponseInputItem],
) -> str:
    prior_specialist = last_specialist_by_channel.get(channel_id)
    if (
        run_context.initial_button_intent is None
        and current_history
        and prior_specialist in {"data", "docs", "bug"}
    ):
        logging.info(
            "Reusing prior specialist '%s' for follow-up in channel %s",
            prior_specialist,
            channel_id,
        )
        return prior_specialist
    return select_starting_agent(aggregated_text, run_context)


def _agent_to_key(agent) -> str | None:
    if agent is yearn_data_agent:
        return "data"
    if agent is yearn_docs_qa_agent:
        return "docs"
    if agent is yearn_bug_triage_agent:
        return "bug"
    if agent is triage_agent:
        return "triage"
    return None


def _ticket_debounce_seconds(channel_id: int, run_context: BotRunContext) -> int:
    if run_context.initial_button_intent == "bug_report" or channel_id in bug_report_debounce_channels:
        return config.BUG_REPORT_COOLDOWN_SECONDS
    return config.COOLDOWN_SECONDS


# Discord Bot Implementation
class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self.runner = Runner

    async def _send_bug_review_status(self, channel: discord.TextChannel, channel_id: int) -> None:
        message = (
            "Reviewing your report against Yearn contracts, tests, and docs. "
            "This can take a couple of minutes."
        )
        try:
            await channel.send(message)
            last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
            logging.info("Sent bug-review status message to channel %s", channel_id)
        except Exception as e:
            logging.warning("Failed to send bug-review status message to channel %s: %s", channel_id, e)

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
                last_specialist_by_channel.pop(channel.id, None)
                stopped_channels.discard(channel.id)
                bug_report_debounce_channels.discard(channel.id)
                pending_messages.pop(channel.id, None)
                if channel.id in pending_tasks:
                    try:
                        pending_tasks.pop(channel.id).cancel()
                    except Exception:
                        pass
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
                    channels_awaiting_initial_button_press.add(channel.id)
                    last_bot_reply_ts_by_channel[channel.id] = datetime.now(timezone.utc)
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

        if channel_id in pending_tasks:
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
                            public_conversations.pop(original_author_id, None)

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
                                group_id=str(original_author_id)
                            )
                            agent_key = select_starting_agent(
                                input_list[-1].get("content", "") if input_list else "",
                                public_run_context
                            )
                            starting_agent = _resolve_agent(agent_key)
                            result: RunResult = await self.runner.run(
                                starting_agent=starting_agent,
                                input=input_list,
                                max_turns=5,
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

                    return

            except discord.NotFound:
                logging.warning(f"Original message for public trigger reply {message.id} not found.")
                return
            except discord.Forbidden:
                logging.warning(f"Missing permissions to fetch original message for public trigger reply {message.id}.")
                return
            except Exception as e:
                logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)
                return

        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category:
            return
        channel_id = message.channel.id
        if channel_id not in monitored_new_channels:
            return

        if channel_id in channels_awaiting_initial_button_press:
            try:
                await message.reply("Please select an option from the buttons on my previous message to get started.", delete_after=20, mention_author=False)
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
            except Exception:
                pass
            return

        current_intent_from_map = channel_intent_after_button.pop(channel_id, None)
        if current_intent_from_map:
            logging.info(f"Message in {channel_id} is a follow-up to button intent: {current_intent_from_map}")

        ticket_run_context = BotRunContext(
            channel_id=channel_id,
            category_id=message.channel.category.id,
            project_context=config.CATEGORY_CONTEXT_MAP.get(message.channel.category.id, "unknown"),
            initial_button_intent=current_intent_from_map
        )

        if ticket_run_context.project_context == "unknown":
            return
        if channel_id in stopped_channels:
            return

        logging.info(f"Processing ticket message in {channel_id} from {message.author.name} (Context: {ticket_run_context.project_context}, Intent: {current_intent_from_map})")

        existing_task = pending_tasks.get(channel_id)
        if existing_task and not existing_task.done():
            existing_task.cancel()
            try:
                await message.channel.send("Got it — updating based on your latest message.")
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
            except Exception:
                pass

        last_reply_ts = last_bot_reply_ts_by_channel.get(channel_id)
        if last_reply_ts and message.created_at <= last_reply_ts:
            pending_messages[channel_id] = message.content
        elif channel_id not in pending_messages:
            pending_messages[channel_id] = message.content
        else:
            pending_messages[channel_id] += "\n" + message.content

        pending_tasks[channel_id] = asyncio.create_task(self.process_ticket_message(channel_id, ticket_run_context))
        debounce_seconds = _ticket_debounce_seconds(channel_id, ticket_run_context)
        logging.debug(f"Scheduled processing task for channel {channel_id} in {debounce_seconds}s")

    async def process_ticket_message(self, channel_id: int, run_context: BotRunContext, is_button_trigger: bool = False, synthetic_user_message_for_log: str = ""):
        current_task = asyncio.current_task()
        try:
            aggregated_text = None
            if is_button_trigger:
                aggregated_text = synthetic_user_message_for_log.strip()
            else:
                debounce_seconds = _ticket_debounce_seconds(channel_id, run_context)
                try:
                    await asyncio.sleep(debounce_seconds)
                except asyncio.CancelledError:
                    logging.debug(f"Processing task for channel {channel_id} cancelled (new message arrived).")
                    return

                aggregated_text = pending_messages.pop(channel_id, None)
            if not aggregated_text:
                return

            channel = self.get_channel(channel_id)
            if not isinstance(channel, discord.TextChannel):
                return

            intent = run_context.initial_button_intent
            is_data_intent = intent in ['data_deposit_check', 'data_withdrawal_flow_start', 'data_vault_search', 'data_deposits_withdrawals_start']
            system_hints: List[str] = []

            if is_data_intent:
                pending_wallet_confirmation_by_channel.pop(channel_id, None)

            pending_wallet = pending_wallet_confirmation_by_channel.get(channel_id)
            if pending_wallet:
                if is_wallet_confirmation(aggregated_text):
                    last_wallet_by_channel[channel_id] = pending_wallet
                    pending_wallet_confirmation_by_channel.pop(channel_id, None)
                    system_hints.append(f"User confirmed their wallet address is {pending_wallet}. Use this address going forward.")
                elif is_wallet_rejection(aggregated_text):
                    pending_wallet_confirmation_by_channel.pop(channel_id, None)
                    system_hints.append("User rejected the suggested wallet address. Ask for the correct wallet address if needed.")

            extracted_address_or_ens = None
            if is_data_intent:
                extracted_address_or_ens = tools_lib.extract_address_or_ens(aggregated_text) or aggregated_text
            elif is_message_primarily_address(aggregated_text) and is_probable_wallet_address(aggregated_text):
                extracted_address_or_ens = aggregated_text

            if extracted_address_or_ens:
                parsed_address = tools_lib.resolve_ens(extracted_address_or_ens)
                if parsed_address:
                    last_wallet = last_wallet_by_channel.get(channel_id)
                    if is_data_intent:
                        last_wallet_by_channel[channel_id] = parsed_address
                        system_hints.append(
                            f"Resolved wallet address for this turn: {parsed_address}. Use this exact address for any wallet-based tool call."
                        )
                    elif last_wallet and parsed_address != last_wallet:
                        pending_wallet_confirmation_by_channel[channel_id] = parsed_address
                        system_hints.append(
                            f"User provided a new address {parsed_address} but previous wallet was {last_wallet}. "
                            "Ask to confirm whether the new address is their wallet before replacing."
                        )
                    else:
                        last_wallet_by_channel[channel_id] = parsed_address

                    if is_data_intent:
                        ack_message = f"Thank you. I've received the address `{parsed_address}` and am looking up the information now. This may take a moment..."
                        try:
                            await channel.send(ack_message)
                            last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                            logging.info(f"Sent pre-run acknowledgement message to channel {channel_id}")
                        except Exception as e:
                            logging.warning(f"Failed to send pre-run acknowledgement message: {e}")

            current_history = conversation_threads.get(channel_id, [])
            input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": aggregated_text}]

            if system_hints:
                input_list = input_list[:-1] + [{"role": "system", "content": " ".join(system_hints)}] + [input_list[-1]]

            if is_migration_issue_query(aggregated_text) or is_bug_report_query(aggregated_text):
                last_wallet = last_wallet_by_channel.get(channel_id)
                if last_wallet and last_wallet not in aggregated_text:
                    input_list = input_list[:-1] + [{"role": "system", "content": f"Context: User previously provided wallet address {last_wallet} for this ticket. Use it if needed; do not ask again unless they want a different address."}] + [input_list[-1]]

            logging.info(f"Processing for ticket {channel_id} (Context: {run_context.project_context}, Initial Button Intent: {run_context.initial_button_intent}): '{aggregated_text[:100]}...'")

            async with channel.typing():
                final_reply = "An unexpected error occurred."
                should_stop_processing = False

                try:
                    run_config = RunConfig(
                        workflow_name=f"Ticket Channel {channel_id} ({run_context.project_context}, Button Intent: {run_context.initial_button_intent})",
                        group_id=str(channel_id)
                    )
                    agent_key = _select_ticket_starting_agent(
                        channel_id,
                        aggregated_text,
                        run_context,
                        current_history,
                    )
                    logging.info(
                        "Selected starting agent '%s' for channel %s (intent=%s)",
                        agent_key,
                        channel_id,
                        run_context.initial_button_intent,
                    )
                    starting_agent = _resolve_agent(agent_key)
                    if agent_key == "bug":
                        await self._send_bug_review_status(channel, channel_id)
                    result: RunResult = await self.runner.run(
                        starting_agent=starting_agent,
                        input=input_list,
                        max_turns=config.MAX_TICKET_CONVERSATION_TURNS,
                        run_config=run_config,
                        context=run_context
                    )
                    conversation_threads[channel_id] = result.to_input_list()
                    completed_agent_key = _agent_to_key(result.last_agent)
                    if completed_agent_key in {"data", "docs", "bug"}:
                        last_specialist_by_channel[channel_id] = completed_agent_key
                    elif completed_agent_key == "triage":
                        last_specialist_by_channel.pop(channel_id, None)

                    raw_final_reply = result.final_output if result.final_output else "I'm not sure how to respond to that."
                    actual_mention = f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}>"
                    final_reply = raw_final_reply.replace(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)

                    if actual_mention in final_reply or config.HUMAN_HANDOFF_TAG_PLACEHOLDER in raw_final_reply:
                        logging.info(
                            "Human handoff tag detected in response for channel %s. Leaving channel active for follow-up.",
                            channel_id,
                        )
                except InputGuardrailTripwireTriggered as e:
                    logging.warning(f"Input Guardrail triggered in channel {channel_id}. Extracting message from output_info.")
                    guardrail_info = e.guardrail_result.output.output_info
                    if isinstance(guardrail_info, dict) and "message" in guardrail_info:
                        final_reply = guardrail_info["message"]
                    else:
                        final_reply = "Your request could not be processed due to input checks."
                    should_stop_processing = True
                    conversation_threads.pop(channel_id, None)
                    last_specialist_by_channel.pop(channel_id, None)
                except MaxTurnsExceeded:
                    logging.warning(f"Max turns ({config.MAX_TICKET_CONVERSATION_TURNS}) exceeded in channel {channel_id}.")
                    if run_context.repo_search_calls:
                        final_reply = (
                            "I hit an internal analysis limit while reviewing repo evidence for this report. "
                            f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}> may need to intervene."
                        )
                    else:
                        final_reply = (
                            f"This conversation has reached its maximum length. "
                            f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}> may need to intervene."
                        )
                    should_stop_processing = True
                    conversation_threads.pop(channel_id, None)
                    last_specialist_by_channel.pop(channel_id, None)
                except AgentsException as e:
                    logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                    final_reply = f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again or notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>."
                    should_stop_processing = True
                    last_specialist_by_channel.pop(channel_id, None)
                except Exception as e:
                    logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                    final_reply = f"An unexpected error occurred. Please notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>."
                    should_stop_processing = True
                    last_specialist_by_channel.pop(channel_id, None)

                try:
                    reply_view = StopBotView() if not should_stop_processing else None
                    await send_long_message(channel, final_reply, view=reply_view)
                    if channel_id in bug_report_debounce_channels:
                        bug_report_debounce_channels.discard(channel_id)
                        logging.info("Cleared bug-report debounce flag for channel %s", channel_id)
                    last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                    logging.info(f"Sent ticket reply/replies in channel {channel_id}. Stop processing flag: {should_stop_processing}")
                    if should_stop_processing and channel_id not in stopped_channels:
                        stopped_channels.add(channel_id)
                        logging.info(f"Added channel {channel_id} to stopped channels due to error/handoff tag.")
                except discord.Forbidden:
                    logging.error(f"Missing permissions to send message in channel {channel_id}")
                    stopped_channels.add(channel_id)
                except Exception as e:
                    logging.error(f"Unexpected error occurred during or after calling send_long_message for channel {channel_id}: {e}", exc_info=True)
        except asyncio.CancelledError:
            logging.info(f"Processing task for channel {channel_id} cancelled mid-run.")
            return
        finally:
            if pending_tasks.get(channel_id) is current_task:
                pending_tasks.pop(channel_id, None)


# Run the Bot
if __name__ == "__main__":
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = TicketBot(intents=intents)
    client.run(config.DISCORD_BOT_TOKEN)
