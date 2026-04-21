import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, List

import discord
from dotenv import load_dotenv

from agents import (
    Runner, TResponseInputItem,
    InputGuardrailTripwireTriggered, MaxTurnsExceeded, AgentsException,
    set_default_openai_key,
)

import config
import tools_lib
from router import (
    is_message_primarily_address,
    is_bug_report_query,
    is_probable_wallet_address,
    is_wallet_confirmation,
    is_wallet_rejection,
)
from support_agents import evaluate_support_boundary
from state import (
    BotRunContext,
    PublicConversation,
    TicketInvestigationJob,
    channels_awaiting_initial_button_press,
    channel_intent_after_button,
    bug_report_debounce_channels,
    clear_public_conversation,
    clear_ticket_channel_state,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    hydrate_public_conversation,
    hydrate_ticket_state,
    last_wallet_by_channel,
    last_bot_reply_ts_by_channel,
    monitored_new_channels,
    pending_messages,
    pending_tasks,
    pending_wallet_confirmation_by_channel,
    persist_public_conversation,
    persist_ticket_state,
    public_conversations,
    stopped_channels,
)
from ticket_investigation_json_endpoint import (
    build_ticket_execution_json_endpoint,
    JsonEndpointTicketExecutionTransport,
)
from ticket_investigation_executor import (
    LocalTicketInvestigationExecutor,
    TicketExecutionHooks,
    TransportTicketInvestigationExecutor,
)
from ticket_investigation_runtime import (
    TicketInvestigationRuntime,
    TicketTurnRequest,
)
from ticket_investigation_worker import TicketInvestigationWorker
from views import InitialInquiryView, StopBotView
from utils import send_long_message


load_dotenv()
set_default_openai_key(config.OPENAI_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)


def _ticket_debounce_seconds(channel_id: int, run_context: BotRunContext) -> int:
    if run_context.initial_button_intent == "bug_report" or channel_id in bug_report_debounce_channels:
        return config.BUG_REPORT_COOLDOWN_SECONDS
    return config.COOLDOWN_SECONDS


def _canonicalize_current_user_message(
    aggregated_text: str,
    run_context: BotRunContext,
    resolved_wallet: str | None,
) -> str:
    if not resolved_wallet:
        return aggregated_text
    if run_context.initial_button_intent in {
        "data_deposit_check",
        "data_withdrawal_flow_start",
        "data_vault_search",
        "data_deposits_withdrawals_start",
    }:
        return resolved_wallet
    if is_message_primarily_address(aggregated_text) and is_probable_wallet_address(aggregated_text):
        return resolved_wallet
    return aggregated_text


def _guardrail_tripwire_reply(exc: InputGuardrailTripwireTriggered) -> str:
    guardrail_info = exc.guardrail_result.output.output_info
    if isinstance(guardrail_info, dict) and "message" in guardrail_info:
        return guardrail_info["message"]
    return "Your request could not be processed due to input checks."


async def _outer_support_boundary_result(text: str) -> dict[str, Any]:
    return await evaluate_support_boundary(text)


async def _outer_support_boundary_reply(text: str) -> str | None:
    output_info = await _outer_support_boundary_result(text)
    if output_info.get("tripwire_triggered") and output_info.get("message"):
        return str(output_info["message"])
    return None


def _boundary_reply_from_output(output_info: dict[str, Any]) -> str | None:
    if output_info.get("tripwire_triggered") and output_info.get("message"):
        return str(output_info["message"])
    return None


def _render_support_reply(raw_reply: str) -> str:
    actual_mention = f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}>"
    return raw_reply.replace(config.HUMAN_HANDOFF_TAG_PLACEHOLDER, actual_mention)


def _build_turn_request(
    *,
    aggregated_text: str,
    input_list: List[TResponseInputItem],
    current_history: List[TResponseInputItem],
    run_context: BotRunContext,
    investigation_job: TicketInvestigationJob,
    workflow_name: str,
    precomputed_boundary: dict[str, Any] | None,
) -> TicketTurnRequest:
    return TicketTurnRequest(
        aggregated_text=aggregated_text,
        input_list=input_list,
        current_history=current_history,
        run_context=run_context,
        investigation_job=investigation_job,
        workflow_name=workflow_name,
        precomputed_boundary=precomputed_boundary,
    )


def _public_workflow_name(channel_id: int) -> str:
    return f"Public Stateful Trigger-{channel_id}"


def _ticket_workflow_name(run_context: BotRunContext) -> str:
    return (
        f"Ticket Channel {run_context.channel_id} ({run_context.project_context}, "
        f"Button Intent: {run_context.initial_button_intent})"
    )


def _project_context_for_category(category_id: int | None) -> str:
    if category_id is None:
        return "unknown"
    return config.CATEGORY_CONTEXT_MAP.get(category_id, "unknown")


def _build_public_run_context(
    *,
    channel_id: int,
    conversation_owner_id: int,
    trigger_char_used: str,
) -> BotRunContext:
    return BotRunContext(
        channel_id=channel_id,
        is_public_trigger=True,
        conversation_owner_id=conversation_owner_id,
        project_context=config.TRIGGER_CONTEXT_MAP.get(trigger_char_used, "unknown"),
    )


def _record_button_requested_intent(
    *,
    channel_id: int,
    investigation_job: TicketInvestigationJob,
) -> str | None:
    current_intent = channel_intent_after_button.pop(channel_id, None)
    if current_intent:
        logging.info(
            "Message in %s is a follow-up to button intent: %s",
            channel_id,
            current_intent,
        )
        investigation_job.record_requested_intent(current_intent)
    return current_intent


def _build_ticket_run_context(
    *,
    channel_id: int,
    category_id: int | None,
    initial_button_intent: str | None,
) -> BotRunContext:
    return BotRunContext(
        channel_id=channel_id,
        category_id=category_id,
        project_context=_project_context_for_category(category_id),
        initial_button_intent=initial_button_intent,
    )


# Discord Bot Implementation
class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self.runner = Runner
        self.investigation_runtime = TicketInvestigationRuntime(self.runner)
        self.investigation_worker = TicketInvestigationWorker(self.investigation_runtime)
        self.local_investigation_executor = LocalTicketInvestigationExecutor(
            self.investigation_worker
        )
        self.investigation_json_endpoint = build_ticket_execution_json_endpoint(
            self.local_investigation_executor
        )
        self.investigation_transport = JsonEndpointTicketExecutionTransport(
            self.investigation_json_endpoint
        )
        self.investigation_executor = TransportTicketInvestigationExecutor(
            self.investigation_transport
        )
        logging.info(
            "Ticket execution runtime configured: %s",
            config.ticket_execution_runtime_summary(),
        )
        for warning in config.ticket_execution_runtime_warnings():
            logging.warning("Ticket execution rollout warning: %s", warning)

    async def _send_bug_review_status(self, channel: discord.TextChannel, channel_id: int) -> None:
        message = (
            "Reviewing your report against Yearn contracts, tests, and docs. "
            "This can take a couple of minutes."
        )
        try:
            await channel.send(message, suppress_embeds=True)
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
                clear_ticket_channel_state(
                    channel.id,
                    keep_stopped=False,
                    delete_persisted=True,
                )
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
                    await channel.send(welcome_message, view=InitialInquiryView(), suppress_embeds=True)
                    channels_awaiting_initial_button_press.add(channel.id)
                    last_bot_reply_ts_by_channel[channel.id] = datetime.now(timezone.utc)
                    persist_ticket_state(channel.id)
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
        run_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=category_id,
            initial_button_intent=intent_category,
        )

        if channel_id in pending_tasks:
            pending_tasks[channel_id].cancel()

        await self.process_ticket_message(channel_id, run_context, is_button_trigger=True, synthetic_user_message_for_log=synthetic_text)

    async def _handle_public_trigger_message(
        self,
        message: discord.Message,
        trigger_char_used: str,
    ) -> bool:
        logging.info(
            "Stateful public trigger '%s' detected by %s in channel %s",
            trigger_char_used,
            message.author.name,
            message.channel.id,
        )

        try:
            await message.delete()
        except Exception as e:
            logging.warning(f"Failed to delete trigger message {message.id}: {e}")

        try:
            original_message = await message.channel.fetch_message(message.reference.message_id)
            if not (original_message and not original_message.author.bot and original_message.content):
                return True

            original_author_id = original_message.author.id
            current_history: List[TResponseInputItem] = []
            public_investigation_job = None
            hydrate_public_conversation(original_author_id)

            conversation = public_conversations.get(original_author_id)
            if conversation:
                time_since_last = datetime.now(timezone.utc) - conversation.last_interaction_time
                if time_since_last <= timedelta(minutes=config.PUBLIC_TRIGGER_TIMEOUT_MINUTES):
                    logging.info(
                        "Continuing public conversation for user %s (last active %.1fs ago).",
                        original_author_id,
                        time_since_last.total_seconds(),
                    )
                    current_history = conversation.history
                    public_investigation_job = conversation.investigation_job
                else:
                    logging.info(
                        "Public conversation for user %s expired (%.1fs ago). Starting new context.",
                        original_author_id,
                        time_since_last.total_seconds(),
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)

            if public_investigation_job is None:
                public_investigation_job = TicketInvestigationJob(
                    channel_id=message.channel.id
                )
            public_run_context = _build_public_run_context(
                channel_id=message.channel.id,
                conversation_owner_id=original_author_id,
                trigger_char_used=trigger_char_used,
            )

            boundary_output = await _outer_support_boundary_result(original_message.content)
            boundary_reply = _boundary_reply_from_output(boundary_output)
            if boundary_reply is not None:
                public_conversations.pop(original_author_id, None)
                clear_public_conversation(original_author_id)
                await original_message.reply(
                    boundary_reply,
                    mention_author=False,
                    suppress_embeds=True,
                )
                return True

            async with message.channel.typing():
                try:
                    worker_result = await self.investigation_executor.execute_turn(
                        _build_turn_request(
                            aggregated_text=original_message.content,
                            input_list=current_history + [
                                {"role": "user", "content": original_message.content}
                            ],
                            current_history=current_history,
                            run_context=public_run_context,
                            investigation_job=public_investigation_job,
                            workflow_name=_public_workflow_name(message.channel.id),
                            precomputed_boundary=boundary_output,
                        )
                    )
                    flow_outcome = worker_result.flow_outcome
                    new_history = flow_outcome.conversation_history
                    public_conversations[original_author_id] = PublicConversation(
                        history=new_history,
                        last_interaction_time=datetime.now(timezone.utc),
                        investigation_job=worker_result.updated_job,
                    )
                    persist_public_conversation(original_author_id)
                    logging.info(
                        "Saved updated public conversation context for user %s. History length: %s items.",
                        original_author_id,
                        len(new_history),
                    )

                    raw_reply = (
                        flow_outcome.raw_final_reply
                        if flow_outcome.raw_final_reply
                        else "I could not determine a response."
                    )
                    final_reply = _render_support_reply(raw_reply)
                    await send_long_message(message.channel, final_reply)
                except InputGuardrailTripwireTriggered as e:
                    logging.warning(
                        "Input Guardrail triggered in public channel %s for user %s.",
                        message.channel.id,
                        original_author_id,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    await original_message.reply(
                        _guardrail_tripwire_reply(e),
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except MaxTurnsExceeded:
                    logging.warning(
                        "Max turns (%s) exceeded during public trigger run for user %s in channel %s.",
                        config.MAX_PUBLIC_TRIGGER_TURNS,
                        original_author_id,
                        message.channel.id,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    if public_run_context.repo_search_calls:
                        reply_text = (
                            "I hit an internal analysis limit while reviewing repo evidence for that request. "
                            f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}> may need to help."
                        )
                    else:
                        reply_text = (
                            "I hit an internal analysis limit while working on that request. "
                            f"<@{config.HUMAN_HANDOFF_TARGET_USER_ID}> may need to help."
                        )
                    await original_message.reply(
                        reply_text,
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except Exception as e:
                    logging.error(
                        "Error during public trigger agent run for user %s: %s",
                        original_author_id,
                        e,
                        exc_info=True,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    await original_message.reply(
                        f"Sorry, an error occurred while processing that request. Please notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>.",
                        mention_author=False,
                        suppress_embeds=True,
                    )
            return True
        except discord.NotFound:
            logging.warning(f"Original message for public trigger reply {message.id} not found.")
            return True
        except discord.Forbidden:
            logging.warning(f"Missing permissions to fetch original message for public trigger reply {message.id}.")
            return True
        except Exception as e:
            logging.error(f"Error handling public trigger for message {message.id}: {e}", exc_info=True)
            return True

    async def _collect_aggregated_ticket_text(
        self,
        channel_id: int,
        run_context: BotRunContext,
        *,
        is_button_trigger: bool,
        synthetic_user_message_for_log: str,
    ) -> str | None:
        if is_button_trigger:
            return synthetic_user_message_for_log.strip()

        debounce_seconds = _ticket_debounce_seconds(channel_id, run_context)
        try:
            await asyncio.sleep(debounce_seconds)
        except asyncio.CancelledError:
            logging.debug(f"Processing task for channel {channel_id} cancelled (new message arrived).")
            return None
        return pending_messages.pop(channel_id, None)

    async def _build_ticket_turn_input(
        self,
        *,
        channel: discord.TextChannel,
        channel_id: int,
        run_context: BotRunContext,
        investigation_job,
        aggregated_text: str,
    ) -> tuple[List[TResponseInputItem], List[TResponseInputItem]]:
        intent = run_context.initial_button_intent
        is_data_intent = intent in [
            'data_deposit_check',
            'data_withdrawal_flow_start',
            'data_vault_search',
            'data_deposits_withdrawals_start',
        ]
        current_user_content = aggregated_text
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
                    investigation_job.remember_wallet(parsed_address)
                    current_user_content = _canonicalize_current_user_message(
                        aggregated_text,
                        run_context,
                        parsed_address,
                    )
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
                    investigation_job.remember_wallet(parsed_address)
                    current_user_content = _canonicalize_current_user_message(
                        aggregated_text,
                        run_context,
                        parsed_address,
                    )

                if is_data_intent:
                    ack_message = f"Thank you. I've received the address `{parsed_address}` and am looking up the information now. This may take a moment..."
                    try:
                        await channel.send(ack_message, suppress_embeds=True)
                        last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                        logging.info(f"Sent pre-run acknowledgement message to channel {channel_id}")
                    except Exception as e:
                        logging.warning(f"Failed to send pre-run acknowledgement message: {e}")

        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [{"role": "user", "content": current_user_content}]

        if system_hints:
            input_list = input_list[:-1] + [{"role": "system", "content": " ".join(system_hints)}] + [input_list[-1]]

        contextual_hints = self.investigation_runtime.build_contextual_hints(
            investigation_job,
            aggregated_text,
            current_history=current_history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [{"role": "system", "content": " ".join(contextual_hints)}] + [input_list[-1]]

        if is_bug_report_query(aggregated_text):
            last_wallet = last_wallet_by_channel.get(channel_id)
            if last_wallet and last_wallet not in aggregated_text:
                input_list = input_list[:-1] + [{
                    "role": "system",
                    "content": f"Context: User previously provided wallet address {last_wallet} for this ticket. Use it if needed; do not ask again unless they want a different address."
                }] + [input_list[-1]]

        return current_history, input_list

    async def on_message(self, message: discord.Message):
        if message.author.bot or message.author.id == self.user.id:
            return

        is_reply = message.reference is not None
        trigger_char_used = message.content.strip()
        is_valid_trigger_char = trigger_char_used in config.TRIGGER_CONTEXT_MAP
        is_trigger_user = str(message.author.id) in config.PUBLIC_TRIGGER_USER_IDS

        if is_reply and is_trigger_user and is_valid_trigger_char:
            await self._handle_public_trigger_message(message, trigger_char_used)
            return

        if not isinstance(message.channel, discord.TextChannel) or not message.channel.category:
            return
        channel_id = message.channel.id
        if message.channel.category.id not in config.CATEGORY_CONTEXT_MAP:
            return
        hydrate_ticket_state(channel_id)
        monitored_new_channels.add(channel_id)

        if channel_id in channels_awaiting_initial_button_press:
            try:
                await message.reply(
                    "Please select an option from the buttons on my previous message to get started.",
                    delete_after=20,
                    mention_author=False,
                    suppress_embeds=True,
                )
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                persist_ticket_state(channel_id)
            except Exception:
                pass
            return

        investigation_job = get_or_create_ticket_investigation_job(channel_id)
        current_intent_from_map = _record_button_requested_intent(
            channel_id=channel_id,
            investigation_job=investigation_job,
        )
        ticket_run_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=message.channel.category.id,
            initial_button_intent=current_intent_from_map,
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
                await message.channel.send("Got it — updating based on your latest message.", suppress_embeds=True)
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
            investigation_job = get_or_create_ticket_investigation_job(channel_id)
            aggregated_text = await self._collect_aggregated_ticket_text(
                channel_id,
                run_context,
                is_button_trigger=is_button_trigger,
                synthetic_user_message_for_log=synthetic_user_message_for_log,
            )
            if not aggregated_text:
                return
            self.investigation_runtime.merge_explicit_evidence(investigation_job, aggregated_text)

            channel = self.get_channel(channel_id)
            if not isinstance(channel, discord.TextChannel):
                return

            current_history, input_list = await self._build_ticket_turn_input(
                channel=channel,
                channel_id=channel_id,
                run_context=run_context,
                investigation_job=investigation_job,
                aggregated_text=aggregated_text,
            )

            logging.info(f"Processing for ticket {channel_id} (Context: {run_context.project_context}, Initial Button Intent: {run_context.initial_button_intent}): '{aggregated_text[:100]}...'")

            async with channel.typing():
                final_reply = "An unexpected error occurred."
                should_stop_processing = False

                boundary_output = await _outer_support_boundary_result(aggregated_text)
                boundary_reply = _boundary_reply_from_output(boundary_output)
                if boundary_reply is not None:
                    final_reply = boundary_reply
                    should_stop_processing = True
                    clear_ticket_channel_state(
                        channel_id,
                        keep_stopped=False,
                        delete_persisted=True,
                    )
                else:
                    try:
                        worker_result = await self.investigation_executor.execute_turn(
                            _build_turn_request(
                                aggregated_text=aggregated_text,
                                input_list=input_list,
                                current_history=current_history,
                                run_context=run_context,
                                investigation_job=investigation_job,
                                workflow_name=_ticket_workflow_name(run_context),
                                precomputed_boundary=boundary_output,
                            ),
                            hooks=TicketExecutionHooks(
                                send_bug_review_status=lambda: self._send_bug_review_status(channel, channel_id),
                            ),
                        )
                        investigation_job.apply_snapshot(worker_result.updated_job)
                        flow_outcome = worker_result.flow_outcome
                        conversation_threads[channel_id] = flow_outcome.conversation_history

                        raw_final_reply = flow_outcome.raw_final_reply
                        final_reply = _render_support_reply(raw_final_reply)
                        if flow_outcome.requires_human_handoff:
                            logging.info(
                                "Human handoff tag detected in response for channel %s. Leaving channel active for follow-up.",
                                channel_id,
                            )
                        persist_ticket_state(channel_id)
                    except InputGuardrailTripwireTriggered as e:
                        logging.warning(f"Input Guardrail triggered in channel {channel_id}. Extracting message from output_info.")
                        final_reply = _guardrail_tripwire_reply(e)
                        should_stop_processing = True
                        clear_ticket_channel_state(channel_id, keep_stopped=False, delete_persisted=True)
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
                        clear_ticket_channel_state(channel_id, keep_stopped=False, delete_persisted=True)
                    except AgentsException as e:
                        logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                        final_reply = f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again or notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>."
                        should_stop_processing = True
                        clear_ticket_channel_state(channel_id, keep_stopped=False, delete_persisted=True)
                    except Exception as e:
                        logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                        final_reply = f"An unexpected error occurred. Please notify <@{config.HUMAN_HANDOFF_TARGET_USER_ID}>."
                        should_stop_processing = True
                        clear_ticket_channel_state(channel_id, keep_stopped=False, delete_persisted=True)

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
                    persist_ticket_state(channel_id)
                except discord.Forbidden:
                    logging.error(f"Missing permissions to send message in channel {channel_id}")
                    stopped_channels.add(channel_id)
                    persist_ticket_state(channel_id)
                except Exception as e:
                    logging.error(f"Unexpected error occurred during or after calling send_long_message for channel {channel_id}: {e}", exc_info=True)
        except asyncio.CancelledError:
            logging.info(f"Processing task for channel {channel_id} cancelled mid-run.")
            return
        finally:
            if pending_tasks.get(channel_id) is current_task:
                pending_tasks.pop(channel_id, None)

    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
        if not isinstance(channel, discord.TextChannel):
            return
        clear_ticket_channel_state(channel.id, keep_stopped=False, delete_persisted=True)


# Run the Bot
if __name__ == "__main__":
    config.validate_runtime_environment_config()
    config.validate_ticket_execution_runtime_config()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True

    client = TicketBot(intents=intents)
    client.run(config.DISCORD_BOT_TOKEN)
