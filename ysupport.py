import asyncio
from contextlib import suppress
import importlib
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, List

import discord
from dotenv import load_dotenv

from agents import (
    TResponseInputItem,
    InputGuardrailTripwireTriggered, MaxTurnsExceeded, AgentsException,
    set_default_openai_key,
)

import config
from discord_support_runtime import (
    _DiscordProgressReporter,
    _attachment_payloads_from_message,
    _boundary_reply_from_output,
    _build_public_run_context,
    _build_staff_summon_history,
    _build_ticket_run_context,
    _build_turn_request,
    _classify_ticket_message_action,
    _detect_ticket_owner_user_id,
    _discard_pending_ticket_payload,
    _execute_ticket_turn,
    _guardrail_tripwire_reply,
    _handoff_delivery_failure_reply,
    _is_support_staff_member,
    _maybe_recover_runtime_stopped_ticket_for_message,
    _merge_pending_ticket_payload,
    _message_text_for_turn,
    _normalize_staff_summon_prompt,
    _notify_handoff,
    _outer_support_boundary_result,
    _public_workflow_name,
    _record_button_requested_intent,
    _record_waiting_for_team_followup,
    _remember_sent_handoff_notice,
    _render_support_reply,
    _restore_active_ticket_payload,
    _run_internal_instruction_turn,
    _should_ack_waiting_for_team,
    _should_stop_for_boundary_output,
    _send_ticket_handoff_notice,
    _ticket_debounce_seconds,
    _ticket_workflow_name,
    _waiting_for_team_reply,
)
from handoff import (
    build_user_handoff_reply,
)
from state import (
    active_ticket_executor_tasks,
    active_ticket_payloads,
    BotRunContext,
    PublicConversation,
    TicketInvestigationJob,
    channels_awaiting_initial_button_press,
    bug_report_debounce_channels,
    clear_public_conversation,
    conversation_threads,
    get_or_create_ticket_investigation_job,
    hydrate_public_conversation,
    hydrate_ticket_state,
    hydrate_persisted_team_handoff_states,
    is_ticket_waiting_for_team,
    last_bot_reply_ts_by_channel,
    mark_ticket_channel_stopped,
    pending_messages,
    pending_attachments_by_channel,
    pending_tasks,
    persist_public_conversation,
    persist_ticket_state,
    prune_expired_public_conversations,
    public_conversations,
    remember_ticket_owner_user_id,
    remember_team_handoff_followup_attachments,
    reset_ticket_channel_for_terminal_reply,
    reset_public_codex_session,
    reset_ticket_codex_session,
    stop_ticket_channel,
    stopped_channels,
    ticket_owner_user_id_by_channel,
)
from ticket_intake import prepare_ticket_turn_input
from ticket_channel_lifecycle import (
    clear_deleted_ticket_channel,
    initialize_ticket_channel,
)
from telegram_handoff_controller import TelegramHandoffController
from ticket_investigation.json_endpoint import (
    build_ticket_execution_json_endpoint,
    JsonEndpointTicketExecutionTransport,
    prune_codex_support_sessions,
)
from ticket_investigation.context import (
    build_contextual_hints,
    merge_explicit_evidence,
)
from ticket_investigation.executor import (
    TicketExecutionHooks,
    TransportTicketInvestigationExecutor,
)
from views import (
    InitialInquiryView,
    StopBotView,
    stop_ticket_for_manual_support,
)
from utils import send_long_message


set_default_openai_key(config.OPENAI_API_KEY)

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)


class TicketBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, **options):
        super().__init__(intents=intents, **options)
        self._state_cleanup_task: asyncio.Task[None] | None = None
        self._public_turn_locks: dict[int, asyncio.Lock] = {}
        local_executor = None
        if "local" in {
            config.TICKET_EXECUTION_ENDPOINT,
            config.TICKET_EXECUTION_FALLBACK_ENDPOINT,
            config.TICKET_EXECUTION_CANARY_ENDPOINT,
            config.TICKET_EXECUTION_SHADOW_ENDPOINT,
        }:
            from ticket_execution.runtime_factory import (
                build_local_ticket_investigation_executor,
            )

            local_executor = build_local_ticket_investigation_executor()
        self.investigation_json_endpoint = build_ticket_execution_json_endpoint(
            local_executor
        )
        self.investigation_transport = JsonEndpointTicketExecutionTransport(
            self.investigation_json_endpoint
        )
        self.investigation_executor = TransportTicketInvestigationExecutor(
            self.investigation_transport
        )
        self.telegram_handoffs = TelegramHandoffController(self)
        logging.info(
            "Ticket execution runtime configured: %s",
            config.ticket_execution_runtime_summary(),
        )
        for warning in config.ticket_execution_runtime_warnings():
            logging.warning("Ticket execution rollout warning: %s", warning)

    async def setup_hook(self) -> None:
        self.add_view(InitialInquiryView())
        self.add_view(StopBotView())

    async def on_ready(self):
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logging.info(f"Monitoring Yearn Ticket Category ID: {config.YEARN_TICKET_CATEGORY_ID}")
        logging.info(f"Support User ID for triggers: {config.PUBLIC_TRIGGER_USER_IDS}")
        logging.info(f"Yearn Public Trigger: '{config.YEARN_PUBLIC_TRIGGER_CHAR}'")
        hydrated_handoffs = hydrate_persisted_team_handoff_states()
        logging.info(
            "Hydrated %s persisted Telegram handoff state(s) before polling.",
            hydrated_handoffs,
        )
        telegram_started = self.telegram_handoffs.start()
        if self._state_cleanup_task is None or self._state_cleanup_task.done():
            self._state_cleanup_task = asyncio.create_task(
                self._state_cleanup_loop()
            )
        logging.info("Telegram handoff reply loop initialized: %s", telegram_started)

    async def close(self) -> None:
        await self.telegram_handoffs.close()
        ticket_tasks = set(pending_tasks.values())
        pending_tasks.clear()
        for task in ticket_tasks:
            task.cancel()
        if ticket_tasks:
            await asyncio.gather(*ticket_tasks, return_exceptions=True)
        if self._state_cleanup_task is not None:
            cleanup_task = self._state_cleanup_task
            self._state_cleanup_task = None
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task
        await super().close()

    async def _state_cleanup_loop(self) -> None:
        cleanup_interval_seconds = max(
            60,
            config.PUBLIC_TRIGGER_TIMEOUT_MINUTES * 60,
        )
        while not self.is_closed():
            removed = prune_expired_public_conversations()
            if removed:
                logging.info(
                    "Removed %s expired public conversation state file(s).",
                    removed,
                )
            try:
                removed_codex_sessions = await prune_codex_support_sessions(
                    self.investigation_json_endpoint
                )
            except Exception:
                logging.exception("Failed to prune expired Codex support sessions.")
            else:
                if removed_codex_sessions:
                    logging.info(
                        "Removed %s expired Codex support session(s).",
                        removed_codex_sessions,
                    )
            await asyncio.sleep(cleanup_interval_seconds)

    async def on_guild_channel_create(self, channel: discord.abc.GuildChannel):
        await initialize_ticket_channel(channel)

    async def _handle_public_trigger_message(
        self,
        message: discord.Message,
        trigger_char_used: str,
    ) -> bool:
        original_message: discord.Message | None = None
        original_author_id: int | None = None
        public_lock: asyncio.Lock | None = None
        public_lock_acquired = False
        preserve_public_state_on_error = False
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
            if not original_message or original_message.author.bot:
                return True
            original_message_text = _message_text_for_turn(original_message)
            if not original_message_text:
                return True

            original_author_id = original_message.author.id
            public_lock = self._public_turn_locks.setdefault(
                original_author_id,
                asyncio.Lock(),
            )
            await public_lock.acquire()
            public_lock_acquired = True
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

            boundary_output = await _outer_support_boundary_result(original_message_text)
            boundary_reply = _boundary_reply_from_output(boundary_output)
            if boundary_reply is not None:
                preserve_public_state_on_error = True
                await original_message.reply(
                    boundary_reply,
                    mention_author=False,
                    suppress_embeds=True,
                )
                clear_public_conversation(original_author_id)
                return True

            async with message.channel.typing():
                progress_reporter = _DiscordProgressReporter(
                    message.channel,
                    message.channel.id,
                )
                next_conversation: PublicConversation | None = None
                try:
                    worker_result = await self.investigation_executor.execute_turn(
                        _build_turn_request(
                            aggregated_text=original_message_text,
                            input_list=current_history + [
                                {"role": "user", "content": original_message_text}
                            ],
                            current_history=current_history,
                            attachments=_attachment_payloads_from_message(original_message),
                            run_context=public_run_context,
                            investigation_job=public_investigation_job,
                            workflow_name=_public_workflow_name(message.channel.id),
                            precomputed_boundary=boundary_output,
                        ),
                        hooks=TicketExecutionHooks(
                            send_progress_update=progress_reporter.update,
                        ),
                    )
                    flow_outcome = worker_result.flow_outcome
                    new_history = flow_outcome.conversation_history
                    next_conversation = PublicConversation(
                        history=new_history,
                        last_interaction_time=datetime.now(timezone.utc),
                        investigation_job=worker_result.updated_job,
                    )

                    raw_reply = (
                        flow_outcome.raw_final_reply
                        if flow_outcome.raw_final_reply
                        else "I could not determine a response."
                    )
                    if flow_outcome.requires_human_handoff:
                        handoff_reason = (
                            flow_outcome.handoff_reason
                            or "manual follow-up needed"
                        )
                        notice = await _notify_handoff(
                            reason=handoff_reason,
                            summary=original_message_text,
                            channel_id=message.channel.id,
                            guild_id=getattr(getattr(message.channel, "guild", None), "id", None),
                            source="public",
                        )
                        if notice is not None:
                            final_reply = build_user_handoff_reply(
                                raw_reply,
                                location="here",
                            )
                            handoff_sent = True
                        else:
                            handoff_sent = False
                            final_reply = _handoff_delivery_failure_reply(
                                raw_reply,
                                location="here",
                            )
                    else:
                        handoff_sent = False
                        final_reply = _render_support_reply(raw_reply)
                    await progress_reporter.close()
                    await send_long_message(message.channel, final_reply)
                    if handoff_sent:
                        clear_public_conversation(original_author_id)
                    else:
                        public_conversations[original_author_id] = next_conversation
                        persist_public_conversation(original_author_id)
                        logging.info(
                            "Saved updated public conversation context for user %s. "
                            "History length: %s items.",
                            original_author_id,
                            len(new_history),
                        )
                except InputGuardrailTripwireTriggered as e:
                    await progress_reporter.close()
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
                    await progress_reporter.close()
                    logging.warning(
                        "Max turns (%s) exceeded during public trigger run for user %s in channel %s.",
                        config.MAX_PUBLIC_TRIGGER_TURNS,
                        original_author_id,
                        message.channel.id,
                    )
                    public_conversations.pop(original_author_id, None)
                    clear_public_conversation(original_author_id)
                    if public_run_context.repo_search_calls:
                        base_reply = (
                            "I hit an internal analysis limit while reviewing repo evidence for that request."
                        )
                    else:
                        base_reply = (
                            "I hit an internal analysis limit while working on that request."
                        )
                    await original_message.reply(
                        f"{base_reply} Please try again.",
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except Exception as e:
                    await progress_reporter.close()
                    logging.error(
                        "Error during public trigger agent run for user %s: %s",
                        original_author_id,
                        e,
                        exc_info=True,
                    )
                    if next_conversation is None:
                        clear_public_conversation(original_author_id)
                        error_reply = (
                            "Sorry, an error occurred while processing that request. "
                            "Please try again."
                        )
                    else:
                        reset_public_codex_session(original_author_id)
                        error_reply = (
                            "I couldn't deliver my complete response. Please try again."
                        )
                    try:
                        await original_message.reply(
                            error_reply,
                            mention_author=False,
                            suppress_embeds=True,
                        )
                    except Exception:
                        logging.warning(
                            "Failed to send public trigger error reply for message %s.",
                            message.id,
                            exc_info=True,
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
            if original_author_id is not None and not preserve_public_state_on_error:
                public_conversations.pop(original_author_id, None)
                clear_public_conversation(original_author_id)
            if original_message is not None:
                try:
                    await original_message.reply(
                        (
                            "I couldn't deliver my complete response. Please try again."
                            if preserve_public_state_on_error
                            else (
                                "Sorry, an error occurred while preparing that request. "
                                "Please try again."
                            )
                        ),
                        mention_author=False,
                        suppress_embeds=True,
                    )
                except Exception:
                    logging.warning(
                        "Failed to send public trigger setup error reply for message %s.",
                        message.id,
                        exc_info=True,
                    )
            return True
        finally:
            if public_lock is not None and public_lock_acquired:
                public_lock.release()

    async def _collect_aggregated_ticket_payload(
        self,
        channel_id: int,
        run_context: BotRunContext,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        debounce_seconds = _ticket_debounce_seconds(channel_id, run_context)
        try:
            await asyncio.sleep(debounce_seconds)
        except asyncio.CancelledError:
            logging.debug(f"Processing task for channel {channel_id} cancelled (new message arrived).")
            return None, []
        aggregated_text = pending_messages.pop(channel_id, None)
        attachments = pending_attachments_by_channel.pop(channel_id, [])
        current_task = asyncio.current_task()
        if aggregated_text and current_task is not None:
            active_ticket_payloads[channel_id] = (
                current_task,
                aggregated_text,
                attachments,
            )
        return aggregated_text, attachments

    def _finish_ticket_task(
        self,
        *,
        channel_id: int,
        current_task: asyncio.Task | None,
        run_context: BotRunContext,
    ) -> None:
        if pending_tasks.get(channel_id) is not current_task:
            return
        pending_tasks.pop(channel_id, None)
        queued_text = pending_messages.get(channel_id)
        if not queued_text:
            return
        if channel_id in stopped_channels:
            _discard_pending_ticket_payload(channel_id)
            return
        if is_ticket_waiting_for_team(channel_id):
            pending_messages.pop(channel_id, None)
            queued_attachments = pending_attachments_by_channel.pop(channel_id, [])
            conversation_threads.setdefault(channel_id, []).append(
                {"role": "user", "content": queued_text}
            )
            if queued_attachments:
                remember_team_handoff_followup_attachments(
                    channel_id,
                    queued_attachments,
                )
            else:
                persist_ticket_state(channel_id)
            return
        followup_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=run_context.category_id,
            initial_button_intent=None,
            conversation_owner_id=run_context.conversation_owner_id,
        )
        pending_tasks[channel_id] = asyncio.create_task(
            self.process_ticket_message(channel_id, followup_context)
        )

    async def _handle_ticket_staff_summon(
        self,
        message: discord.Message,
        run_context: BotRunContext,
        prompt_text: str,
        *,
        was_stopped: bool,
    ) -> None:
        channel_id = message.channel.id
        logging.info(
            "Processing staff-directed ySupport turn in ticket %s from %s; "
            "ticket_stopped=%s.",
            channel_id,
            message.author.name,
            was_stopped,
        )
        instruction_text = (
            "This is an explicit instruction from authorized Yearn support staff. "
            "Write the reply ySupport should send to the ticket user. Treat the "
            "current message and any system-labeled staff transcript entries as "
            "internal staff direction, not as claims or requests from the ticket "
            "user. Use the ticket context, do not expose the internal instruction, "
            "and do not request a human or Telegram handoff."
        )
        if was_stopped:
            instruction_text += (
                " This ticket remains under manual staff control after this one "
                "reply. Give a complete answer and do not ask the user to reply or "
                "promise unattended follow-up from the bot."
            )

        current_task = asyncio.current_task()
        turn_result = None
        try:
            current_history = await _build_staff_summon_history(
                message.channel,
                exclude_message_id=message.id,
                ticket_owner_user_id=ticket_owner_user_id_by_channel.get(channel_id),
                bot_user_id=self.user.id if self.user is not None else None,
            )
            async with message.channel.typing():
                turn_result = await _run_internal_instruction_turn(
                    executor=self.investigation_executor,
                    channel=message.channel,
                    channel_id=channel_id,
                    run_context=run_context,
                    prompt_text=prompt_text,
                    instruction_text=instruction_text,
                    workflow_suffix="staff summon",
                    attachments=_attachment_payloads_from_message(message),
                    current_history_override=current_history,
                )
            await send_long_message(
                message.channel,
                turn_result.reply,
                view=StopBotView() if not was_stopped else None,
            )
            last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
            if not was_stopped:
                conversation_threads[channel_id] = turn_result.conversation_history
                persist_ticket_state(channel_id)
        except asyncio.CancelledError:
            logging.info(
                "Staff-directed ySupport turn in ticket %s was cancelled.",
                channel_id,
            )
            raise
        except Exception as exc:
            if turn_result is not None:
                reset_ticket_codex_session(channel_id)
            logging.error(
                "Staff-directed ySupport turn failed in ticket %s: %s",
                channel_id,
                exc,
                exc_info=True,
            )
            failure_reply = "I couldn't complete that ySupport request."
            if was_stopped:
                failure_reply += " The ticket remains under manual staff control."
            try:
                await send_long_message(
                    message.channel,
                    failure_reply,
                    view=StopBotView() if not was_stopped else None,
                )
            except Exception:
                logging.warning(
                    "Could not send staff-directed ySupport failure reply in "
                    "ticket %s.",
                    channel_id,
                    exc_info=True,
                )
        finally:
            if was_stopped and not stop_ticket_channel(channel_id):
                logging.error(
                    "Could not preserve stopped state after staff-directed turn "
                    "in ticket %s.",
                    channel_id,
                )
            self._finish_ticket_task(
                channel_id=channel_id,
                current_task=current_task,
                run_context=run_context,
            )

    async def _build_ticket_turn_input(
        self,
        *,
        channel: discord.TextChannel,
        channel_id: int,
        run_context: BotRunContext,
        investigation_job,
        aggregated_text: str,
    ) -> tuple[List[TResponseInputItem], List[TResponseInputItem]]:
        preparation = await prepare_ticket_turn_input(
            channel_id=channel_id,
            run_context=run_context,
            investigation_job=investigation_job,
            aggregated_text=aggregated_text,
        )
        for ack_message in preparation.ack_messages:
            try:
                await channel.send(ack_message, suppress_embeds=True)
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                logging.info(
                    "Sent ticket intake acknowledgement message to channel %s",
                    channel_id,
                )
            except Exception as exc:
                logging.warning(
                    "Failed to send ticket intake acknowledgement message: %s",
                    exc,
                )

        current_history = conversation_threads.get(channel_id, [])
        input_list: List[TResponseInputItem] = current_history + [
            {"role": "user", "content": preparation.current_user_content}
        ]

        if preparation.system_hints:
            input_list = input_list[:-1] + [
                {"role": "system", "content": " ".join(preparation.system_hints)}
            ] + [input_list[-1]]

        contextual_hints = build_contextual_hints(
            investigation_job,
            aggregated_text,
            current_history=current_history,
        )
        if contextual_hints:
            input_list = input_list[:-1] + [{"role": "system", "content": " ".join(contextual_hints)}] + [input_list[-1]]

        return current_history, input_list

    async def on_message(self, message: discord.Message):
        if message.author.bot or (self.user is not None and message.author.id == self.user.id):
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
        ticket_owner_user_id = ticket_owner_user_id_by_channel.get(channel_id)
        if ticket_owner_user_id is None and isinstance(message.channel, discord.TextChannel):
            ticket_owner_user_id = await _detect_ticket_owner_user_id(message.channel)
            if ticket_owner_user_id is not None:
                remember_ticket_owner_user_id(channel_id, ticket_owner_user_id)
                logging.info(
                    "Recovered ticket owner %s for channel %s from opener history during message processing.",
                    ticket_owner_user_id,
                    channel_id,
                )
        _maybe_recover_runtime_stopped_ticket_for_message(
            channel_id=channel_id,
            author=message.author,
            ticket_owner_user_id=ticket_owner_user_id,
        )
        action = _classify_ticket_message_action(
            author=message.author,
            content=message.content,
            ticket_owner_user_id=ticket_owner_user_id,
            stopped=channel_id in stopped_channels,
        )

        if action == "ignore":
            logging.info(
                "Ignoring ticket message in %s from %s. owner=%s stopped=%s staff=%s",
                channel_id,
                message.author.name,
                ticket_owner_user_id,
                channel_id in stopped_channels,
                _is_support_staff_member(message.author),
            )
            return

        if action == "staff_takeover":
            stopped_durably = await stop_ticket_for_manual_support(channel_id)
            logging.info(
                "Support staff took manual control of ticket %s; persisted=%s.",
                channel_id,
                stopped_durably,
            )
            return

        if action == "staff_summon_usage":
            try:
                await message.channel.send(
                    "Add an instruction after `y:` to ask ySupport to reply.",
                    delete_after=20,
                    suppress_embeds=True,
                )
            except Exception:
                logging.warning(
                    "Could not send staff summon usage help in ticket %s.",
                    channel_id,
                )
            return

        if action == "staff_summon":
            summon_prompt = _normalize_staff_summon_prompt(message.content)
            if summon_prompt is None:
                return
            was_stopped = channel_id in stopped_channels
            investigation_job = get_or_create_ticket_investigation_job(channel_id)
            summon_run_context = _build_ticket_run_context(
                channel_id=channel_id,
                category_id=message.channel.category.id,
                initial_button_intent=investigation_job.requested_intent,
                conversation_owner_id=ticket_owner_user_id,
            )
            existing_task = pending_tasks.get(channel_id)
            if existing_task is not None and not existing_task.done():
                existing_task.cancel()
            _discard_pending_ticket_payload(channel_id)
            summon_task = asyncio.create_task(
                self._handle_ticket_staff_summon(
                    message,
                    summon_run_context,
                    summon_prompt,
                    was_stopped=was_stopped,
                )
            )
            pending_tasks[channel_id] = summon_task
            return

        if ticket_owner_user_id is None and not _is_support_staff_member(message.author):
            ticket_owner_user_id = message.author.id
            remember_ticket_owner_user_id(channel_id, ticket_owner_user_id)
            logging.info(
                "Assigned fallback ticket owner %s for channel %s from first "
                "non-staff message.",
                ticket_owner_user_id,
                channel_id,
            )

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
        if action == "process" and is_ticket_waiting_for_team(channel_id):
            _record_waiting_for_team_followup(channel_id, message)
            if _should_ack_waiting_for_team(channel_id):
                try:
                    await message.channel.send(
                        _waiting_for_team_reply(),
                        suppress_embeds=True,
                    )
                    last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                    persist_ticket_state(channel_id)
                except Exception:
                    pass
            return
        current_intent_from_map = _record_button_requested_intent(
            channel_id=channel_id,
            investigation_job=investigation_job,
        )
        ticket_run_context = _build_ticket_run_context(
            channel_id=channel_id,
            category_id=message.channel.category.id,
            initial_button_intent=current_intent_from_map,
            conversation_owner_id=ticket_owner_user_id,
        )

        if ticket_run_context.project_context == "unknown":
            return
        logging.info(f"Processing ticket message in {channel_id} from {message.author.name} (Context: {ticket_run_context.project_context}, Intent: {current_intent_from_map})")

        normalized_message_text = _message_text_for_turn(message)
        message_attachments = _attachment_payloads_from_message(message)
        existing_task = pending_tasks.get(channel_id)
        interrupted_active_executor = False
        if existing_task and not existing_task.done():
            if (
                channel_id not in active_ticket_payloads
                and active_ticket_executor_tasks.get(channel_id) is not existing_task
            ):
                _merge_pending_ticket_payload(
                    channel_id,
                    normalized_message_text,
                    message_attachments,
                )
                return
            interrupted_active_executor = (
                active_ticket_executor_tasks.get(channel_id) is existing_task
            )
            _restore_active_ticket_payload(channel_id)
            existing_task.cancel()

        _merge_pending_ticket_payload(
            channel_id,
            normalized_message_text,
            message_attachments,
        )

        pending_tasks[channel_id] = asyncio.create_task(self.process_ticket_message(channel_id, ticket_run_context))
        if interrupted_active_executor:
            try:
                await message.channel.send(
                    "Got it. I’ve added your follow-up to your previous request for context, "
                    "and I’m continuing to work on it now. Please wait for my response. "
                    "There’s no need to resend anything.",
                    suppress_embeds=True,
                )
                last_bot_reply_ts_by_channel[channel_id] = datetime.now(
                    timezone.utc
                )
            except Exception:
                pass
        debounce_seconds = _ticket_debounce_seconds(channel_id, ticket_run_context)
        logging.debug(f"Scheduled processing task for channel {channel_id} in {debounce_seconds}s")

    async def process_ticket_message(
        self,
        channel_id: int,
        run_context: BotRunContext,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            investigation_job = get_or_create_ticket_investigation_job(channel_id)
            aggregated_text, attachments = await self._collect_aggregated_ticket_payload(
                channel_id,
                run_context,
            )
            if not aggregated_text:
                return
            merge_explicit_evidence(investigation_job, aggregated_text)

            channel = self.get_channel(channel_id)
            if not isinstance(channel, discord.TextChannel):
                return
            guild_id = getattr(getattr(channel, "guild", None), "id", None)

            current_history, input_list = await self._build_ticket_turn_input(
                channel=channel,
                channel_id=channel_id,
                run_context=run_context,
                investigation_job=investigation_job,
                aggregated_text=aggregated_text,
            )

            logging.info(f"Processing for ticket {channel_id} (Context: {run_context.project_context}, Initial Button Intent: {run_context.initial_button_intent}): '{aggregated_text[:100]}...'")

            async with channel.typing():
                progress_reporter = _DiscordProgressReporter(channel, channel_id)
                final_reply = "An unexpected error occurred."
                should_stop_processing = False
                stop_reason = None
                proposed_job: TicketInvestigationJob | None = None
                proposed_history: List[TResponseInputItem] | None = None
                handoff_state_committed = False

                boundary_output = await _outer_support_boundary_result(aggregated_text)
                boundary_reply = _boundary_reply_from_output(boundary_output)
                if boundary_reply is not None:
                    final_reply = boundary_reply
                    should_stop_processing = _should_stop_for_boundary_output(
                        boundary_output
                    )
                    if should_stop_processing:
                        stop_reason = "boundary_stop"
                else:
                    try:
                        worker_result = await _execute_ticket_turn(
                            executor=self.investigation_executor,
                            channel_id=channel_id,
                            request=_build_turn_request(
                                aggregated_text=aggregated_text,
                                input_list=input_list,
                                current_history=current_history,
                                attachments=attachments,
                                run_context=run_context,
                                investigation_job=investigation_job,
                                workflow_name=_ticket_workflow_name(run_context),
                                precomputed_boundary=boundary_output,
                            ),
                            hooks=TicketExecutionHooks(
                                send_progress_update=progress_reporter.update,
                            ),
                        )
                        active_payload = active_ticket_payloads.get(channel_id)
                        if active_payload is not None and active_payload[0] is current_task:
                            active_ticket_payloads.pop(channel_id, None)
                        flow_outcome = worker_result.flow_outcome
                        proposed_job = worker_result.updated_job
                        proposed_history = flow_outcome.conversation_history

                        raw_final_reply = flow_outcome.raw_final_reply
                        if flow_outcome.requires_human_handoff:
                            handoff_reason = (
                                flow_outcome.handoff_reason
                                or "manual follow-up needed"
                            )
                            notice = await _send_ticket_handoff_notice(
                                reason=handoff_reason,
                                summary=aggregated_text,
                                channel_id=channel_id,
                                guild_id=guild_id,
                                investigation_job=proposed_job,
                            )
                            handoff_sent = notice is not None
                            if handoff_sent:
                                proposed_job.mark_escalated_to_human()
                                investigation_job.apply_snapshot(proposed_job)
                                conversation_threads[channel_id] = proposed_history
                                _remember_sent_handoff_notice(
                                    channel_id=channel_id,
                                    reason=handoff_reason,
                                    notice=notice,
                                )
                                handoff_state_committed = True
                            else:
                                proposed_job.mark_waiting_for_user()
                            final_reply = (
                                build_user_handoff_reply(raw_final_reply)
                                if handoff_sent
                                else _handoff_delivery_failure_reply(raw_final_reply)
                            )
                        else:
                            final_reply = _render_support_reply(raw_final_reply)
                        if flow_outcome.requires_human_handoff:
                            logging.info(
                                "Support turn requested a human handoff for channel %s. "
                                "Leaving channel active for follow-up.",
                                channel_id,
                            )
                    except InputGuardrailTripwireTriggered as e:
                        logging.warning(f"Input Guardrail triggered in channel {channel_id}. Extracting message from output_info.")
                        final_reply = _guardrail_tripwire_reply(e)
                        guardrail_info = getattr(
                            getattr(getattr(e, "guardrail_result", None), "output", None),
                            "output_info",
                            None,
                        )
                        should_stop_processing = _should_stop_for_boundary_output(
                            guardrail_info if isinstance(guardrail_info, dict) else None
                        )
                        if should_stop_processing:
                            stop_reason = "boundary_stop"
                    except MaxTurnsExceeded:
                        logging.warning(f"Max turns ({config.MAX_TICKET_CONVERSATION_TURNS}) exceeded in channel {channel_id}.")
                        if run_context.repo_search_calls:
                            final_reply = (
                                "I hit an internal analysis limit while reviewing repo evidence for this report."
                            )
                        else:
                            final_reply = (
                                "This conversation has reached its maximum length."
                            )
                        should_stop_processing = True
                        stop_reason = "runtime_error"
                    except AgentsException as e:
                        logging.error(f"Agent SDK error during ticket processing for channel {channel_id}: {e}")
                        final_reply = (
                            f"Sorry, an error occurred while processing the request ({type(e).__name__}). Please try again."
                        )
                        should_stop_processing = True
                        stop_reason = "runtime_error"
                    except Exception as e:
                        logging.error(f"Unexpected error during ticket processing for channel {channel_id}: {e}", exc_info=True)
                        final_reply = "An unexpected error occurred."
                        should_stop_processing = True
                        stop_reason = "runtime_error"
                    finally:
                        await progress_reporter.close()

                active_payload = active_ticket_payloads.get(channel_id)
                if active_payload is not None and active_payload[0] is current_task:
                    active_ticket_payloads.pop(channel_id, None)

                try:
                    reply_view = StopBotView() if not should_stop_processing else None
                    await send_long_message(channel, final_reply, view=reply_view)
                    if proposed_job is not None and not handoff_state_committed:
                        investigation_job.apply_snapshot(proposed_job)
                        if proposed_history is not None:
                            conversation_threads[channel_id] = proposed_history
                    if stop_reason == "boundary_stop":
                        reset_ticket_channel_for_terminal_reply(channel_id)
                    if channel_id in bug_report_debounce_channels:
                        bug_report_debounce_channels.discard(channel_id)
                        logging.info("Cleared bug-report debounce flag for channel %s", channel_id)
                    last_bot_reply_ts_by_channel[channel_id] = datetime.now(timezone.utc)
                    logging.info(f"Sent ticket reply/replies in channel {channel_id}. Stop processing flag: {should_stop_processing}")
                    if should_stop_processing and channel_id not in stopped_channels:
                        mark_ticket_channel_stopped(
                            channel_id,
                            reason=stop_reason or "runtime_error",
                        )
                        logging.info(f"Added channel {channel_id} to stopped channels due to error/handoff tag.")
                    elif not should_stop_processing:
                        persist_ticket_state(channel_id)
                except discord.Forbidden:
                    logging.error(f"Missing permissions to send message in channel {channel_id}")
                    if not handoff_state_committed:
                        reset_ticket_codex_session(channel_id)
                        mark_ticket_channel_stopped(
                            channel_id,
                            reason="runtime_error",
                        )
                except Exception as e:
                    logging.error(f"Unexpected error occurred during or after calling send_long_message for channel {channel_id}: {e}", exc_info=True)
                    if not handoff_state_committed:
                        reset_ticket_codex_session(channel_id)
                        mark_ticket_channel_stopped(
                            channel_id,
                            reason="runtime_error",
                        )
                    try:
                        await channel.send(
                            (
                                "Your request reached the support team, but I couldn't "
                                "deliver my complete response here. You can add more "
                                "details while you wait."
                                if handoff_state_committed
                                else (
                                    "I couldn't deliver my complete response. Please send "
                                    "a new message to retry."
                                )
                            ),
                            suppress_embeds=True,
                        )
                    except Exception:
                        logging.warning(
                            "Could not send delivery-failure notice in ticket %s.",
                            channel_id,
                            exc_info=True,
                        )
        except asyncio.CancelledError:
            logging.info(f"Processing task for channel {channel_id} cancelled mid-run.")
            return
        finally:
            active_payload = active_ticket_payloads.get(channel_id)
            if active_payload is not None and active_payload[0] is current_task:
                active_ticket_payloads.pop(channel_id, None)
            self._finish_ticket_task(
                channel_id=channel_id,
                current_task=current_task,
                run_context=run_context,
            )

    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel):
        await clear_deleted_ticket_channel(channel)


def _build_discord_intents() -> discord.Intents:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.messages = True
    return intents


def _reload_runtime_env_and_config() -> None:
    load_dotenv(config.BASE_DIR / ".env", override=True)
    importlib.reload(config)


def _run_ticket_bot_once() -> None:
    config.validate_runtime_environment_config()
    config.validate_ticket_execution_runtime_config()
    client = TicketBot(intents=_build_discord_intents())
    client.run(config.DISCORD_BOT_TOKEN, log_handler=None)


def _run_ticket_bot_with_fatal_startup_backoff() -> None:
    while True:
        _reload_runtime_env_and_config()
        try:
            _run_ticket_bot_once()
            return
        except (
            discord.errors.LoginFailure,
            discord.errors.PrivilegedIntentsRequired,
        ):
            backoff_seconds = max(
                float(config.DISCORD_FATAL_STARTUP_BACKOFF_SECONDS or 0.0),
                60.0,
            )
            logging.critical(
                "Fatal Discord startup failure. Backing off for %.0f seconds before retrying.",
                backoff_seconds,
                exc_info=True,
            )
            time.sleep(backoff_seconds)


# Run the Bot
if __name__ == "__main__":
    _run_ticket_bot_with_fatal_startup_backoff()
