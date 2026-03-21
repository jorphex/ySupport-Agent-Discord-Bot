import logging

import discord
from discord.ui import View, Button, button

import config
from bot_behavior import STANDARD_REDIRECT_MESSAGE
from state import (
    channels_awaiting_initial_button_press,
    channel_intent_after_button,
    bug_report_debounce_channels,
    clear_ticket_investigation_job,
    get_or_create_ticket_investigation_job,
    stopped_channels,
    pending_messages,
    pending_tasks,
)


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
        investigation_job = get_or_create_ticket_investigation_job(interaction.channel.id)
        investigation_job.record_requested_intent(intent_category)
        if intent_category == "investigate_issue":
            investigation_job.begin_collecting()
        else:
            investigation_job.mark_waiting_for_user()

        if interaction.channel:
            await interaction.channel.send(prompt_message)
        else:
            await interaction.followup.send(prompt_message, ephemeral=False)

        logging.info(f"Button '{button_custom_id}' clicked in {interaction.channel.id}. Intent set to '{intent_category}'. Sent follow-up prompt.")

    @button(label="💼 Deposits / Withdrawals", style=discord.ButtonStyle.secondary, custom_id="initial_deposits_withdrawals", row=0)
    async def deposits_withdrawals_button(self, interaction: discord.Interaction, button: Button):
        prompt = (
            "Got it. If you want deposit or withdrawal help, send your wallet address (0x...). "
            "If you are trying to find a vault or check vault info first, send the token, vault name, "
            "or vault address instead."
        )
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "data_deposits_withdrawals_start")

    @button(label="📖 General Info/How-To", style=discord.ButtonStyle.secondary, custom_id="initial_general_info", row=1)
    async def general_info_button(self, interaction: discord.Interaction, button: Button):
        project_ctx_name = "Yearn"
        if interaction.channel.category and interaction.channel.category.id in config.CATEGORY_CONTEXT_MAP:
            project_ctx_name = config.CATEGORY_CONTEXT_MAP[interaction.channel.category.id].capitalize()
        prompt = f"Great! What specific information or product are you looking for, or what how-to question do you have about {project_ctx_name}?"
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "docs_qa")

    @button(label="🔎 Investigate Issue", style=discord.ButtonStyle.secondary, custom_id="initial_bug_report", row=1)
    async def bug_report_button(self, interaction: discord.Interaction, button: Button):
        prompt = (
            "Describe the issue in as much detail as you have. You can send multiple messages now and I will review them together. "
            "Include the product or vault involved, the exact page or URL if relevant, what you expected to happen, what actually happened, "
            "any error text, any tx hash, and your browser/device or wallet if that matters."
        )
        await self.handle_button_click_and_prompt(interaction, button.custom_id, prompt, "investigate_issue")
        bug_report_debounce_channels.add(interaction.channel.id)
        logging.info(f"Issue investigation initiated in {interaction.channel.id}. Awaiting issue details.")

    @button(label="🤝 BD / Partnerships / Listings", style=discord.ButtonStyle.secondary, custom_id="initial_bd_partner", row=2)
    async def bd_partner_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        if interaction.channel:
            await interaction.channel.send(STANDARD_REDIRECT_MESSAGE)
        else:
            await interaction.followup.send(STANDARD_REDIRECT_MESSAGE, ephemeral=False)
        stopped_channels.add(interaction.channel.id)
        logging.info(f"BD/Partner inquiry redirected in {interaction.channel.id}. Bot stopped.")

    @button(label="❓ Other/My Issue Isn't Listed", style=discord.ButtonStyle.secondary, custom_id="initial_other_issue", row=2)
    async def other_issue_button(self, interaction: discord.Interaction, button: Button):
        await self.handle_button_click(interaction, button.custom_id)
        prompt = "Okay, please describe your issue or question in detail below. I'll do my best to assist or find the right help for you."
        if interaction.channel:
            await interaction.channel.send(prompt)
        else:
            await interaction.followup.send(prompt, ephemeral=False)
        channel_intent_after_button[interaction.channel.id] = "other_free_form"
        investigation_job = get_or_create_ticket_investigation_job(interaction.channel.id)
        investigation_job.record_requested_intent("other_free_form")
        investigation_job.mark_waiting_for_user()
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


class StopBotView(View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @button(label="Stop Bot", style=discord.ButtonStyle.secondary, custom_id="stop_bot_button")
    async def stop_button_callback(self, interaction: discord.Interaction, button: Button):
        channel_id = interaction.channel.id
        user_who_clicked = interaction.user

        logging.info(f"Stop Bot button clicked in channel {channel_id} by {user_who_clicked.name} ({user_who_clicked.id})")
        if not interaction.response.is_done():
            await interaction.response.defer()

        stopped_channels.add(channel_id)
        bug_report_debounce_channels.discard(channel_id)
        clear_ticket_investigation_job(channel_id)
        pending_messages.pop(channel_id, None)
        task = pending_tasks.pop(channel_id, None)
        if task:
            task.cancel()

        try:
            if interaction.message:
                disabled_view = View.from_message(interaction.message)
                if disabled_view:
                    for child_button in disabled_view.children:
                        if isinstance(child_button, Button):
                            child_button.disabled = True
                    await interaction.message.edit(view=disabled_view)
                    logging.info(f"Disabled Stop Bot button on message {interaction.message.id}")
        except discord.HTTPException as e:
            logging.warning(f"Could not disable Stop Bot button on message {interaction.message.id if interaction.message else 'Unknown'}: {e.status} {e.text}")
        except Exception as e:
            logging.error(f"Unexpected error disabling Stop Bot button on message {interaction.message.id if interaction.message else 'Unknown'}: {e}", exc_info=True)

        confirmation_message = "Support bot stopped for this channel. ySupport contributors are available for further inquiries."
        try:
            await interaction.followup.send(confirmation_message, ephemeral=False)
        except Exception:
            if interaction.channel:
                await interaction.channel.send(confirmation_message)
