from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Literal

from agents import TResponseInputItem


# State & Context Definitions
@dataclass
class BotRunContext:
    channel_id: int
    category_id: Optional[int] = None
    is_public_trigger: bool = False
    project_context: Literal["yearn", "unknown"] = "unknown"
    initial_button_intent: Optional[str] = None


# State Management for Public Conversations
@dataclass
class PublicConversation:
    """Stores the state for a temporary public conversation."""
    history: List[TResponseInputItem]
    last_interaction_time: datetime


# Globals
stopped_channels: set[int] = set()
conversation_threads: Dict[int, List[TResponseInputItem]] = {}
pending_messages: Dict[int, str] = {}
pending_tasks: Dict[int, asyncio.Task] = {}
monitored_new_channels: set[int] = set()

channels_awaiting_initial_button_press: set[int] = set()
channel_intent_after_button: Dict[int, str] = {}

public_conversations: Dict[int, PublicConversation] = {}

# Last known wallet address per channel (checksummed)
last_wallet_by_channel: Dict[int, str] = {}

# Pending confirmation for a new wallet address per channel
pending_wallet_confirmation_by_channel: Dict[int, str] = {}

# Timestamp of the last bot reply per channel
last_bot_reply_ts_by_channel: Dict[int, datetime] = {}
