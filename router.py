import re
from typing import Literal


AgentKey = Literal["data", "docs", "bug", "triage"]


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


def is_probable_wallet_address(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    if not is_message_primarily_address(q):
        return False
    vault_hints = ["vault", "strategy", "contract", "vault address", "vault addr"]
    wallet_hints = ["wallet", "my address", "my wallet", "my account", "for my wallet", "my acct"]
    if any(hint in q for hint in vault_hints):
        return False
    return any(hint in q for hint in wallet_hints)


def is_wallet_confirmation(text: str) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-z']+", " ", text.lower()).strip()
    return normalized in {
        "yes",
        "yep",
        "yeah",
        "yup",
        "correct",
        "confirm",
        "confirmed",
        "that's my wallet",
        "that is my wallet",
        "use that",
        "use this",
        "use it",
        "that's right",
        "that is right",
        "yes that's correct",
        "yes that is correct",
        "yes use that",
        "yes use this",
        "yes use it",
    }


def is_wallet_rejection(text: str) -> bool:
    if not text:
        return False
    q = text.lower().strip()
    exact_rejections = {"no", "nope", "nah"}
    if q in exact_rejections:
        return True
    phrase_rejections = [
        "not my wallet",
        "that's not it",
        "that is not it",
        "wrong address",
    ]
    return any(phrase in q for phrase in phrase_rejections)


def select_starting_agent(text: str, run_context) -> AgentKey:
    intent = run_context.initial_button_intent
    if intent in ["data_deposit_check", "data_withdrawal_flow_start", "data_vault_search", "data_deposits_withdrawals_start"]:
        return "data"
    if intent == "docs_qa":
        return "docs"
    if intent == "bug_report":
        return "bug"
    if intent == "investigate_issue":
        return "triage"
    # Free-form messages should stay in triage unless the runtime already has a
    # stronger structured reason to start elsewhere. That keeps lane selection
    # owned by the triage/router agents instead of duplicated keyword policy
    # here in the wrapper.
    return "triage"
