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


def is_migration_issue_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    migration_issue_terms = [
        "can't migrate", "cant migrate", "cannot migrate", "unable to migrate",
        "not showing migrate", "missing migrate", "migration failed",
        "migrate card", "migration card", "cta missing", "no migrate",
        "stuck migrating", "migration error"
    ]
    return any(t in q for t in migration_issue_terms)


def is_bug_report_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    strong_terms = [
        "bug",
        "ui issue",
        "ui error",
        "exploit",
        "vulnerability",
        "security issue",
        "unexpected behavior",
        "reentrancy",
        "broken",
    ]
    failure_terms = [
        "not working",
        "doesn't work",
        "doesnt work",
        "failed",
        "fails",
        "failing",
        "stuck",
        "error",
        "wrong",
        "unable",
        "cannot",
        "can't",
        "cant",
        "missing",
    ]
    product_terms = [
        "styfi",
        "veyfi",
        "vault",
        "router",
        "strategy",
        "migration",
        "deposit",
        "withdraw",
        "app",
        "site",
        "page",
        "button",
    ]
    return any(term in q for term in strong_terms) or (
        any(term in q for term in failure_terms) and any(term in q for term in product_terms)
    )


def is_account_specific_veyfi_query(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    veyfi_terms = ["veyfi", "ve yfi"]
    account_terms = ["my", "mine", "balance", "eligibility", "eligible", "can you see", "can you check", "check", "see"]
    if not any(t in q for t in veyfi_terms):
        return False
    return any(t in q for t in account_terms)


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
    q = text.lower().strip()
    confirmations = [
        "yes", "yep", "yeah", "yup", "correct", "confirm", "confirmed",
        "that's my wallet", "that is my wallet", "use that", "use this",
        "use it", "that's right", "that is right"
    ]
    return any(phrase in q for phrase in confirmations)


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


def should_force_docs_route(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    if is_migration_issue_query(q):
        return False
    if is_bug_report_query(q):
        return False
    if is_account_specific_veyfi_query(q):
        return False
    docs_keywords = [
        "veyfi", "styfi", "dyfi", "yip", "governance", "staking"
    ]
    if any(k in q for k in docs_keywords):
        return True
    if "contract address" in q:
        data_keywords = [
            "vault", "vaults", "deposit", "withdraw", "apy", "tvl",
            "balance", "strategy", "strategies", "pps", "price per share", "yield"
        ]
        if not any(k in q for k in data_keywords):
            return True
    return False


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
    if should_force_docs_route(text):
        return "docs"
    return "triage"
