import re
from urllib.parse import urlsplit

from agents import TResponseInputItem

from ysupport import config
from ysupport.bot.state import TicketInvestigationJob
from ysupport.tools.yearn_targets import extract_yearn_vault_url_target


TX_HASH_RE = re.compile(r"\b0x[a-fA-F0-9]{64}\b")
_TX_CORRECTION = re.compile(
    rf"\buse\s+`?({TX_HASH_RE.pattern})`?\s+instead\b|"
    rf"\b(?:instead\s+use\s+|(?:right|correct) (?:tx(?: hash)?|transaction(?: hash)?|hash)\s*(?:is\b|:)\s*)"
    rf"`?({TX_HASH_RE.pattern})", re.I,
)
CHAIN_NAMES = tuple(config.CHAIN_NAME_TO_ID)
_CHAIN_PATTERN = "|".join(map(re.escape, CHAIN_NAMES))
CHAIN_NAME_RE = re.compile(rf"\b({_CHAIN_PATTERN})\b", re.I)
_CHAIN_CORRECTION = re.compile(
    r"(?:^|[.!?\n])[ \t]*(?:correction|actually)\b[ \t]*[:,]?[ \t]*"
    r"(?:(?:the[ \t]+)?(?:chain|network)[ \t]*(?::|(?:is|was)\b)[ \t]*|"
    r"it(?:['’]s|[ \t]+(?:is|was))[ \t]+(?:on[ \t]+)?|on[ \t]+)?"
    rf"({_CHAIN_PATTERN})\b(?=[ \t]*(?:[,.!?;\n—–-]|$))",
    re.I,
)
_EXPLORER_CHAINS = {urlsplit(url).netloc: chain for chain, url in config.BLOCK_EXPLORER_URLS.items()}
_HTTPS_URL = re.compile(r"https://[^\s<>\[\]()\"'`]+", re.I)
_EXPLORER_TX_PATH = re.compile(rf"/tx/({TX_HASH_RE.pattern})/?", re.I)
_TX_ON_CHAIN = re.compile(rf"({TX_HASH_RE.pattern})`?[ \t]+on[ \t]+({_CHAIN_PATTERN})\b", re.I)
_TX_TOKEN = rf"`?{TX_HASH_RE.pattern}`?"
_CHAIN_TX_LABEL = re.compile(
    rf"\b({_CHAIN_PATTERN})(?:[ \t]+(?:source|destination|outgoing|incoming))?"
    rf"(?:[ \t]+(?:txs?|transactions?)(?:[ \t]+hash(?:es)?)?)?[ \t]*:[ \t]*"
    rf"({_TX_TOKEN}(?:[ \t]*(?:,[ \t]*(?:and[ \t]+)?|and[ \t]+){_TX_TOKEN})*)",
    re.I,
)


def contains_tx_hash(text: str) -> bool:
    return bool(TX_HASH_RE.search(text))


def corrected_tx_hash(text: str) -> str | None:
    matches = list(_TX_CORRECTION.finditer(text))
    if matches:
        replacement = matches[-1]
        return (replacement[1] or replacement[2]).lower()
    return None


def corrected_transaction_chain(text: str) -> str | None:
    matches = _CHAIN_CORRECTION.findall(text)
    if not matches or (not contains_tx_hash(text) and extract_yearn_vault_url_target(text)):
        return None
    return matches[-1].lower()


def _transaction_chains(text: str) -> dict[str, str | None]:
    candidates: dict[str, set[str]] = {}

    def parse_link(match: re.Match) -> str:
        try:
            url = urlsplit(match[0].rstrip(".,;"))
        except ValueError:
            return " "
        chain = _EXPLORER_CHAINS.get(url.netloc.lower().removeprefix("www."))
        tx = _EXPLORER_TX_PATH.fullmatch(url.path)
        if chain and tx:
            tx_hash = tx[1].lower()
            candidates.setdefault(tx_hash, set()).add(chain)
            return tx_hash
        return " "

    prose = _HTTPS_URL.sub(parse_link, text)
    for tx_hash, chain in _TX_ON_CHAIN.findall(prose):
        candidates.setdefault(tx_hash.lower(), set()).add(chain.lower())
    for chain, hashes in _CHAIN_TX_LABEL.findall(prose):
        for tx_hash in TX_HASH_RE.findall(hashes):
            candidates.setdefault(tx_hash.lower(), set()).add(chain.lower())
    return {tx: next(iter(chains)) if len(chains) == 1 else None for tx, chains in candidates.items()}


def merge_explicit_evidence(
    investigation_job: TicketInvestigationJob,
    text: str,
) -> None:
    if not text:
        return

    if corrected := corrected_tx_hash(text):
        investigation_job.evidence.withdrawal_tx_hash = corrected
    hashes = list(dict.fromkeys(tx.lower() for tx in TX_HASH_RE.findall(text)))
    for tx_hash in hashes:
        investigation_job.remember_tx_hash(tx_hash)
    evidence = investigation_job.evidence
    stated_chains = _transaction_chains(text)
    evidence.tx_chains.update(stated_chains)

    if chain := corrected_transaction_chain(text):
        investigation_job.remember_chain(chain)
        targets = [corrected] if corrected else hashes or evidence.tx_hashes
        if len(targets) == 1:
            target = targets[0]
            evidence.tx_chains[target] = chain if stated_chains.get(target, chain) == chain else None
    elif chain_match := CHAIN_NAME_RE.search(text):
        investigation_job.remember_chain(chain_match.group(1).lower())


def build_contextual_hints(
    investigation_job: TicketInvestigationJob,
    aggregated_text: str,
    current_history: list[TResponseInputItem] | None = None,
) -> list[str]:
    del current_history
    hints: list[str] = []
    lowered_text = aggregated_text.lower()
    known_wallet = investigation_job.evidence.wallet
    known_chain = investigation_job.evidence.chain
    known_tx_hashes = investigation_job.evidence.tx_hashes
    if known_tx_hashes:
        tx_chains = investigation_job.evidence.tx_chains
        references = ", ".join(f"{tx} ({tx_chains.get(tx) or 'chain unresolved'})" for tx in known_tx_hashes[-2:])
        hints.append(f"Recent ticket transactions: {references}. Use each transaction's own chain.")
    elif known_chain and known_chain not in lowered_text:
        hints.append(
            f"Known chain for this ticket is {known_chain}. "
            "Use that chain for continued investigation unless the user explicitly corrects it."
        )
    if known_tx_hashes and not any(
        tx_hash in aggregated_text for tx_hash in known_tx_hashes
    ):
        hints.append(
            "This is a follow-up to an existing transaction investigation. "
            "Continue the onchain investigation yourself and do not ask the user to choose between receipt decoding, log inspection, or contract calls."
        )
    wallet_hint_added = False
    if known_wallet and known_wallet not in aggregated_text:
        hints.append(
            f"Known wallet for this ticket is {known_wallet}. "
            "Use it if needed; do not ask again unless the user corrects it."
        )
        wallet_hint_added = True
    deposit_chain = investigation_job.evidence.withdrawal_target_chain
    deposit_vault = investigation_job.evidence.withdrawal_target_vault
    if deposit_chain and deposit_vault and _is_withdrawal_followup(aggregated_text):
        if known_wallet:
            if wallet_hint_added:
                hints.pop()
            hints.append(
                f"This withdrawal follow-up already has the needed details from earlier in the ticket: wallet {known_wallet}, "
                f"vault {deposit_vault}, and chain {deposit_chain}. Use those exact values for withdrawal instructions unless the user explicitly corrects them. "
                "Do not re-check deposits, do not ask which vault they mean, and do not default to ethereum."
            )
        else:
            hints.append(
                f"The current ticket already has a single withdrawal target: {deposit_vault} on {deposit_chain}. "
                "For this withdrawal follow-up, use that vault and chain unless the user explicitly corrects them. "
                "Do not re-check deposits or default to ethereum if the ticket already identified the target vault."
            )
    return hints


def _is_withdrawal_followup(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in lowered
        for keyword in ("withdraw", "withdrawing", "redeem", "redeeming")
    )
