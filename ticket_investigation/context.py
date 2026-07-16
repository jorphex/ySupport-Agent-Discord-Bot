import re

from agents import TResponseInputItem

from state import TicketInvestigationJob


TX_HASH_RE = re.compile(r"\b0x[a-fA-F0-9]{64}\b")
CHAIN_NAMES = (
    "ethereum",
    "base",
    "arbitrum",
    "optimism",
    "polygon",
    "sonic",
    "katana",
)


def contains_tx_hash(text: str) -> bool:
    return bool(TX_HASH_RE.search(text))


def merge_explicit_evidence(
    investigation_job: TicketInvestigationJob,
    text: str,
) -> None:
    if not text:
        return

    for tx_hash in TX_HASH_RE.findall(text):
        investigation_job.remember_tx_hash(tx_hash)

    lowered = text.lower()
    for chain_name in CHAIN_NAMES:
        if chain_name in lowered:
            investigation_job.remember_chain(chain_name)
            break


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
    combined_chain_and_tx_hint_added = False
    if known_chain and known_tx_hashes:
        recent_hashes = ", ".join(known_tx_hashes[-2:])
        hints.append(
            f"For onchain investigation on this ticket, use chain '{known_chain}' and transaction hash(es) {recent_hashes}. "
            "Do not substitute a different chain or transaction unless the user explicitly corrects you."
        )
        combined_chain_and_tx_hint_added = True
    elif known_chain and known_chain not in lowered_text:
        hints.append(
            f"Known chain for this ticket is {known_chain}. "
            "Use that chain for continued investigation unless the user explicitly corrects it."
        )
    if (
        known_tx_hashes
        and not combined_chain_and_tx_hint_added
        and not any(tx_hash in aggregated_text for tx_hash in known_tx_hashes)
    ):
        recent_hashes = ", ".join(known_tx_hashes[-2:])
        hints.append(
            f"Known transaction hashes for this ticket: {recent_hashes}. "
            "Reuse them for continued onchain investigation unless the user provides a different transaction."
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
