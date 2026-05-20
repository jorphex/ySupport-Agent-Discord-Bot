from dataclasses import dataclass, field

from router import (
    is_message_primarily_address,
    is_probable_wallet_address,
    is_wallet_confirmation,
    is_wallet_rejection,
)
from state import (
    BotRunContext,
    TicketInvestigationJob,
    last_wallet_by_channel,
    pending_wallet_confirmation_by_channel,
)
import tools_lib


@dataclass
class TicketTurnInputPreparation:
    current_user_content: str
    system_hints: list[str] = field(default_factory=list)
    ack_messages: list[str] = field(default_factory=list)


def canonicalize_current_user_message(
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
    if is_message_primarily_address(aggregated_text) and is_probable_wallet_address(
        aggregated_text
    ):
        return resolved_wallet
    return aggregated_text


def is_wallet_first_data_intent(intent: str | None) -> bool:
    return intent in {
        "data_deposit_check",
        "data_withdrawal_flow_start",
        "data_deposits_withdrawals_start",
    }


def known_yearn_address_hint(resolved_target) -> str:
    kind = getattr(resolved_target, "kind", "unknown")
    label = getattr(resolved_target, "label", None) or "Known Yearn object"
    chain = getattr(resolved_target, "chain", None)
    chain_text = f" on {chain}" if chain else ""
    if kind == "vault":
        return (
            f"Resolved address is a known Yearn vault{chain_text}: {label}. "
            "Treat it as a vault/product target, not as the user's wallet. "
            "Do not switch into wallet deposit lookup unless the user separately provides a wallet address."
        )
    if kind == "strategy":
        vault_label = getattr(resolved_target, "vault_label", None)
        context = f" attached to {vault_label}" if vault_label else ""
        return (
            f"Resolved address is a known Yearn strategy{chain_text}: {label}{context}. "
            "Treat it as a strategy target, not as the user's wallet."
        )
    vault_label = getattr(resolved_target, "vault_label", None)
    context = f" for {vault_label}" if vault_label else ""
    return (
        f"Resolved address is a known Yearn staking wrapper or gauge{chain_text}: {label}{context}. "
        "Treat it as a staking/auxiliary contract target, not as the user's wallet."
    )


def known_yearn_address_acknowledgement(resolved_target) -> str:
    label = getattr(resolved_target, "label", None) or "that address"
    kind = getattr(resolved_target, "kind", "unknown")
    if kind == "vault":
        return f"Thank you. I've identified `{label}` as a Yearn vault and am checking it now. This may take a moment..."
    if kind == "strategy":
        return f"Thank you. I've identified `{label}` as a Yearn strategy and am checking it now. This may take a moment..."
    return f"Thank you. I've identified `{label}` as a Yearn staking contract and am checking it now. This may take a moment..."


def vault_url_target_hint(chain: str, vault_address: str) -> str:
    return (
        f"User referenced a Yearn vault page URL for chain {chain} and address {vault_address}. "
        "Treat the linked address as a vault/product target, not as the user's wallet."
    )


async def prepare_ticket_turn_input(
    *,
    channel_id: int,
    run_context: BotRunContext,
    investigation_job: TicketInvestigationJob,
    aggregated_text: str,
) -> TicketTurnInputPreparation:
    intent = run_context.initial_button_intent
    is_data_intent = intent in {
        "data_deposit_check",
        "data_withdrawal_flow_start",
        "data_vault_search",
        "data_deposits_withdrawals_start",
    }
    wallet_first_data_intent = is_wallet_first_data_intent(intent)
    preparation = TicketTurnInputPreparation(current_user_content=aggregated_text)

    known_tx_hashes = investigation_job.evidence.tx_hashes
    tx_hash_present_in_message = bool(
        known_tx_hashes and any(tx_hash in aggregated_text for tx_hash in known_tx_hashes)
    )
    vault_url_target = tools_lib.extract_yearn_vault_url_target(aggregated_text)

    if vault_url_target is not None:
        vault_url_chain, vault_url_address = vault_url_target
        investigation_job.remember_chain(vault_url_chain)
        preparation.system_hints.append(
            vault_url_target_hint(vault_url_chain, vault_url_address)
        )
    else:
        vault_url_address = None

    if tx_hash_present_in_message:
        preparation.system_hints.append(
            "User provided a transaction hash in this message. Investigate the transaction directly before falling back to wallet-position lookup. "
            "Do not replace the user's message with a bare address or assume any included address is their wallet unless they clearly identify it as such."
        )

    if is_data_intent:
        pending_wallet_confirmation_by_channel.pop(channel_id, None)

    pending_wallet = pending_wallet_confirmation_by_channel.get(channel_id)
    if pending_wallet:
        if is_wallet_confirmation(aggregated_text):
            last_wallet_by_channel[channel_id] = pending_wallet
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            preparation.system_hints.append(
                f"User confirmed their wallet address is {pending_wallet}. Use this address going forward."
            )
        elif is_wallet_rejection(aggregated_text):
            pending_wallet_confirmation_by_channel.pop(channel_id, None)
            preparation.system_hints.append(
                "User rejected the suggested wallet address. Ask for the correct wallet address if needed."
            )

    extracted_address_or_ens = None
    if vault_url_address:
        extracted_address_or_ens = vault_url_address
    elif is_data_intent:
        extracted_address_or_ens = (
            tools_lib.extract_address_or_ens(aggregated_text) or aggregated_text
        )
    elif is_message_primarily_address(aggregated_text) and is_probable_wallet_address(
        aggregated_text
    ):
        extracted_address_or_ens = aggregated_text

    if not extracted_address_or_ens:
        return preparation

    parsed_address = tools_lib.resolve_ens(extracted_address_or_ens)
    if not parsed_address:
        return preparation

    resolved_target = await tools_lib.resolve_yearn_address_target(
        parsed_address,
        chain_hint=investigation_job.evidence.chain,
    )
    resolved_kind = getattr(resolved_target, "kind", "unknown")
    last_wallet = last_wallet_by_channel.get(channel_id)

    if resolved_kind in {"vault", "strategy", "wrapper_or_gauge"}:
        preparation.system_hints.append(known_yearn_address_hint(resolved_target))
        if is_data_intent:
            preparation.ack_messages.append(
                known_yearn_address_acknowledgement(resolved_target)
            )
        return preparation

    if is_data_intent and resolved_kind == "contract_unknown":
        preparation.system_hints.append(
            "Resolved address has contract code but is not a known Yearn vault, strategy, or staking wrapper on the current chain. "
            "Do not assume it is the user's wallet. If wallet ownership changes the next step, ask whether this is the user's wallet address or a target contract address."
        )
        return preparation

    if is_data_intent and wallet_first_data_intent and not tx_hash_present_in_message:
        last_wallet_by_channel[channel_id] = parsed_address
        investigation_job.remember_wallet(parsed_address)
        preparation.current_user_content = canonicalize_current_user_message(
            aggregated_text,
            run_context,
            parsed_address,
        )
        preparation.system_hints.append(
            f"Resolved wallet address for this turn: {parsed_address}. Use this exact address for any wallet-based tool call."
        )
        preparation.ack_messages.append(
            f"Thank you. I've received the address `{parsed_address}` and am looking up the information now. This may take a moment..."
        )
        return preparation

    if resolved_kind == "eoa_or_missing_code" and last_wallet and parsed_address != last_wallet:
        pending_wallet_confirmation_by_channel[channel_id] = parsed_address
        preparation.system_hints.append(
            f"User provided a new address {parsed_address} but previous wallet was {last_wallet}. "
            "Ask to confirm whether the new address is their wallet before replacing."
        )
        return preparation

    if resolved_kind == "eoa_or_missing_code" and not tx_hash_present_in_message:
        last_wallet_by_channel[channel_id] = parsed_address
        investigation_job.remember_wallet(parsed_address)
        preparation.current_user_content = canonicalize_current_user_message(
            aggregated_text,
            run_context,
            parsed_address,
        )

    return preparation
