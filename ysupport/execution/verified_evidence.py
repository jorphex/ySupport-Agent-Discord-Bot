"""Keep receipt and simulation claims tied to observations from this turn."""

from dataclasses import dataclass, replace
import re

from ysupport import config
from ysupport.execution.context import CHAIN_NAMES, TX_HASH_RE, corrected_tx_hash
from ysupport.tools.yearn_targets import extract_yearn_vault_url_target


@dataclass(frozen=True)
class VerifiedEvidence:
    kind: str
    chain: str
    wallet: str
    vault: str
    summary: str
    instructions: str | None = None
    tx_hash: str | None = None
    shares_raw: int | None = None
    max_shares_raw: int | None = None
    receiver: str | None = None


def _payouts(chain, data):
    if data["coverage"].get("truncated"):
        return
    wallet, vault = data["wallet"], data["token"]
    for tx in data["transactions"]:
        asset = tx.get("underlying_asset")
        if tx["status"] != "success" or not asset:
            continue
        tx_hash, block = tx["transaction_hash"], tx["block_number"]
        exits = [row for row in data["withdrawals"] if row["transaction_hash"] == tx_hash
                 and row["receiver"] == wallet and int(row["assets_raw"]) > 0]
        transfers = [row for row in data["asset_transfers_in"] if row["transaction_hash"] == tx_hash
                     and row["token"] == asset and row["from"] == vault and row["to"] == wallet
                     and int(row["amount_raw"]) > 0]
        # Different or multiple movements need separate reconciliation before claiming a payout.
        if len(exits) != 1 or len(transfers) != 1 or exits[0]["assets_raw"] != transfers[0]["amount_raw"]:
            continue
        amount = transfers[0]["amount_raw"]
        explorer = config.BLOCK_EXPLORER_URLS[chain]
        yield VerifiedEvidence("payout", chain, wallet, vault,
            f"Vault `{vault}` paid {amount} raw units of `{asset}` to `{wallet}` on {chain.capitalize()} "
            f"at block {block}. {explorer}/tx/{tx_hash}", tx_hash=tx_hash)


def _burns(chain, data):
    for row in data["transfers"]:
        if not row["is_burn"] or row["direction"] != "out" or int(row["amount_raw"]) <= 0:
            continue
        tx_hash, block = row["transaction_hash"], row["block_number"]
        yield VerifiedEvidence("burn", chain, data["wallet"], data["token"],
            f"The receipt shows a share burn at block {block} on {chain.capitalize()}. "
            f"{config.BLOCK_EXPLORER_URLS[chain]}/tx/{tx_hash}", tx_hash=tx_hash)


def evidence_from_read(operation, chain, data):
    if operation == "get_wallet_history":
        payouts = list(_payouts(chain, data))
        paid = {proof.tx_hash for proof in payouts}
        return payouts + [proof for proof in _burns(chain, data) if proof.tx_hash not in paid]
    if data.get("kind") != "v3_protected_redemption":
        return []
    if data["status"] == "no_shares":
        return [VerifiedEvidence("zero_shares", chain, data["wallet"], data["vault"],
            f"The wallet held zero vault shares at block {data['as_of_block']} on {chain.capitalize()}. "
            f"Share the relevant transaction hash to trace past holdings. "
            f"{data['explorer_url']}")]
    if data["status"] != "ready" or data.get("max_loss_basis_points") != 0:
        return []
    shares, amount = data["shares_to_redeem_raw"], data["simulated_assets_raw"]
    receiver = data.get("receiver", data["wallet"])
    max_shares = min(int(data["balance_shares_raw"]), int(data["max_redeem_zero_loss_raw"]))
    block = data["as_of_block"]
    scope = "partial redemption" if data["partial"] else "redemption"
    instructions = data["instructions"]
    if data["partial"]:
        instructions = "This simulation covers only part of your vault position.\n\n" + instructions
    return [VerifiedEvidence("redemption", chain, data["wallet"], data["vault"],
        f"Zero-loss {scope} simulation at block {block} on {chain.capitalize()}: "
        f"{shares} raw shares for {amount} raw underlying units to `{receiver}` with `max_loss=0`. "
        f"{data['explorer_url']}", instructions=instructions,
        shares_raw=int(shares), max_shares_raw=max_shares, receiver=receiver)]


_PAYOUT_QUESTION = re.compile(
    r"\b(?:did (?:i|it) (?:get paid|receive)\b[^.!?\n]{0,100}\b(?:withdrawal|redemption|payout)|"
    r"did it pay out|"
    r"(?:where (?:is|are)|what happened (?:to|with)) (?:my|this|the) (?:withdrawal|payout)"
    r"(?=\s*(?:[.!?\n]|$)|\s+(?:on|from|after)\b)|"
    r"(?:where (?:is|are)|what happened (?:to|with)) (?:my|this|the) funds\b[^.!?\n]{0,100}"
    r"\b(?:withdrawal|withdrew|redemption|redeemed|payout)|"
    r"did (?:my|this|the) (?:withdrawal|redemption|payout) (?:complete|finish|succeed|land|go through|pay out))\b", re.I,
)
_REDEEM_RECIPE = re.compile(
    r"\bredeem\s*\(\s*(?:0x[0-9a-f]+|\d+)[\d_]*\s*,|"
    r"\bwithdraw\s*\(\s*(?:0x[0-9a-f]+|\d[\d_]*)\s*,(?:[^,()]+,){2}[^,()]+\)|"
    r"\b_?(?:shares|assets)\b[\s`*]*(?:[:=]|\bto\b)[\s`*]*\d+.{0,220}\b_?max[_ ]?loss\b|"
    r"\b_?max[_ ]?loss\b.{0,220}\b_?(?:shares|assets)\b[\s`*]*(?:[:=]|\bto\b)[\s`*]*\d+|"
    r"\b(?:call|invoke|run|select|choose|use)\b[^.\n]{0,80}\bredeem\b.{0,240}"
    r"\b(?:set|enter|click|submit|maxRedeem|zero\s+max\s+loss|\d[\d,_]*\s+(?:raw\s+)?shares)\b|"
    r"\b(?:call|invoke|run|select|choose|use)\b[^.\n]{0,80}\bfour[- ](?:argument|input|parameter)\s+(?:redeem|withdraw)\b",
    re.I | re.S,
)
_WITHDRAWAL_RECEIPT = re.compile(
    r"\b(?:withdrew|redeemed|(?:withdrawal|redemption|payout) (?:transaction|tx|hash)|"
    r"after (?:my|the|this) (?:withdrawal|redemption))\b|"
    rf"\b(?:withdrawal|redemption|payout)\s*(?:(?:is|was)\s+|:\s*)[`(<]*{TX_HASH_RE.pattern}", re.I,
)
_LABELED_WALLET = re.compile(
    r"\bwallet(?:\s+address)?(?:\s+(?:is|at))?\s*[:=]?\s*[`(<]*(0x[0-9a-f]{40})\b", re.I,
)
_LABELED_VAULT = re.compile(r"\bvault(?:\s+address)?\s*[:=]?\s*[`(<]*(0x[0-9a-f]{40})\b", re.I)
_LABELED_CHAIN = re.compile(rf"\b(?:on|chain:)\s+({'|'.join(CHAIN_NAMES)})\b", re.I)
_PAIRED_RECEIPTS = re.compile(r"\b(?:both|(?:the|these|those)\s+(?:two|2))\b", re.I)
_RECEIPT_OUTCOME = re.compile(
    r"\b(?:did|has|have|was|were|is)\b[^.!?\n]{0,100}"
    r"\b(?:land(?:ed)?|arriv(?:e|ed)|reach(?:ed)?|pay|paid|receive(?:d)?|sent|transferred|show(?:ed)? up|go through)\b|"
    r"\bwhere (?:did|have) (?:my|the|these) (?:funds|tokens|assets|proceeds) (?:go|gone|end up)\b", re.I,
)
_PAYOUT_CLAIM = re.compile(
    r"\b(?:received|paid|credited|returned)\b[^.!?\n]{0,100}\b(?:withdrawal|redemption)\b|"
    r"\b(?:withdrawal|redemption)\b[^.!?\n]{0,100}\b(?:paid|credited|returned|reached)\b", re.I,
)
_RAW_SHARES = re.compile(r"(?<![\w.,+-])([0-9]+(?:_[0-9]+)*|[0-9]{1,3}(?:,[0-9]{3})+)\s+raw\s+shares\b", re.I)
_CUSTOM_AMOUNT = re.compile(
    r"\b(?:half|quarter|third|some)\s+(?:of\s+)?(?:my\s+|the\s+)?shares\b|\d\s*%|"
    r"\b(?:withdraw|redeem)\s+(?:(?:only|exactly|just|around|about|up to)\s+)*(?:\$?\s*\d|half\b|some\b)|"
    r"\b(?:withdraw|redeem)\s+(?!(?:all|my|the)\b)[a-z-]+\s+(?:raw\s+)?shares\b|"
    r"\b(?:take|get)\s+(?:out\s+\d[\d,.]*\s+\w+\b|\d[\d,.]*\s+\w+\s+out\b)|"
    r"\b\d[\d,.]*\s+(?:(?:raw\s+)?shares\b|[a-z][\w-]*\s+instead\b)", re.I,
)
_FULL_BALANCE = re.compile(
    r"\b(?:withdraw|redeem)\s+(?:all|everything|(?:my|the)\s+(?:full|entire)\s+(?:position|balance))\b|"
    r"^\s*(?:all(?: of)?\s+(?:my\s+|the\s+)?shares|(?:my\s+|the\s+)?(?:full|entire)\s+(?:position|balance)|"
    r"all|everything)[.!]?\s*$", re.I,
)
_REDEMPTION_RESTRICTION = re.compile(
    r"\b(?:not|never|cannot|can['’]t|don['’]t|instead of)\b[^.!?;\n]{0,60}"
    r"\b(?:withdraw|redeem|shares|all|everything)\b"
    r"|\b(?:all(?:\s+(?:of\s+)?(?:(?:my|the)\s+)?shares)?|everything|(?:full|entire)\s+(?:position|balance))[,–—]?\s+"
    r"(?:(?:except|but)\b|or\s+(?:(?:only|just)\s+)?(?:half\b|quarter\b|some\b|\d|[a-z-]+\s+shares\b))", re.I,
)
_RECIPIENT = re.compile(
    r"\b(?:(?:recipient|receiver)(?:\s+(?:wallet(?:\s+address)?|address))?(?:\s+is)?\s*[:=]?\s*|"
    r"to\s+(?:(?:wallet(?:\s+address)?|address)\b\s*[:=]?\s*)?)[`(<]*(0x[0-9a-f]{40})\b", re.I,
)
_RECIPIENT_NEGATION = re.compile(r"\b(?:not|never|cannot|can['’]t|don['’]t|avoid|instead of)\b[^.!?;]{0,120}$", re.I)
_OWNER_RECIPIENT = re.compile(r"\bto\s+my\s+(?:own\s+)?(?:wallet|address)\b", re.I)
_RECIPIENT_CUE = re.compile(
    r"\b(?:recipient|receiver)\b|\bto\s+(?:(?:another|a different)\s+(?:wallet|address)|[\w.-]+\.eth)\b", re.I,
)
_NON_AMOUNT_TEXT = re.compile(r"https?://\S+|\b0x[0-9a-f]+\b", re.I)


def _message_hashes(text):
    if replacement := corrected_tx_hash(text):
        return {replacement}
    return {value.lower() for value in TX_HASH_RE.findall(text)}


def _prior_user_messages(request):
    for item in reversed(request.recent_transcript):
        text = item.get("content")
        if item.get("role") == "user" and isinstance(text, str):
            yield text


def _requested_hashes(request):
    current = request.current_user_message
    count = 2 if _PAIRED_RECEIPTS.search(current) else 1
    hashes = _message_hashes(current)
    if len(hashes) >= count or corrected_tx_hash(current):
        return hashes
    for text in _prior_user_messages(request):
        corrected = corrected_tx_hash(text)
        hashes.update({corrected} if corrected else (value.lower() for value in TX_HASH_RE.findall(text)))
        if corrected or len(hashes) >= count:
            return hashes
    if count == 1 and (replacement := request.evidence.get("withdrawal_tx_hash")):
        return {replacement}
    for value in reversed(request.evidence.get("tx_hashes") or []):
        hashes.add(value.lower())
        if len(hashes) >= count:
            break
    return hashes


def _labeled_wallets(text):
    recipient_spans = {match.span(1) for match in _RECIPIENT.finditer(text)}
    return {match[1].lower() for match in _LABELED_WALLET.finditer(text) if match.span(1) not in recipient_spans}


def _requested_wallets(request, messages):
    wallet = (request.evidence.get("wallet") or "").lower()
    for text in messages:
        if wallets := _labeled_wallets(text):
            return wallets
        # Intake already resolved a newer address-only message to this wallet.
        if wallet.startswith("0x") and text.strip().lower() == wallet:
            return {wallet}
    return {wallet} if wallet.startswith("0x") else set()


def _current_target_messages(text):
    # Debounced and steered inputs share one text field. Resolve explicit
    # corrections and target changes before considering older instructions.
    messages = []
    correction = r"(?:correction\b|actually\b|instead\b|no[:,])"
    parts = re.split(rf"\n|(?<=[.!?])(?=[ \t]+{correction})", text, flags=re.I)
    start = 0
    network = vault = None
    wallets = set()
    for index, part in enumerate(parts):
        part_network, part_vault = _explicit_vault(part)
        part_wallets = _labeled_wallets(part)
        if index > start and (_target_changed(network, vault, part_network, part_vault)
                              or (wallets and part_wallets and wallets.isdisjoint(part_wallets))
                              or re.match(rf"\s*{correction}", part, re.I)):
            messages.append("\n".join(parts[start:index]))
            start = index
        network, vault = part_network or network, part_vault or vault
        wallets = part_wallets or wallets
    messages.append("\n".join(parts[start:]))
    return reversed(messages)


def _requested_vault(request, messages):
    network = vault = None
    for text in messages:
        linked_network, linked_vault = _explicit_vault(text)
        network = network or linked_network
        vault = vault or linked_vault
        if network and vault:
            break
    target = request.evidence
    return (
        network or target.get("withdrawal_target_chain") or target.get("chain"),
        vault or target.get("withdrawal_target_vault"),
    )


def _explicit_vault(text):
    if linked := extract_yearn_vault_url_target(text):
        return linked[0].lower(), linked[1].lower()
    network, vault = _LABELED_CHAIN.search(text), _LABELED_VAULT.search(text)
    return network.group(1).lower() if network else None, vault.group(1).lower() if vault else None


def _target_changed(network, vault, text_network, text_vault):
    return (network and text_network and network.lower() != text_network
            or vault and text_vault and vault.lower() != text_vault)


def _requested_redemption(messages, network, vault, wallets):
    if len(wallets) > 1:
        return None, None, "Confirm which wallet should redeem the shares."
    shares = receiver = None
    amount_known = recipient_known = False
    for index, text in enumerate(messages):
        text_network, text_vault = _explicit_vault(text)
        text_wallets = _labeled_wallets(text)
        if (_target_changed(network, vault, text_network, text_vault)
            or (wallets and text_wallets and text_wallets != wallets)):
            break
        if not recipient_known:
            matches = list(_RECIPIENT.finditer(text))
            if any(_RECIPIENT_NEGATION.search(text[:match.start()]) for match in matches):
                return shares, None, "Confirm the intended recipient's full wallet address."
            recipients = {match[1].lower() for match in matches}
            owner = bool(_OWNER_RECIPIENT.search(text))
            if len(recipients) > 1 or (owner and recipients) or (not (owner or recipients) and _RECIPIENT_CUE.search(text)):
                return shares, None, "Share the intended recipient's full wallet address."
            receiver = next(iter(recipients), None)
            recipient_known = owner or receiver is not None
        if not amount_known:
            amount_text = _NON_AMOUNT_TEXT.sub("", text)
            amounts = {value.replace(",", "").replace("_", "") for value in _RAW_SHARES.findall(amount_text)}
            full = _FULL_BALANCE.search(text)
            custom = _CUSTOM_AMOUNT.search(_RAW_SHARES.sub("", amount_text))
            if (amounts or full) and (_REDEMPTION_RESTRICTION.search(text) or custom or (amounts and full)):
                return None, receiver, "Confirm the raw share amount or say you want to redeem the maximum."
            if amounts:
                raw = next(iter(amounts))
                if len(amounts) != 1 or len(raw) > 78 or not 0 < int(raw) < 2**256:
                    return None, receiver, "Give one positive share amount in raw units."
                shares, amount_known = int(raw), True
            elif full:
                if index != 0:
                    return None, receiver, "Confirm you want to redeem the maximum."
                amount_known = True
            elif custom:
                return None, receiver, "Give the exact share amount in raw units so I can verify that amount."
        if amount_known and recipient_known:
            break
    if not amount_known:
        return None, receiver, "Specify the raw share amount or say you want to redeem the maximum."
    if receiver is not None and not wallets:
        return shares, receiver, "Share the wallet address that owns the shares."
    return shares, receiver, None


def _withdrawal_receipt_context(request):
    current = request.current_user_message
    if _WITHDRAWAL_RECEIPT.search(current):
        return True
    hashes = _message_hashes(current)
    # An explicit hash can reuse earlier context only for that transaction.
    for text in _prior_user_messages(request):
        earlier = _message_hashes(text)
        if not earlier or (hashes and not hashes.intersection(earlier)):
            continue
        if _WITHDRAWAL_RECEIPT.search(text) or _PAYOUT_QUESTION.search(text):
            return True
        if not hashes:
            break
    return False


def _matches_target(proof, chain, vault, hashes, wallets):
    if chain and chain.lower() != proof.chain:
        return False
    if wallets and proof.wallet.lower() not in wallets:
        return False
    if vault and vault.lower() != proof.vault.lower():
        return False
    return not (proof.tx_hash and hashes and proof.tx_hash.lower() not in hashes)


def ground_material_claims(result, request, observations, read_scopes):
    scopes = set()
    current = request.current_user_message
    if (_PAYOUT_QUESTION.search(current)
        or (_RECEIPT_OUTCOME.search(current) and _withdrawal_receipt_context(request))
        or (_PAYOUT_CLAIM.search(result.answer) and (
            "wallet_history" in read_scopes or TX_HASH_RE.search(current) or _WITHDRAWAL_RECEIPT.search(current)
        ))):
        scopes.add("payout")
    if _REDEEM_RECIPE.search(result.answer) or (not result.answer and "redemption" in read_scopes):
        scopes.add("redemption")
    refs = result.evidence_refs or []
    if result.answer and not scopes:
        return result, []
    hashes = _requested_hashes(request)
    messages = tuple(part for text in (current, *_prior_user_messages(request)) for part in _current_target_messages(text))
    wallets = _requested_wallets(request, messages)
    network, vault = _requested_vault(request, messages)
    proofs = [observations.get(ref) for ref in refs]
    if (len(refs) > 3 or len(set(refs)) != len(refs) or
        any(proof is None or not _matches_target(proof, network, vault, hashes, wallets) for proof in proofs)):
        proofs = []
    payouts = list(dict.fromkeys(proof for proof in proofs if proof.kind == "payout"))
    shares, receiver, clarification = (
        _requested_redemption(messages, network, vault, wallets) if "redemption" in scopes else (None, None, None)
    )
    redemptions = list(dict.fromkeys(
        proof for proof in proofs if proof.kind == "redemption" and not clarification
        and proof.shares_raw == (shares if shares is not None else proof.max_shares_raw)
        and proof.receiver and proof.receiver.lower() == (receiver or proof.wallet.lower())
    ))
    zero_shares = list(dict.fromkeys(proof for proof in proofs if proof.kind == "zero_shares"))
    sections, summaries = [], []
    if payouts:
        sections.extend(proof.summary for proof in payouts)
        summaries.extend(sections)
        if hashes - {proof.tx_hash for proof in payouts}:
            sections.append("I couldn't verify the remaining payouts from the available receipts.")
    elif "payout" in scopes or not ("redemption" in scopes or redemptions):
        burns = list(dict.fromkeys(proof.summary for proof in observations.values()
                                  if proof.kind == "burn" and _matches_target(proof, network, vault, hashes, wallets)))[:3]
        sections.extend(burns)
        summaries.extend(burns)
        sections.append("I couldn't verify a payout to this wallet from the available receipts. "
                        "The underlying token transfer hash would help verify payment.")
    if len(redemptions) == 1:
        sections.append(redemptions[0].instructions)
        summaries.append(redemptions[0].summary)
    elif len(zero_shares) == 1 and not redemptions:
        sections.append(zero_shares[0].summary)
        summaries.append(zero_shares[0].summary)
    elif "redemption" in scopes or redemptions:
        if "redemption_needs_wallet" in read_scopes and not wallets:
            sections.append("Share your wallet address so I can verify a zero-loss withdrawal.")
        else:
            sections.append("I couldn't verify an executable zero-loss withdrawal for this wallet and vault.")
            if clarification:
                sections.append(clarification)
    return replace(result, answer="\n\n".join(sections), handoff_reason=(
        "Review the withdrawal evidence and advise on the unresolved issue." if result.requires_human_handoff else None
    )), summaries
