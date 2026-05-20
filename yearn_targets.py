from dataclasses import dataclass
import logging
import re
from time import monotonic
from typing import Any, Literal

import aiohttp
from web3 import Web3

import config
from chain_access import get_web3_instance, inspect_contract_profile


LOGGER = logging.getLogger(__name__)

_YDAEMON_ADDRESS_CACHE_TTL_SECONDS = 300.0
_YDAEMON_VAULT_CATALOG_CACHE: dict[str, Any] = {
    "fetched_at": 0.0,
    "vaults": None,
    "address_index": None,
}

ResolvedYearnAddressKind = Literal[
    "vault",
    "strategy",
    "wrapper_or_gauge",
    "contract_unknown",
    "eoa_or_missing_code",
    "unknown",
]


@dataclass
class ResolvedYearnAddressTarget:
    address: str
    kind: ResolvedYearnAddressKind
    chain: str | None = None
    label: str | None = None
    vault_address: str | None = None
    vault_label: str | None = None
    contract_profile_kind: str | None = None
    is_contract: bool | None = None


def _normalize_chain(chain: str) -> str:
    return (chain or "").strip().lower()


def _safe_checksum_address(value: str | None) -> str | None:
    if not value or not Web3.is_address(value):
        return None
    return Web3.to_checksum_address(value)


def _describe_vault_entry(vault_data: dict[str, Any]) -> str:
    name = str(vault_data.get("name") or "").strip()
    symbol = str(vault_data.get("symbol") or "").strip()
    if name and symbol:
        return f"{name} ({symbol})"
    return name or symbol or "Yearn vault"


def _build_ydaemon_address_index(
    vaults: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}

    def add_entry(address: str | None, entry: dict[str, Any]) -> None:
        checksum = _safe_checksum_address(address)
        if checksum is None:
            return
        payload = dict(entry)
        payload["address"] = checksum
        index.setdefault(checksum, []).append(payload)

    for vault in vaults:
        chain_name = _normalize_chain(config.ID_TO_CHAIN_NAME.get(vault.get("chainID"), ""))
        vault_address = _safe_checksum_address(vault.get("address"))
        vault_label = _describe_vault_entry(vault)
        if vault_address is not None:
            add_entry(
                vault_address,
                {
                    "kind": "vault",
                    "chain": chain_name or None,
                    "label": vault_label,
                    "vault_address": vault_address,
                    "vault_label": vault_label,
                },
            )

        for strategy in vault.get("strategies") or []:
            add_entry(
                strategy.get("address"),
                {
                    "kind": "strategy",
                    "chain": chain_name or None,
                    "label": str(strategy.get("name") or "Yearn strategy").strip(),
                    "vault_address": vault_address,
                    "vault_label": vault_label,
                },
            )

        staking = vault.get("staking") or {}
        if staking.get("available"):
            staking_label = str(staking.get("source") or "Yearn staking wrapper").strip()
            add_entry(
                staking.get("address"),
                {
                    "kind": "wrapper_or_gauge",
                    "chain": chain_name or None,
                    "label": staking_label,
                    "vault_address": vault_address,
                    "vault_label": vault_label,
                },
            )

    return index


async def _fetch_ydaemon_vault_catalog() -> list[dict[str, Any]]:
    cache_age = monotonic() - float(_YDAEMON_VAULT_CATALOG_CACHE["fetched_at"] or 0.0)
    cached_vaults = _YDAEMON_VAULT_CATALOG_CACHE.get("vaults")
    if isinstance(cached_vaults, list) and cache_age < _YDAEMON_ADDRESS_CACHE_TTL_SECONDS:
        return cached_vaults

    api_url = (
        "https://ydaemon.yearn.fi/vaults/detected"
        "?hideAlways=false&strategiesDetails=withDetails&strategiesCondition=all&limit=2000"
    )
    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(api_url) as response:
            response.raise_for_status()
            vaults = await response.json()
    if not isinstance(vaults, list):
        raise ValueError("Unexpected yDaemon vault catalog response.")
    _YDAEMON_VAULT_CATALOG_CACHE["vaults"] = vaults
    _YDAEMON_VAULT_CATALOG_CACHE["address_index"] = _build_ydaemon_address_index(vaults)
    _YDAEMON_VAULT_CATALOG_CACHE["fetched_at"] = monotonic()
    return vaults


async def _get_ydaemon_address_index() -> dict[str, list[dict[str, Any]]]:
    cache_age = monotonic() - float(_YDAEMON_VAULT_CATALOG_CACHE["fetched_at"] or 0.0)
    cached_index = _YDAEMON_VAULT_CATALOG_CACHE.get("address_index")
    if isinstance(cached_index, dict) and cache_age < _YDAEMON_ADDRESS_CACHE_TTL_SECONDS:
        return cached_index
    await _fetch_ydaemon_vault_catalog()
    refreshed_index = _YDAEMON_VAULT_CATALOG_CACHE.get("address_index")
    return refreshed_index if isinstance(refreshed_index, dict) else {}


def _select_yearn_address_entry(
    entries: list[dict[str, Any]],
    *,
    chain_hint: str | None,
) -> dict[str, Any] | None:
    if not entries:
        return None
    normalized_chain = _normalize_chain(chain_hint or "")
    candidates = entries
    if normalized_chain:
        filtered = [
            entry
            for entry in entries
            if _normalize_chain(str(entry.get("chain") or "")) == normalized_chain
        ]
        if len(filtered) == 1:
            return filtered[0]
        if filtered:
            candidates = filtered
    if len(candidates) == 1:
        return candidates[0]
    return None


async def resolve_yearn_address_target(
    address: str,
    *,
    chain_hint: str | None = None,
) -> ResolvedYearnAddressTarget | None:
    checksum_address = _safe_checksum_address(address)
    if checksum_address is None:
        return None

    try:
        address_index = await _get_ydaemon_address_index()
    except Exception as exc:
        LOGGER.warning(
            "Failed to fetch yDaemon address index for %s: %s",
            checksum_address,
            exc,
        )
        address_index = {}

    entry = _select_yearn_address_entry(
        list(address_index.get(checksum_address, [])),
        chain_hint=chain_hint,
    )
    if entry is not None:
        return ResolvedYearnAddressTarget(
            address=checksum_address,
            kind=str(entry.get("kind") or "unknown"),  # type: ignore[arg-type]
            chain=entry.get("chain"),
            label=entry.get("label"),
            vault_address=entry.get("vault_address"),
            vault_label=entry.get("vault_label"),
            is_contract=True,
        )

    normalized_chain = _normalize_chain(chain_hint or "")
    web3_instance = get_web3_instance(normalized_chain)
    if web3_instance is None:
        return ResolvedYearnAddressTarget(
            address=checksum_address,
            kind="unknown",
            chain=normalized_chain or None,
        )

    try:
        profile = await inspect_contract_profile(
            web3_instance,
            checksum_address,
        )
    except Exception as exc:
        LOGGER.warning(
            "Failed to inspect contract profile for %s on %s: %s",
            checksum_address,
            normalized_chain,
            exc,
        )
        return ResolvedYearnAddressTarget(
            address=checksum_address,
            kind="unknown",
            chain=normalized_chain or None,
        )

    profile_kind = str(profile.get("kind") or "unknown")
    if profile_kind == "eoa_or_missing_code":
        return ResolvedYearnAddressTarget(
            address=checksum_address,
            kind="eoa_or_missing_code",
            chain=normalized_chain or None,
            is_contract=False,
            contract_profile_kind=profile_kind,
        )

    return ResolvedYearnAddressTarget(
        address=checksum_address,
        kind="contract_unknown",
        chain=normalized_chain or None,
        label=str(profile.get("name") or profile.get("symbol") or "").strip() or None,
        is_contract=bool(profile.get("has_code")),
        contract_profile_kind=profile_kind,
    )


_VAULT_URL_PATTERN = re.compile(
    r"https?://(?:legacy\.)?yearn\.fi/(?:vaults|v3)/(?P<chain_id>\d+)/(?P<address>0x[a-fA-F0-9]{40})",
    re.IGNORECASE,
)


def extract_yearn_vault_url_target(text: str) -> tuple[str, str] | None:
    match = _VAULT_URL_PATTERN.search(text or "")
    if not match:
        return None

    chain_id_raw = match.group("chain_id")
    address = match.group("address")
    try:
        chain_id = int(chain_id_raw)
    except ValueError:
        return None

    chain_name = str(config.ID_TO_CHAIN_NAME.get(chain_id) or "").strip().lower()
    checksum_address = _safe_checksum_address(address)
    if not chain_name or checksum_address is None:
        return None
    return (chain_name, checksum_address)
