from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from web3 import Web3

import chain_access
import config
from kong_vaults import fetch_kong_vault_catalog, fetch_kong_vault_snapshot


ensure_web3_instances = chain_access.ensure_web3_instances
get_web3_instance = chain_access.get_web3_instance
resolve_ens_name = chain_access.resolve_ens_name

try:
    V1_VAULTS = json.loads(
        (Path(__file__).resolve().parent / "v1_vaults.json").read_text(encoding="utf-8")
    )
except Exception as exc:
    logging.warning(
        "Could not load v1_vaults.json: %s. V1 deposit checks will fail.", exc
    )
    V1_VAULTS = []


@dataclass(frozen=True)
class DepositScanResult:
    text: str
    found: bool
    complete: bool


@dataclass(frozen=True)
class _VaultBalanceResult:
    vault_info: Dict
    wallet_balance: int = 0
    staked_balance: int = 0
    complete: bool = True


def _format_token_balance(value: float) -> str:
    if 0 < abs(value) < 0.000001:
        return f"{value:.3e}"
    return f"{value:,.6f}"


async def _fetch_vault_and_gauge_balances(
    vault_info: Dict,
    web3_instance: Web3,
    user_checksum_addr: str,
    semaphore: asyncio.Semaphore,
) -> _VaultBalanceResult:
    """
    Fetches a user's balance from a vault AND its associated staking gauge concurrently.
    Returns the known balances plus whether every applicable balance call completed.
    """
    async with semaphore:
        vault_addr_str = vault_info.get("address")
        if not vault_addr_str:
            return _VaultBalanceResult(vault_info=vault_info, complete=False)

        try:
            vault_checksum_addr = Web3.to_checksum_address(vault_addr_str)
            vault_contract = web3_instance.eth.contract(
                address=vault_checksum_addr, abi=config.ERC20_ABI
            )

            # --- Create all web3 call coroutines first ---
            wallet_balance_coro = asyncio.to_thread(
                vault_contract.functions.balanceOf(user_checksum_addr).call
            )

            complete = True
            gauge_balance_coro = None
            gauge_contract = None
            staking_info = vault_info.get("staking")
            if staking_info and staking_info.get("available"):
                gauge_addr_str = staking_info.get("address")
                if Web3.is_address(gauge_addr_str):
                    gauge_checksum_addr = Web3.to_checksum_address(gauge_addr_str)
                    gauge_contract = web3_instance.eth.contract(
                        address=gauge_checksum_addr, abi=config.GAUGE_ABI
                    )
                    gauge_balance_coro = asyncio.to_thread(
                        gauge_contract.functions.balanceOf(user_checksum_addr).call
                    )
                else:
                    complete = False
                    logging.warning(
                        "Invalid staking gauge address for vault %s: %s",
                        vault_addr_str,
                        gauge_addr_str,
                    )

            # --- Run wallet and gauge balance checks in parallel ---
            if gauge_balance_coro is not None:
                results = await asyncio.gather(
                    wallet_balance_coro, gauge_balance_coro, return_exceptions=True
                )
                if isinstance(results[0], Exception):
                    wallet_balance = 0
                    complete = False
                    logging.warning(
                        f"Error fetching wallet balance for {vault_addr_str}: {results[0]}"
                    )
                else:
                    wallet_balance = int(results[0])
                if isinstance(results[1], Exception):
                    gauge_token_balance = 0
                    complete = False
                    logging.warning(
                        f"Error fetching gauge balance for {gauge_addr_str}: {results[1]}"
                    )
                else:
                    gauge_token_balance = int(results[1])
            else:
                # Only run the wallet balance check if no gauge
                wallet_balance = int(await wallet_balance_coro)
                gauge_token_balance = 0

            logging.debug(
                f"Vault {vault_addr_str}: Wallet balance={wallet_balance}, Gauge token balance={gauge_token_balance}"
            )

            # --- If staked, perform the final conversion call ---
            staked_balance_in_yvtoken = 0
            if gauge_token_balance > 0:
                try:
                    staked_balance_in_yvtoken = int(
                        await asyncio.to_thread(
                            gauge_contract.functions.convertToAssets(
                                gauge_token_balance
                            ).call
                        )
                    )
                    logging.debug(
                        f"Vault {vault_addr_str}: Staked balance (in yvToken) is {staked_balance_in_yvtoken}"
                    )
                except Exception as exc:
                    complete = False
                    logging.warning(
                        "Error converting gauge balance for %s: %s",
                        gauge_addr_str,
                        exc,
                    )

            return _VaultBalanceResult(
                vault_info=vault_info,
                wallet_balance=wallet_balance,
                staked_balance=staked_balance_in_yvtoken,
                complete=complete,
            )

        except Exception as e:
            logging.error(
                f"Critical error in balance fetching logic for vault {vault_addr_str}: {e}",
                exc_info=True,
            )
            return _VaultBalanceResult(vault_info=vault_info, complete=False)


async def query_active_deposits_logic(
    resolved_address: str,
    chain: Optional[str] = None,
    token_symbol: Optional[str] = None,
) -> DepositScanResult:
    logging.info(
        f"[Logic:query_active_deposits] Checking Active for {resolved_address}, Chain: {chain}, Token: {token_symbol}"
    )

    chains_to_check = []
    web3_instances = ensure_web3_instances()
    if chain:
        chain_lower = chain.strip().lower()
        if chain_lower in web3_instances:
            chains_to_check.append(chain_lower)
        else:
            return DepositScanResult(
                text=f"Unsupported chain: {chain}.", found=False, complete=False
            )
    else:
        chains_to_check = [c for c in web3_instances.keys() if c != "berachain"]

    try:
        all_vaults_data = await fetch_kong_vault_catalog()
        logging.info(
            "[Logic:query_active_deposits] Fetched %s Yearn vault definitions from Kong.",
            len(all_vaults_data),
        )
    except Exception as exc:
        logging.error(
            "[Logic:query_active_deposits] Failed to fetch Kong vault list: %s",
            exc,
        )
        return DepositScanResult(
            text=f"Error: Could not fetch the list of active vaults: {exc}",
            found=False,
            complete=False,
        )

    user_checksum_addr = Web3.to_checksum_address(resolved_address)

    all_results = []
    total_deposits_found = 0
    incomplete_checks = 0
    for chain_name in chains_to_check:
        web3_instance = web3_instances.get(chain_name)
        if not web3_instance:
            continue

        chain_id = config.CHAIN_NAME_TO_ID.get(chain_name)
        if not chain_id:
            continue

        chain_vaults = [
            vault
            for vault in all_vaults_data
            if vault.get("chainId") == chain_id
        ]
        if token_symbol:
            token_lower = token_symbol.lower()
            chain_vaults = [
                v
                for v in chain_vaults
                if token_lower in v.get("asset", {}).get("symbol", "").lower()
            ]

        if not chain_vaults:
            logging.info(
                f"No vaults to check for chain '{chain_name}' after filtering."
            )
            continue

        logging.info(
            f"Creating {len(chain_vaults)} balance check tasks for chain '{chain_name}'."
        )
        semaphore = asyncio.Semaphore(25)
        tasks = [
            _fetch_vault_and_gauge_balances(
                v, web3_instance, user_checksum_addr, semaphore
            )
            for v in chain_vaults
        ]

        balance_results = await asyncio.gather(*tasks, return_exceptions=True)
        logging.info(f"Completed balance checks for chain '{chain_name}'.")

        chain_deposits = []
        for result in balance_results:
            if isinstance(result, Exception):
                incomplete_checks += 1
                continue
            if not result.complete:
                incomplete_checks += 1

            total_balance = result.wallet_balance + result.staked_balance

            if total_balance > 0:
                vault_info = result.vault_info
                try:
                    decimals = int(vault_info.get("decimals", 18))
                    total_display_balance = total_balance / (10**decimals)
                    vault_address = Web3.to_checksum_address(vault_info.get("address"))
                    vault_url = f"https://yearn.fi/vaults/{chain_id}/{vault_address}"
                    vault_name = vault_info.get("name", "Unknown Vault")
                    vault_symbol = vault_info.get("symbol", "N/A")
                    deposit_lines = [
                        f"**Vault:** [{vault_name}]({vault_url}) (Symbol: {vault_symbol})",
                        f"  Address: `{vault_address}`",
                        f"  Total Position: **{_format_token_balance(total_display_balance)} {vault_symbol}**",
                    ]
                    staked_balance = result.staked_balance
                    if staked_balance > 0:
                        wallet_balance = result.wallet_balance
                        wallet_display = wallet_balance / (10**decimals)
                        staked_display = staked_balance / (10**decimals)
                        if wallet_balance > 0:
                            breakdown = (
                                f"(Breakdown: {_format_token_balance(wallet_display)} liquid + "
                                f"{_format_token_balance(staked_display)} staked)"
                            )
                        else:
                            breakdown = "(Staked in gauge)"
                        deposit_lines.append(f"    {breakdown}")
                    chain_deposits.append("\n".join(deposit_lines))
                    total_deposits_found += 1
                except Exception as e:
                    incomplete_checks += 1
                    logging.error(
                        f"Error processing deposit for vault {vault_info.get('address')} on {chain_name}: {e}"
                    )

        if chain_deposits:
            all_results.append(
                f"**{chain_name.capitalize()} Active Deposits:**\n"
                + "\n\n".join(chain_deposits)
            )
    if total_deposits_found > 0:
        result_text = "\n\n---\n\n".join(all_results)
        if incomplete_checks:
            result_text += (
                f"\n\nWarning: {incomplete_checks} active vault balance check(s) "
                "failed, so the positions above may be incomplete."
            )
        return DepositScanResult(
            text=result_text,
            found=True,
            complete=incomplete_checks == 0,
        )
    if incomplete_checks:
        return DepositScanResult(
            text=(
                f"Active vault balance checks were incomplete because {incomplete_checks} "
                "check(s) failed. No confident no-deposit conclusion can be made for "
                "this address."
            ),
            found=False,
            complete=False,
        )
    if not chain:
        result_text = (
            "No active vault deposits found for that address on any supported Yearn chain"
            + (f" matching token '{token_symbol}'." if token_symbol else ".")
        )
    else:
        result_text = (
            f"No active deposits found on {chain.capitalize()} for this address"
            + (f" matching token '{token_symbol}'." if token_symbol else ".")
        )
    return DepositScanResult(text=result_text, found=False, complete=True)


async def query_v1_deposits_logic(
    resolved_address: str, token_symbol: Optional[str] = None
) -> DepositScanResult:
    logging.info(
        f"[Logic:query_v1_deposits] Checking V1 for {resolved_address}, Token: {token_symbol}"
    )
    if not V1_VAULTS:
        return DepositScanResult(
            text="V1 vault data is not loaded; deprecated vault deposits could not be checked.",
            found=False,
            complete=False,
        )
    web3_eth = get_web3_instance("ethereum")
    if web3_eth is None:
        return DepositScanResult(
            text="Ethereum connection unavailable; deprecated V1 vault deposits could not be checked.",
            found=False,
            complete=False,
        )
    try:
        user_checksum_addr = Web3.to_checksum_address(resolved_address)
    except ValueError:
        return DepositScanResult(
            text=f"Invalid Ethereum address format: {resolved_address}",
            found=False,
            complete=False,
        )

    found_deposits = []
    incomplete_checks = 0
    for vault in V1_VAULTS:
        if token_symbol and token_symbol.lower() not in vault.get("symbol", "").lower():
            continue
        try:
            vault_checksum_addr = Web3.to_checksum_address(vault.get("address"))
            contract = web3_eth.eth.contract(
                address=vault_checksum_addr, abi=config.ERC20_ABI
            )
            balance = await asyncio.to_thread(
                contract.functions.balanceOf(user_checksum_addr).call
            )

            if balance > 0:
                decimals = int(vault.get("decimals", 18))
                display_balance = balance / (10**decimals)
                etherscan_link = f"https://etherscan.io/address/{vault_checksum_addr}"
                vault_name = vault.get("name", "Unknown V1 Vault")
                vault_symbol = vault.get("symbol", "N/A")
                instruction_verb = (
                    "'withdrawAll'"
                    if vault.get("withdraw_all", False)
                    else "'withdraw' (entering your balance)"
                )
                instr = (
                    f"**Vault:** [{vault_name}]({etherscan_link}) (Symbol: {vault_symbol})\n"
                    f"  Deposit Found: {_format_token_balance(display_balance)} tokens.\n"
                    f"  *Withdrawal:* Go to Etherscan link -> 'Contract' tab -> 'Write Contract' -> Connect Wallet -> Use the {instruction_verb} function."
                )
                found_deposits.append(instr)
        except Exception as e:
            incomplete_checks += 1
            logging.error(
                f"[Logic:query_v1_deposits] Error checking V1 vault {vault.get('address')} for {user_checksum_addr}: {e}"
            )

    if found_deposits:
        result_text = (
            "**Deprecated V1 Vault Deposits Found (Ethereum Only):**\n\n"
            + "\n\n".join(found_deposits)
        )
        if incomplete_checks:
            result_text += (
                f"\n\nWarning: {incomplete_checks} deprecated V1 vault balance "
                "check(s) failed, so the positions above may be incomplete."
            )
        return DepositScanResult(
            text=result_text,
            found=True,
            complete=incomplete_checks == 0,
        )
    if incomplete_checks:
        return DepositScanResult(
            text=(
                f"Deprecated V1 vault balance checks were incomplete because {incomplete_checks} "
                "check(s) failed. No confident no-deposit conclusion can be made for "
                "these vaults."
            ),
            found=False,
            complete=False,
        )
    return DepositScanResult(
        text="No deposits found in deprecated V1 vaults for this address.",
        found=False,
        complete=True,
    )


async def core_check_all_deposits(
    user_address_or_ens: str, token_symbol: Optional[str] = None
) -> str:
    """
    Checks for user deposits in BOTH active (v2/v3) and deprecated (v1) Yearn vaults across all supported chains.
    Provide the user's wallet address or ENS name. Optionally filter by token symbol (e.g., 'USDC').
    Returns a combined summary of any deposits found in either type of vault.
    """
    logging.info(f"[CoreTool:check_deposits] User: '{user_address_or_ens}'")
    resolved_address = resolve_ens_name(user_address_or_ens)
    if not resolved_address:
        return f"Could not resolve '{user_address_or_ens}' to a valid Ethereum address."

    # Run checks concurrently
    v1_task = asyncio.create_task(
        query_v1_deposits_logic(resolved_address, token_symbol)
    )
    active_task = asyncio.create_task(
        query_active_deposits_logic(
            resolved_address, chain=None, token_symbol=token_symbol
        )
    )

    v1_results, active_results = await asyncio.gather(v1_task, active_task)

    if (
        v1_results.complete
        and active_results.complete
        and not v1_results.found
        and not active_results.found
    ):
        return (
            f"No deposits found in any active or deprecated Yearn vaults for address {resolved_address}"
            + (f" matching token '{token_symbol}'." if token_symbol else ".")
        )

    final_output = []
    for result, empty_note in (
        (v1_results, "(No deposits found in deprecated V1 vaults)"),
        (active_results, "(No deposits found in active V2/V3 vaults)"),
    ):
        if result.found or not result.complete:
            final_output.append(result.text)
        else:
            final_output.append(empty_note)

    combined_result = "\n\n---\n\n".join(final_output)
    logging.info(
        f"[CoreTool:check_deposits] Combined Result for {resolved_address}:\n{combined_result}"
    )
    return combined_result


async def core_get_withdrawal_instructions(
    user_address_or_ens: Optional[str], vault_address: str, chain: str
) -> str:
    """
    Generates step-by-step instructions for withdrawing from a specific Yearn vault (v1, v2, or v3) using a block explorer.
    Also provides the direct link to the vault on the Yearn website for reference (for v2/v3).
    Provide the vault's address, the chain name (e.g., 'ethereum', 'optimism', 'arbitrum'), and optionally the user's address/ENS.
    Use this when a user asks how to withdraw or reports issues using the Yearn website for a specific vault.
    """
    logging.info(
        f"[Tool:get_withdrawal_instructions] Args: User={user_address_or_ens}, Vault={vault_address}, Chain={chain}"
    )

    # --- Step 1: Input Validation & Setup ---
    resolved_user_address = None
    user_checksum_addr = None
    if user_address_or_ens:
        resolved_user_address = resolve_ens_name(user_address_or_ens)
        if not resolved_user_address:
            logging.warning(
                f"Could not resolve provided user address/ENS: '{user_address_or_ens}'. Proceeding without it."
            )
        else:
            try:
                user_checksum_addr = Web3.to_checksum_address(resolved_user_address)
            except ValueError:
                logging.warning(
                    f"Invalid format for resolved user address: '{resolved_user_address}'. Proceeding without it."
                )
                user_checksum_addr = None

    chain_lower = chain.lower()
    web3_instance = get_web3_instance(chain_lower)
    chain_id = config.CHAIN_NAME_TO_ID.get(chain_lower)
    explorer_base_url = config.BLOCK_EXPLORER_URLS.get(chain_lower)

    if not web3_instance or not chain_id or not explorer_base_url:
        return f"Unsupported or invalid chain: '{chain}'. Cannot generate instructions."

    try:
        vault_checksum_addr = Web3.to_checksum_address(vault_address)
    except ValueError as e:
        return f"Invalid vault address format provided: {e}. Please provide a valid vault address."

    explorer_vault_url = f"{explorer_base_url}/address/{vault_checksum_addr}"

    # --- Step 2: Check V1 List (Ethereum Only) ---
    if chain_lower == "ethereum":
        v1_vault_info = next(
            (
                v
                for v in V1_VAULTS
                if v.get("address", "").lower() == vault_checksum_addr.lower()
            ),
            None,
        )

        if v1_vault_info:
            logging.info(
                f"Vault {vault_checksum_addr} identified as V1. Generating V1 instructions."
            )
            vault_name = v1_vault_info.get("name", vault_checksum_addr)

            # --- REVERTED V1 INSTRUCTION WORDING ---
            instructions = [
                f"This vault (**{vault_name}** `{vault_checksum_addr}`) is a **deprecated Yearn V1 vault**.",
                "Withdrawals must be done directly via the block explorer:",
                f"1. Go to the vault contract page: {explorer_vault_url}",
                "2. Click the **'Contract'** tab.",
                "3. Click the **'Write Contract'** tab.",
                f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}.",
                "5. Look for a suitable withdrawal function (often named like 'withdraw', 'withdrawAll', or similar). Prioritize functions that take no arguments if available.",  # Generic guidance
                "6. Click the **'Write'** button next to the chosen function.",
                "7. Confirm the transaction in your wallet.",
                "\nOnce the transaction confirms, your funds should be back in your wallet.",
            ]
            final_instructions = "\n".join(instructions)
            logging.info(
                f"Generated V1 instructions for {vault_checksum_addr}:\n{final_instructions}"
            )
            return final_instructions
        else:
            logging.info(
                f"Vault {vault_checksum_addr} not found in V1 list. Proceeding to check V2/V3."
            )

    # --- Step 3: Fetch exact V2/V3 details from Kong ---
    vault_details_json: Optional[Dict] = None
    try:
        vault_details_json = await fetch_kong_vault_snapshot(
            chain_id,
            vault_checksum_addr,
        )
    except Exception as exc:
        logging.error(
            "Error fetching Kong vault snapshot for withdrawal tool: %s",
            exc,
        )

    if not vault_details_json or vault_details_json.get("origin") != "yearn":
        logging.warning(
            "Failed to fetch V2/V3 vault details for %s on chain %s via Kong.",
            vault_checksum_addr,
            chain_id,
        )
        return (
            f"Could not fetch vault details from the Yearn API for `{vault_checksum_addr}` on {chain.capitalize()} "
            f"to determine the correct withdrawal method for V2/V3 vaults. \n"
            f"Please double-check the vault address and chain name. \n"
            f"You can view the contract directly here: {explorer_vault_url}"
        )

    # --- Step 4: Determine V2/V3 Version and Format Instructions ---
    api_version_str = vault_details_json.get("apiVersion", "")
    vault_name = vault_details_json.get("name", vault_checksum_addr)
    yearn_ui_link = f"https://yearn.fi/vaults/{chain_id}/{vault_checksum_addr}"

    intro_message = (
        f"Okay, here are instructions for withdrawing from the **{vault_name}** vault (`{vault_checksum_addr}`) on **{chain.capitalize()}** using the block explorer.\n\n"
        f"You can also try the Yearn website here: {yearn_ui_link}\n\n"
        f"If using the block explorer:"
    )
    instructions = [intro_message]
    instructions.append(f"1. Go to the vault contract page: {explorer_vault_url}")
    instructions.append("2. Click the **'Contract'** tab.")
    write_tab_options = (
        "'Write Contract as Proxy' (if available, otherwise 'Write Contract')"
    )
    instructions.append(f"3. Click the **{write_tab_options}** tab.")
    instructions.append(
        f"4. Click the **'Connect to Web3'** button and connect your wallet {f'(`{user_checksum_addr}`)' if user_checksum_addr else '(the one you used to deposit)'}."
    )

    is_v3 = api_version_str.startswith("3.")
    is_v2 = api_version_str.startswith("0.")

    if is_v3:
        logging.info(
            f"Vault {vault_checksum_addr} identified as V3 (version: {api_version_str}). Generating 'redeem' instructions."
        )
        user_balance: Optional[int] = None
        balance_input_value = "**(Enter your full share balance manually)**"

        if user_checksum_addr:
            try:
                contract = web3_instance.eth.contract(
                    address=vault_checksum_addr, abi=config.ERC20_ABI
                )
                balance_raw = await asyncio.to_thread(
                    contract.functions.balanceOf(user_checksum_addr).call
                )
                user_balance = int(balance_raw)
                if user_balance == 0:
                    return (
                        f"Confirmed user `{user_checksum_addr}` has zero balance in V3 vault `{vault_checksum_addr}`.\n"
                        f"Vault link for reference: {yearn_ui_link}\n"
                        f"No withdrawal needed or possible via explorer."
                    )
                balance_input_value = (
                    f"`{user_balance}` (This should be your full share balance)"
                )
            except Exception as e:
                logging.error(
                    f"Failed to fetch user balance for V3 vault {vault_checksum_addr}: {e}"
                )
                logging.warning("Generating V3 instructions without fetched balance.")
        else:
            logging.warning(
                "User address not provided; cannot fetch V3 balance. Instructions require manual input."
            )

        instructions.extend(
            [
                "5. Find the **'redeem'** function.",
                "6. Enter the following values:",
                f"   - `shares (uint256)`: {balance_input_value}",
                f"   - `receiver (address)`: {f'`{user_checksum_addr}` (Your wallet address)' if user_checksum_addr else '**(Your wallet address)**'}",
                f"   - `owner (address)`: {f'`{user_checksum_addr}` (Your wallet address again)' if user_checksum_addr else '**(Your wallet address again)**'}",
                "7. Click the **'Write'** button.",
                "8. Confirm the transaction in your wallet.",
            ]
        )

    elif is_v2:
        logging.info(
            f"Vault {vault_checksum_addr} identified as V2 (version: {api_version_str}). Generating 'withdraw' (no args) instructions."
        )
        instructions.extend(
            [
                "5. Find the **'withdraw()'** function that takes **no arguments**.",
                "   *Important: Do NOT use a 'withdraw' function that asks for '_shares' or an amount.*",
                "6. Click the **'Write'** button next to that specific 'withdraw()' function.",
                "7. Confirm the transaction in your wallet.",
            ]
        )

    else:
        logging.warning(
            f"Vault {vault_checksum_addr} has unknown or missing V2/V3 version: '{api_version_str}'. Providing generic instructions."
        )
        balance_info = "(Cannot check balance without your address)"
        if user_checksum_addr:
            try:
                contract = web3_instance.eth.contract(
                    address=vault_checksum_addr, abi=config.ERC20_ABI
                )
                balance_raw = await asyncio.to_thread(
                    contract.functions.balanceOf(user_checksum_addr).call
                )
                user_balance = int(balance_raw)
                if user_balance == 0:
                    return (
                        f"Confirmed user `{user_checksum_addr}` has zero balance in vault `{vault_checksum_addr}` (Version Unknown).\n"
                        f"Vault link for reference: {yearn_ui_link}\n"
                        f"No withdrawal needed or possible via explorer."
                    )
                balance_info = f"`{user_balance}`"
            except Exception:
                logging.warning("Could not fetch balance for unknown version vault.")
                balance_info = "(Could not fetch balance)"

        instructions.extend(
            [
                f"5. **Unknown Vault Version (API Version: {api_version_str or 'Not Found'})**: Look for a function named **'withdraw'** or **'redeem'**.",
                "   - Try **'withdraw()'** (with no input fields) first.",
                f"   - If that's not present or doesn't work, try **'redeem'**. If it asks for `shares`, `receiver`, and `owner`, enter your balance {balance_info} for `shares` and your address ({f'`{user_checksum_addr}`' if user_checksum_addr else 'Your Address'}) for `receiver` and `owner`.",
                "   - If unsure, please ask for human help again, mentioning the vault version is unclear.",
                "6. Click the **'Write'** button for the chosen function.",
                "7. Confirm the transaction in your wallet.",
            ]
        )

    instructions.append(
        "\nOnce the transaction confirms on the blockchain, your funds should be back in your wallet."
    )
    final_instructions = "\n".join(instructions)
    logging.info(
        f"Generated V2/V3/Fallback instructions for {vault_checksum_addr}:\n{final_instructions}"
    )
    return final_instructions
