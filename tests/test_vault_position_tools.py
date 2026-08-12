import tests as _test_environment  # noqa: F401

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import vault_position_tools


class VaultPositionToolTests(unittest.IsolatedAsyncioTestCase):
    def test_positive_tiny_balance_never_formats_as_zero(self) -> None:
        self.assertEqual(
            vault_position_tools._format_token_balance(0.000000000000000001),
            "1.000e-18",
        )

    async def test_check_all_deposits_combines_v1_and_active_results(self) -> None:
        with (
            patch(
                "vault_position_tools.resolve_ens_name",
                return_value="0x0000000000000000000000000000000000000001",
            ),
            patch(
                "vault_position_tools.query_v1_deposits_logic",
                new=AsyncMock(
                    return_value=vault_position_tools.DepositScanResult(
                        text="V1 deposit result", found=True, complete=True
                    )
                ),
            ) as query_v1,
            patch(
                "vault_position_tools.query_active_deposits_logic",
                new=AsyncMock(
                    return_value=vault_position_tools.DepositScanResult(
                        text="Active deposit result", found=True, complete=True
                    )
                ),
            ) as query_active,
        ):
            result = await vault_position_tools.core_check_all_deposits(
                "vitalik.eth",
                "USDC",
            )

        self.assertEqual(result, "V1 deposit result\n\n---\n\nActive deposit result")
        query_v1.assert_awaited_once_with(
            "0x0000000000000000000000000000000000000001",
            "USDC",
        )
        query_active.assert_awaited_once_with(
            "0x0000000000000000000000000000000000000001",
            chain=None,
            token_symbol="USDC",
        )

    async def test_check_all_deposits_does_not_turn_incomplete_scan_into_zero_balance(
        self,
    ) -> None:
        incomplete = vault_position_tools.DepositScanResult(
            text="Active vault checks were incomplete.",
            found=False,
            complete=False,
        )
        empty = vault_position_tools.DepositScanResult(
            text="No deposits found in deprecated V1 vaults for this address.",
            found=False,
            complete=True,
        )
        with (
            patch(
                "vault_position_tools.resolve_ens_name",
                return_value="0x0000000000000000000000000000000000000001",
            ),
            patch(
                "vault_position_tools.query_v1_deposits_logic",
                new=AsyncMock(return_value=empty),
            ),
            patch(
                "vault_position_tools.query_active_deposits_logic",
                new=AsyncMock(return_value=incomplete),
            ),
        ):
            result = await vault_position_tools.core_check_all_deposits("vitalik.eth")

        self.assertIn("Active vault checks were incomplete.", result)
        self.assertIn("No deposits found in deprecated V1 vaults", result)
        self.assertNotIn("No deposits found in any active or deprecated", result)

    async def test_balance_fetch_preserves_wallet_position_when_gauge_conversion_fails(
        self,
    ) -> None:
        vault_address = "0x1111111111111111111111111111111111111111"
        gauge_address = "0x2222222222222222222222222222222222222222"

        class _Call:
            def __init__(self, value=None, error: Exception | None = None) -> None:
                self.value = value
                self.error = error

            def call(self):
                if self.error is not None:
                    raise self.error
                return self.value

        class _VaultFunctions:
            def balanceOf(self, _address):
                return _Call(100)

        class _GaugeFunctions:
            def balanceOf(self, _address):
                return _Call(50)

            def convertToAssets(self, _balance):
                return _Call(error=RuntimeError("conversion unavailable"))

        class _Contract:
            def __init__(self, functions) -> None:
                self.functions = functions

        class _Eth:
            def contract(self, *, address, abi):
                if address.lower() == gauge_address.lower():
                    return _Contract(_GaugeFunctions())
                return _Contract(_VaultFunctions())

        class _Web3:
            eth = _Eth()

        result = await vault_position_tools._fetch_vault_and_gauge_balances(
            {
                "address": vault_address,
                "staking": {"available": True, "address": gauge_address},
            },
            _Web3(),
            "0x3333333333333333333333333333333333333333",
            asyncio.Semaphore(1),
        )

        self.assertEqual(result.wallet_balance, 100)
        self.assertEqual(result.staked_balance, 0)
        self.assertFalse(result.complete)

    async def test_active_scan_reports_inconclusive_when_balance_call_fails(
        self,
    ) -> None:
        vault_info = {
            "chainId": 1,
            "address": "0x1111111111111111111111111111111111111111",
            "decimals": 18,
        }

        with (
            patch(
                "vault_position_tools.ensure_web3_instances",
                return_value={"ethereum": object()},
            ),
            patch(
                "vault_position_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[vault_info]),
            ),
            patch(
                "vault_position_tools._fetch_vault_and_gauge_balances",
                new=AsyncMock(
                    return_value=vault_position_tools._VaultBalanceResult(
                        vault_info=vault_info,
                        complete=False,
                    )
                ),
            ),
        ):
            result = await vault_position_tools.query_active_deposits_logic(
                "0x3333333333333333333333333333333333333333"
            )

        self.assertFalse(result.found)
        self.assertFalse(result.complete)
        self.assertIn("No confident no-deposit conclusion", result.text)

    async def test_active_scan_preserves_known_position_when_other_evidence_fails(
        self,
    ) -> None:
        vault_info = {
            "chainId": 1,
            "address": "0x1111111111111111111111111111111111111111",
            "name": "Known Vault",
            "symbol": "yvTEST",
            "decimals": 2,
        }

        with (
            patch(
                "vault_position_tools.ensure_web3_instances",
                return_value={"ethereum": object()},
            ),
            patch(
                "vault_position_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[vault_info]),
            ),
            patch(
                "vault_position_tools._fetch_vault_and_gauge_balances",
                new=AsyncMock(
                    return_value=vault_position_tools._VaultBalanceResult(
                        vault_info=vault_info,
                        wallet_balance=125,
                        complete=False,
                    )
                ),
            ),
        ):
            result = await vault_position_tools.query_active_deposits_logic(
                "0x3333333333333333333333333333333333333333"
            )

        self.assertTrue(result.found)
        self.assertFalse(result.complete)
        self.assertIn("1.250000 yvTEST", result.text)
        self.assertIn("positions above may be incomplete", result.text)

    async def test_active_scan_includes_retired_vaults_with_stranded_positions(
        self,
    ) -> None:
        vault_info = {
            "chainId": 1,
            "address": "0x1111111111111111111111111111111111111111",
            "name": "Retired Vault",
            "symbol": "yvOLD",
            "decimals": 2,
            "isRetired": True,
        }
        balance_check = AsyncMock(
            return_value=vault_position_tools._VaultBalanceResult(
                vault_info=vault_info,
                wallet_balance=100,
            )
        )
        with (
            patch(
                "vault_position_tools.ensure_web3_instances",
                return_value={"ethereum": object()},
            ),
            patch(
                "vault_position_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[vault_info]),
            ),
            patch(
                "vault_position_tools._fetch_vault_and_gauge_balances",
                new=balance_check,
            ),
        ):
            result = await vault_position_tools.query_active_deposits_logic(
                "0x3333333333333333333333333333333333333333"
            )

        self.assertTrue(result.found)
        self.assertIn("Retired Vault", result.text)
        balance_check.assert_awaited_once()

    async def test_v1_scan_reports_inconclusive_when_balance_call_fails(self) -> None:
        class _Call:
            def call(self):
                raise RuntimeError("RPC unavailable")

        class _Functions:
            def balanceOf(self, _address):
                return _Call()

        class _Contract:
            functions = _Functions()

        class _Eth:
            def contract(self, *, address, abi):
                return _Contract()

        class _Web3:
            eth = _Eth()

        with (
            patch("vault_position_tools.get_web3_instance", return_value=_Web3()),
            patch.object(
                vault_position_tools,
                "V1_VAULTS",
                [
                    {
                        "address": "0x1111111111111111111111111111111111111111",
                        "symbol": "yvTEST",
                        "decimals": 18,
                    }
                ],
            ),
        ):
            result = await vault_position_tools.query_v1_deposits_logic(
                "0x3333333333333333333333333333333333333333"
            )

        self.assertFalse(result.found)
        self.assertFalse(result.complete)
        self.assertIn("No confident no-deposit conclusion", result.text)

    async def test_v1_withdrawal_uses_loaded_catalog_without_network(self) -> None:
        vault = vault_position_tools.V1_VAULTS[0]
        with patch(
            "vault_position_tools.get_web3_instance",
            return_value=object(),
        ):
            result = await vault_position_tools.core_get_withdrawal_instructions(
                None,
                vault["address"],
                "ethereum",
            )

        self.assertIn("deprecated Yearn V1 vault", result)
        self.assertIn(vault["name"], result)
        self.assertIn("withdraw", result)

    async def test_v3_withdrawal_uses_exact_kong_snapshot(self) -> None:
        vault_address = "0x1111111111111111111111111111111111111111"
        fetch_snapshot = AsyncMock(
            return_value={
                "chainId": 1,
                "address": vault_address,
                "name": "USDC yVault",
                "apiVersion": "3.0.4",
                "origin": "yearn",
            }
        )
        with (
            patch("vault_position_tools.get_web3_instance", return_value=object()),
            patch(
                "vault_position_tools.fetch_kong_vault_snapshot",
                new=fetch_snapshot,
            ),
        ):
            result = await vault_position_tools.core_get_withdrawal_instructions(
                None,
                vault_address,
                "ethereum",
            )

        self.assertIn("Find the **'redeem'** function", result)
        fetch_snapshot.assert_awaited_once_with(1, vault_address)

    async def test_withdrawal_rejects_non_yearn_kong_snapshot(self) -> None:
        vault_address = "0x1111111111111111111111111111111111111111"
        with (
            patch("vault_position_tools.get_web3_instance", return_value=object()),
            patch(
                "vault_position_tools.fetch_kong_vault_snapshot",
                new=AsyncMock(
                    return_value={
                        "chainId": 1,
                        "address": vault_address,
                        "name": "External Vault",
                        "apiVersion": "3.0.4",
                        "origin": None,
                    }
                ),
            ),
        ):
            result = await vault_position_tools.core_get_withdrawal_instructions(
                None,
                vault_address,
                "ethereum",
            )

        self.assertIn("Could not fetch vault details from the Yearn API", result)
        self.assertNotIn("Find the **'redeem'** function", result)

    def test_v1_catalog_loads_outside_repository_working_directory(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(repository_root), environment.get("PYTHONPATH")),
            )
        )
        with tempfile.TemporaryDirectory() as working_directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import vault_position_tools; print(len(vault_position_tools.V1_VAULTS))",
                ],
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            int(completed.stdout.strip()), len(vault_position_tools.V1_VAULTS)
        )
        self.assertGreater(len(vault_position_tools.V1_VAULTS), 0)
