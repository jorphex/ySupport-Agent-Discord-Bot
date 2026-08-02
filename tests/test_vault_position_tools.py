import tests as _test_environment  # noqa: F401

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import vault_position_tools


class VaultPositionToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_check_all_deposits_combines_v1_and_active_results(self) -> None:
        with (
            patch(
                "vault_position_tools.resolve_ens_name",
                return_value="0x0000000000000000000000000000000000000001",
            ),
            patch(
                "vault_position_tools.query_v1_deposits_logic",
                new=AsyncMock(return_value="V1 deposit result"),
            ) as query_v1,
            patch(
                "vault_position_tools.query_active_deposits_logic",
                new=AsyncMock(return_value="Active deposit result"),
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
