import unittest
from unittest.mock import AsyncMock, patch


import onchain_tools
import vault_search_tools


class _VaultSearchResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        return None

    async def json(self):
        return self.payload


class _VaultSearchSession:
    def __init__(self, payload) -> None:
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, *args, **kwargs):
        return _VaultSearchResponse(self.payload)


class OnchainInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_tx_summary_profiles_contracts_and_formats_transfers(self) -> None:
        class _HexLike:
            def __init__(self, value: str) -> None:
                self._value = value

            def hex(self) -> str:
                return self._value

        class _FakeEth:
            def get_transaction_receipt(self, tx_hash: str) -> dict:
                return {
                    "status": 1,
                    "blockNumber": 27331428,
                    "gasUsed": 771090,
                    "logs": [
                        {
                            "address": "0x2222222222222222222222222222222222222222",
                            "logIndex": 0,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        }
                    ],
                }

            def get_transaction(self, tx_hash: str) -> dict:
                return {
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x3333333333333333333333333333333333333333",
                }

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        async def fake_inspect_contract_profile(
            web3_instance, contract_address: str, *, block_identifier=None
        ):
            self.assertEqual(block_identifier, 27331428)
            return {
                "address": contract_address,
                "symbol": "yvWBUSDT",
                "name": "Yearn Katana Vault",
                "decimals": 18,
                "asset": "0x4444444444444444444444444444444444444444",
                "asset_symbol": "USDT",
                "asset_decimals": 6,
                "kind": "erc4626_like",
                "has_code": True,
            }

        with patch.dict(
            onchain_tools.chain_access.WEB3_INSTANCES,
            {"katana": _FakeWeb3()},
            clear=True,
        ):
            with patch(
                "onchain_tools._decode_logs_with_abis",
                return_value=[
                    {
                        "event": "Transfer",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 0,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "from": "0x1111111111111111111111111111111111111111",
                            "to": "0x5555555555555555555555555555555555555555",
                            "value": 650914700000000000000,
                        },
                    }
                ],
            ):
                with patch(
                    "onchain_tools._inspect_contract_profile",
                    new=fake_inspect_contract_profile,
                ):
                    summary = await onchain_tools.core_inspect_onchain(
                        chain="katana",
                        mode="tx_summary",
                        tx_hash="0x" + "a" * 64,
                    )

        self.assertIn("Transaction summary", summary)
        self.assertIn("contracts_profiled", summary)
        self.assertIn("yvWBUSDT", summary)
        self.assertIn("650.9147", summary)
        self.assertIn("erc4626_like", summary)

    async def test_tx_investigate_summarizes_transfer_and_approval_activity(
        self,
    ) -> None:
        class _HexLike:
            def __init__(self, value: str) -> None:
                self._value = value

            def hex(self) -> str:
                return self._value

        class _FakeEth:
            def get_transaction_receipt(self, tx_hash: str) -> dict:
                return {
                    "status": 1,
                    "blockNumber": 27331428,
                    "gasUsed": 771090,
                    "logs": [
                        {
                            "address": "0x2222222222222222222222222222222222222222",
                            "logIndex": 0,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        },
                        {
                            "address": "0x3333333333333333333333333333333333333333",
                            "logIndex": 1,
                            "transactionHash": _HexLike(tx_hash),
                            "topics": [],
                            "data": "0x",
                        },
                    ],
                }

            def get_transaction(self, tx_hash: str) -> dict:
                return {
                    "from": "0x1111111111111111111111111111111111111111",
                    "to": "0x4444444444444444444444444444444444444444",
                }

        class _FakeWeb3:
            def __init__(self) -> None:
                self.eth = _FakeEth()

        async def fake_inspect_contract_profile(
            web3_instance, contract_address: str, *, block_identifier=None
        ):
            self.assertEqual(block_identifier, 27331428)
            if contract_address.lower() == "0x2222222222222222222222222222222222222222":
                return {
                    "address": contract_address,
                    "symbol": "frxUSD",
                    "name": "Frax USD",
                    "decimals": 18,
                    "asset": None,
                    "asset_symbol": None,
                    "asset_decimals": None,
                    "kind": "erc20_like",
                    "has_code": True,
                }
            return {
                "address": contract_address,
                "symbol": "yvWBUSDT",
                "name": "Yearn Katana Vault",
                "decimals": 18,
                "asset": "0x5555555555555555555555555555555555555555",
                "asset_symbol": "USDT",
                "asset_decimals": 6,
                "kind": "erc4626_like",
                "has_code": True,
            }

        with patch.dict(
            onchain_tools.chain_access.WEB3_INSTANCES,
            {"katana": _FakeWeb3()},
            clear=True,
        ):
            with patch(
                "onchain_tools._decode_logs_with_abis",
                return_value=[
                    {
                        "event": "Transfer",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 0,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "from": "0x1111111111111111111111111111111111111111",
                            "to": "0x3333333333333333333333333333333333333333",
                            "value": 1990379783000000000000,
                        },
                    },
                    {
                        "event": "Approval",
                        "address": "0x2222222222222222222222222222222222222222",
                        "log_index": 1,
                        "transaction_hash": "0x" + "a" * 64,
                        "args": {
                            "owner": "0x1111111111111111111111111111111111111111",
                            "spender": "0x3333333333333333333333333333333333333333",
                            "value": 1990379783000000000000,
                        },
                    },
                ],
            ):
                with patch(
                    "onchain_tools._inspect_contract_profile",
                    new=fake_inspect_contract_profile,
                ):
                    investigation = await onchain_tools.core_inspect_onchain(
                        chain="katana",
                        mode="tx_investigate",
                        tx_hash="0x" + "a" * 64,
                    )

        self.assertIn("Transaction investigation", investigation)
        self.assertIn("user_transfers_out", investigation)
        self.assertIn("approvals", investigation)
        self.assertIn("Observed 1 transfer(s) out from the tx sender.", investigation)
        self.assertIn("Decoded 1 approval event(s).", investigation)




class SearchVaultsTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _vault(
        *,
        address: str,
        name: str,
        symbol: str,
        chain_id: int = 1,
        asset_symbol: str = "USDC",
        tvl=1_000_000,
        net_apy=0.04,
        kind: str = "Multi Strategy",
        strategies_count: int = 2,
        retired: bool = False,
        highlighted: bool = False,
    ) -> dict:
        return {
            "chainId": chain_id,
            "address": address,
            "name": name,
            "symbol": symbol,
            "apiVersion": "3.0.4",
            "decimals": 6,
            "asset": {
                "name": asset_symbol,
                "symbol": asset_symbol,
                "address": "0x2222222222222222222222222222222222222222",
            },
            "tvl": tvl,
            "performance": {
                "oracle": {"netAPY": net_apy},
                "historical": {
                    "net": net_apy,
                    "weeklyNet": net_apy,
                    "monthlyNet": net_apy,
                    "inceptionNet": net_apy,
                },
            },
            "fees": {"performanceFee": 1000, "managementFee": 0},
            "kind": kind,
            "inclusion": {"isYearn": True},
            "strategiesCount": strategies_count,
            "riskLevel": 1,
            "isRetired": retired,
            "isHidden": False,
            "isHighlighted": highlighted,
            "origin": "yearn",
            "migration": False,
            "pricePerShare": 1_000_000,
        }

    async def test_core_search_vaults_rejects_unknown_chain(self) -> None:
        fetch_catalog = AsyncMock(return_value=[])
        with patch(
            "vault_search_tools.fetch_kong_vault_catalog",
            new=fetch_catalog,
        ):
            result = await vault_search_tools.core_search_vaults(
                "USDC", chain="mainnet"
            )

        self.assertIn("Unsupported chain: 'mainnet'", result)
        self.assertNotIn("Found", result)
        fetch_catalog.assert_not_awaited()

    async def test_core_search_vaults_reports_provider_failure(self) -> None:
        with patch(
            "vault_search_tools.fetch_kong_vault_catalog",
            new=AsyncMock(side_effect=ValueError("invalid catalog")),
        ):
            result = await vault_search_tools.core_search_vaults("USDC")

        self.assertEqual(
            result,
            "Error: An unexpected error occurred while fetching vault data: "
            "invalid catalog.",
        )

    async def test_core_search_vaults_does_not_recommend_retired_vault(self) -> None:
        retired = self._vault(
            address="0x1111111111111111111111111111111111111111",
            name="Retired Yearn Vault",
            symbol="yvOLD",
            retired=True,
        )
        with patch(
            "vault_search_tools.fetch_kong_vault_catalog",
            new=AsyncMock(return_value=[retired]),
        ):
            result = await vault_search_tools.core_search_vaults(
                "all", recommended_only=True
            )

        self.assertEqual(
            result,
            "No recommendation-grade active Yearn vaults found matching your criteria.",
        )

    async def test_core_search_vaults_tolerates_malformed_numeric_fields(self) -> None:
        vault = self._vault(
            address="0x1111111111111111111111111111111111111111",
            name="Yearn USDC",
            symbol="yvUSDC",
            tvl="unknown",
            net_apy="unknown",
        )
        with (
            patch(
                "vault_search_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[vault]),
            ),
            patch(
                "vault_search_tools.fetch_kong_vault_snapshots",
                new=AsyncMock(return_value=[None]),
            ),
        ):
            result = await vault_search_tools.core_search_vaults(
                "all", recommended_only=True
            )

        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)

    def test_format_single_vault_data_handles_nullable_fields(self) -> None:
        vault = self._vault(
            address="0x1111111111111111111111111111111111111111",
            name="Nullable Vault",
            symbol="yvNULL",
            tvl=None,
            net_apy=None,
        )
        vault.update(
            {
                "fees": {"performanceFee": None, "managementFee": None},
                "composition": [
                    {
                        "name": "Strat One",
                        "address": "0x3333333333333333333333333333333333333333",
                        "status": "active",
                        "performance": {"historical": {"net": None}},
                        "currentDebtUsd": None,
                        "lastReport": None,
                    }
                ],
                "staking": {
                    "available": True,
                    "source": "VeYFI",
                    "address": "0x4444444444444444444444444444444444444444",
                },
            }
        )

        formatted = vault_search_tools.format_single_vault_data_for_llm(vault, 1)

        self.assertIn("Current Estimated Net APY: N/A", formatted)
        self.assertIn("Vault Fees: Performance=N/A, Management=N/A", formatted)
        self.assertIn("Historical Realized Net APY: Week=N/A, Month=N/A", formatted)
        self.assertIn("Realized APY: N/A", formatted)
        self.assertIn("Staking Opportunity: Yes", formatted)

    async def test_core_search_vaults_supports_all_query(self) -> None:
        vaults = [
            self._vault(
                address="0x1111111111111111111111111111111111111111",
                name="Yearn USDC",
                symbol="yvUSDC",
            ),
            self._vault(
                address="0x3333333333333333333333333333333333333333",
                name="Yearn DAI",
                symbol="yvDAI",
                chain_id=42161,
                asset_symbol="DAI",
                tvl=500_000,
            ),
        ]
        with (
            patch(
                "vault_search_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=vaults),
            ),
            patch(
                "vault_search_tools.fetch_kong_vault_snapshots",
                new=AsyncMock(return_value=[None, None]),
            ),
        ):
            result = await vault_search_tools.core_search_vaults("all")

        self.assertIn("Found 2 Yearn vault(s) matching 'all'.", result)
        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)
        self.assertIn("Vault: Yearn DAI (yvDAI)", result)

    async def test_core_search_vaults_enriches_match_with_exact_snapshot(self) -> None:
        vault = self._vault(
            address="0x1111111111111111111111111111111111111111",
            name="Catalog Name",
            symbol="yvUSDC",
        )
        snapshot = dict(vault)
        snapshot["name"] = "Snapshot Name"
        fetch_snapshots = AsyncMock(return_value=[snapshot])
        with (
            patch(
                "vault_search_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[vault]),
            ),
            patch(
                "vault_search_tools.fetch_kong_vault_snapshots",
                new=fetch_snapshots,
            ),
        ):
            result = await vault_search_tools.core_search_vaults("USDC")

        self.assertIn("Vault: Snapshot Name (yvUSDC)", result)
        fetch_snapshots.assert_awaited_once_with([(1, vault["address"])])

    async def test_core_search_vaults_falls_back_when_snapshot_fails(self) -> None:
        vault = self._vault(
            address="0x1111111111111111111111111111111111111111",
            name="Catalog Name",
            symbol="yvUSDC",
        )
        with (
            patch(
                "vault_search_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[vault]),
            ),
            patch(
                "vault_search_tools.fetch_kong_vault_snapshots",
                new=AsyncMock(return_value=[RuntimeError("snapshot unavailable")]),
            ),
        ):
            result = await vault_search_tools.core_search_vaults("USDC")

        self.assertIn("Vault: Catalog Name (yvUSDC)", result)

    async def test_core_search_vaults_recommended_only_filters_single_strategy(self) -> None:
        saver = self._vault(
            address="0x1111111111111111111111111111111111111111",
            name="Yearn Saver Strategy",
            symbol="ysUSDC",
            kind="Single Strategy",
            strategies_count=1,
        )
        multi = self._vault(
            address="0x3333333333333333333333333333333333333333",
            name="Yearn USDC",
            symbol="yvUSDC",
            highlighted=True,
        )
        with (
            patch(
                "vault_search_tools.fetch_kong_vault_catalog",
                new=AsyncMock(return_value=[saver, multi]),
            ),
            patch(
                "vault_search_tools.fetch_kong_vault_snapshots",
                new=AsyncMock(return_value=[None]),
            ),
        ):
            result = await vault_search_tools.core_search_vaults(
                "all", recommended_only=True
            )

        self.assertIn("Vault: Yearn USDC (yvUSDC)", result)
        self.assertNotIn("Vault: Yearn Saver Strategy (ysUSDC)", result)
