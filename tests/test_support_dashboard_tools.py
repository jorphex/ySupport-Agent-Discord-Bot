import tests as _test_environment  # noqa: F401

import json
import unittest
from unittest import mock

import support_dashboard_tools


class SupportDashboardToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_fetch_uses_configured_tls_verification(self) -> None:
        class _Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def raise_for_status(self) -> None:
                return None

            async def json(self):
                return {}

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def get(self, _url):
                return _Response()

        with (
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_BASE_URL",
                "https://dashboard.example",
            ),
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_VERIFY_SSL",
                True,
            ),
            mock.patch(
                "support_dashboard_tools.aiohttp.TCPConnector"
            ) as connector_mock,
            mock.patch(
                "support_dashboard_tools.aiohttp.ClientSession",
                return_value=_Session(),
            ),
        ):
            await support_dashboard_tools._fetch_dashboard_json("/api/test", {})

        connector_mock.assert_called_once_with(ssl=True)

    async def test_dashboard_tools_require_base_url(self) -> None:
        with mock.patch.object(
            support_dashboard_tools.config,
            "SUPPORT_DASHBOARD_BASE_URL",
            "",
        ):
            with self.assertRaises(RuntimeError):
                await support_dashboard_tools.core_support_dashboard_discover(
                    chain_id=1,
                    market="stablecoins",
                )

    async def test_core_support_dashboard_harvests_formats_recent_rows(self) -> None:
        payload = {
            "event": "StrategyReported",
            "trailing_24h": {"report_count": 2},
            "available_chains": [{"chain_id": 1, "chain_label": "Ethereum"}],
            "recent": [
                {
                    "block_time": "2026-04-20T08:18:23+00:00",
                    "tx_hash": "0xtx",
                    "vault_address": "0xvault",
                    "vault_symbol": "yvTEST",
                    "token_symbol": "USDC",
                    "strategy_address": "0xstrategy",
                    "gain": "1",
                    "loss": "0",
                    "debt_after": "2",
                    "fee_assets": "0",
                    "refund_assets": "0",
                }
            ],
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_harvests(
                days=7,
                chain_id=1,
                vault_address="0xvault",
                limit=5,
            )

        fetch_mock.assert_awaited_once_with(
            "/api/reports",
            {
                "days": 7,
                "chain_id": 1,
                "vault_address": "0xvault",
                "limit": 5,
            },
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/reports")
        self.assertEqual(parsed["filters"]["vault_address"], "0xvault")
        self.assertEqual(parsed["event"], "StrategyReported")
        self.assertEqual(parsed["recent"][0]["strategy_address"], "0xstrategy")

    async def test_core_support_dashboard_discover_compacts_rows(self) -> None:
        payload = {
            "filters": {"chain_id": 1, "market": "stablecoins"},
            "pagination": {"limit": 2, "total": 1},
            "summary": {"vaults": 1},
            "coverage": {"coverage_ratio": 1.0},
            "rows": [
                {
                    "vault_address": "0xvault",
                    "chain_id": 1,
                    "symbol": "yvUSDC-1",
                    "tvl_usd": 123.0,
                    "est_apy": 0.02,
                    "realized_apy_30d": 0.03,
                    "momentum_7d_30d": 0.001,
                    "market": "stablecoins",
                }
            ],
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_discover(
                chain_id=1,
                market="stablecoins",
                limit=2,
            )

        fetch_mock.assert_awaited_once_with(
            "/api/discover",
            {
                "chain_id": 1,
                "market": "stablecoins",
                "universe": "core",
                "sort_by": "tvl",
                "limit": 2,
            },
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/discover")
        self.assertEqual(parsed["rows"][0]["symbol"], "yvUSDC-1")

    async def test_core_support_dashboard_token_venues_uses_symbol_path(self) -> None:
        payload = {
            "filters": {"universe": "core"},
            "summary": {"vaults": 2},
            "rows": [
                {
                    "vault_address": "0x1",
                    "chain_id": 1,
                    "symbol": "yvUSDC-1",
                    "tvl_usd": 100.0,
                    "est_apy": 0.01,
                    "realized_apy_30d": 0.02,
                    "momentum_7d_30d": 0.001,
                }
            ],
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_token_venues(
                token_symbol="USDC",
                universe="core",
            )

        fetch_mock.assert_awaited_once_with(
            "/api/assets/USDC/vaults",
            {"universe": "core", "limit": 25},
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/assets/USDC/vaults")
        self.assertEqual(parsed["summary"]["vaults"], 2)

    async def test_core_support_dashboard_token_venues_escapes_symbol_path(self) -> None:
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value={"rows": []}),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_token_venues(
                token_symbol=" ../US DC ",
            )

        fetch_mock.assert_awaited_once_with(
            "/api/assets/..%2FUS%20DC/vaults",
            {"universe": "core", "limit": 25},
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/assets/..%2FUS%20DC/vaults")

    async def test_core_support_dashboard_token_venues_rejects_empty_symbol(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "token symbol is required"):
            await support_dashboard_tools.core_support_dashboard_token_venues(
                token_symbol="   "
            )

    async def test_core_support_dashboard_changes_limits_movers(self) -> None:
        payload = {
            "filters": {"window": "7d"},
            "summary": {"vaults_eligible": 1},
            "freshness": {"window_stale_vaults": 0},
            "movers": {
                "risers": [{"symbol": "A"}, {"symbol": "B"}],
                "fallers": [{"symbol": "C"}, {"symbol": "D"}],
            },
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ):
            result = await support_dashboard_tools.core_support_dashboard_changes(
                window="7d",
                universe="core",
                limit=1,
            )

        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(len(parsed["risers"]), 1)
        self.assertEqual(len(parsed["fallers"]), 1)

    async def test_core_support_dashboard_styfi_trims_snapshot_tail(self) -> None:
        payload = {
            "filters": {"days": 30},
            "summary": {"reward_epoch": 5},
            "reward_token": {"symbol": "yvUSDC-1"},
            "current_reward_state": {"styfi_current_apr": 0.38},
            "series": {
                "snapshots": [{"observed_at": f"2026-04-20T00:0{i}:00Z"} for i in range(7)]
            },
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_styfi(
                days=30,
                epoch_limit=12,
            )

        fetch_mock.assert_awaited_once_with(
            "/api/styfi",
            {"days": 30, "epoch_limit": 12},
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/styfi")
        self.assertEqual(len(parsed["latest_snapshots"]), 5)
        self.assertEqual(parsed["latest_snapshots"][0]["observed_at"], "2026-04-20T00:02:00Z")
