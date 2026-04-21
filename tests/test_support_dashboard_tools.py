import json
import unittest
from unittest import mock

import support_dashboard_tools


class SupportDashboardToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_tools_require_base_url(self) -> None:
        with mock.patch.object(
            support_dashboard_tools.config,
            "SUPPORT_DASHBOARD_BASE_URL",
            "",
        ):
            with self.assertRaises(RuntimeError):
                await support_dashboard_tools.core_support_dashboard_discover(
                    chain_id=1,
                    token_symbol="USDC",
                )

    async def test_core_support_dashboard_harvests_formats_recent_rows(self) -> None:
        payload = {
            "generated_at_utc": "2026-04-20T14:27:45+00:00",
            "scope": {"level": "vault"},
            "filters": {"chain_id": 1, "vault_address": "0xvault"},
            "trailing_24h": {"harvest_count": 2},
            "chain_rollups": [{"chain_id": 1, "last_harvest_at": "2026-04-20T08:18:23+00:00"}],
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
            "last_run": {"status": "success"},
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ):
            result = await support_dashboard_tools.core_support_dashboard_harvests(
                days=7,
                chain_id=1,
                vault_address="0xvault",
                limit=5,
            )

        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/harvests")
        self.assertEqual(parsed["filters"]["vault_address"], "0xvault")
        self.assertEqual(parsed["recent"][0]["strategy_address"], "0xstrategy")

    async def test_core_support_dashboard_discover_compacts_rows(self) -> None:
        payload = {
            "filters": {"chain_id": 1, "token_symbol": "USDC"},
            "pagination": {"limit": 2, "total": 1},
            "summary": {"vaults": 1},
            "coverage": {"coverage_ratio": 1.0},
            "rows": [
                {
                    "vault_address": "0xvault",
                    "chain_id": 1,
                    "name": "USDC-1 yVault",
                    "symbol": "yvUSDC-1",
                    "token_symbol": "USDC",
                    "category": "Stablecoin",
                    "tvl_usd": 123.0,
                    "est_apy": 0.02,
                    "safe_apy_30d": 0.03,
                    "realized_apy_30d": 0.03,
                    "strategies_count": 7,
                    "last_point_time": "2026-04-20T00:00:00Z",
                    "regime": "stable",
                    "is_retired": False,
                    "migration_available": False,
                }
            ],
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ):
            result = await support_dashboard_tools.core_support_dashboard_discover(
                chain_id=1,
                token_symbol="USDC",
                limit=2,
            )

        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/discover")
        self.assertEqual(parsed["rows"][0]["symbol"], "yvUSDC-1")

    async def test_core_support_dashboard_token_venues_uses_symbol_path(self) -> None:
        payload = {
            "filters": {"universe": "core"},
            "summary": {"venues": 2},
            "rows": [
                {
                    "vault_address": "0x1",
                    "chain_id": 1,
                    "name": "USDC-1 yVault",
                    "symbol": "yvUSDC-1",
                    "category": "Stablecoin",
                    "tvl_usd": 100.0,
                    "est_apy": 0.01,
                    "safe_apy_30d": 0.02,
                    "realized_apy_30d": 0.02,
                    "regime": "stable",
                    "last_point_time": "2026-04-20T00:00:00Z",
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

        fetch_mock.assert_awaited_once()
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/assets/USDC/venues")
        self.assertEqual(parsed["summary"]["venues"], 2)

    async def test_core_support_dashboard_changes_limits_movers(self) -> None:
        payload = {
            "filters": {"window": "7d"},
            "summary": {"vaults_eligible": 1},
            "freshness": {"window_stale_vaults": 0},
            "movers": {
                "risers": [{"symbol": "A"}, {"symbol": "B"}],
                "fallers": [{"symbol": "C"}, {"symbol": "D"}],
                "largest_abs_delta": [{"symbol": "E"}, {"symbol": "F"}],
            },
            "stale": [{"symbol": "G"}, {"symbol": "H"}],
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
        self.assertEqual(len(parsed["stale"]), 1)

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
        ):
            result = await support_dashboard_tools.core_support_dashboard_styfi(
                days=30,
                epoch_limit=12,
                chain_id=1,
            )

        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/styfi")
        self.assertEqual(len(parsed["latest_snapshots"]), 5)
        self.assertEqual(parsed["latest_snapshots"][0]["observed_at"], "2026-04-20T00:02:00Z")
