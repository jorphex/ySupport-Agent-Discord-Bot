import tests as _test_environment  # noqa: F401

import asyncio
import json
import unittest
from unittest import mock

import aiohttp

import support_dashboard_tools


class _Response:
    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payload = {} if payload is None else payload
        self.error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    async def json(self):
        return self.payload


class _Session:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.get_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, _url):
        self.get_calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _http_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        request_info=mock.Mock(real_url="https://dashboard.example/api/test"),
        history=(),
        status=status,
        message="test response",
    )


class SupportDashboardToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_fetch_uses_configured_tls_verification(self) -> None:
        session = _Session([_Response()])
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
                return_value=session,
            ),
        ):
            await support_dashboard_tools._fetch_dashboard_json("/api/test", {})

        connector_mock.assert_called_once_with(ssl=True)

    async def test_dashboard_fetch_retries_one_server_error(self) -> None:
        session = _Session(
            [_Response(error=_http_error(503)), _Response({"status": "ok"})]
        )
        with (
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_BASE_URL",
                "https://dashboard.example",
            ),
            mock.patch(
                "support_dashboard_tools.aiohttp.ClientSession",
                return_value=session,
            ),
        ):
            payload = await support_dashboard_tools._fetch_dashboard_json(
                "/api/test", {}
            )

        self.assertEqual(payload, {"status": "ok"})
        self.assertEqual(session.get_calls, 2)

    async def test_dashboard_fetch_retries_one_connection_failure(self) -> None:
        session = _Session(
            [aiohttp.ClientConnectionError("temporary"), _Response({"status": "ok"})]
        )
        with (
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_BASE_URL",
                "https://dashboard.example",
            ),
            mock.patch(
                "support_dashboard_tools.aiohttp.ClientSession",
                return_value=session,
            ),
        ):
            payload = await support_dashboard_tools._fetch_dashboard_json(
                "/api/test", {}
            )

        self.assertEqual(payload, {"status": "ok"})
        self.assertEqual(session.get_calls, 2)

    async def test_dashboard_fetch_does_not_retry_client_error(self) -> None:
        session = _Session([_Response(error=_http_error(422))])
        with (
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_BASE_URL",
                "https://dashboard.example",
            ),
            mock.patch(
                "support_dashboard_tools.aiohttp.ClientSession",
                return_value=session,
            ),
        ):
            with self.assertRaises(aiohttp.ClientResponseError):
                await support_dashboard_tools._fetch_dashboard_json("/api/test", {})

        self.assertEqual(session.get_calls, 1)

    async def test_dashboard_fetch_rejects_non_object_json_without_retry(self) -> None:
        session = _Session([_Response([])])
        with (
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_BASE_URL",
                "https://dashboard.example",
            ),
            mock.patch(
                "support_dashboard_tools.aiohttp.ClientSession",
                return_value=session,
            ),
        ):
            with self.assertRaisesRegex(TypeError, "non-object JSON"):
                await support_dashboard_tools._fetch_dashboard_json("/api/test", {})

        self.assertEqual(session.get_calls, 1)

    async def test_dashboard_fetch_stops_after_second_timeout(self) -> None:
        session = _Session([asyncio.TimeoutError(), asyncio.TimeoutError()])
        with (
            mock.patch.object(
                support_dashboard_tools.config,
                "SUPPORT_DASHBOARD_BASE_URL",
                "https://dashboard.example",
            ),
            mock.patch(
                "support_dashboard_tools.aiohttp.ClientSession",
                return_value=session,
            ),
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await support_dashboard_tools._fetch_dashboard_json("/api/test", {})

        self.assertEqual(session.get_calls, 2)

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

    async def test_reports_require_exact_vault_and_preserve_event_semantics(
        self,
    ) -> None:
        payload = {
            "event": "StrategyReported",
            "trailing_24h": {"report_count": 2},
            "recent": [
                {
                    "block_time": "2026-04-20T08:18:23+00:00",
                    "tx_hash": "0xtx",
                    "log_index": 7,
                    "chain_id": 1,
                    "vault_address": "0xvault",
                    "vault_symbol": "yvTEST",
                    "token_symbol": "USDC",
                    "strategy_address": "0xstrategy",
                    "strategy_name": "Strategy",
                    "report_type": "accounting_update",
                    "gain": "0",
                    "loss": "0",
                    "fee_assets": None,
                    "refund_assets": None,
                    "token_decimals": 6,
                    "vault_version": "3.0.4",
                    "debt_after": "2000000",
                }
            ],
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_reports(
                days=7,
                chain_id=1,
                vault_address=" 0xvault ",
                limit=5,
            )

        fetch_mock.assert_awaited_once_with(
            "/api/reports",
            {
                "chain_id": 1,
                "vault_address": "0xvault",
                "days": 7,
                "limit": 5,
                "meaningful_only": False,
            },
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/reports")
        self.assertFalse(parsed["filters"]["meaningful_only"])
        self.assertEqual(parsed["recent"][0]["report_type"], "accounting_update")
        self.assertEqual(parsed["recent"][0]["token_decimals"], 6)
        self.assertIsNone(parsed["recent"][0]["fee_assets"])
        self.assertIn("inconclusive", parsed["interpretation"]["history_limit"])

    async def test_reports_reject_empty_vault_address_before_provider_call(
        self,
    ) -> None:
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(),
        ) as fetch_mock:
            with self.assertRaisesRegex(ValueError, "vault address is required"):
                await support_dashboard_tools.core_support_dashboard_reports(
                    chain_id=1,
                    vault_address="   ",
                )

        fetch_mock.assert_not_awaited()

    async def test_discover_compacts_rows_and_requests_descending_order(self) -> None:
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
                "direction": "desc",
                "limit": 2,
            },
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/discover")
        self.assertEqual(parsed["rows"][0]["symbol"], "yvUSDC-1")

    async def test_freshness_is_compact_and_marks_system_only_scope(self) -> None:
        payload = {
            "as_of_utc": "2026-08-03T00:00:00Z",
            "threshold": "24h",
            "stale_threshold_seconds": 86400,
            "latest_pps_at": "2026-08-02T00:00:00Z",
            "latest_pps_age_seconds": 86400,
            "metrics_newest_point_at": "2026-08-02T00:00:00Z",
            "metrics_newest_age_seconds": 86400,
            "metrics_rows": 100,
            "pps_vaults_total": 10,
            "pps_vaults_stale": 2,
            "pps_stale_ratio": 0.2,
            "stale_by_chain": [{"chain_id": 1, "vaults": 10}],
            "stale_by_category": [{"category": "core", "vaults": 10}],
            "ingestion_jobs": {"kong_pps_metrics": {"running": False}},
            "alerts": {
                "ingestion_stale:kong_pps_metrics": {
                    "status": "healthy",
                    "is_firing": False,
                    "last_success_at": "2026-08-03T00:00:00Z",
                    "current_age_seconds": 10,
                    "threshold_seconds": 3600,
                    "notify_channels": ["private"],
                }
            },
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_freshness()

        fetch_mock.assert_awaited_once_with(
            "/api/meta/freshness",
            {"threshold": "24h"},
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertIn("cannot prove", parsed["scope"])
        self.assertEqual(parsed["pps_vaults_stale"], 2)
        alert = parsed["alerts"]["ingestion_stale:kong_pps_metrics"]
        self.assertNotIn("notify_channels", alert)

    async def test_styfi_returns_only_tolerant_support_subset(self) -> None:
        payload = {
            "summary": {"reward_epoch": 5},
            "current_reward_state": {"styfi_current_apr": 0.38},
            "freshness": {"latest_snapshot_age_seconds": 20},
            "ingestion": {"last_run": {"status": "success"}},
            "recent_activity": [{"type": "deposit"}],
            "series": {"snapshots": [{"observed_at": "unused"}]},
            "components": {"layout": "unstable"},
        }
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value=payload),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_styfi(
                days=7,
                epoch_limit=3,
            )

        fetch_mock.assert_awaited_once_with(
            "/api/styfi",
            {"days": 7, "epoch_limit": 3},
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertEqual(parsed["source"], "/api/styfi")
        self.assertEqual(parsed["ingestion_last_run"]["status"], "success")
        self.assertNotIn("series", parsed)
        self.assertNotIn("components", parsed)

    async def test_styfi_tolerates_non_object_ingestion(self) -> None:
        with mock.patch(
            "support_dashboard_tools._fetch_dashboard_json",
            new=mock.AsyncMock(return_value={"ingestion": []}),
        ) as fetch_mock:
            result = await support_dashboard_tools.core_support_dashboard_styfi()

        fetch_mock.assert_awaited_once_with(
            "/api/styfi",
            {"days": 7, "epoch_limit": 3},
        )
        parsed = json.loads(result.split("\n", 1)[1])
        self.assertIsNone(parsed["ingestion_last_run"])
