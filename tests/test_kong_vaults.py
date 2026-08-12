import unittest
from unittest.mock import Mock, patch

import kong_vaults
import yearn_targets


class _Response:
    def __init__(self, payload, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    async def json(self):
        return self.payload


class _Session:
    def __init__(self, *, get_payload=None, post_payload=None, status: int = 200):
        self.get_payload = get_payload
        self.post_payload = post_payload
        self.status = status
        self.get_urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **_kwargs):
        self.get_urls.append(url)
        return _Response(self.get_payload, status=self.status)

    def post(self, *_args, **_kwargs):
        return _Response(self.post_payload, status=self.status)


class KongVaultProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        for cache in (
            kong_vaults._VAULT_CATALOG_CACHE,
            kong_vaults._ADDRESS_CATALOG_CACHE,
        ):
            cache.update({"fetched_at": 0.0, "value": None})

    async def test_catalog_keeps_only_canonical_yearn_origin(self) -> None:
        payload = [
            {"address": "0x1", "origin": "yearn"},
            {"address": "0x2", "origin": "external"},
            {"address": "0x3", "origin": None},
        ]
        with patch.object(
            kong_vaults.aiohttp,
            "ClientSession",
            return_value=_Session(get_payload=payload),
        ):
            result = await kong_vaults.fetch_kong_vault_catalog()

        self.assertEqual(result, [payload[0]])

    async def test_catalog_rejects_malformed_rows(self) -> None:
        with patch.object(
            kong_vaults.aiohttp,
            "ClientSession",
            return_value=_Session(get_payload=["not-an-object"]),
        ):
            with self.assertRaisesRegex(ValueError, "Unexpected Kong vault catalog"):
                await kong_vaults.fetch_kong_vault_catalog()

    async def test_snapshot_returns_none_for_not_found(self) -> None:
        with patch.object(
            kong_vaults.aiohttp,
            "ClientSession",
            return_value=_Session(status=404),
        ):
            result = await kong_vaults.fetch_kong_vault_snapshot(1, "0x1")

        self.assertIsNone(result)

    async def test_snapshot_batch_reuses_one_http_session(self) -> None:
        session = _Session(get_payload={"origin": "yearn"})
        session_factory = Mock(return_value=session)
        with patch.object(
            kong_vaults.aiohttp,
            "ClientSession",
            new=session_factory,
        ):
            result = await kong_vaults.fetch_kong_vault_snapshots(
                [(1, "0x1"), (10, "0x2")]
            )

        self.assertEqual(result, [{"origin": "yearn"}, {"origin": "yearn"}])
        session_factory.assert_called_once()
        self.assertEqual(len(session.get_urls), 2)

    async def test_address_catalog_rejects_graphql_errors(self) -> None:
        with patch.object(
            kong_vaults.aiohttp,
            "ClientSession",
            return_value=_Session(post_payload={"errors": [{"message": "failed"}]}),
        ):
            with self.assertRaisesRegex(ValueError, "address catalog query failed"):
                await kong_vaults.fetch_kong_address_catalog()


class KongAddressIndexTests(unittest.TestCase):
    def test_index_preserves_vault_strategy_and_staking_relationships(self) -> None:
        vault = "0x1111111111111111111111111111111111111111"
        strategy = "0x2222222222222222222222222222222222222222"
        staking = "0x3333333333333333333333333333333333333333"
        index = yearn_targets._build_kong_address_index(
            {
                "vaults": [
                    {
                        "chainId": 1,
                        "address": vault,
                        "name": "USDC yVault",
                        "symbol": "yvUSDC",
                        "strategies": [strategy],
                        "staking": {
                            "available": True,
                            "address": staking,
                            "source": "VeYFI",
                        },
                    }
                ],
                "strategies": [
                    {
                        "chainId": 1,
                        "address": strategy,
                        "name": "USDC Lender",
                    }
                ],
            }
        )

        self.assertEqual(index[vault][0]["kind"], "vault")
        self.assertEqual(index[strategy][0]["kind"], "strategy")
        self.assertEqual(index[strategy][0]["vault_address"], vault)
        self.assertEqual(index[staking][0]["kind"], "wrapper_or_gauge")

    def test_canonical_vault_wins_over_nested_strategy_relationships(self) -> None:
        address = "0x1111111111111111111111111111111111111111"
        selected = yearn_targets._select_yearn_address_entry(
            [
                {
                    "kind": "strategy",
                    "chain": "ethereum",
                    "vault_address": "0x2222222222222222222222222222222222222222",
                },
                {
                    "kind": "vault",
                    "chain": "ethereum",
                    "vault_address": address,
                },
                {
                    "kind": "strategy",
                    "chain": "ethereum",
                    "vault_address": "0x3333333333333333333333333333333333333333",
                },
            ],
            chain_hint="ethereum",
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["kind"], "vault")
        self.assertEqual(selected["vault_address"], address)

    def test_index_ignores_strategy_for_non_yearn_vault(self) -> None:
        strategy = "0x2222222222222222222222222222222222222222"
        index = yearn_targets._build_kong_address_index(
            {
                "vaults": [],
                "strategies": [
                    {
                        "chainId": 1,
                        "address": strategy,
                        "name": "External Strategy",
                    }
                ],
            }
        )

        self.assertNotIn(strategy, index)
