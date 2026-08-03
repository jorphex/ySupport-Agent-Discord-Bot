import tests as _test_environment  # noqa: F401

import unittest
import warnings

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from mcp.server.auth.middleware.bearer_auth import (
    BearerAuthBackend,
    RequireAuthMiddleware,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from starlette.testclient import TestClient

from mcp_server import _build_mcp_server, _StaticBearerTokenVerifier, mcp


class StaticBearerTokenVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_only_configured_token(self) -> None:
        verifier = _StaticBearerTokenVerifier("expected-token")

        accepted = await verifier.verify_token("expected-token")
        rejected = await verifier.verify_token("wrong-token")

        self.assertIsNotNone(accepted)
        self.assertEqual(accepted.client_id, "ysupport-codex")
        self.assertEqual(accepted.scopes, ["ysupport"])
        self.assertIsNone(rejected)

    async def test_missing_configured_token_fails_closed(self) -> None:
        verifier = _StaticBearerTokenVerifier(None)

        self.assertIsNone(await verifier.verify_token(""))
        self.assertIsNone(await verifier.verify_token("any-token"))


class MCPToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_tool_schemas_match_current_api_contract(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        discover = tools["support_dashboard_discover"].inputSchema["properties"]
        harvests = tools["support_dashboard_harvests"].inputSchema["properties"]
        styfi = tools["support_dashboard_styfi"].inputSchema["properties"]

        self.assertEqual(
            set(discover),
            {"chain_id", "market", "universe", "sort_by", "limit"},
        )
        self.assertEqual(
            discover["market"]["enum"],
            ["all", "stablecoins", "eth", "bitcoin", "other"],
        )
        self.assertIsNone(harvests["chain_id"]["default"])
        self.assertEqual(harvests["days"]["minimum"], 7)
        self.assertEqual(set(styfi), {"days", "epoch_limit"})
        self.assertEqual(styfi["days"]["maximum"], 122)
        self.assertEqual(styfi["epoch_limit"]["minimum"], 3)


class MCPAuthenticationIntegrationTests(unittest.TestCase):
    def _client(self) -> TestClient:
        verifier = _StaticBearerTokenVerifier("expected-token")

        async def protected_endpoint(scope, receive, send) -> None:
            response = PlainTextResponse("authorized")
            await response(scope, receive, send)

        app = Starlette(
            routes=[
                Route(
                    "/mcp",
                    endpoint=RequireAuthMiddleware(
                        protected_endpoint,
                        ["ysupport"],
                    ),
                    methods=["POST"],
                )
            ],
            middleware=[
                Middleware(
                    AuthenticationMiddleware,
                    backend=BearerAuthBackend(verifier),
                )
            ],
        )
        return TestClient(
            app,
            base_url="http://127.0.0.1:18002",
        )

    def test_fastmcp_server_uses_static_verifier_and_required_scope(self) -> None:
        server = _build_mcp_server(
            host="127.0.0.1",
            port=18002,
            api_key="expected-token",
        )

        self.assertIsInstance(
            server._token_verifier,
            _StaticBearerTokenVerifier,
        )
        self.assertEqual(
            server.settings.auth.required_scopes,
            ["ysupport"],
        )

    def test_missing_bearer_token_is_rejected(self) -> None:
        with self._client() as client:
            response = client.post("/mcp")

        self.assertEqual(response.status_code, 401)

    def test_wrong_bearer_token_is_rejected(self) -> None:
        with self._client() as client:
            response = client.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer wrong-token",
                },
            )

        self.assertEqual(response.status_code, 401)

    def test_correct_bearer_token_initializes_mcp(self) -> None:
        with self._client() as client:
            response = client.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer expected-token",
                },
            )

        self.assertEqual(response.status_code, 200)
