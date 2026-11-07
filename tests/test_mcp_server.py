import tests as _test_environment  # noqa: F401

import unittest

from ysupport.integrations.mcp_server import _StaticBearerTokenVerifier, mcp


class StaticBearerTokenVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_configured_token_fails_closed(self) -> None:
        verifier = _StaticBearerTokenVerifier(None)

        self.assertIsNone(await verifier.verify_token(""))
        self.assertIsNone(await verifier.verify_token("any-token"))


class MCPToolContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_docs_and_repo_tools_enforce_bounded_inputs(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        docs = tools["search_documentation"].inputSchema["properties"]
        repo = tools["search_repo_context"].inputSchema["properties"]
        artifacts = tools["fetch_repo_artifacts"].inputSchema["properties"]

        self.assertEqual(docs["query"]["maxLength"], 2000)
        self.assertEqual(
            set(repo),
            {"query", "limit", "include_legacy"},
        )
        self.assertEqual(repo["query"]["maxLength"], 2000)
        self.assertEqual(repo["limit"]["maximum"], 12)
        self.assertEqual(artifacts["artifact_refs_text"]["maxLength"], 512)
