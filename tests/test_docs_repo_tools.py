import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import AsyncMock, Mock, patch

import docs_repo_tools
from docs_repo_tools import (
    _docs_query_specs,
    _get_pinecone_index,
    _should_include_flex_docs,
)


class DocsRepoToolsTests(unittest.TestCase):
    def test_pinecone_index_is_initialized_lazily_and_reused(self) -> None:
        fake_index = object()
        fake_client = Mock()
        fake_client.Index.return_value = fake_index

        with (
            patch.object(docs_repo_tools, "pinecone_client", None),
            patch.object(docs_repo_tools, "pinecone_index", None),
            patch("docs_repo_tools.Pinecone", return_value=fake_client) as pinecone,
        ):
            self.assertIs(_get_pinecone_index(), fake_index)
            self.assertIs(_get_pinecone_index(), fake_index)

        pinecone.assert_called_once_with(api_key=docs_repo_tools.config.PINECONE_API_KEY)
        fake_client.Index.assert_called_once_with(
            docs_repo_tools.config.PINECONE_INDEX_NAME
        )

    def test_should_include_flex_docs_for_explicit_product_name(self) -> None:
        self.assertTrue(_should_include_flex_docs("what is flex?"))

    def test_should_include_flex_docs_for_unique_flex_terms(self) -> None:
        self.assertTrue(_should_include_flex_docs("how do troves get redeemed?"))
        self.assertTrue(_should_include_flex_docs("what is a lender vault?"))
        self.assertTrue(_should_include_flex_docs("how is this different from liquity?"))

    def test_should_not_include_flex_docs_for_generic_redemption_wording(self) -> None:
        self.assertFalse(_should_include_flex_docs("what does redeem mean in yearn?"))

    def test_docs_query_specs_use_only_current_namespaces(self) -> None:
        self.assertEqual(
            _docs_query_specs(["yearn-docs"], include_yips=False),
            [("yearn-docs", "documentation")],
        )

    def test_docs_query_specs_include_yips_from_yearn_docs(self) -> None:
        self.assertEqual(
            _docs_query_specs(
                ["yearn-docs", "flex-docs"], include_yips=True
            ),
            [
                ("yearn-docs", "documentation"),
                ("yearn-docs", "yip"),
                ("flex-docs", "documentation"),
            ],
        )


class DocsRepoToolsAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_context_search_disables_hyde_and_returns_sources(self) -> None:
        with patch.object(
            docs_repo_tools,
            "_build_docs_context",
            new=AsyncMock(
                return_value=(
                    "Source: Vaults (https://docs.yearn.fi/vaults)\nContent:\nVault text",
                    "YIP-61: Implemented",
                    False,
                )
            ),
        ) as build_context:
            result = await docs_repo_tools.core_search_docs_context("How do vaults work?")

        build_context.assert_awaited_once_with(
            "How do vaults work?",
            use_hyde=False,
        )
        self.assertIn("Ranked Yearn documentation excerpts:", result)
        self.assertIn("https://docs.yearn.fi/vaults", result)
        self.assertIn("Relevant YIP status: YIP-61: Implemented", result)

    async def test_direct_context_search_reports_no_relevant_context(self) -> None:
        with patch.object(
            docs_repo_tools,
            "_build_docs_context",
            new=AsyncMock(return_value=("", "", True)),
        ):
            result = await docs_repo_tools.core_search_docs_context("unknown topic")

        self.assertEqual(
            result,
            "No sufficiently relevant Yearn documentation context was found for this query.",
        )

    async def test_direct_context_build_skips_hyde_call(self) -> None:
        with (
            patch.object(docs_repo_tools, "_get_openai_async_client") as async_client,
            patch.object(
                docs_repo_tools,
                "_get_openai_sync_client",
            ) as sync_client,
        ):
            sync_client.return_value.embeddings.create.side_effect = RuntimeError(
                "stop after embedding boundary"
            )
            with (
                self.assertLogs(level="ERROR"),
                self.assertRaisesRegex(RuntimeError, "Error generating embedding"),
            ):
                await docs_repo_tools._build_docs_context(
                    "direct query",
                    use_hyde=False,
                )

        async_client.assert_not_called()

    async def test_synthesized_answer_retains_hyde_path(self) -> None:
        with (
            patch.object(
                docs_repo_tools,
                "_build_docs_context",
                new=AsyncMock(return_value=("context", "", False)),
            ) as build_context,
            patch.object(
                docs_repo_tools,
                "_synthesize_docs_answer",
                new=AsyncMock(return_value="answer"),
            ) as synthesize,
        ):
            result = await docs_repo_tools.core_answer_from_docs("question")

        self.assertEqual(result, "answer")
        build_context.assert_awaited_once_with("question", use_hyde=True)
        synthesize.assert_awaited_once_with("question", "context", "", False)
