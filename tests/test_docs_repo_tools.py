import tests as _test_environment  # noqa: F401

import unittest
from unittest.mock import Mock, patch

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
