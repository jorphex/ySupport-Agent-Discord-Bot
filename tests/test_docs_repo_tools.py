import unittest

from docs_repo_tools import _should_include_flex_docs


class DocsRepoToolsTests(unittest.TestCase):
    def test_should_include_flex_docs_for_explicit_product_name(self) -> None:
        self.assertTrue(_should_include_flex_docs("what is flex?"))

    def test_should_include_flex_docs_for_unique_flex_terms(self) -> None:
        self.assertTrue(_should_include_flex_docs("how do troves get redeemed?"))
        self.assertTrue(_should_include_flex_docs("what is a lender vault?"))
        self.assertTrue(_should_include_flex_docs("how is this different from liquity?"))

    def test_should_not_include_flex_docs_for_generic_redemption_wording(self) -> None:
        self.assertFalse(_should_include_flex_docs("what does redeem mean in yearn?"))

