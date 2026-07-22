import unittest
from unittest.mock import Mock, patch

with patch.dict("sys.modules", {"frontmatter": Mock(), "tiktoken": Mock()}):
    from yearn_rag.process_docs import YipMetadataError, extract_yip_metadata


class ProcessDocsTests(unittest.TestCase):
    def test_extracts_canonical_yip_metadata_table(self) -> None:
        content = """
| Metadata | Details |
| --- | --- |
| YIP | 91 |
| Outcome | **Active** |
| Created | 2026-07-06 |
| Forum discussion | [View discussion](https://gov.yearn.fi/t/yip-91) |
"""

        self.assertEqual(
            extract_yip_metadata(
                "contributing/governance/yips/yip-91.md", content
            ),
            {
                "yip_number": 91,
                "status": "Active",
                "created_date": "2026-07-06",
                "discussion_link": "https://gov.yearn.fi/t/yip-91",
            },
        )

    def test_preserves_yip_zero_metadata(self) -> None:
        content = """
| Metadata | Details |
| --- | --- |
| YIP | 0 |
| Outcome | **Passed** |
| Created | 2020-07-22 |
| Forum discussion | [View discussion](https://gov.yearn.fi/) |
"""

        metadata = extract_yip_metadata(
            "contributing/governance/yips/yip-0.md", content
        )

        self.assertEqual(metadata["yip_number"], 0)

    def test_rejects_mismatched_yip_metadata(self) -> None:
        content = """
| Metadata | Details |
| --- | --- |
| YIP | 90 |
| Outcome | **Passed** |
| Created | 2025-12-12 |
| Forum discussion | [View discussion](https://gov.yearn.fi/t/yip-90) |
"""

        with self.assertRaisesRegex(YipMetadataError, "metadata mismatch"):
            extract_yip_metadata(
                "contributing/governance/yips/yip-91.md", content
            )
