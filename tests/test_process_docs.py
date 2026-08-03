import tests as _test_environment  # noqa: F401

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

with patch.dict("sys.modules", {"frontmatter": Mock(), "tiktoken": Mock()}):
    from yearn_rag import process_docs
    from yearn_rag.process_docs import YipMetadataError, extract_yip_metadata


class ProcessDocsTests(unittest.TestCase):
    def test_missing_source_directory_fails_without_replacing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"

            with self.assertRaisesRegex(RuntimeError, "source directory is missing"):
                process_docs.process_markdown_files(missing, [])

    def test_file_processing_failure_aborts_the_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            (source_dir / "broken.md").write_text("# Broken", encoding="utf-8")

            with (
                patch.object(
                    process_docs.frontmatter,
                    "load",
                    side_effect=ValueError("bad frontmatter"),
                ),
                self.assertRaisesRegex(RuntimeError, "bad frontmatter"),
            ):
                process_docs.process_markdown_files(source_dir, [])

    def test_atomic_writer_preserves_unchanged_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "docs.json"

            self.assertTrue(process_docs.write_text_if_changed(path, "content\n"))
            original_mtime = path.stat().st_mtime_ns
            self.assertFalse(process_docs.write_text_if_changed(path, "content\n"))

            self.assertEqual(path.stat().st_mtime_ns, original_mtime)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

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
