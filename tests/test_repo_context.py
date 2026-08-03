import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import repo_context
import repo_context_build


class RepoContextStatusTests(unittest.TestCase):
    def test_status_inspection_does_not_create_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "empty.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE unrelated (value TEXT)")

            status = repo_context.get_repo_context_status(
                db_path=db_path,
                enabled=True,
            )

            self.assertEqual(status["state"], "error")
            with sqlite3.connect(db_path) as conn:
                table_count = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
                ).fetchone()[0]
            self.assertEqual(table_count, 1)

    def test_manifest_failure_is_reported_as_status_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "context.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                repo_context_build._ensure_schema(conn)
                repo_context_build._set_build_meta(
                    conn,
                    "repo_context.schema_version",
                    repo_context_build.REPO_CONTEXT_SCHEMA_VERSION,
                )
                repo_context_build._set_build_meta(
                    conn,
                    "repo_context.built_at",
                    datetime.now(timezone.utc).isoformat(),
                )
                repo_context_build._set_build_meta(
                    conn,
                    "repo_context.manifest_hash",
                    "old-hash",
                )
                conn.commit()

            with patch.object(
                repo_context,
                "manifest_hash",
                side_effect=ValueError("invalid manifest"),
            ):
                status = repo_context.get_repo_context_status(
                    db_path=db_path,
                    enabled=True,
                )

            self.assertEqual(status["state"], "error")
            self.assertIn("invalid manifest", status["reason"])


class RepoContextBoundaryTests(unittest.TestCase):
    def test_artifact_ids_stay_stable_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "context.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                repo_context_build._ensure_schema(conn)
                original = repo_context_build.RepoFile(
                    repo_name="yearn/example",
                    repo_ref="main",
                    product_tag="vaults",
                    authority_tag="contract_code",
                    legacy=False,
                    path="README.md",
                    language="markdown",
                    content="# Withdrawals\n\nOriginal behavior.",
                )
                updated = replace(
                    original,
                    content="# Withdrawals\n\nUpdated behavior.",
                )

                repo_context_build._insert_repo_file(conn, original)
                original_ids = conn.execute(
                    "SELECT id FROM segments ORDER BY id"
                ).fetchall()
                repo_context_build._clear_repo_rows(conn, original.repo_name)
                repo_context_build._insert_repo_file(conn, updated)
                updated_ids = conn.execute(
                    "SELECT id FROM segments ORDER BY id"
                ).fetchall()

            self.assertEqual(original_ids, updated_ids)

    def test_removed_manifest_source_rows_are_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "context.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                repo_context_build._ensure_schema(conn)
                for repo_name in ("yearn/current", "yearn/removed"):
                    repo_context_build._insert_repo_file(
                        conn,
                        repo_context_build.RepoFile(
                            repo_name=repo_name,
                            repo_ref="main",
                            product_tag="vaults",
                            authority_tag="contract_code",
                            legacy=False,
                            path="README.md",
                            language="markdown",
                            content=f"# {repo_name}\n\nRepository content.",
                        ),
                    )

                repo_context_build._clear_unconfigured_repo_rows(
                    conn,
                    {"yearn/current"},
                )
                remaining = conn.execute(
                    "SELECT DISTINCT repo_name FROM files"
                ).fetchall()

            self.assertEqual([row[0] for row in remaining], ["yearn/current"])

    def test_removed_manifest_source_cache_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            current = cache_dir / "yearn__current@main"
            removed = cache_dir / "yearn__removed@main"
            current.mkdir()
            removed.mkdir()

            repo_context_build._cleanup_unconfigured_repo_cache_dirs(
                cache_dir,
                {"yearn/current"},
            )

            self.assertTrue(current.exists())
            self.assertFalse(removed.exists())

    def test_search_rejects_unbounded_result_limit(self) -> None:
        with (
            patch.object(
                repo_context,
                "get_repo_context_status",
                return_value={"state": "ready"},
            ),
            self.assertRaisesRegex(ValueError, "Repo search limit"),
        ):
            repo_context.search_repo_context(
                "vault accounting",
                limit=repo_context.MAX_REPO_SEARCH_RESULTS + 1,
            )

    def test_fetch_rejects_unbounded_artifact_count(self) -> None:
        refs = [
            f"segment:{index}"
            for index in range(repo_context.MAX_REPO_ARTIFACTS + 1)
        ]
        with (
            patch.object(
                repo_context,
                "get_repo_context_status",
                return_value={"state": "ready"},
            ),
            self.assertRaisesRegex(ValueError, "repo artifacts"),
        ):
            repo_context.fetch_repo_artifacts(refs)

    def test_build_rejects_manifest_source_with_no_matching_files(self) -> None:
        source = repo_context_build.RepoSource(
            owner="yearn",
            repo="empty",
            ref="main",
            product_tag="test",
            authority_tag="contract_code",
            legacy=False,
            include_globs=("README.md",),
            exclude_globs=(),
        )
        session = MagicMock()
        session.__enter__.return_value = session
        session.__exit__.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    repo_context_build,
                    "load_repo_manifest",
                    return_value=[source],
                ),
                patch.object(
                    repo_context_build,
                    "manifest_hash",
                    return_value="manifest-hash",
                ),
                patch.object(
                    repo_context_build.requests,
                    "Session",
                    return_value=session,
                ),
                patch.object(
                    repo_context_build,
                    "_resolve_default_branch",
                    return_value="main",
                ),
                patch.object(
                    repo_context_build,
                    "_download_repo_archive",
                    return_value=b"archive",
                ),
                patch.object(
                    repo_context_build,
                    "_extract_repo_files",
                    return_value=[],
                ),
                self.assertRaisesRegex(RuntimeError, "matched no files"),
            ):
                repo_context_build.build_repo_context_index(
                    cache_dir=Path(temp_dir) / "cache",
                    db_path=Path(temp_dir) / "context.sqlite3",
                )
