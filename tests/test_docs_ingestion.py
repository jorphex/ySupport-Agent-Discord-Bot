from __future__ import annotations

import tests as _test_environment  # noqa: F401

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from yearn_rag import embed_and_store
from yearn_rag import fetch_flex_docs
from yearn_rag.fetch_flex_docs import write_markdown_file


def _docs() -> list[dict[str, object]]:
    return [
        {
            "doc_id": "doc-a",
            "chunk_index": 0,
            "chunk_id": 0,
            "source_path": "a.md",
            "text": "Alpha documentation content.",
        },
        {
            "doc_id": "doc-b",
            "chunk_index": 0,
            "chunk_id": 0,
            "source_path": "b.md",
            "text": "Beta documentation content.",
        },
    ]


class _FakeEmbeddings:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: list[str], encoding_format: str):
        self.calls.append(input)
        if self.failure:
            raise self.failure
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(index), 1.0])
                for index, _ in enumerate(input)
            ]
        )


class _FakeEmbeddingClient:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.embeddings = _FakeEmbeddings(failure=failure)


class _FakeIndex:
    def __init__(self, namespaces: dict[str, set[str]] | None = None) -> None:
        self.namespaces = {
            namespace: set(vector_ids)
            for namespace, vector_ids in (namespaces or {}).items()
        }
        self.upserts: list[tuple[str, list[str]]] = []
        self.deletes: list[tuple[str, list[str]]] = []
        self.events: list[str] = []

    def describe_index_stats(self):
        return {
            "namespaces": {
                namespace: {"vector_count": len(vector_ids)}
                for namespace, vector_ids in self.namespaces.items()
            }
        }

    def list(self, *, namespace: str):
        vector_ids = sorted(self.namespaces.get(namespace, set()))
        if vector_ids:
            yield vector_ids

    def upsert(self, *, vectors, namespace: str):
        vector_ids = [vector["id"] for vector in vectors]
        self.namespaces.setdefault(namespace, set()).update(vector_ids)
        self.upserts.append((namespace, vector_ids))
        self.events.append("upsert")
        return {"upserted_count": len(vector_ids)}

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        delete_all: bool | None = None,
        namespace: str,
    ):
        if delete_all:
            self.namespaces.pop(namespace, None)
            return {}
        vector_ids = ids or []
        self.namespaces.setdefault(namespace, set()).difference_update(vector_ids)
        self.deletes.append((namespace, vector_ids))
        self.events.append("delete")
        return {}


class DocsIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "docs.json"
        self.state_path = self.root / "state.json"
        self.docs = _docs()
        self.input_path.write_text(json.dumps(self.docs), encoding="utf-8")
        self.config = {
            "input_json": str(self.input_path),
            "namespace": "test-docs",
        }
        self.desired_ids = {
            embed_and_store.vector_id_for_doc(doc) for doc in self.docs
        }

    def _matching_state(self) -> dict[str, object]:
        return {
            "schema_version": embed_and_store.STATE_SCHEMA_VERSION,
            "sources": {
                "test-docs": {
                    "embedding_model": embed_and_store.EMBEDDING_MODEL,
                    "fingerprint": embed_and_store.source_fingerprint(self.docs),
                    "pipeline_version": embed_and_store.PIPELINE_VERSION,
                    "vector_count": len(self.docs),
                }
            },
        }

    def test_unchanged_source_skips_embeddings_and_writes(self) -> None:
        state = self._matching_state()
        index = _FakeIndex({"test-docs": self.desired_ids})
        client = _FakeEmbeddingClient()

        result = embed_and_store.process_and_embed_source(
            self.config,
            index=index,
            embedding_client=client,
            state=state,
            state_path=self.state_path,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(client.embeddings.calls, [])
        self.assertEqual(index.upserts, [])
        self.assertEqual(index.deletes, [])
        self.assertFalse(self.state_path.exists())

    def test_missing_state_refreshes_before_deleting_stale_ids(self) -> None:
        state = {
            "schema_version": embed_and_store.STATE_SCHEMA_VERSION,
            "sources": {},
        }
        index = _FakeIndex({"test-docs": {"stale-id"}})
        client = _FakeEmbeddingClient()

        result = embed_and_store.process_and_embed_source(
            self.config,
            index=index,
            embedding_client=client,
            state=state,
            state_path=self.state_path,
        )

        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(index.namespaces["test-docs"], self.desired_ids)
        self.assertEqual(index.deletes, [("test-docs", ["stale-id"])])
        self.assertEqual(index.events, ["upsert", "delete"])
        self.assertTrue(self.state_path.exists())
        saved_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_state["sources"]["test-docs"]["vector_count"], 2)

    def test_live_id_drift_forces_repair_despite_matching_count(self) -> None:
        state = self._matching_state()
        index = _FakeIndex({"test-docs": {"wrong-a", "wrong-b"}})
        client = _FakeEmbeddingClient()

        result = embed_and_store.process_and_embed_source(
            self.config,
            index=index,
            embedding_client=client,
            state=state,
            state_path=self.state_path,
        )

        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(index.namespaces["test-docs"], self.desired_ids)
        self.assertEqual(result["deleted_count"], 2)

    def test_changed_source_fingerprint_forces_refresh(self) -> None:
        state = self._matching_state()
        changed_docs = [dict(doc) for doc in self.docs]
        changed_docs[0]["text"] = "Updated alpha documentation content."
        self.input_path.write_text(json.dumps(changed_docs), encoding="utf-8")
        index = _FakeIndex({"test-docs": self.desired_ids})
        client = _FakeEmbeddingClient()

        result = embed_and_store.process_and_embed_source(
            self.config,
            index=index,
            embedding_client=client,
            state=state,
            state_path=self.state_path,
        )

        self.assertEqual(result["status"], "refreshed")
        self.assertEqual(len(client.embeddings.calls), 1)
        saved_state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(
            saved_state["sources"]["test-docs"]["fingerprint"],
            embed_and_store.source_fingerprint(changed_docs),
        )

    def test_embedding_failure_keeps_old_namespace_and_state_unwritten(self) -> None:
        state = {
            "schema_version": embed_and_store.STATE_SCHEMA_VERSION,
            "sources": {},
        }
        index = _FakeIndex({"test-docs": {"old-id"}})
        client = _FakeEmbeddingClient(failure=RuntimeError("provider down"))

        with patch.object(embed_and_store.time, "sleep"), self.assertRaisesRegex(
            RuntimeError,
            "Embedding generation failed",
        ):
            embed_and_store.process_and_embed_source(
                self.config,
                index=index,
                embedding_client=client,
                state=state,
                state_path=self.state_path,
            )

        self.assertEqual(index.namespaces["test-docs"], {"old-id"})
        self.assertEqual(index.deletes, [])
        self.assertFalse(self.state_path.exists())

    def test_corrupt_state_is_treated_as_missing(self) -> None:
        self.state_path.write_text("{not-json", encoding="utf-8")

        self.assertEqual(
            embed_and_store.load_state(self.state_path),
            {
                "schema_version": embed_and_store.STATE_SCHEMA_VERSION,
                "sources": {},
            },
        )

    def test_content_aware_flex_write_preserves_unchanged_file(self) -> None:
        path = self.root / "flex.md"
        self.assertTrue(write_markdown_file(path, "same content"))
        original_mtime = path.stat().st_mtime_ns

        self.assertFalse(write_markdown_file(path, "same content"))

        self.assertEqual(path.stat().st_mtime_ns, original_mtime)

    def test_flex_index_is_discovery_only_not_a_stored_source(self) -> None:
        llms_text = "\n".join(
            [
                "[Docs](https://flexmeow.com/docs)",
                "[Risks](https://flexmeow.com/risks)",
                "[Info](https://flexmeow.com/info)",
            ]
        )
        with patch.object(fetch_flex_docs, "fetch_text", return_value=llms_text):
            targets = fetch_flex_docs.build_fetch_targets()

        self.assertEqual(
            {target["filename"] for target in targets},
            {"docs.md", "risks.md", "info.md"},
        )
        self.assertNotIn(fetch_flex_docs.LLMS_URL, {target["url"] for target in targets})

    def test_flex_index_must_include_every_required_source(self) -> None:
        with (
            patch.object(
                fetch_flex_docs,
                "fetch_text",
                return_value="[Docs](https://flexmeow.com/docs)",
            ),
            self.assertRaisesRegex(RuntimeError, "missing required pages"),
        ):
            fetch_flex_docs.build_fetch_targets()

    def test_refresh_entrypoint_prevents_overlapping_runs(self) -> None:
        script = (
            embed_and_store.BASE_DIR / "yearn_rag" / "update_docs.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('exec 9>"$LOCK_DIR/update_docs.lock"', script)
        self.assertIn("flock -n 9", script)
        self.assertIn("git pull --ff-only origin master", script)

    def test_flex_refresh_removes_sources_no_longer_in_the_target_set(self) -> None:
        output_dir = self.root / "flex-docs"
        output_dir.mkdir()
        stale_path = output_dir / "stale.md"
        stale_path.write_text("stale", encoding="utf-8")

        with (
            patch.object(fetch_flex_docs, "OUTPUT_DIR", output_dir),
            patch.object(
                fetch_flex_docs,
                "convert_html_page",
                return_value="current",
            ),
        ):
            fetch_flex_docs.fetch_all_targets(
                [
                    {
                        "url": "https://flexmeow.com/docs",
                        "filename": "docs.md",
                    }
                ]
            )

        self.assertFalse(stale_path.exists())
        self.assertEqual(
            (output_dir / "docs.md").read_text(encoding="utf-8"),
            "current",
        )

    def test_failed_flex_refresh_does_not_prune_existing_sources(self) -> None:
        output_dir = self.root / "flex-docs"
        output_dir.mkdir()
        stale_path = output_dir / "stale.md"
        stale_path.write_text("stale", encoding="utf-8")

        with (
            patch.object(fetch_flex_docs, "OUTPUT_DIR", output_dir),
            patch.object(
                fetch_flex_docs,
                "convert_html_page",
                side_effect=RuntimeError("fetch failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "fetch failed"),
        ):
            fetch_flex_docs.fetch_all_targets(
                [
                    {
                        "url": "https://flexmeow.com/docs",
                        "filename": "docs.md",
                    }
                ]
            )

        self.assertTrue(stale_path.exists())
