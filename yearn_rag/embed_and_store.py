from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
DEFAULT_STATE_PATH = BASE_DIR / ".cache" / "docs_ingestion" / "embedding_state.json"

load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "docs")
PINECONE_HOST = os.getenv("PINECONE_HOST")
STATE_PATH = Path(os.getenv("DOCS_INGESTION_STATE_PATH", str(DEFAULT_STATE_PATH)))
FORCE_REFRESH = os.getenv("DOCS_INGESTION_FORCE_REFRESH", "0") == "1"

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_RETRIES = 3
EMBEDDING_BATCH_SIZE = 100
DELETE_BATCH_SIZE = 1000
FETCH_BATCH_SIZE = 250
STATE_SCHEMA_VERSION = 1
PIPELINE_VERSION = 1
VERIFY_ATTEMPTS = 6
VERIFY_DELAY_SECONDS = 2.0

EMBEDDING_SOURCES = {
    "yearn_docs": {
        "input_json": "cleaned_yearn_docs.json",
        "namespace": "yearn-docs",
    },
    "flex_docs": {
        "input_json": "cleaned_flex_docs.json",
        "namespace": "flex-docs",
    },
}
RETIRED_NAMESPACES = ("yearn-yips",)


def sanitize_for_id(text: str) -> str:
    text = text.replace("/", "-").replace("\\", "-").replace(" ", "-")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\-.]", "", text)
    return text.lower()


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def vector_id_for_doc(doc: dict[str, Any]) -> str:
    id_base = doc.get("doc_id") or doc.get("source_path", "unknown")
    chunk_index = doc.get("chunk_index", doc.get("chunk_id", -1))
    return f"{sanitize_for_id(str(id_base))}-{sanitize_for_id(str(chunk_index))}"


def metadata_for_doc(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "text": doc.get("text", ""),
        "filename": doc.get("filename", "unknown"),
        "doc_title": doc.get("doc_title", "Unknown Title"),
        "section_heading": doc.get("section_heading", "Unknown Section"),
        "source_path": doc.get("source_path", "unknown"),
        "chunk_id": doc.get("chunk_id", -1),
        "chunk_index": doc.get("chunk_index", doc.get("chunk_id", -1)),
        "doc_id": doc.get("doc_id"),
        "doc_last_modified": doc.get("doc_last_modified"),
        "source_type": doc.get("source_type"),
        "source_url": doc.get("source_url"),
        "content_hash": content_hash(doc.get("text", "")),
        "yip_status": doc.get("yip_status"),
        "yip_number": doc.get("yip_number"),
        "yip_created": doc.get("yip_created"),
        "yip_discussion_link": doc.get("yip_discussion_link"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def source_fingerprint(docs: list[dict[str, Any]]) -> str:
    payload = {
        "embedding_model": EMBEDDING_MODEL,
        "pipeline_version": PIPELINE_VERSION,
        "documents": docs,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"schema_version": STATE_SCHEMA_VERSION, "sources": {}}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != STATE_SCHEMA_VERSION
        or not isinstance(payload.get("sources"), dict)
    ):
        return {"schema_version": STATE_SCHEMA_VERSION, "sources": {}}
    return payload


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _namespace_vector_count(stats: Any, namespace: str) -> int:
    namespaces = stats.get("namespaces", {}) if hasattr(stats, "get") else {}
    namespace_stats = namespaces.get(namespace, {})
    if hasattr(namespace_stats, "get"):
        return int(namespace_stats.get("vector_count", 0))
    return int(getattr(namespace_stats, "vector_count", 0))


def list_namespace_ids(index: Any, namespace: str) -> set[str]:
    ids: set[str] = set()
    for page in index.list(namespace=namespace):
        ids.update(str(vector_id) for vector_id in page)
    return ids


def fetch_namespace_ids(
    index: Any,
    namespace: str,
    expected_ids: set[str],
) -> set[str]:
    fetched_ids: set[str] = set()
    ordered_ids = sorted(expected_ids)
    for offset in range(0, len(ordered_ids), FETCH_BATCH_SIZE):
        response = index.fetch(
            ids=ordered_ids[offset : offset + FETCH_BATCH_SIZE],
            namespace=namespace,
        )
        vectors = (
            response.get("vectors", {})
            if hasattr(response, "get")
            else getattr(response, "vectors", {})
        )
        fetched_ids.update(str(vector_id) for vector_id in vectors)
    return fetched_ids


def namespace_matches_expected(
    index: Any,
    namespace: str,
    expected_ids: set[str],
    *,
    stats_count: int | None = None,
    listed_ids: set[str] | None = None,
) -> bool:
    if stats_count is None:
        stats_count = _namespace_vector_count(index.describe_index_stats(), namespace)
    if stats_count != len(expected_ids):
        return False
    if listed_ids is None:
        listed_ids = list_namespace_ids(index, namespace)
    if listed_ids == expected_ids:
        return True
    return fetch_namespace_ids(index, namespace, expected_ids) == expected_ids


def _load_docs(input_json: str) -> list[dict[str, Any]]:
    path = Path(input_json)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Input file not found: {input_json}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Input file is invalid JSON: {input_json}") from exc
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Input file has no document chunks: {input_json}")
    if not all(isinstance(doc, dict) and isinstance(doc.get("text"), str) for doc in payload):
        raise RuntimeError(f"Input file contains invalid document chunks: {input_json}")
    return payload


def _generate_embeddings(client: Any, texts: list[str]) -> list[list[float]]:
    print(f"Generating embeddings for a batch of {len(texts)} texts...")
    last_error: Exception | None = None
    for attempt in range(EMBEDDING_RETRIES):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                encoding_format="float",
            )
            embeddings = [item.embedding for item in response.data]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Embedding response returned {len(embeddings)} vectors for {len(texts)} texts"
                )
            return embeddings
        except Exception as exc:
            last_error = exc
            print(
                "Embedding generation failed "
                f"(attempt {attempt + 1}/{EMBEDDING_RETRIES}): {exc}"
            )
            if attempt < EMBEDDING_RETRIES - 1:
                time.sleep(2**attempt)
    raise RuntimeError("Embedding generation failed after all retries") from last_error


def _upserted_count(response: Any) -> int:
    if hasattr(response, "get"):
        return int(response.get("upserted_count", 0))
    return int(getattr(response, "upserted_count", 0))


def _build_vectors(
    docs: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": vector_id_for_doc(doc),
            "values": embedding,
            "metadata": metadata_for_doc(doc),
        }
        for doc, embedding in zip(docs, embeddings)
    ]


def _verify_namespace_ids(
    index: Any,
    namespace: str,
    expected_ids: set[str],
) -> None:
    actual_ids: set[str] = set()
    for attempt in range(VERIFY_ATTEMPTS):
        actual_ids = list_namespace_ids(index, namespace)
        stats_count = _namespace_vector_count(index.describe_index_stats(), namespace)
        if namespace_matches_expected(
            index,
            namespace,
            expected_ids,
            stats_count=stats_count,
            listed_ids=actual_ids,
        ):
            return
        if attempt < VERIFY_ATTEMPTS - 1:
            time.sleep(VERIFY_DELAY_SECONDS)
    missing = len(expected_ids - actual_ids)
    stale = len(actual_ids - expected_ids)
    raise RuntimeError(
        f"Namespace verification failed for {namespace}: "
        f"expected={len(expected_ids)} actual={len(actual_ids)} "
        f"missing={missing} stale={stale}"
    )


def process_and_embed_source(
    config: dict[str, str],
    *,
    index: Any,
    embedding_client: Any,
    state: dict[str, Any],
    state_path: Path = STATE_PATH,
    force_refresh: bool = FORCE_REFRESH,
) -> dict[str, Any]:
    input_json = config["input_json"]
    namespace = config["namespace"]
    print(f"\n--- Processing Source: {input_json} -> Namespace: {namespace} ---")

    docs = _load_docs(input_json)
    desired_ids = [vector_id_for_doc(doc) for doc in docs]
    desired_id_set = set(desired_ids)
    if len(desired_id_set) != len(desired_ids):
        raise RuntimeError(f"Duplicate vector IDs generated for namespace {namespace}")

    fingerprint = source_fingerprint(docs)
    stats_count = _namespace_vector_count(index.describe_index_stats(), namespace)
    live_ids = list_namespace_ids(index, namespace)
    prior = state["sources"].get(namespace, {})

    state_matches = (
        prior.get("fingerprint") == fingerprint
        and prior.get("vector_count") == len(docs)
        and prior.get("embedding_model") == EMBEDDING_MODEL
        and prior.get("pipeline_version") == PIPELINE_VERSION
    )
    live_matches = namespace_matches_expected(
        index,
        namespace,
        desired_id_set,
        stats_count=stats_count,
        listed_ids=live_ids,
    )
    if not force_refresh and state_matches and live_matches:
        print(
            f"Unchanged: {namespace} already has the expected {len(docs)} vectors; "
            "skipping embeddings and writes."
        )
        return {
            "namespace": namespace,
            "status": "skipped",
            "vector_count": len(docs),
            "upserted_count": 0,
            "deleted_count": 0,
        }

    if force_refresh:
        reason = "forced refresh"
    elif not state_matches:
        reason = "source fingerprint/state changed or missing"
    else:
        reason = "live namespace drift"
    print(f"Refreshing {namespace}: {reason}.")

    total_upserted = 0
    for offset in range(0, len(docs), EMBEDDING_BATCH_SIZE):
        batch_docs = docs[offset : offset + EMBEDDING_BATCH_SIZE]
        embeddings = _generate_embeddings(
            embedding_client,
            [doc["text"] for doc in batch_docs],
        )
        vectors = _build_vectors(batch_docs, embeddings)
        response = index.upsert(vectors=vectors, namespace=namespace)
        upserted_count = _upserted_count(response)
        if upserted_count != len(vectors):
            raise RuntimeError(
                f"Pinecone upsert stored {upserted_count}/{len(vectors)} vectors "
                f"for {namespace} batch {offset // EMBEDDING_BATCH_SIZE + 1}"
            )
        total_upserted += upserted_count
        print(
            f"Upserted batch {offset // EMBEDDING_BATCH_SIZE + 1}: "
            f"stored {upserted_count} vectors in {namespace}."
        )

    stale_ids = sorted(live_ids - desired_id_set)
    for offset in range(0, len(stale_ids), DELETE_BATCH_SIZE):
        batch_ids = stale_ids[offset : offset + DELETE_BATCH_SIZE]
        index.delete(ids=batch_ids, namespace=namespace)

    _verify_namespace_ids(index, namespace, desired_id_set)
    state["sources"][namespace] = {
        "embedding_model": EMBEDDING_MODEL,
        "fingerprint": fingerprint,
        "pipeline_version": PIPELINE_VERSION,
        "vector_count": len(docs),
    }
    save_state(state, state_path)
    print(
        f"Verified {namespace}: vectors={len(docs)} "
        f"upserted={total_upserted} deleted={len(stale_ids)}."
    )
    return {
        "namespace": namespace,
        "status": "refreshed",
        "vector_count": len(docs),
        "upserted_count": total_upserted,
        "deleted_count": len(stale_ids),
    }


def _build_clients() -> tuple[Any, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is required")
    if not PINECONE_INDEX_NAME:
        raise RuntimeError("PINECONE_INDEX_NAME is required")

    embedding_client = OpenAI(api_key=OPENAI_API_KEY)
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_HOST:
        index = pinecone_client.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
    else:
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
    index.describe_index_stats()
    return embedding_client, index


def _retire_namespaces(index: Any) -> None:
    stats = index.describe_index_stats()
    for namespace in RETIRED_NAMESPACES:
        if _namespace_vector_count(stats, namespace) > 0:
            print(f"Deleting retired namespace {namespace}.")
            index.delete(delete_all=True, namespace=namespace)


def main() -> None:
    embedding_client, index = _build_clients()
    state = load_state()
    _retire_namespaces(index)

    results = [
        process_and_embed_source(
            config,
            index=index,
            embedding_client=embedding_client,
            state=state,
        )
        for config in EMBEDDING_SOURCES.values()
    ]
    refreshed = sum(result["status"] == "refreshed" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    upserted = sum(result["upserted_count"] for result in results)
    deleted = sum(result["deleted_count"] for result in results)
    print(
        "\nEmbedding sync complete: "
        f"refreshed_sources={refreshed} skipped_sources={skipped} "
        f"upserted={upserted} deleted={deleted}."
    )


if __name__ == "__main__":
    main()
