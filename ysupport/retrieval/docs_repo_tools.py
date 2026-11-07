# docs_repo_tools.py
import asyncio
import logging
import re
from time import monotonic
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pinecone import Pinecone

from ysupport import config
from ysupport.retrieval.repo_context import (
    MAX_REPO_ARTIFACTS,
    fetch_repo_artifacts,
    format_repo_artifacts,
    format_repo_search_results,
    get_repo_context_status,
    search_repo_context,
)

openai_sync_client: OpenAI | None = None
pinecone_client: Pinecone | None = None
pinecone_index: Any | None = None
_NAMESPACE_CACHE_TTL_SECONDS = 300
_namespace_cache: tuple[Any, set[str], float] | None = None
_namespace_lookup: tuple[Any, asyncio.Task[set[str]]] | None = None


async def _existing_namespaces(index: Any) -> set[str]:
    global _namespace_lookup
    if (
        _namespace_cache is not None
        and _namespace_cache[0] is index
        and monotonic() < _namespace_cache[2]
    ):
        return _namespace_cache[1]

    if (
        _namespace_lookup is None
        or _namespace_lookup[0] is not index
        or _namespace_lookup[1].done()
    ):
        async def discover() -> set[str]:
            global _namespace_cache
            stats = await asyncio.to_thread(index.describe_index_stats)
            namespaces = set((stats or {}).get("namespaces", {}))
            if _namespace_lookup is not None and _namespace_lookup[1] is asyncio.current_task():
                _namespace_cache = (index, namespaces, monotonic() + _NAMESPACE_CACHE_TTL_SECONDS)
            return namespaces

        task = asyncio.create_task(discover())
        _namespace_lookup = (index, task)
    return await asyncio.shield(_namespace_lookup[1])


def _get_pinecone_client() -> Pinecone:
    global pinecone_client
    if pinecone_client is None:
        pinecone_client = Pinecone(api_key=config.PINECONE_API_KEY)
    return pinecone_client


def _get_pinecone_index() -> Any:
    global pinecone_index
    if pinecone_index is None:
        pinecone_index = _get_pinecone_client().Index(config.PINECONE_INDEX_NAME)
    return pinecone_index


def _normalize_match(match: Any, namespace: str) -> Dict[str, Any]:
    if isinstance(match, dict):
        return {
            "id": match.get("id", ""),
            "score": match.get("score", 0.0) or 0.0,
            "metadata": match.get("metadata", {}) or {},
            "namespace": namespace,
        }
    return {
        "id": getattr(match, "id", ""),
        "score": getattr(match, "score", 0.0) or 0.0,
        "metadata": getattr(match, "metadata", {}) or {},
        "namespace": namespace,
    }


def _get_match_id(match: Any) -> str:
    if isinstance(match, dict):
        ns = match.get("namespace", "")
        mid = match.get("id", "")
        return f"{ns}:{mid}" if ns else mid
    return getattr(match, "id", "")


def _extract_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "have", "has",
        "you", "your", "about", "what", "when", "where", "why", "how", "is",
        "are", "can", "cant", "cannot", "able", "unable", "not", "show",
        "showing", "missing", "does", "do", "did"
    }
    return [t for t in tokens if len(t) > 3 and t not in stop]


def _truncate_rerank_document(text: str, max_chars: int = 3000) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip()


def _top_matches_by_score(
    matches: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    return sorted(
        matches,
        key=lambda match: float(match.get("score", 0.0) or 0.0),
        reverse=True,
    )[:limit]


def _should_include_flex_docs(query_lower: str) -> bool:
    if re.search(r"\bflex\b", query_lower):
        return True
    strong_markers = (
        r"\btroves?\b",
        r"\bredeemers?\b",
        r"\blender vaults?\b",
        r"\bfixed[- ]rate (?:money market|lending market)\b",
        r"\bliquity\b",
    )
    if any(re.search(pattern, query_lower) for pattern in strong_markers):
        return True
    if re.search(r"\bredemptions?\b", query_lower):
        return any(
            marker in query_lower
            for marker in ("fixed-rate", "fixed rate", "trove", "lender vault", "liquity")
        )
    return False


def _docs_query_specs(
    available_namespaces: list[str], *, include_yips: bool
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for namespace in available_namespaces:
        specs.append((namespace, "documentation"))
        if namespace == "yearn-docs" and include_yips:
            specs.append((namespace, "yip"))
    return specs


def _get_openai_sync_client() -> OpenAI:
    global openai_sync_client
    if openai_sync_client is None:
        openai_sync_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return openai_sync_client


async def close_shared_clients() -> None:
    global openai_sync_client, pinecone_client, pinecone_index
    global _namespace_cache, _namespace_lookup

    if _namespace_lookup is not None and not _namespace_lookup[1].done():
        _namespace_lookup[1].cancel()
        await asyncio.gather(_namespace_lookup[1], return_exceptions=True)
    _namespace_lookup = None
    _namespace_cache = None


    if openai_sync_client is not None:
        try:
            openai_sync_client.close()
        except Exception:
            pass
        openai_sync_client = None

    if pinecone_client is not None:
        try:
            pinecone_client.close()
        except Exception:
            pass
        pinecone_client = None
        pinecone_index = None


async def _build_docs_context(
    user_query: str,
) -> tuple[str, str, bool]:
    initial_retrieval_k = 15
    rerank_top_n = 8
    query_lower = user_query.lower()
    yip_terms = ["yip", "proposal", "governance vote", "snapshot vote"]
    include_yips = any(t in query_lower for t in yip_terms)
    namespaces_to_query = ["yearn-docs"]
    if _should_include_flex_docs(query_lower):
        namespaces_to_query.append("flex-docs")

    embedding_text = user_query

    try:
        response = await asyncio.to_thread(
            _get_openai_sync_client().embeddings.create,
            model="text-embedding-3-large",
            input=[embedding_text],
            encoding_format="float"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        raise RuntimeError("Error generating embedding.") from e

    all_matches: List[Dict[str, Any]] = []
    try:
        index = _get_pinecone_index()
        client = _get_pinecone_client()
        try:
            existing_namespaces = await _existing_namespaces(index)
            available_namespaces = [ns for ns in namespaces_to_query if ns in existing_namespaces]
            if not available_namespaces:
                available_namespaces = ["yearn-docs"]
        except Exception:
            available_namespaces = ["yearn-docs"]

        query_tasks: List[tuple[str, Any]] = []
        for namespace, source_type in _docs_query_specs(
            available_namespaces, include_yips=include_yips
        ):
            query_tasks.append((
                namespace,
                asyncio.to_thread(
                    index.query,
                    namespace=namespace,
                    vector=query_embedding,
                    top_k=initial_retrieval_k,
                    include_metadata=True,
                    filter={"source_type": {"$eq": source_type}},
                ),
            ))

        results_list = await asyncio.gather(*(task for _, task in query_tasks))
        for (namespace, _), res in zip(query_tasks, results_list):
            for match in res.get("matches", []):
                all_matches.append(_normalize_match(match, namespace))

    except Exception as e:
        logging.error(f"Pinecone error: {e}")
        raise RuntimeError("Error searching docs.") from e

    reranked_matches: List[Dict[str, Any]] = []
    if all_matches:
        try:
            unique_matches_map: Dict[str, Any] = {}
            for match in all_matches:
                match_id = _get_match_id(match)
                if match_id and match_id not in unique_matches_map:
                    unique_matches_map[match_id] = match
            unique_matches = list(unique_matches_map.values()) if unique_matches_map else all_matches
            docs_to_rerank = []
            for match in unique_matches:
                metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {}
                text_chunk = _truncate_rerank_document(metadata.get("text") or "")
                source_type = metadata.get("source_type", "unknown")
                docs_to_rerank.append(f"[source_type={source_type}]\n{text_chunk}")

            rerank_response = await asyncio.to_thread(
                client.inference.rerank,
                model="bge-reranker-v2-m3",
                query=user_query,
                documents=docs_to_rerank,
                top_n=rerank_top_n,
                return_documents=False
            )
            unique_matches_list = list(unique_matches)
            reranked_matches = [unique_matches_list[result.index] for result in rerank_response.data]
        except Exception as e:
            logging.error(f"Rerank error: {e}")
            reranked_matches = _top_matches_by_score(
                all_matches,
                rerank_top_n,
            )
    else:
        logging.info("[CoreTool:docs_context] No matches found; proceeding with empty context.")

    context_pieces = []
    yip_status_entries: List[str] = []
    for match in reranked_matches:
        meta = match.get("metadata", {})
        text = meta.get("text")
        if not text:
            continue

        doc_title = meta.get("doc_title", "Unk")
        section_heading = meta.get("section_heading")
        source_url = meta.get("source_url")
        source_path = meta.get("source_path", "")
        source_type = meta.get("source_type")
        doc_last_modified = meta.get("doc_last_modified")

        yip_number = meta.get("yip_number")
        yip_status = meta.get("yip_status")
        yip_created = meta.get("yip_created")
        yip_discussion_link = meta.get("yip_discussion_link")

        title_bits = [doc_title]
        if section_heading and section_heading != doc_title:
            title_bits.append(section_heading)
        title_str = " > ".join(title_bits)

        location = source_url or source_path
        source_bits = [title_str]
        if location:
            source_bits.append(f"({location})")
        if source_type:
            source_bits.append(f"[{source_type}]")
        if doc_last_modified:
            source_bits.append(f"last_modified={doc_last_modified}")
        if yip_number is not None:
            yip_bits = [f"YIP-{yip_number}"]
            if yip_status:
                yip_bits.append(f"status={yip_status}")
            if yip_created:
                yip_bits.append(f"created={yip_created}")
            if yip_discussion_link:
                yip_bits.append(f"discussion={yip_discussion_link}")
            source_bits.append(" ".join(yip_bits))
            if yip_status:
                yip_status_entries.append(f"YIP-{yip_number}: {yip_status}")

        source = " ".join(source_bits).strip()
        context_pieces.append(f"Source: {source}\nContent:\n{text}")

    context_text = "\n\n---\n\n".join(context_pieces)
    yip_status_summary = "; ".join(sorted(set(yip_status_entries)))

    keywords = _extract_keywords(user_query)
    context_lower = context_text.lower()
    if not context_text.strip():
        no_context = True
    elif not keywords:
        no_context = False
    else:
        no_context = not any(k in context_lower for k in keywords)
    if no_context:
        context_text = ""

    logging.info(
        "[CoreTool:docs_context] Built docs context. no_context=%s source_count=%s",
        no_context,
        len(context_pieces),
    )
    return context_text, yip_status_summary, no_context


async def core_search_docs_context(user_query: str) -> str:
    """Return bounded, reranked Yearn docs context for a reasoning caller."""
    logging.info("[CoreTool:search_docs_context] Query: '%s'", user_query)
    try:
        context_text, yip_status_summary, no_context = await _build_docs_context(
            user_query,
        )
    except RuntimeError as exc:
        return str(exc)
    if no_context:
        return (
            "No sufficiently relevant Yearn documentation context was found for "
            "this query."
        )

    sections = ["Ranked Yearn documentation excerpts:", context_text]
    if yip_status_summary:
        sections.append(f"Relevant YIP status: {yip_status_summary}")
    return "\n\n".join(sections)


async def core_search_repo_context(
    query: str,
    limit: Optional[int] = None,
    include_legacy: bool = False,
) -> str:
    logging.info(
        "[CoreTool:search_repo_context] Query='%s' limit=%s include_legacy=%s",
        query,
        limit,
        include_legacy,
    )
    status = get_repo_context_status()
    if status["state"] not in {"ready", "available"}:
        logging.warning(
            "[CoreTool:search_repo_context] Repo context unavailable. state=%s reason=%s",
            status["state"],
            status["reason"],
        )
        return (
            f"Repo context unavailable. state={status['state']} reason={status['reason']} "
            f"db_path={status['db_path']}"
        )

    try:
        results = search_repo_context(
            query,
            limit=limit,
            include_legacy=include_legacy,
        )
        logging.info("[CoreTool:search_repo_context] Returned %s results.", len(results))
        return format_repo_search_results(results)
    except Exception as exc:
        logging.error(f"[CoreTool:search_repo_context] Error: {exc}", exc_info=True)
        return f"Error searching repo context: {exc}"


async def core_fetch_repo_artifacts(artifact_refs_text: str) -> str:
    logging.info("[CoreTool:fetch_repo_artifacts] artifact_refs_text='%s'", artifact_refs_text)
    status = get_repo_context_status()
    if status["state"] not in {"ready", "available"}:
        logging.warning(
            "[CoreTool:fetch_repo_artifacts] Repo context unavailable. state=%s reason=%s",
            status["state"],
            status["reason"],
        )
        return (
            f"Repo context unavailable. state={status['state']} reason={status['reason']} "
            f"db_path={status['db_path']}"
        )

    artifact_refs = re.findall(r"(?:segment|fact):\d+", artifact_refs_text)
    if not artifact_refs:
        return "No valid repo artifact references were provided. Use references like 'segment:12' or 'fact:34'."
    if len(dict.fromkeys(artifact_refs)) > MAX_REPO_ARTIFACTS:
        return f"At most {MAX_REPO_ARTIFACTS} repo artifact references can be fetched at once."

    try:
        artifacts = fetch_repo_artifacts(artifact_refs)
        logging.info("[CoreTool:fetch_repo_artifacts] Returned %s artifacts.", len(artifacts))
        return format_repo_artifacts(artifacts)
    except Exception as exc:
        logging.error(f"[CoreTool:fetch_repo_artifacts] Error: {exc}", exc_info=True)
        return f"Error fetching repo artifacts: {exc}"


async def core_repo_context_status() -> str:
    status = get_repo_context_status()
    summary = status.get("summary") or {}
    repo_count = summary.get("repos_indexed", 0) if isinstance(summary, dict) else 0
    return (
        f"Repo context status: state={status['state']} available={status['available']} "
        f"fresh={status['fresh']} built_at={status['built_at']} age_hours={status['age_hours']} "
        f"repo_count={repo_count} reason={status['reason']}"
    )
