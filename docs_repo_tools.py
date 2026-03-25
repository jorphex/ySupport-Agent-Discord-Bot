# docs_repo_tools.py
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import requests
from openai import AsyncOpenAI, OpenAI
from pinecone import Pinecone

import config
from repo_context import (
    fetch_repo_artifacts,
    format_repo_artifacts,
    format_repo_search_results,
    get_repo_context_status,
    search_repo_context,
)

openai_async_client: AsyncOpenAI | None = None
openai_sync_client: OpenAI | None = None
pc = Pinecone(api_key=config.PINECONE_API_KEY)
pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)

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


def _extract_repo_artifact_refs(text: str) -> list[str]:
    refs = re.findall(r"(?:segment|fact):\d+", text or "")
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
    return ordered


def _get_openai_async_client() -> AsyncOpenAI:
    global openai_async_client
    if openai_async_client is None:
        openai_async_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return openai_async_client


def _get_openai_sync_client() -> OpenAI:
    global openai_sync_client
    if openai_sync_client is None:
        openai_sync_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return openai_sync_client


async def close_shared_openai_clients() -> None:
    global openai_async_client, openai_sync_client

    if openai_async_client is not None:
        try:
            await openai_async_client.close()
        except Exception:
            pass
        openai_async_client = None

    if openai_sync_client is not None:
        try:
            openai_sync_client.close()
        except Exception:
            pass
        openai_sync_client = None


def _github_api_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ysupport-report-artifact-fetcher",
    }
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _parse_gist_id(report_url: str) -> Optional[str]:
    match = re.match(r"^https://gist\.github\.com/[^/]+/([0-9a-fA-F]+)", report_url)
    if match:
        return match.group(1)
    match = re.match(r"^https://gist\.githubusercontent\.com/[^/]+/([0-9a-fA-F]+)/", report_url)
    if match:
        return match.group(1)
    return None


def _normalize_supported_report_url(report_url: str) -> tuple[str, str]:
    stripped = (report_url or "").strip()
    gist_id = _parse_gist_id(stripped)
    if gist_id:
        return "gist", gist_id

    if stripped.startswith("https://raw.githubusercontent.com/"):
        return "raw", stripped

    blob_match = re.match(r"^https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", stripped)
    if blob_match:
        owner, repo, ref, path = blob_match.groups()
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
        return "raw", raw_url

    if stripped.startswith("https://github.com/") and "/raw/" in stripped:
        return "raw", stripped.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/raw/", "/", 1)

    raise ValueError(
        "Unsupported report URL. Supported public artifact URLs are gist.github.com, "
        "gist.githubusercontent.com, raw.githubusercontent.com, and github.com/.../blob/... links."
    )


async def _fetch_gist_content(gist_id: str, *, max_chars: int) -> str:
    api_url = f"https://api.github.com/gists/{gist_id}"

    def _do_request() -> dict[str, Any]:
        response = requests.get(api_url, headers=_github_api_headers(), timeout=20)
        response.raise_for_status()
        return response.json()

    gist_payload = await asyncio.to_thread(_do_request)
    files = gist_payload.get("files") or {}
    if not files:
        return "The gist is public but contains no files or readable content."

    blocks: list[str] = []
    remaining_chars = max_chars
    truncated = False

    for filename, file_meta in files.items():
        content = file_meta.get("content") or ""
        if not content:
            raw_url = file_meta.get("raw_url")
            if raw_url:
                try:
                    content = await _fetch_raw_report_content(raw_url, max_chars=max(remaining_chars, 1000))
                except Exception:
                    content = ""
        header = f"File: {filename}"
        block = f"{header}\n{content}".strip()
        if len(block) > remaining_chars:
            block = block[:remaining_chars].rstrip()
            truncated = True
        blocks.append(block)
        remaining_chars -= len(block) + 2
        if remaining_chars <= 0:
            truncated = True
            break

    body = "\n\n".join(blocks)
    if truncated:
        body += "\n\n[artifact truncated]"
    return body


async def _fetch_raw_report_content(raw_url: str, *, max_chars: int) -> str:
    def _do_request() -> str:
        response = requests.get(raw_url, headers=_github_api_headers(), timeout=20)
        response.raise_for_status()
        return response.text

    raw_text = await asyncio.to_thread(_do_request)
    if len(raw_text) > max_chars:
        return raw_text[:max_chars].rstrip() + "\n\n[artifact truncated]"
    return raw_text


async def _build_docs_context(user_query: str) -> tuple[str, str, bool]:
    initial_retrieval_k = 15
    rerank_top_n = 8
    query_lower = user_query.lower()
    yip_terms = ["yip", "proposal", "governance vote", "snapshot vote"]
    include_yips = any(t in query_lower for t in yip_terms)
    namespaces_to_query = ["yearn-docs"] + (["yearn-yips"] if include_yips else [])

    try:
        hyde_prompt = (
            f"You are a Yearn documentation expert. A user has asked: '{user_query}'.\n"
            "Generate a concise, hypothetical answer..."
        )
        hyde_response = await _get_openai_async_client().chat.completions.create(
            model=config.LLM_DOCS_HYDE_MODEL,
            messages=[{"role": "system", "content": hyde_prompt}],
            reasoning_effort=config.LLM_DOCS_HYDE_REASONING_EFFORT,
            verbosity=config.LLM_DOCS_HYDE_VERBOSITY,
        )
        hypothetical_answer = hyde_response.choices[0].message.content.strip()
        embedding_text = f"{user_query}\n\n{hypothetical_answer}"
    except Exception as e:
        logging.error(f"HyDE error: {e}")
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
        try:
            stats = await asyncio.to_thread(pinecone_index.describe_index_stats)
            existing_namespaces = set((stats or {}).get("namespaces", {}).keys())
            available_namespaces = [ns for ns in namespaces_to_query if ns in existing_namespaces]
            if not available_namespaces:
                available_namespaces = ["yearn-docs"]
        except Exception:
            available_namespaces = ["yearn-docs"]

        meta_like = any(
            k in query_lower
            for k in ["veyfi", "styfi", "dyfi", "yip", "governance", "delegat", "migration"]
        )
        if meta_like:
            meta_k, docs_k = 6, 9
        else:
            meta_k, docs_k = 4, 10
        meta_k = max(1, meta_k)
        docs_k = max(1, docs_k)

        query_tasks = []
        for ns in available_namespaces:
            if ns == "yearn-docs":
                query_tasks.append(
                    asyncio.to_thread(
                        pinecone_index.query,
                        namespace=ns,
                        vector=query_embedding,
                        top_k=meta_k,
                        include_metadata=True,
                        filter={"source_type": {"$eq": "meta_context"}}
                    )
                )
                query_tasks.append(
                    asyncio.to_thread(
                        pinecone_index.query,
                        namespace=ns,
                        vector=query_embedding,
                        top_k=docs_k,
                        include_metadata=True,
                        filter={"source_type": {"$eq": "documentation"}}
                    )
                )
            else:
                query_tasks.append(
                    asyncio.to_thread(
                        pinecone_index.query,
                        namespace=ns,
                        vector=query_embedding,
                        top_k=docs_k,
                        include_metadata=True,
                        filter={"source_type": {"$eq": "yip"}}
                    )
                )

        results_list = await asyncio.gather(*query_tasks)
        for idx, res in enumerate(results_list):
            namespace = available_namespaces[idx // 2] if len(available_namespaces) > 1 else available_namespaces[0]
            for match in res.get("matches", []):
                all_matches.append(_normalize_match(match, namespace))

        if not all_matches:
            fallback_tasks = [
                asyncio.to_thread(
                    pinecone_index.query,
                    namespace=ns,
                    vector=query_embedding,
                    top_k=initial_retrieval_k,
                    include_metadata=True
                ) for ns in available_namespaces
            ]
            fallback_results = await asyncio.gather(*fallback_tasks)
            for idx, res in enumerate(fallback_results):
                namespace = available_namespaces[idx] if available_namespaces else "yearn-docs"
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
                pc.inference.rerank,
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
            all_matches.sort(key=lambda x: x.score, reverse=True)
            reranked_matches = all_matches[:rerank_top_n]
    else:
        logging.info("[CoreTool:answer_from_docs] No matches found; proceeding with empty context.")

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
        if yip_number:
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
        "[CoreTool:answer_from_docs] Built docs context. no_context=%s source_count=%s",
        no_context,
        len(context_pieces),
    )
    return context_text, yip_status_summary, no_context


async def _synthesize_docs_answer(
    user_query: str,
    context_text: str,
    yip_status_summary: str,
    no_context: bool,
) -> str:
    system_prompt = (
        "You are an expert Yearn assistant. Answer based SOLELY on the context.\n"
        "Answer the user's question directly using only this knowledge.\n"
        "Lead with the grounded conclusion, not with commentary about checking docs or sources.\n"
        "1. If a source line includes a URL, include that link in your answer.\n"
        "2. If YIP_STATUS_METADATA is provided and not 'none', include a final line: 'YIP Status: ...' using that metadata.\n"
        "3. If a source has no URL, do not invent one.\n"
        "4. If NO_CONTEXT is true or the context is empty, say you do not have a confirmed Yearn answer for that from the available context. Do not invent details.\n"
        "5. If the context only partially answers the question, say exactly what is documented and what is not documented.\n"
        "6. For procedural or product-navigation questions, give the concrete next step or destination first.\n"
        "7. Do not inflate thin evidence into a broad or authoritative answer.\n"
        "8. For onboarding or getting-started questions, do not imply the user must first buy YFI or another 'Yearn token' unless the context explicitly says that. Prefer the docs-backed explanation that users deposit the token accepted by the chosen vault, and describe asset/network examples only as examples if that is all the docs provide.\n"
        "9. If the context does not recommend a specific asset, network, or onboarding path, say the docs do not specify a single best choice instead of improvising one.\n"
        "10. If the context says a system is legacy or deprecated and replaced by a newer supported path, say that directly. For questions about new deposits or whether the user should still use the old system, point to the current supported path first and only mention the legacy path for managing existing positions or exits if the context supports that distinction.\n"
        "11. If the context already resolves a current-vs-legacy or destination question, stop after the grounded answer. Do not append option menus like 'if you want, I can also...' unless the context still leaves a real unresolved branch.\n"
        "12. For builder or setup questions, if the context includes an official deployment or management guide, include that guide link and the concrete documented setup steps before broader explanation.\n"
        "13. If the context only supports a high-level setup path, say that clearly and separate it from any missing implementation detail the docs do not specify.\n"
        "14. Do not open with phrases like 'based on the docs', 'according to the docs', 'the docs support', or 'in the official sources I checked'. If useful, add a short final 'Source:' line instead of narrating your grounding process.\n"
        "15. When an exact detail is undocumented, say 'The docs do not say whether ...' or 'That is not documented here' instead of saying you checked sources.\n"
        "16. Respond in your own words; do not use canned responses or templates.\n"
        "17. NO META-COMMENTARY.\n"
    )

    try:
        response = await _get_openai_async_client().chat.completions.create(
            model=config.LLM_DOCS_SYNTH_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nYIP_STATUS_METADATA: {yip_status_summary or 'none'}\nNO_CONTEXT: {str(no_context).lower()}\n\nQuestion: {user_query}"}
            ],
            reasoning_effort=config.LLM_DOCS_SYNTH_REASONING_EFFORT,
            verbosity=config.LLM_DOCS_SYNTH_VERBOSITY,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Gen error: {e}")
        return "Error generating final answer."


async def core_answer_from_docs(user_query: str) -> str:
    """
    Answers questions from Yearn docs and YIPs only.
    """
    logging.info(f"[CoreTool:answer_from_docs] Query: '{user_query}'")
    try:
        context_text, yip_status_summary, no_context = await _build_docs_context(user_query)
    except RuntimeError as exc:
        return str(exc)
    return await _synthesize_docs_answer(user_query, context_text, yip_status_summary, no_context)


async def core_search_repo_context(
    query: str,
    limit: Optional[int] = None,
    include_legacy: bool = False,
    include_ui: bool = False,
) -> str:
    logging.info(
        "[CoreTool:search_repo_context] Query='%s' limit=%s include_legacy=%s include_ui=%s",
        query,
        limit,
        include_legacy,
        include_ui,
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
            limit=limit or config.REPO_CONTEXT_TOP_K,
            include_legacy=include_legacy,
            include_ui=include_ui,
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

    try:
        artifacts = fetch_repo_artifacts(artifact_refs)
        logging.info("[CoreTool:fetch_repo_artifacts] Returned %s artifacts.", len(artifacts))
        return format_repo_artifacts(artifacts)
    except Exception as exc:
        logging.error(f"[CoreTool:fetch_repo_artifacts] Error: {exc}", exc_info=True)
        return f"Error fetching repo artifacts: {exc}"


async def core_pretriage_repo_claim(
    claim_text: str,
    *,
    include_docs: bool = True,
    limit: Optional[int] = None,
    include_legacy: bool = False,
    include_ui: bool = False,
) -> str:
    logging.info(
        "[CoreTool:pretriage_repo_claim] claim='%s' include_docs=%s limit=%s include_legacy=%s include_ui=%s",
        claim_text,
        include_docs,
        limit,
        include_legacy,
        include_ui,
    )
    sections: list[str] = []

    repo_search = await core_search_repo_context(
        claim_text,
        limit=limit,
        include_legacy=include_legacy,
        include_ui=include_ui,
    )
    sections.append(f"Repo search:\n{repo_search}")

    artifact_refs = _extract_repo_artifact_refs(repo_search)[:2]
    if artifact_refs:
        artifact_text = await core_fetch_repo_artifacts(", ".join(artifact_refs))
        sections.append(f"Fetched repo artifacts:\n{artifact_text}")

    if include_docs:
        docs_answer = await core_answer_from_docs(claim_text)
        sections.append(f"Docs context:\n{docs_answer}")

    return "\n\n".join(section for section in sections if section.strip())


async def core_fetch_report_artifact(report_url: str, max_chars: int = 12000) -> str:
    logging.info("[CoreTool:fetch_report_artifact] report_url='%s' max_chars=%s", report_url, max_chars)
    try:
        artifact_kind, normalized_value = _normalize_supported_report_url(report_url)
        if artifact_kind == "gist":
            artifact_text = await _fetch_gist_content(normalized_value, max_chars=max_chars)
            source_label = f"https://gist.github.com/.../{normalized_value}"
        else:
            artifact_text = await _fetch_raw_report_content(normalized_value, max_chars=max_chars)
            source_label = normalized_value
        return f"Fetched public report artifact from: {source_label}\n\n{artifact_text}"
    except ValueError as exc:
        return str(exc)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        logging.warning(
            "[CoreTool:fetch_report_artifact] HTTP error for '%s': status=%s",
            report_url,
            status_code,
        )
        return f"Could not fetch the public report artifact (HTTP {status_code})."
    except Exception as exc:
        logging.error(f"[CoreTool:fetch_report_artifact] Error: {exc}", exc_info=True)
        return f"Error fetching the public report artifact: {exc}"


async def core_repo_context_status() -> str:
    status = get_repo_context_status()
    summary = status.get("summary") or {}
    repo_count = summary.get("repos_indexed", 0) if isinstance(summary, dict) else 0
    return (
        f"Repo context status: state={status['state']} available={status['available']} "
        f"fresh={status['fresh']} built_at={status['built_at']} age_hours={status['age_hours']} "
        f"repo_count={repo_count} reason={status['reason']}"
    )
