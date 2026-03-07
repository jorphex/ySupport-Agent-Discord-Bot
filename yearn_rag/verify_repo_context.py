import argparse
import json
import sys
from pathlib import Path


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _bootstrap_repo_root()

    import config
    from repo_context import (
        build_repo_context_index,
        fetch_repo_artifacts,
        get_repo_context_status,
        search_repo_context,
    )

    parser = argparse.ArgumentParser(description="Verify the local Yearn repo-context index and explicit search/fetch path.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the repo-context index before verification.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        default=None,
        help="Optional repo name/full_name filter for build mode. Repeat to limit to a subset.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        default=None,
        help="Query to verify against the repo index. Repeat for multiple queries.",
    )
    args = parser.parse_args()

    if args.build:
        build_repo_context_index(
            manifest_path=config.REPO_CONTEXT_MANIFEST_PATH,
            cache_dir=config.REPO_CONTEXT_CACHE_DIR,
            db_path=config.REPO_CONTEXT_DB_PATH,
            repo_names=args.repos,
        )

    status = get_repo_context_status(enabled=True, require_fresh=True)
    summary: dict[str, object] = {
        "status": status,
        "queries": [],
    }

    failures: list[str] = []
    if status["state"] != "ready":
        failures.append(f"Repo context status is not ready: {status['state']} ({status['reason']})")

    queries = args.queries or []
    for query in queries:
        results = search_repo_context(query, limit=3, include_legacy=True, include_ui=False)
        query_summary: dict[str, object] = {
            "query": query,
            "result_count": len(results),
            "top_artifact_refs": [result.artifact_ref for result in results[:3]],
        }
        if not results:
            failures.append(f"No repo-context results for query: {query}")
            summary["queries"].append(query_summary)
            continue

        artifacts = fetch_repo_artifacts([results[0].artifact_ref])
        query_summary["fetched_artifacts"] = [artifact["artifact_ref"] for artifact in artifacts]
        if not artifacts:
            failures.append(f"Search returned results but fetch returned nothing for query: {query}")
        summary["queries"].append(query_summary)

    summary["ok"] = not failures
    summary["failures"] = failures
    print(json.dumps(summary, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
