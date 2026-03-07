import argparse
import json
import logging
import sys
from pathlib import Path


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _bootstrap_repo_root()

    import config
    from repo_context import build_repo_context_index

    parser = argparse.ArgumentParser(description="Build the local Yearn repo-context cache and SQLite index.")
    parser.add_argument(
        "--manifest",
        default=str(config.REPO_CONTEXT_MANIFEST_PATH),
        help="Path to repo manifest JSON.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(config.REPO_CONTEXT_CACHE_DIR),
        help="Directory for extracted repo files.",
    )
    parser.add_argument(
        "--db-path",
        default=str(config.REPO_CONTEXT_DB_PATH),
        help="SQLite database path for the repo-context index.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        default=None,
        help="Optional repo name/full_name to build. Repeat to limit to a subset.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary = build_repo_context_index(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        db_path=args.db_path,
        repo_names=args.repos,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
