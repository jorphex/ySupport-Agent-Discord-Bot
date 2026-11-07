import argparse
import json
import logging


def main() -> None:
    from ysupport import config
    from ysupport.retrieval.repo_context import build_repo_context_index

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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    summary = build_repo_context_index(
        manifest_path=args.manifest,
        cache_dir=args.cache_dir,
        db_path=args.db_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
