#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCK_DIR="$SCRIPT_DIR/../.cache/docs_ingestion"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/update_docs.lock"
if ! flock -n 9; then
    echo "Documentation refresh is already running; skipping this invocation."
    exit 0
fi

echo "Pulling latest Yearn docs..."
cd yearn-devdocs
git pull --ff-only origin master
cd ..

echo "Fetching Flex docs..."
python3 fetch_flex_docs.py

echo "Running process_docs.py..."
python3 process_docs.py

echo "Updating vector store..."
python3 embed_and_store.py

echo "Rebuilding repo context..."
python3 build_repo_context.py

echo "Verifying repo context..."
python3 verify_repo_context.py

echo "Yearn docs sync, vector refresh, and repo-context refresh complete."
