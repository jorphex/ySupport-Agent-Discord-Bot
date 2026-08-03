#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/../.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing project Python environment: $PYTHON" >&2
    exit 1
fi

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
"$PYTHON" fetch_flex_docs.py

echo "Running process_docs.py..."
"$PYTHON" process_docs.py

echo "Updating vector store..."
"$PYTHON" embed_and_store.py

echo "Rebuilding repo context..."
"$PYTHON" build_repo_context.py

echo "Verifying repo context..."
"$PYTHON" verify_repo_context.py

echo "Yearn docs sync, vector refresh, and repo-context refresh complete."
