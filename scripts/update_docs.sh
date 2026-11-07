#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing project Python environment: $PYTHON" >&2
    exit 1
fi

LOCK_DIR="$REPO_ROOT/.cache/docs_ingestion"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/update_docs.lock"
if ! flock -n 9; then
    echo "Documentation refresh is already running; skipping this invocation."
    exit 0
fi

refresh_docs() {
    echo "Refreshing shallow Yearn docs checkout..."
    "$PYTHON" -m ysupport.ingestion.fetch_yearn_docs || return "$?"

    echo "Fetching Flex docs..."
    "$PYTHON" -m ysupport.ingestion.fetch_flex_docs || return "$?"

    echo "Running process_docs.py..."
    "$PYTHON" -m ysupport.ingestion.process_docs || return "$?"

    echo "Updating vector store..."
    "$PYTHON" -m ysupport.ingestion.embed_and_store
}

refresh_repo() {
    echo "Rebuilding repo context..."
    "$PYTHON" -m ysupport.ingestion.build_repo_context || return "$?"

    echo "Verifying repo context..."
    "$PYTHON" -m ysupport.ingestion.verify_repo_context
}

run_stage() {
    local stage="$1"
    shift
    local result=0
    "$@" || result=$?

    local finished_at last_success="never" status="failed"
    finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ) || return "$?"
    if [[ "$result" -eq 0 ]]; then
        status="succeeded"
        printf '%s\n' "$finished_at" > "$LOCK_DIR/$stage.last_success.tmp" || return "$?"
        mv "$LOCK_DIR/$stage.last_success.tmp" "$LOCK_DIR/$stage.last_success" || return "$?"
    fi
    if [[ -f "$LOCK_DIR/$stage.last_success" ]]; then
        read -r last_success < "$LOCK_DIR/$stage.last_success" || return "$?"
    fi
    printf 'status=%s\nfinished_at=%s\nlast_success=%s\n' \
        "$status" "$finished_at" "$last_success" > "$LOCK_DIR/$stage.status.tmp" || return "$?"
    mv "$LOCK_DIR/$stage.status.tmp" "$LOCK_DIR/$stage.status" || return "$?"
    echo "Refresh stage=$stage status=$status finished_at=$finished_at last_success=$last_success"
    return "$result"
}

refresh_result=0
run_stage docs refresh_docs || refresh_result=1
run_stage repo refresh_repo || refresh_result=1
exit "$refresh_result"
