#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export TICKET_EXECUTION_CODEX_AUTH_SOURCE=""
export TICKET_EXECUTION_CODEX_AUTH_SYNC_SOURCE=""
: "${TICKET_EXECUTION_CODEX_AUTH_LINK_SOURCE:=${HOME}/.codex/auth.json}"

exec python3 ysupport.py
