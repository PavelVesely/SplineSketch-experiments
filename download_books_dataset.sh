#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  python_executable="$PYTHON"
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
  python_executable="$repo_dir/.venv/bin/python"
else
  python_executable="python3"
fi
exec "$python_executable" "$repo_dir/reproduce.py" download books "$@"
