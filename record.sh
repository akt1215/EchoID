#!/bin/bash
set -euo pipefail

# Wrapper that prevents the cv2 <-> av FFmpeg dylib conflict on macOS.
# The required env vars live in _env.sh (shared with tui.sh) and must be set at
# process level before Python starts — see that file for the full rationale.
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_env.sh
source "$ROOT/_env.sh"
exec "$ROOT/.venv/bin/python3" "$ROOT/main.py" "$@"
