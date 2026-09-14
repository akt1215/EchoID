#!/bin/bash
set -euo pipefail

# Launches the Textual TUI with the same macOS dylib environment as record.sh.
# The TUI records in-process (imports cv2 via VisualIngestion) and shells out to
# main.py for the pipeline; that subprocess inherits these vars from os.environ,
# so torchaudio.load in biometrics.py gets the soundfile backend.
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_env.sh
source "$ROOT/_env.sh"
exec "$ROOT/.venv/bin/python3" "$ROOT/zoomrecorder_tui.py" "$@"
