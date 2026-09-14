# Shared macOS launch environment for EchoID (sourced by record.sh and tui.sh).
#
# These must be exported at the PROCESS level, before Python starts:
#   OBJC_DISABLE_INITIALIZE_FORK_SAFETY  The Objective-C runtime reads this at
#       startup. It prevents a SIGKILL from the cv2 <-> av FFmpeg dylibs, which
#       both register duplicate Objective-C classes. Setting it from inside
#       Python is too late — the runtime has already initialized.
#   TORCHAUDIO_USE_BACKEND=soundfile     Forces torchaudio (biometrics.py's
#       torchaudio.load) onto the soundfile backend, avoiding the FFmpeg dylib
#       path entirely. The TUI's reprocess subprocess inherits this via os.environ.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TORCHAUDIO_USE_BACKEND=soundfile
# Let any op the Metal (MPS) backend doesn't implement fall back to CPU instead
# of crashing — diarization + speaker embeddings run on MPS (~7x faster).
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Machine-local API keys (GROQ_API_KEY, GEMINI_API_KEY) go in _env.local.sh,
# which is gitignored so secrets are never committed. Sourced here (when present)
# so the pipeline and the TUI's reprocess subprocess both inherit the keys.
# ROOT is set by record.sh/tui.sh, but this file is also sourced directly (a
# manual pipeline run, a batch reprocess). Gating on ROOT alone meant those runs
# silently got no keys and fell back to the local backend, which looks like a
# config that did not take rather than a missing secret. Fall back to this
# file's own directory so sourcing it always works.
_ZR_ENV_DIR="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
if [ -f "$_ZR_ENV_DIR/_env.local.sh" ]; then
    source "$_ZR_ENV_DIR/_env.local.sh"
fi
unset _ZR_ENV_DIR
