#!/bin/bash
set -euo pipefail

# ────────────────────────────────────────────────────────────
# Post-venv setup for EchoID
# Patches the conflicting libavdevice dylib and ensures the
# environment is ready to run.
# ────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

echo "==> Setting up EchoID environment..."

# 1. Ensure virtualenv exists
if [ ! -f "$VENV/bin/python3" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$VENV"
fi

# 2. Install dependencies
echo "Installing dependencies..."
"$VENV/bin/pip" install -r "$ROOT/requirements.txt" --quiet

# Resolve the venv's site-packages once, after pip install has populated it.
# Globbing python3.* keeps this working across Python minor-version bumps.
SITE_PACKAGES="$(ls -d "$VENV"/lib/python3.*/site-packages 2>/dev/null | head -1 || true)"

# 3. Patch conflicting libavdevice (cv2 ↔ av dylib)
# Glob the bundled ffmpeg version so an opencv upgrade doesn't silently skip this.
CV2_DEVICE="$(ls "$SITE_PACKAGES"/cv2/.dylibs/libavdevice*.dylib 2>/dev/null | head -1 || true)"

# llvm-objcopy is keg-only (not on PATH); prefer it there, then Homebrew's prefix.
LLVM_OBJCOPY="$(command -v llvm-objcopy || true)"
if [ -z "$LLVM_OBJCOPY" ]; then
    BREW_LLVM="$(brew --prefix llvm 2>/dev/null || true)"
    if [ -n "$BREW_LLVM" ] && [ -x "$BREW_LLVM/bin/llvm-objcopy" ]; then
        LLVM_OBJCOPY="$BREW_LLVM/bin/llvm-objcopy"
    fi
fi

if [ -n "$CV2_DEVICE" ] && [ -f "$CV2_DEVICE" ]; then
    # Check if section is already removed
    if otool -l "$CV2_DEVICE" 2>/dev/null | grep -q "__objc_classlist"; then
        echo "Patching libavdevice (removing Objective-C classlist to prevent duplicate registration)..."
        if [ -n "$LLVM_OBJCOPY" ] && [ -f "$LLVM_OBJCOPY" ]; then
            cp "$CV2_DEVICE" "${CV2_DEVICE}.bak"
            "$LLVM_OBJCOPY" --remove-section=__DATA_CONST,__objc_classlist "$CV2_DEVICE"
            echo "  ✓ Patched"
        else
            echo "  ⚠ llvm-objcopy not found (looked on PATH and via 'brew --prefix llvm')"
            echo "    Install with: brew install llvm"
        fi
    else
        echo "  ✓ libavdevice already patched"
    fi
else
    echo "  ⚠ cv2 libavdevice not found under $SITE_PACKAGES/cv2/.dylibs/"
    echo "    (may need to install opencv-python first)"
fi

# 4. Torchcodec rpath fix (if torchcodec is installed)
TCODEC="$SITE_PACKAGES/torchcodec"
FFMPEG_LIB="/opt/homebrew/opt/ffmpeg/lib"
if [ -d "$TCODEC" ]; then
    for f in "$TCODEC"/libtorchcodec_core*.dylib; do
        if [ -f "$f" ]; then
            if ! otool -l "$f" 2>/dev/null | grep -q "$FFMPEG_LIB"; then
                echo "Patching torchcodec rpath..."
                install_name_tool -add_rpath "$FFMPEG_LIB" "$f" 2>/dev/null || true
            fi
        fi
    done
    # Compat symlinks for libavutil
    for ver in 56 57 58 59; do
        target="$FFMPEG_LIB/libavutil.$ver.dylib"
        if [ ! -f "$target" ]; then
            ln -sf libavutil.60.dylib "$target" 2>/dev/null || true
        fi
    done
    echo "  ✓ torchcodec rpath set"
fi

# 5. Patch pyannote io.py to skip hanging torchcodec import
PYANN_IO="$SITE_PACKAGES/pyannote/audio/core/io.py"
if [ -f "$PYANN_IO" ]; then
    if grep -q "TORCHCODEC_AVAILABLE = False" "$PYANN_IO" && grep -q "import torchcodec" "$PYANN_IO"; then
        echo "Patching pyannote io.py (skip torchcodec import — hangs on FFmpeg version mismatch)..."
        "$VENV/bin/python3" -c "
import re
with open('$PYANN_IO') as f:
    content = f.read()
# Replace the try/except block that imports torchcodec
old = '''try:
    import torchcodec
    from torchcodec import AudioSamples
    from torchcodec.decoders import AudioDecoder, AudioStreamMetadata
    TORCHCODEC_AVAILABLE = True
except Exception as e:
    warnings.warn(
        \"\\\\ntorchcodec is not installed correctly so built-in audio decoding will fail. Solutions are:\\\\n\"
        \"* use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary;\\\\n\"
        \"* fix torchcodec installation. Error message was:\\\\n\\\\n\"
        f\"{e}\"
    )
    TORCHCODEC_AVAILABLE = False
    AudioDecoder = None
    AudioStreamMetadata = None'''
new = '''# Patched: torchcodec import hangs (FFmpeg dylib version mismatch) — we pass audio
# in-memory so it's never needed anyway.
TORCHCODEC_AVAILABLE = False
AudioDecoder = None
AudioStreamMetadata = None'''
content = content.replace(old, new)
with open('$PYANN_IO', 'w') as f:
    f.write(content)
"
        echo "  ✓ Patched"
    else
        echo "  ✓ pyannote io.py already patched"
    fi
fi

# 6. Build the ScreenCaptureKit audio helper (replaces BlackHole on the sck backend)
SCK_SRC="$ROOT/native/sck_capture.swift"
SCK_BIN="$ROOT/native/sck_capture"
if [ -f "$SCK_SRC" ]; then
    if command -v swiftc >/dev/null 2>&1; then
        echo "Building ScreenCaptureKit helper (native/sck_capture)..."
        swiftc -parse-as-library -O "$SCK_SRC" -o "$SCK_BIN"
        echo "  ✓ Built"
    else
        echo "  ⚠ swiftc not found — skipping SCK helper build."
        echo "    Install Xcode / Command Line Tools, or set audio.capture_backend: blackhole in config.yaml."
    fi
fi

echo ""
echo "Setup complete. Run: ./record.sh --list"
