# Contributing to EchoID

Thank you for your interest in contributing to EchoID. EchoID is a local-first, identity-aware meeting notetaker for macOS.

## Architectural Invariants

Before proposing changes, please note several load-bearing design decisions that must be preserved:

- **Dual-Channel Audio Invariant**: Capture always preserves two channels: Channel 0 = local microphone, Channel 1 = remote participants (system audio). Diarization, speaker identification, and energy dominance rely on this channel separation. Never collapse or pre-downmix native recordings at capture time.
- **Mean-Centered Voiceprints**: ECAPA and ResNet voice embeddings share a dominant common-mean vector (~75% of magnitude) across different speakers. All biometric cosine distance calculations and database comparisons must occur in a mean-centered space (`core/embedding_mean.py`).
- **Privacy & Offline First**: Audio and meeting data must never leave the machine by default. Cloud integrations (Groq transcription, Gemini vision) must remain opt-in and require explicit user keys.
- **Single Source of Truth**: Tunable thresholds, paths, model names, and feature flags belong in `config.example.yaml` and `config.yaml`, never hardcoded into module internals.
- **macOS Dylib Compatibility**: Launching through `./record.sh` and `./tui.sh` sources `_env.sh`, which sets `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to prevent duplicate Objective-C class conflicts between OpenCV and PyAV dylibs.

## Development Setup

1. **Prerequisites**:
   - macOS 15+ (Sequoia) on Apple Silicon (M1/M2/M3/M4)
   - Homebrew (`brew install ollama ffmpeg`)
   - Python 3.13+

2. **Environment Initialization**:
   ```bash
   cp config.example.yaml config.yaml
   ./setup.sh
   ```
   Note: `./setup.sh` applies binary patches to `cv2` and `torchcodec` dylibs via `llvm-objcopy` to resolve Apple Silicon shared-library conflicts.

3. **Running the Test Suite**:
   ```bash
   .venv/bin/python -m pytest
   ```
   All tests in `tests/` should pass. An autouse guard in `tests/conftest.py` protects real configuration and workspace data during test execution.

## Pull Request Guidelines

1. **Fork and Branch**: Create a descriptive feature branch from `main` (e.g., `git checkout -b feature/improved-vad`).
2. **Test Coverage**: Accompany any bug fix or new feature with unit or regression tests under `tests/`.
3. **Commit Discipline**:
   - Write clear, present-tense commit messages (e.g., `add drift compensation test for channel pairer`).
   - Keep logically independent changes in separate commits.
4. **No Secrets or Personal Data**: Never commit `.wav` files, real `speakers.json` databases, private notes, or API keys. Verify git status and diff before pushing.
