# Security and Privacy Policy

## Privacy Architecture & Threat Model

EchoID is designed from the ground up as a **local-first, privacy-preserving** meeting notetaker for macOS. Meeting audio often contains sensitive, confidential, or proprietary information. The architecture reflects this reality with strict technical boundaries:

- **100% Local by Default**: Meeting audio capture, speaker diarization, voiceprint biometrics, speech-to-text transcription, LLM summarization, and note generation run locally on Apple Silicon.
- **Zero Telemetry**: The application contains no analytics, telemetry, phone-home mechanisms, or background crash reporters.
- **Data Boundaries**:
  - Raw audio recordings (`.wav`), extracted video frames (`.frames/`), biometric databases (`speakers.json`), and generated notes land strictly inside the local `workspace/` directory.
  - The `workspace/`, `config.yaml`, and `_env.local.sh` paths are gitignored to prevent accidental exposure of personal data or credentials.
- **Explicit Cloud Opt-Ins**:
  - External cloud services are disabled by default.
  - Optional cloud transcription (`transcription.backend: groq`) and vision OCR (`name_reader.backend: gemini`) are strictly opt-in and require user-provided API keys in `_env.local.sh`. Audio or frame data leaves the machine only when a cloud backend is intentionally configured.
- **System Permissions**: EchoID relies entirely on standard macOS TCC permissions (Screen Recording and Microphone) and ScreenCaptureKit system APIs without requiring root privileges or third-party kernel extensions.

## Supported Versions

| Version | Supported |
| :--- | :--- |
| main (latest) | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability or privacy leak (such as unintended network egress or insecure file permission defaults), please report it responsibly:

1. **GitHub Security Advisory**: Use the [Private Vulnerability Reporting](https://github.com/akt1215/EchoID/security/advisories/new) tab on GitHub.
2. **Direct Contact**: If GitHub Advisory is unavailable, contact the maintainer directly via the contact methods listed on [akt1215.github.io](https://akt1215.github.io).

Please include:
- A description of the issue and potential impact.
- Step-by-step reproduction steps or proof-of-concept.
- The operating environment (macOS version, hardware architecture).

Vulnerability reports will be acknowledged within 48 hours, and patches will be developed and released through the public repository.
