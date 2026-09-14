"""Resolve the torch device for the PyTorch models (diarization + embeddings).

`auto` picks Metal (MPS) on Apple Silicon, else CPU — measured ~7x faster than
CPU for pyannote diarization. Resolved here rather than passing the literal
'auto' to torch.device (which raises). Transcription is unaffected: ctranslate2
has no MPS, so it stays on CPU (or the Groq backend).
"""

import torch


def resolve_device(setting="auto"):
    if setting and setting != "auto":
        return setting
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
