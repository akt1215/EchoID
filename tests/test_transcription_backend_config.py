"""The public configuration template must keep audio on the machine by default."""

import yaml


def _config():
    with open("config.example.yaml") as f:
        return yaml.safe_load(f)


def test_transcription_backend_is_local():
    assert _config()["transcription"]["backend"] == "local"


def test_local_transcription_stays_on_cpu():
    """torch.device rejects "auto", so whisperx crashes on anything else; the
    local path must remain usable as the fallback it is."""
    assert _config()["transcription"]["device"] == "cpu"
