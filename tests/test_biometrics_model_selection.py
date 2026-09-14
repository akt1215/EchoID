"""The model id must follow the backend.

main.py passed config's `speechbrain_model` whatever the backend was, so
selecting wespeaker_onnx handed the ONNX loader an ECAPA repo id and the
pipeline died at model load with a 404 for
speechbrain/spkrec-ecapa-voxceleb/voxceleb_resnet34.onnx. Unit tests could not
see it: every one of them builds the manager directly with an explicit model
name, so only a full pipeline run reached the mismatch.
"""

import pytest
import yaml

from ai.biometrics import model_for


def test_speechbrain_backend_uses_the_speechbrain_model():
    cfg = {"backend": "speechbrain",
           "speechbrain_model": "speechbrain/spkrec-ecapa-voxceleb",
           "wespeaker_model": "WESPEAKER/wespeaker-voxceleb-resnet34"}
    assert model_for(cfg) == "speechbrain/spkrec-ecapa-voxceleb"


def test_wespeaker_backend_uses_the_wespeaker_model():
    cfg = {"backend": "wespeaker_onnx",
           "speechbrain_model": "speechbrain/spkrec-ecapa-voxceleb",
           "wespeaker_model": "WESPEAKER/wespeaker-voxceleb-resnet34"}
    assert model_for(cfg) == "WESPEAKER/wespeaker-voxceleb-resnet34"


def test_an_unknown_backend_is_loud():
    with pytest.raises(ValueError):
        model_for({"backend": "nope"})


def test_a_missing_model_id_is_loud():
    # Silently falling back to the other backend's model is what caused the
    # original failure; an absent id must stop the run instead.
    with pytest.raises(ValueError):
        model_for({"backend": "wespeaker_onnx", "speechbrain_model": "x"})


def test_the_shipped_config_resolves_to_a_wespeaker_repo():
    with open("config.example.yaml") as f:
        bio = yaml.safe_load(f)["biometrics"]
    name = model_for(bio)
    if bio["backend"] == "wespeaker_onnx":
        assert "wespeaker" in name.lower(), \
            f"backend is wespeaker_onnx but the model id is {name!r}"
