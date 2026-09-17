import json

import numpy as np
import soundfile as sf

from tools import deecho_native_recording as deecho


def _write_drifted_fixture(path, sidecar, sr=8000, seconds=20,
                           intercept=0.10, slope=0.01):
    rng = np.random.default_rng(7)
    n = sr * seconds
    system = rng.normal(0, 0.15, n).astype(np.float32)
    system[8 * sr:12 * sr] = 0
    mic_times = np.arange(n) / sr
    system_positions = (intercept + (1 + slope) * mic_times) * sr
    echo = np.interp(system_positions, np.arange(n), system,
                     left=0, right=0).astype(np.float32)
    local = np.zeros(n, dtype=np.float32)
    local[9 * sr:11 * sr] = 0.25 * np.sin(
        2 * np.pi * 300 * np.arange(2 * sr) / sr)
    sf.write(path, np.column_stack((0.6 * echo + local, system)), sr,
             subtype="PCM_16")
    sidecar.write_text(json.dumps({
        "diarization": [
            {"start": 0, "end": 8},
            {"start": 12, "end": 20},
        ]
    }))


def test_repair_preserves_system_suppresses_echo_and_keeps_local(tmp_path):
    source = tmp_path / "meeting.wav"
    sidecar = tmp_path / "meeting.segments.json"
    output = tmp_path / "meeting_deechoed.wav"
    _write_drifted_fixture(source, sidecar)

    deecho.repair(str(source), str(output), str(sidecar),
                  drift=(0.10, 0.01), activity_threshold=1e-3)
    original, _ = sf.read(source, always_2d=True)
    repaired, _ = sf.read(output, always_2d=True)

    assert np.allclose(repaired[:, 1], original[:, 1], atol=4e-5)
    assert np.sqrt(np.mean(repaired[2 * 8000:7 * 8000, 0] ** 2)) < 1e-4
    assert np.sqrt(np.mean(repaired[9 * 8000:11 * 8000, 0] ** 2)) > 0.1
    assert sf.info(output).channels == 2


def test_repair_refuses_to_overwrite_source(tmp_path):
    source = tmp_path / "meeting.wav"
    sf.write(source, np.zeros((800, 2), dtype=np.float32), 8000)
    try:
        deecho.repair(str(source), str(source), drift=(0.0, 0.0))
    except ValueError as exc:
        assert "overwrite" in str(exc)
    else:
        raise AssertionError("repair accepted the source as its output")
