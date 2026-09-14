"""Clips for labeling must contain audible speech, not merely be long.

Real failure that motivated this: meeting 2026-07-23 SPEAKER_01 has 39 turns.
Every audible one is 0.6-0.9s; the only turn over 3s is near-silent. Picking the
longest turn handed the user 7.7 seconds of silence, and they labeled the
cluster "I_DON'T_HEAR_ANYTHING".
"""

import json

import numpy as np
import soundfile as sf

from evaluation import clusters


def _meeting(tmp_path, turns, sr=8000):
    """turns: [(start, end, amplitude)] written into ch1 of a stereo wav."""
    total = int(max(e for _, e, _ in turns) * sr) + sr
    right = np.zeros(total, dtype="float32")
    rng = np.random.default_rng(0)
    for start, end, amp in turns:
        a, b = int(start * sr), int(end * sr)
        right[a:b] = rng.normal(0, amp, b - a).astype("float32")
    wav = tmp_path / "meeting_2026-01-01_00-00-00.wav"
    sf.write(str(wav), np.stack([np.zeros(total, dtype="float32"), right], axis=1), sr)

    side = tmp_path / "meeting_2026-01-01_00-00-00.segments.json"
    side.write_text(json.dumps({
        "wav": str(wav),
        "diarization": [
            {"start": s, "end": e, "label": "X", "cluster": "S0",
             "source": "visual", "embedding": [1.0, 0.0]}
            for s, e, _ in turns
        ],
    }))
    return str(tmp_path)


def test_loudest_turns_skips_a_long_silent_turn(tmp_path):
    # The 6s turn is silent; the audible speech is in two short bursts.
    ws = _meeting(tmp_path, [(0.0, 6.0, 0.0), (7.0, 7.9, 0.2), (9.0, 9.8, 0.2)])
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.loudest_turns(g)
    assert picked, "no clip selected"
    assert all(not (s < 6.0) for s, _ in picked), \
        f"the silent 0-6s turn was selected: {picked}"


def test_loudest_turns_stitches_several_short_bursts(tmp_path):
    ws = _meeting(tmp_path, [(0.0, 0.9, 0.2), (2.0, 2.9, 0.2), (4.0, 4.9, 0.2)])
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.loudest_turns(g, max_seconds=8.0)
    assert len(picked) >= 2, "short bursts should be combined into one clip"


def test_loudest_turns_respects_the_total_budget(tmp_path):
    ws = _meeting(tmp_path, [(float(i), float(i) + 0.9, 0.2) for i in range(0, 40, 2)])
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.loudest_turns(g, max_seconds=4.0)
    assert sum(e - s for s, e in picked) <= 4.0 + 1e-6


def test_loudest_turns_returns_spans_in_time_order(tmp_path):
    ws = _meeting(tmp_path, [(0.0, 0.9, 0.1), (2.0, 2.9, 0.3), (4.0, 4.9, 0.2)])
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.loudest_turns(g, max_seconds=8.0)
    assert picked == sorted(picked), "clip should play in chronological order"


def test_silence_ratio_flags_a_dead_cluster(tmp_path):
    ws = _meeting(tmp_path, [(0.0, 2.0, 0.0), (3.0, 5.0, 0.0)])
    g = clusters.discover(ws, min_turns=1)[0]
    assert clusters.peak_rms(g) < 1e-3


def test_peak_rms_reports_a_live_cluster(tmp_path):
    ws = _meeting(tmp_path, [(0.0, 2.0, 0.2)])
    g = clusters.discover(ws, min_turns=1)[0]
    assert clusters.peak_rms(g) > 0.05


def test_loudest_turns_spread_across_the_cluster_timeline(tmp_path):
    """A clip drawn only from the loudest turns can come entirely from one
    speaker, so a cluster holding two people sounds like one. Sampling across
    the cluster's span is what makes a second voice audible. Here the loud turns
    sit early and the quieter ones late; the clip must reach the late half."""
    # Enough loud early turns to exhaust the budget on their own, which is the
    # shape of a real cluster where one speaker dominates the first half.
    early = [(float(i), float(i) + 0.9, 0.30) for i in range(0, 40, 2)]
    late = [(float(i), float(i) + 0.9, 0.12) for i in range(60, 70, 2)]
    ws = _meeting(tmp_path, early + late)
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.loudest_turns(g, max_seconds=8.0)
    assert any(s >= 55.0 for s, _ in picked), \
        f"clip never reaches the second half of the cluster: {picked}"
    assert any(s < 20.0 for s, _ in picked), "clip should still include the first half"


def test_spread_still_rejects_silence(tmp_path):
    # Reaching across the timeline must not mean including dead turns.
    ws = _meeting(tmp_path, [(0.0, 0.9, 0.25), (2.0, 2.9, 0.25),
                             (60.0, 66.0, 0.0), (70.0, 76.0, 0.0)])
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.loudest_turns(g, max_seconds=8.0)
    assert all(s < 50.0 for s, _ in picked), f"silent turns were included: {picked}"


def test_ranked_turns_embeds_loud_speech_not_long_silence(tmp_path):
    """Turns chosen for embedding must be picked by level, not length.

    A cluster's longest turn is often a diarization artifact with nothing in it.
    Embedding those builds a voiceprint out of room noise that matches nobody --
    including the same speaker in another meeting. This was a real bug: one
    cluster's voiceprint scored 0.075 against the same person elsewhere.
    """
    ws = _meeting(tmp_path, [(0.0, 20.0, 0.0),          # long and silent
                             (30.0, 32.0, 0.25),        # short and audible
                             (40.0, 42.0, 0.25)])
    g = clusters.discover(ws, min_turns=1)[0]
    picked = clusters.ranked_turns(g, max_turns=2)
    assert all(s >= 30.0 for s, _ in picked), \
        f"the long silent turn was chosen for embedding: {picked}"


def test_ranked_turns_falls_back_to_longest_without_audio(tmp_path):
    side = tmp_path / "meeting_2026-01-09_00-00-00.segments.json"
    side.write_text(json.dumps({
        "wav": str(tmp_path / "gone.wav"),
        "diarization": [
            {"start": 0.0, "end": 9.0, "label": "X", "cluster": "S0",
             "source": "visual", "embedding": [1.0, 0.0]},
            {"start": 10.0, "end": 11.0, "label": "X", "cluster": "S0",
             "source": "visual", "embedding": [1.0, 0.0]},
        ],
    }))
    g = clusters.discover(str(tmp_path), min_turns=1)[0]
    assert clusters.ranked_turns(g, max_turns=1) == [(0.0, 9.0)]


def test_missing_wav_falls_back_to_the_metadata_pick(tmp_path):
    side = tmp_path / "meeting_2026-01-02_00-00-00.segments.json"
    side.write_text(json.dumps({
        "wav": str(tmp_path / "gone.wav"),
        "diarization": [
            {"start": 0.0, "end": 5.0, "label": "X", "cluster": "S0",
             "source": "visual", "embedding": [1.0, 0.0]},
            {"start": 6.0, "end": 6.5, "label": "X", "cluster": "S0",
             "source": "visual", "embedding": [1.0, 0.0]},
        ],
    }))
    g = clusters.discover(str(tmp_path), min_turns=1)[0]
    assert clusters.loudest_turns(g) == [(0.0, 5.0)]   # longest, as before
    assert clusters.peak_rms(g) is None
