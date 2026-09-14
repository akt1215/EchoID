import json

import numpy as np

from evaluation import ground_truth as gt
from evaluation.clusters import Group
from tools import rebuild_speakers_db as rebuild


def _g(meeting, cluster, vec, n=10):
    return Group(meeting=meeting, cluster=cluster, wav="w.wav", n_turns=n,
                 total_seconds=float(n), turns=[(0.0, 1.0)] * n,
                 embeddings=np.tile(np.asarray(vec, float), (n, 1)))


def test_one_print_per_confirmed_person():
    groups = [_g("m1", "S0", [1, 0, 0]), _g("m2", "S0", [0, 1, 0])]
    labels = gt.set_label(gt.set_label({}, "m1", "S0", "confirmed", "A"),
                          "m2", "S0", "confirmed", "B")
    db = rebuild.build(groups, labels, mean=None)
    assert set(db) == {"A", "B"}
    assert len(db["A"]) == 3


def test_prints_are_unit_length():
    groups = [_g("m1", "S0", [3, 4, 0])]
    labels = gt.set_label({}, "m1", "S0", "confirmed", "A")
    db = rebuild.build(groups, labels, mean=None)
    assert abs(np.linalg.norm(db["A"]) - 1.0) < 1e-9


def test_multiple_meetings_average_into_one_print():
    groups = [_g("m1", "S0", [1, 0, 0]), _g("m2", "S0", [0, 1, 0])]
    labels = gt.set_label(gt.set_label({}, "m1", "S0", "confirmed", "A"),
                          "m2", "S0", "confirmed", "A")
    db = rebuild.build(groups, labels, mean=None)
    assert set(db) == {"A"}
    assert np.allclose(db["A"], [2 ** -0.5, 2 ** -0.5, 0.0])


def test_unconfirmed_clusters_are_excluded():
    groups = [_g("m1", "S0", [1, 0, 0]), _g("m2", "S0", [0, 1, 0])]
    labels = gt.set_label(gt.set_label({}, "m1", "S0", "confirmed", "A"),
                          "m2", "S0", "multiple", None)
    assert set(rebuild.build(groups, labels, mean=None)) == {"A"}


def test_the_global_mean_is_subtracted_before_normalizing():
    groups = [_g("m1", "S0", [2, 1, 0])]
    labels = gt.set_label({}, "m1", "S0", "confirmed", "A")
    db = rebuild.build(groups, labels, mean=np.asarray([1.0, 1.0, 0.0]))
    assert np.allclose(db["A"], [1.0, 0.0, 0.0])


def test_spelling_variants_fold_into_one_identity():
    groups = [_g("m1", "S0", [1, 0, 0]), _g("m2", "S0", [1, 0, 0])]
    labels = gt.set_label(gt.set_label({}, "m1", "S0", "confirmed", "Riley Stone"),
                          "m2", "S0", "confirmed", "Riley Stone")
    assert len(rebuild.build(groups, labels, mean=None)) == 1


def _mixed(meeting, cluster, spans):
    """spans: [(duration, vector)] — one turn each, in order."""
    turns, embs, t = [], [], 0.0
    for dur, vec in spans:
        turns.append((t, t + dur))
        embs.append(np.asarray(vec, float))
        t += dur + 1.0
    return Group(meeting=meeting, cluster=cluster, wav="w.wav", n_turns=len(spans),
                 total_seconds=sum(d for d, _ in spans), turns=turns,
                 embeddings=np.stack(embs))


def test_short_turns_do_not_drag_the_print_off_the_long_ones():
    """Sub-second turns embed so poorly they miss even their own speaker, so a
    plain mean over every turn is how a blended voiceprint gets built."""
    group = _mixed("m1", "S0", [(30.0, [1, 0, 0])] + [(0.4, [0, 1, 0])] * 20)
    labels = gt.set_label({}, "m1", "S0", "confirmed", "A")
    db = rebuild.build(groups=[group], gt=labels, mean=None)
    assert np.allclose(db["A"], [1.0, 0.0, 0.0], atol=1e-6)


def test_longer_turns_carry_more_weight_than_brief_ones():
    group = _mixed("m1", "S0", [(30.0, [1, 0, 0]), (2.0, [0, 1, 0])])
    labels = gt.set_label({}, "m1", "S0", "confirmed", "A")
    db = rebuild.build(groups=[group], gt=labels, mean=None)
    assert db["A"][0] > db["A"][1] * 10          # 30s dominates 2s


def test_a_cluster_of_only_short_turns_still_yields_a_print():
    """Dropping it entirely would silently lose the speaker; the caller's
    minimum-speech gate is what handles a cluster too small to trust."""
    group = _mixed("m1", "S0", [(0.3, [1, 0, 0]), (0.3, [1, 0, 0])])
    labels = gt.set_label({}, "m1", "S0", "confirmed", "A")
    db = rebuild.build(groups=[group], gt=labels, mean=None)
    assert np.allclose(db["A"], [1.0, 0.0, 0.0], atol=1e-6)


def test_main_backs_up_before_overwriting(tmp_path):
    # The live DB is irreplaceable; the tool must write a .bak before replacing.
    ws = tmp_path / "workspace"
    (ws / "eval").mkdir(parents=True)
    (ws / "speakers.json").write_text(json.dumps({"Old": [1.0, 0.0, 0.0]}))
    gt.save(str(ws / "eval" / "ground_truth.json"), {})
    rc = rebuild.main(["--workspace", str(ws), "--out", str(ws / "speakers.json")])
    assert rc == 0
    backup = ws / "speakers.json.pre-rebuild.bak"
    assert backup.exists()
    assert json.loads(backup.read_text()) == {"Old": [1.0, 0.0, 0.0]}
