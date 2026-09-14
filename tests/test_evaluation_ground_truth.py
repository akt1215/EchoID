import pytest

from evaluation import ground_truth as gt


def test_set_and_read_confirmed():
    d = gt.set_label({}, "m1", "SPEAKER_00", "confirmed", "Jordan Lee")
    assert gt.confirmed_name(d, "m1", "SPEAKER_00") == "Jordan Lee"
    assert gt.labeled_keys(d) == {("m1", "SPEAKER_00")}


def test_multiple_and_skip_carry_no_name():
    d = gt.set_label({}, "m1", "SPEAKER_00", "multiple", "ignored")
    assert gt.get(d, "m1", "SPEAKER_00")["name"] is None
    # "multiple" means the cluster holds >1 person: not usable as a positive
    assert gt.confirmed_name(d, "m1", "SPEAKER_00") is None
    d = gt.set_label(d, "m1", "SPEAKER_01", "skip")
    assert gt.confirmed_name(d, "m1", "SPEAKER_01") is None


def test_confirmed_requires_a_name():
    with pytest.raises(ValueError):
        gt.set_label({}, "m1", "SPEAKER_00", "confirmed", "  ")


def test_unknown_verdict_rejected():
    with pytest.raises(ValueError):
        gt.set_label({}, "m1", "SPEAKER_00", "maybe", "X")


def test_relabel_overwrites_in_place():
    d = gt.set_label({}, "m1", "S0", "confirmed", "A")
    d = gt.set_label(d, "m1", "S0", "confirmed", "B")
    assert gt.confirmed_name(d, "m1", "S0") == "B"
    assert len(gt.labeled_keys(d)) == 1


def test_roundtrip_through_disk(tmp_path):
    p = tmp_path / "eval" / "ground_truth.json"
    d = gt.set_label({}, "m1", "S0", "confirmed", "Jordan Lee")
    gt.save(str(p), d)
    assert gt.load(str(p)) == d


def test_load_missing_file_returns_empty(tmp_path):
    assert gt.load(str(tmp_path / "nope.json")) == {}


def test_names_are_whitespace_normalized():
    d = gt.set_label({}, "m1", "S0", "confirmed", "  Jordan   Lee ")
    assert gt.confirmed_name(d, "m1", "S0") == "Jordan Lee"
