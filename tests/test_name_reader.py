"""VLM name reader: cleaning, cluster voting, and backend dispatch (no network)."""

from ai.name_reader import (
    NameReader, clean_reading, is_local_name, name_clusters, vote_cluster_name,
)

LOCAL = ["Alex Morgan", "Alex M.", "A. Morgan", "Alex"]


# ── clean_reading ──────────────────────────────────────────────────────────

def test_clean_reading_accepts_a_name():
    assert clean_reading("Emilia Hermann") == "Emilia Hermann"
    assert clean_reading('"Riley Stone"') == "Riley Stone"   # strips quotes
    assert clean_reading("Emilia Hermann\n(speaking)") == "Emilia Hermann"  # first line


def test_clean_reading_rejects_none_and_garbage():
    assert clean_reading("NONE") is None
    assert clean_reading("none") is None
    assert clean_reading("") is None
    assert clean_reading("   ") is None
    assert clean_reading("0 a: ay") is None


def test_clean_reading_rejects_the_local_user():
    # The local user's own name (read off their self-view tile) must never surface
    # as a speaker — they are channel 0, resolved by the channel-energy override.
    assert clean_reading("Alex Morgan", local_names=LOCAL) is None
    assert clean_reading("Alex Morga", local_names=LOCAL) is None   # truncated read
    assert clean_reading("Alex Morgan", local_names=LOCAL) is None
    assert clean_reading("Alex", local_names=LOCAL) is None           # given-name-only
    # A genuine remote participant still passes, even with the exclusion active.
    assert clean_reading("Emilia Hermann", local_names=LOCAL) == "Emilia Hermann"


# ── is_local_name ──────────────────────────────────────────────────────────

def test_is_local_name_fuzzy_multi_token_exact_single_token():
    assert is_local_name("Alex Morgan", LOCAL)
    assert is_local_name("alex  morgan", LOCAL)         # spacing/case tolerant
    assert is_local_name("Alex Morga", LOCAL)         # truncation (fuzzy)
    assert is_local_name("Alex", LOCAL)                 # single-token exact
    # Single-token names match ONLY exactly, so a similar real name is NOT excluded.
    assert not is_local_name("Akira Tan", LOCAL)
    assert not is_local_name("Emilia Hermann", LOCAL)
    assert not is_local_name("Alex Morgan", [])      # no local names configured


# ── vote_cluster_name ──────────────────────────────────────────────────────

def test_vote_cluster_name_takes_valid_majority():
    reads = ["Emilia Hermann", "Emilia Hermann", "NONE", "0 a: ay"]
    assert vote_cluster_name(reads) == "Emilia Hermann"


def test_vote_cluster_name_none_when_no_valid_reads():
    assert vote_cluster_name(["NONE", "", "~ re oi ~"]) is None
    assert vote_cluster_name([]) is None


def test_vote_cluster_name_requires_agreement():
    # Two valid but DISAGREEING reads -> Unknown, not a coin-flip guess.
    assert vote_cluster_name(["Emilia Hermann", "Riley Stone"], min_agreement=2) is None
    # Two that agree -> accepted; majority over a lone dissenter -> accepted.
    assert vote_cluster_name(["Emilia Hermann", "Emilia Hermann"], min_agreement=2) == "Emilia Hermann"
    assert vote_cluster_name(["Emilia Hermann", "Emilia Hermann", "Riley Stone"],
                             min_agreement=2) == "Emilia Hermann"
    # A single read has nothing to corroborate against, so it is still accepted.
    assert vote_cluster_name(["Emilia Hermann"], min_agreement=2) == "Emilia Hermann"


def test_vote_cluster_name_excludes_local():
    assert vote_cluster_name(["Alex Morgan", "Alex Morgan"], local_names=LOCAL) is None


# ── NameReader dispatch (injected fake backend, no network) ────────────────

def test_name_reader_returns_cleaned_backend_output():
    r = NameReader(backend_fn=lambda path: "Taylor Kim")
    assert r.read("frame.jpg") == "Taylor Kim"


def test_name_reader_returns_none_on_none_reading():
    r = NameReader(backend_fn=lambda path: "NONE")
    assert r.read("frame.jpg") is None


def test_name_reader_excludes_local_name():
    r = NameReader(backend_fn=lambda path: "Alex Morgan")
    assert r.read("frame.jpg", local_names=LOCAL) is None


def test_name_reader_swallows_backend_errors():
    def boom(path):
        raise RuntimeError("network down")
    r = NameReader(backend_fn=boom)
    assert r.read("frame.jpg") is None


# ── name_clusters (frame selection + per-cluster vote) ─────────────────────

class _FakeReader:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def read(self, path, known=(), local_names=()):
        self.calls.append(path)
        name = self.mapping.get(path)
        return None if is_local_name(name or "", local_names) else name


def test_name_clusters_reads_only_nearby_frames_and_votes():
    segs = [{"start": 0, "end": 5, "speaker": "SPEAKER_00"},
            {"start": 20, "end": 25, "speaker": "SPEAKER_01"}]
    frame_events = [(2.0, "a.jpg"), (4.0, "b.jpg"), (22.0, "c.jpg"), (100.0, "far.jpg")]
    reader = _FakeReader({"a.jpg": "Emilia Hermann", "b.jpg": "Emilia Hermann",
                          "c.jpg": "Riley Stone"})
    out = name_clusters(segs, ["SPEAKER_00", "SPEAKER_01"], frame_events, reader)
    assert out["SPEAKER_00"] == "Emilia Hermann"
    assert out["SPEAKER_01"] == "Riley Stone"   # lone read, still accepted
    assert "far.jpg" not in reader.calls          # frame far from any turn is skipped


def test_name_clusters_skips_cluster_with_no_valid_reads():
    segs = [{"start": 0, "end": 5, "speaker": "S0"}]
    reader = _FakeReader({})                       # every read returns None
    out = name_clusters(segs, ["S0"], [(2.0, "f.jpg")], reader)
    assert "S0" not in out


def test_name_clusters_excludes_local_user():
    # A cluster the model reads as the local user must not be labeled with his name.
    segs = [{"start": 0, "end": 5, "speaker": "S0"}]
    reader = _FakeReader({"a.jpg": "Alex Morgan", "b.jpg": "Alex Morgan"})
    out = name_clusters(segs, ["S0"], [(1.0, "a.jpg"), (3.0, "b.jpg")], reader,
                        local_names=LOCAL)
    assert "S0" not in out


def test_name_clusters_disagreeing_frames_yield_unknown():
    segs = [{"start": 0, "end": 5, "speaker": "S0"}]
    reader = _FakeReader({"a.jpg": "Emilia Hermann", "b.jpg": "Riley Stone"})
    out = name_clusters(segs, ["S0"], [(1.0, "a.jpg"), (3.0, "b.jpg")], reader,
                        min_agreement=2)
    assert "S0" not in out
