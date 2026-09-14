import json

from core import glossary


def test_clean_term_keeps_real_terms():
    assert glossary.clean_term("Sam Rivera") == "Sam Rivera"
    assert glossary.clean_term("  Virtual PBV  ") == "Virtual PBV"


def test_clean_term_rejects_speaker_label_junk():
    assert glossary.clean_term("Unknown (SPEAKER_01)") is None
    assert glossary.clean_term("Unknown Example Group 01") is None
    assert glossary.clean_term("Me (Local)") is None
    assert glossary.clean_term("") is None
    assert glossary.clean_term("ab") is None  # too short


def test_clean_term_strips_trailing_parenthetical_decoration():
    assert glossary.clean_term("HSF (Hippocampal Subfields)") == "HSF"
    assert glossary.clean_term("Photon counting CT (PCC)") == "Photon counting CT"


def test_active_terms_filters_existing_store_junk(tmp_path):
    # Pre-existing pollution in the store must not reach any consumer, even
    # though we never rewrite the user's file.
    p = str(tmp_path / "g.json")
    with open(p, "w") as f:
        json.dump({"Jordan Lee": 2, "Unknown Example Group 01": 2, "Me (Local)": 5}, f)
    assert glossary.active_terms(p) == ["Jordan Lee"]


def test_harvest_rejects_junk_names_and_entities(tmp_path):
    p = str(tmp_path / "g.json")
    active = glossary.harvest(p, names=["Unknown Example Group 01", "Jordan Lee"])
    assert "Jordan Lee" in active
    assert "Unknown Example Group 01" not in active
    glossary.harvest(p, entities=["Me (Local)"])
    glossary.harvest(p, entities=["Me (Local)"])
    assert "Me (Local)" not in glossary.active_terms(p)


def test_harvest_recovers_from_corrupt_store(tmp_path):
    p = str(tmp_path / "g.json")
    with open(p, "w") as f:
        f.write("{ truncated not json")
    active = glossary.harvest(p, names=["Jordan Lee"])  # must not raise
    assert "Jordan Lee" in active
    with open(p) as f:
        assert json.load(f)["Jordan Lee"] >= glossary.MIN_COUNT  # valid JSON written
    assert (tmp_path / "g.json.bad").exists()  # corrupt file preserved aside


def test_names_pinned_active_immediately(tmp_path):
    p = str(tmp_path / "g.json")
    active = glossary.harvest(p, names=["Sam Rivera"])
    assert "Sam Rivera" in active
    # persists across reload
    assert "Sam Rivera" in glossary.active_terms(p)


def test_entities_active_only_after_recurrence(tmp_path):
    p = str(tmp_path / "g.json")
    a1 = glossary.harvest(p, entities=["Virtual PBV"])
    assert "Virtual PBV" not in a1            # first sighting: not yet active
    a2 = glossary.harvest(p, entities=["Virtual PBV"])
    assert "Virtual PBV" in a2                # second sighting: active


def test_short_entities_ignored(tmp_path):
    p = str(tmp_path / "g.json")
    glossary.harvest(p, entities=["ok"])       # len < 3
    glossary.harvest(p, entities=["ok"])
    assert "ok" not in glossary.active_terms(p)


def test_missing_file_is_empty(tmp_path):
    assert glossary.active_terms(str(tmp_path / "nope.json")) == []


TERMS = ["FreeSurfer", "NIfTI", "AI-PET", "ResUMamba", "3D Slicer", "PET", "tau",
         "Sam Rivera", "Jordan Lee", "Siemens Sensation 64", "Virtual PET",
         "MAPE", "ANTs"]


def correct(text):
    return glossary.TermCorrector(TERMS)(text)


def test_exact_key_match_restores_canonical_formatting():
    # Tier 1: same letters, different casing/punctuation. No judgement involved.
    assert correct("we ran freesurfer on it") == "we ran FreeSurfer on it"
    assert correct("saved as a nifti") == "saved as a NIfTI"
    assert correct("open it in 3dslicer") == "open it in 3D Slicer"
    assert correct("the 3d slicer version") == "the 3D Slicer version"


def test_near_miss_is_corrected():
    # Tier 2: what Whisper actually produced once the prompt was removed.
    assert correct("run FeeSurfer first") == "run FreeSurfer first"
    assert correct("the ResUMamba model") == "the ResUMamba model"
    assert correct("Sam Rvera presented") == "Sam Rivera presented"


def test_already_correct_text_is_untouched():
    for s in ("we ran FreeSurfer on the NIfTI", "AI-PET and Virtual PET",
              "Jordan Lee said so"):
        assert correct(s) == s


def test_short_tokens_are_never_fuzzy_matched():
    """"pad"/"pet" and "top"/"tau" are one edit apart. Only an exact key match
    may rewrite something this short, or the corrector invents errors."""
    assert correct("put the pad down") == "put the pad down"
    assert correct("at the top of the file") == "at the top of the file"
    assert correct("the maps are ready") == "the maps are ready"


def test_common_english_words_are_never_rewritten():
    """A glossary of surnames and acronyms sits one or two edits from ordinary
    words; rewriting those is worse than leaving a spelling wrong."""
    assert correct("raise the bar for the group") == "raise the bar for the group"
    assert correct("the ants were everywhere") == "the ants were everywhere"


def test_longest_phrase_wins():
    assert correct("the virtual pet project uses a siemens sensation 64") == (
        "the Virtual PET project uses a Siemens Sensation 64")


def test_multi_word_terms_survive_punctuation_between_words():
    assert correct("ask jordan  lee about it") == "ask Jordan Lee about it"


def test_empty_glossary_and_empty_text_are_no_ops():
    assert glossary.TermCorrector([])("freesurfer stays") == "freesurfer stays"
    assert not glossary.TermCorrector([])
    assert correct("") == ""
    assert correct(None) is None


def test_correction_preserves_surrounding_text_exactly():
    src = "  So, freesurfer -- and then nifti (v2)!  "
    assert correct(src) == "  So, FreeSurfer -- and then NIfTI (v2)!  "


def test_a_possessive_is_not_swallowed():
    """"Amelia's" keys as "amelias" and fuzzy-matches "Amelia" at 0.92; without
    the possessive split the correction silently deletes the "'s"."""
    c = glossary.TermCorrector(["Amelia", "FreeSurfer"])
    assert c("Amelia's paper is due") == "Amelia's paper is due"
    assert c("FeeSurfer's output") == "FreeSurfer's output"


def test_a_fuzzy_match_may_not_swallow_a_neighbouring_word():
    """Both endpoints of the key must survive. These are the real corruptions
    seen over three meetings before the guard existed."""
    c = glossary.TermCorrector(["CT emphysema subtypes", "SPIROMICS", "Taylor Morgan",
                                "Virtual PBV"])
    # trailing word eaten: "CT-emposema subtypes in" -> "CT emphysema subtypes"
    out = c("the V2 CT-emposema subtypes in spironics, we have")
    assert "V2" in out and " in " in out
    assert "SPIROMICS" in out
    # leading word eaten: "Dr-Taylor" -> "Taylor"
    assert c("Dr-Taylor Morgan said").startswith("Dr-")
    # single letter carrying a phrase match: "Virtual, P.D.:" -> "Virtual PBV.D.:"
    assert c("Virtual, P.D.: yes") == "Virtual, P.D.: yes"


def test_phonetically_distant_manglings_are_left_alone():
    """"Noah" is well below the cutoff against "Avery". Fixing
    those needs a phonetic model, and a wrong rewrite beats a wrong spelling."""
    assert glossary.TermCorrector(["Avery"])("Noah said") == "Noah said"


def test_clean_term_rejects_speaker_placeholders():
    # "Speaker 1" is a placeholder label, never a spoken name -- it must never
    # be treated as vocabulary.
    from core import glossary
    assert glossary.clean_term("Speaker 1") is None
    assert glossary.clean_term("Speaker 12") is None
    assert glossary.clean_term("Speaker Pelosi") == "Speaker Pelosi"  # a real name survives
