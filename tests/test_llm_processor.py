from export.llm_processor import LLMProcessor as L


def test_clean_ts():
    assert L._clean_ts("12:34") == "12:34"
    assert L._clean_ts("1:02") == "1:02"
    assert L._clean_ts("bad") is None
    assert L._clean_ts(None) is None
    assert L._clean_ts("") is None
    assert L._clean_ts("12") is None          # no colon
    assert L._clean_ts("12:34:56") is None     # too many parts
    assert L._clean_ts("aa:bb") is None        # non-digit


def test_normalize_entry():
    assert L._normalize_entry({"point": "hi", "timestamp": "12:34"}, "point") == {"text": "hi", "timestamp": "12:34"}
    assert L._normalize_entry({"item": "do x", "timestamp": "bad"}, "item") == {"text": "do x", "timestamp": None}
    assert L._normalize_entry("plain string", "point") == {"text": "plain string", "timestamp": None}
    assert L._normalize_entry({"text": "y", "timestamp": "3:00"}, "point") == {"text": "y", "timestamp": "3:00"}
    assert L._normalize_entry({"timestamp": "1:00"}, "point") == {"text": "", "timestamp": "1:00"}
