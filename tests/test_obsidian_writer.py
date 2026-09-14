from export.obsidian_writer import ObsidianWriter, render_note


def test_render_note():
    llm = {
        "executive_summary": [{"text": "point one", "timestamp": "01:00"}],
        "action_items": [{"text": "Alice does X", "timestamp": "02:00"}],
        "entities": ["Alice"],
    }
    md = render_note("2026-07-08T09:00:00", "**Alice** (01:00):\nhi ", llm, ["Alice", "Me (Local)"])
    assert "date: 2026-07-08T09:00:00" in md
    assert "# Meeting Summary" in md and "- point one  (01:00)" in md
    assert "## Action Items" in md and "[ ] [[Alice]] does X  (02:00)" in md
    assert "## Transcript" in md
    assert 'participants:' in md and '"[[Alice]]"' in md


def test_fmt_entry():
    f = ObsidianWriter._fmt_entry
    assert f({"text": "point", "timestamp": "12:34"}) == "- point  (12:34)"
    assert f({"text": "point", "timestamp": None}) == "- point"
    assert f({"text": "do x", "timestamp": "28:10"}, checkbox=True) == "- [ ] do x  (28:10)"
    assert f({"text": "do y", "timestamp": None}, checkbox=True) == "- [ ] do y"
    assert f("bare string") == "- bare string"  # defensive: tolerate a plain str
