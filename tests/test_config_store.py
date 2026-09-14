from core.config_store import apply_overrides, load_config, save_config


def test_save_config_changes_target_section_and_keeps_comments(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "transcription:\n"
        "  backend: local            # local | groq\n"
        "  model: large-v3-turbo\n"
        "name_reader:\n"
        "  backend: gemini           # gemini | ollama | off\n"
    )
    save_config(str(cfg), "transcription", "backend", "groq")

    d = load_config(str(cfg))
    assert d["transcription"]["backend"] == "groq"
    assert d["transcription"]["model"] == "large-v3-turbo"   # sibling untouched
    assert d["name_reader"]["backend"] == "gemini"           # other section untouched
    assert "# local | groq" in cfg.read_text()               # comment preserved


def test_save_config_targets_the_right_section_for_a_shared_key(tmp_path):
    # `backend` exists under both sections; editing name_reader must not touch
    # transcription.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "transcription:\n  backend: local\n"
        "name_reader:\n  backend: gemini\n"
    )
    save_config(str(cfg), "name_reader", "backend", "ollama")

    d = load_config(str(cfg))
    assert d["name_reader"]["backend"] == "ollama"
    assert d["transcription"]["backend"] == "local"


def test_save_config_quotes_yaml_keyword_values(tmp_path):
    # A bare `off` reloads as boolean False in YAML; save_config must quote it so
    # it round-trips as the string "off".
    cfg = tmp_path / "config.yaml"
    cfg.write_text("name_reader:\n  backend: gemini           # gemini | ollama | off\n")
    save_config(str(cfg), "name_reader", "backend", "off")

    d = load_config(str(cfg))
    assert d["name_reader"]["backend"] == "off"
    assert d["name_reader"]["backend"] is not False
    assert "# gemini | ollama | off" in cfg.read_text()   # comment still preserved


def test_apply_overrides_only_sets_provided():
    cfg = {"transcription": {"backend": "local"},
           "name_reader": {"backend": "gemini"},
           "llm": {"model": "minimax-m3:cloud"}}
    apply_overrides(cfg, transcription_backend="groq", llm_model="gpt-oss:20b")
    assert cfg["transcription"]["backend"] == "groq"
    assert cfg["name_reader"]["backend"] == "gemini"   # None → unchanged
    assert cfg["llm"]["model"] == "gpt-oss:20b"
