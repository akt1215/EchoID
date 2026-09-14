"""Read config.yaml and persist single-value changes without losing comments.

The TUI's Models area writes the user's global model choices back here so they
stick across sessions; per-run overrides are applied to the in-memory config
only. A line-based rewrite keeps every comment and the file's ordering intact
(a full YAML round-trip would strip the documented comments).
"""

import os
import re
import shutil

import yaml

_SECTION_RE = re.compile(r"^(\w[\w-]*):\s*(?:#.*)?$")

# YAML 1.1 parses these bare tokens as bool/null, so a written value like `off`
# would reload as False. Quote them (and empty strings) to keep them strings.
_YAML_SPECIAL = {"true", "false", "yes", "no", "on", "off", "null", "none", "~"}


def _fmt_value(value):
    if isinstance(value, str) and (value.strip() == "" or value.lower() in _YAML_SPECIAL):
        return f'"{value}"'
    return str(value)


def load_config(path="config.yaml"):
    if not os.path.exists(path) and path == "config.yaml":
        path = "config.example.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(path, section, key, value):
    """Rewrite `section.key`'s value in place, preserving comments and ordering.
    Only the first `key:` under the matching top-level `section:` is changed."""
    if not os.path.exists(path) and os.path.basename(path) == "config.yaml":
        template = os.path.join(os.path.dirname(os.path.abspath(path)),
                                "config.example.yaml")
        if not os.path.exists(template):
            raise FileNotFoundError(f"configuration template not found: {template}")
        shutil.copyfile(template, path)
    with open(path) as f:
        lines = f.readlines()

    key_re = re.compile(rf"^(\s+)({re.escape(key)}):\s*(.*?)(\s+#.*)?\s*$")
    current, done, out = None, False, []
    for line in lines:
        body = line.rstrip("\n")
        section_hdr = _SECTION_RE.match(body)
        if section_hdr:
            current = section_hdr.group(1)
            out.append(line)
            continue
        if current == section and not done:
            m = key_re.match(body)
            if m:
                indent, k, _old, comment = m.groups()
                out.append(f"{indent}{k}: {_fmt_value(value)}{comment or ''}\n")
                done = True
                continue
        out.append(line)

    with open(path, "w") as f:
        f.writelines(out)


def apply_overrides(cfg, transcription_backend=None, name_reader_backend=None,
                    llm_model=None):
    """Apply per-run model overrides to an in-memory config (None = keep)."""
    if transcription_backend:
        cfg.setdefault("transcription", {})["backend"] = transcription_backend
    if name_reader_backend:
        cfg.setdefault("name_reader", {})["backend"] = name_reader_backend
    if llm_model:
        cfg.setdefault("llm", {})["model"] = llm_model
    return cfg
