import os
import re


def inject_wikilinks(text, entities):
    """Wrap known entities in [[ ]] for Obsidian graph view, using word
    boundaries and skipping anything already linked. Module-level so both the
    writer and the speaker-review core can reuse it when rebuilding a note.
    """
    if not entities:
        return text

    linked_text = text
    for entity in entities:
        if not entity or len(entity) < 3:
            continue

        escaped_entity = re.escape(entity)
        # Match the entity as a whole word, not already wrapped in [[ ]]
        pattern = re.compile(r'(?<!\[\[)\b(' + escaped_entity + r')\b(?!\]\])', re.IGNORECASE)
        linked_text = pattern.sub(r'[[\1]]', linked_text)

    return linked_text


MEETING_PREFIX = "meeting_"


def note_stem(audio_path):
    """The note's filename stem for the recording at `audio_path`.

    Keyed on the audio the pipeline actually consumed, the same way the review
    sidecar's path is, and NOT on the meeting timestamp: two imports can derive
    one timestamp (`derive_timestamp` falls back to mtime, whose one-second
    granularity a `cp -p` bulk copy ties routinely), and a note named for the
    timestamp alone would then be written twice. The second import would
    replace the first's note, and the first's sidecar -- which still names that
    path -- would later rewrite it from the wrong meeting's transcript.

    A leading `meeting_` is dropped so a recording this tool made keeps the
    `<ts>_Meeting.md` name the vault already uses; ingest's `-2`/`-3`
    disambiguation of a colliding import rides through in the same field. Only
    the filename disambiguates -- the note's date still comes from the meeting
    time.
    """
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    if stem.startswith(MEETING_PREFIX):
        stem = stem[len(MEETING_PREFIX):]
    return stem


def _fmt_entry(entry, checkbox=False):
    """Render a summary/action entry as a bullet, appending its source timestamp
    when present. Accepts {"text","timestamp"} or a bare string."""
    prefix = "- [ ] " if checkbox else "- "
    if isinstance(entry, dict):
        text = entry.get("text", "")
        ts = entry.get("timestamp")
    else:
        text, ts = str(entry), None
    suffix = f"  ({ts})" if ts else ""
    return f"{prefix}{text}{suffix}"


def render_note(date_iso, transcript_text, llm_data, unique_speakers):
    """Build the full Obsidian note markdown. Shared by ObsidianWriter (pipeline)
    and SpeakerReview.commit (post-naming re-render), so a named summary and the
    rebuilt transcript render identically."""
    entities = llm_data.get('entities', [])
    all_entities = list(set(entities + unique_speakers))

    exec_summary = inject_wikilinks(
        "\n".join(_fmt_entry(e) for e in llm_data.get('executive_summary', [])), all_entities)
    action_items = inject_wikilinks(
        "\n".join(_fmt_entry(e, checkbox=True) for e in llm_data.get('action_items', [])), all_entities)
    linked_transcript = inject_wikilinks(transcript_text, all_entities)
    participants_yaml = "[" + ", ".join(f'"[[{s}]]"' for s in unique_speakers) + "]"

    return f"""---
date: {date_iso}
type: meeting
participants: {participants_yaml}
tags: [zoom, auto-generated]
---

# Meeting Summary
{exec_summary}

## Action Items
{action_items}

## Transcript
{linked_transcript}
"""


def timestamp_to_iso(timestamp):
    """meeting_ timestamp 'YYYY-MM-DD_HH-MM-SS' -> ISO 'YYYY-MM-DDTHH:MM:SS'."""
    parts = timestamp.replace("_", "T").split("T")
    if len(parts) == 2:
        return f"{parts[0]}T{parts[1].replace('-', ':')}"
    return timestamp


class ObsidianWriter:
    def __init__(self, output_dir="Obsidian_Vault/Meetings"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _inject_wikilinks(self, text, entities):
        return inject_wikilinks(text, entities)

    @staticmethod
    def _fmt_entry(entry, checkbox=False):
        return _fmt_entry(entry, checkbox)

    def write_markdown(self, stem, timestamp, transcript_text, llm_data, unique_speakers):
        """Generate the Obsidian Markdown file.

        `stem` names the file (see `note_stem`); `timestamp`
        (YYYY-MM-DD_HH-MM-SS) is the meeting time the note displays. They are
        separate because a disambiguated import needs a distinct filename while
        still being dated when the meeting happened.
        """
        filepath = os.path.join(self.output_dir, f"{stem}_Meeting.md")
        content = render_note(timestamp_to_iso(timestamp), transcript_text, llm_data, unique_speakers)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Obsidian Note successfully created at: {filepath}")
        return filepath
