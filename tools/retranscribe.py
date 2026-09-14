"""Re-transcribe recorded meetings in place, keeping their reviewed speakers.

The shipped transcripts carry Whisper's own vocabulary prompt continued as
speech: on low-information audio the model kept listing glossary terms instead
of transcribing, destroying real speech in those windows. That prompt is gone
now — replaced by a style sentence carrying no vocabulary, with canonical
spellings applied after recognition — but existing sidecars predate the fix.

A full `--from-file` reprocess would fix them and cost too much: it re-runs
diarization, which reassigns cluster ids, and the ground-truth labels are keyed
by cluster id — so every hand-verified verdict would be pruned as stale, taking
the voiceprint DB's provenance with it.

So this re-runs ONLY transcription. Diarization turns, cluster ids, embeddings,
sources and the resolved speaker names are read from the sidecar and reused
exactly, which also means the new transcript is attributed by the *reviewed*
speakers rather than by a fresh guess. Ground truth, `speakers.json` and the
learned glossary are never touched.

Run:  .venv/bin/python -m tools.retranscribe --dry-run
      .venv/bin/python -m tools.retranscribe --limit 2
      .venv/bin/python -m tools.retranscribe --only 2026-07-17
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import yaml

from core import atomic_write, channel_health, glossary


def _backup_once(path, suffix=".pre-retranscribe.bak"):
    """Copy `path` aside, but never over an existing backup.

    Re-running after a filter change is expected, and a plain copy would
    overwrite the backup with the already-rewritten file — destroying the only
    copy of the original transcript on the second run.
    """
    dest = path + suffix
    if not os.path.exists(dest):
        shutil.copy2(path, dest)
    return dest


def speaker_segments(diarization):
    """`[{start, end, speaker}]` for the transcriber, from the sidecar's turns.

    Uses the resolved `label`, so the rebuilt transcript inherits the names the
    user already confirmed instead of re-deriving them.
    """
    return [{"start": float(d["start"]), "end": float(d["end"]),
             "speaker": d["label"]} for d in diarization]


def replace_transcript_section(content, transcript_md, names):
    """Swap a note's `## Transcript` body, leaving summary and frontmatter alone.

    The summary above it was generated from the old transcript. Regenerating it
    means another LLM pass, so that is opt-in (`--summarize`) rather than a
    silent side effect of cleaning the transcript.
    """
    from export.obsidian_writer import inject_wikilinks

    head = content.split("## Transcript")[0]
    return head + "## Transcript\n" + inject_wikilinks(transcript_md, names) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--only", default=None,
                        help="only meetings whose name contains this text")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many meetings (try a few first)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, transcribe nothing")
    parser.add_argument("--summarize", action="store_true",
                        help="also re-run the LLM summary from the clean transcript")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t_cfg = cfg["transcription"]

    paths = sorted(glob.glob(os.path.join(args.workspace, "*.segments.json")))
    if args.only:
        paths = [p for p in paths if args.only.lower() in os.path.basename(p).lower()]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("No matching sidecars.")
        return 1

    # Canonical spellings for the post-recognition correction pass, matching what
    # main.py assembles. Per-meeting speaker names are added inside the loop.
    base_terms = (list(t_cfg.get("glossary") or [])
                  + glossary.active_terms(
                      os.path.join(args.workspace, "glossary_learned.json")))
    print(f"backend: {t_cfg.get('backend')}   {len(paths)} meeting(s)\n")

    transcriber = None
    grand = 0.0
    for path in paths:
        meeting = os.path.basename(path)[: -len(".segments.json")]
        with open(path) as f:
            data = json.load(f)
        wav, dia = data.get("wav", ""), data.get("diarization", [])
        if not wav or not os.path.exists(wav):
            print(f"{meeting}: WAV missing — skipped")
            continue
        # `dia` may legitimately be empty: a partial recording (system audio
        # never captured) has no turns to reuse. Its transcript is still worth
        # rebuilding — with nothing to assign, every segment falls back to the
        # same label it already carries, so nothing is reattributed.
        before = data.get("transcript", [])
        labels = sorted({d["label"] for d in dia})
        if args.dry_run:
            print(f"{meeting}: {len(dia)} turns, {len(before)} transcript segs, "
                  f"speakers={labels}")
            continue

        if transcriber is None:
            from ai.transcription import Transcriber
            transcriber = Transcriber(
                backend=t_cfg.get("backend", "local"), model_name=t_cfg["model"],
                device=t_cfg["device"],
                compute_type=t_cfg.get("compute_type", "default"),
                beam_size=t_cfg.get("beam_size", 5),
                style_prompt=t_cfg.get("style_prompt"),
                groq_model=t_cfg.get("groq_model", "whisper-large-v3-turbo"),
                chunk_seconds=t_cfg.get("chunk_seconds", 600))
        # This meeting's reviewed speaker names belong in its own corrector, so
        # rebuild it per meeting rather than paying for another model load.
        transcriber.corrector = glossary.TermCorrector(base_terms + labels)

        # Same fallback the live pipeline uses: a railed remote channel would
        # otherwise bury the mic in the downmix and yield pure hallucination.
        mixed = data.get("mixed", False)
        remote_dead = not mixed and channel_health.is_unusable(wav, 1)
        t0 = time.perf_counter()
        transcript_md, segments = transcriber.transcribe(
            wav, speaker_segments(dia), mixed=mixed,
            channel=(0 if remote_dead else None))
        elapsed = time.perf_counter() - t0
        grand += elapsed

        # The sidecar is the user's reviewed work: back it up before rewriting,
        # and touch only the transcript.
        _backup_once(path)
        data["transcript"] = segments
        atomic_write.write_text(path, json.dumps(data, indent=2))

        note = data.get("note", "")
        wrote_note = False
        if note and os.path.exists(note):
            _backup_once(note)
            with open(note) as f:
                content = f.read()
            if args.summarize:
                from export.llm_processor import LLMProcessor
                from export.obsidian_writer import render_note, timestamp_to_iso
                llm = LLMProcessor(model_name=cfg["llm"]["model"],
                                   timeout=cfg["llm"].get("timeout", 120))
                content = render_note(timestamp_to_iso(meeting), transcript_md,
                                      llm.generate_summary(transcript_md),
                                      sorted({s["label"] for s in segments}))
            else:
                content = replace_transcript_section(
                    content, transcript_md, sorted({s["label"] for s in segments}))
            atomic_write.write_text(note, content)
            wrote_note = True

        dur = max((float(d["end"]) for d in dia), default=0.0)
        print(f"{meeting}: {len(before)} -> {len(segments)} segs, "
              f"{elapsed:5.1f}s for {dur/60:.0f} min audio "
              f"({dur/elapsed if elapsed else 0:.0f}x realtime)"
              f"{'' if wrote_note else '  [note not found]'}")

    if grand:
        print(f"\ntotal transcription time: {grand:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
