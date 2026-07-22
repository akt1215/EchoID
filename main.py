import os
import json
import argparse
import datetime
import time
import yaml
import sounddevice as sd
import soundfile as sf

from core.audio_recorder import AudioRecorder, CaptureStartupError
from core.visual_ingestion import VisualIngestion
from core.denoiser import Denoiser
from core import channel_health, embedding_mean, glossary, ingest, review_sidecar
from core.config_store import apply_overrides
from core.frames_store import frames_dir_for, read_frames_manifest
from ai.diarization import Diarizer
from ai.biometrics import BiometricsManager, model_for
from ai.identity_resolution import cluster_db_names, merge_clusters, resolve_clusters
from ai.name_reader import NameReader, is_local_name, name_clusters
from ai.transcription import Transcriber
from ai.speaker_review import SpeakerReview
from export.obsidian_writer import ObsidianWriter, note_stem


def load_config(path="config.yaml"):
    if not os.path.exists(path) and path == "config.yaml":
        path = "config.example.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def _auto_detect_devices():
    devices = sd.query_devices()
    bh = mic = None
    for i, d in enumerate(devices):
        name = d['name'].lower()
        if 'blackhole' in name and d['max_input_channels'] > 0:
            bh = i
        elif d['max_input_channels'] > 0 and mic is None:
            mic = i
    return mic, bh


def _review_speakers_cli(sidecar_path, db_path, llm_model=None, llm_timeout=120,
                          placeholder_min_duration=60.0, local_names=None):
    """After the pipeline completes, review speakers over the sidecar: play a
    clip, rename, or reassign to an existing person, then enroll on commit.
    commit also summarizes the named transcript (name-first summary), so this is
    where the note's summary is produced. Merge/split are TUI-only.
    """
    review = SpeakerReview(sidecar_path, db_path, llm_model=llm_model, llm_timeout=llm_timeout,
                           placeholder_min_duration=placeholder_min_duration,
                           local_names=local_names)
    speakers = review.list_speakers()

    if speakers:
        print("\n--- Review Speakers ---")
        known = list(review.db.keys())
        if known:
            print(f"Known voiceprints (type one to reassign): {', '.join(known)}")
        print("Per speaker: type a name, 'p' to hear a clip, or Enter to keep.\n")
        for s in speakers:
            gid = s["id"]
            while True:
                resp = input(
                    f"  {s['name']} ({s['total_dur']:.0f}s, {s['n_segments']} segs, "
                    f"{s['source']}) \u2192 name / 'p' / Enter: "
                ).strip()
                if resp.lower() == "p":
                    try:
                        review.play_clip(gid)
                    except Exception as e:
                        print(f"    (playback failed: {e})")
                    continue
                if resp:
                    review.rename(gid, resp)
                break

    review.stop_playback()  # nothing should keep playing during summarization

    # Always commit: this generates the summary from the (named) transcript and
    # writes the final note, even for a mic-only meeting with no diarized speakers.
    print("Generating summary and finalizing note...")
    review.commit()
    print("Done.\n")


def _ensure_ollama_running():
    import socket, subprocess, time
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("localhost", 11434))
        s.close()
        return
    except ConnectionRefusedError:
        pass
    finally:
        s.close()
    print("Ollama not running — starting ollama serve...")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(0.5)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("localhost", 11434))
            s.close()
            print("Ollama is ready.")
            return
        except ConnectionRefusedError:
            pass
        finally:
            s.close()
    print("Warning: Ollama did not start in time.")


def _extract_timestamp_from_path(path):
    """Try to parse timestamp from a meeting_YYYY-MM-DD_HH-MM-SS.wav filename."""
    basename = os.path.splitext(os.path.basename(path))[0]
    if basename.startswith("meeting_"):
        ts = basename[len("meeting_"):]
        try:
            datetime.datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
            return ts
        except ValueError:
            pass
    # Fallback: use file modification time
    mtime = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H-%M-%S")


def _read_cluster_names(cfg, segments, unmatched, frame_events, mixed=False):
    """VLM-name the DB-unmatched clusters from their frames. Returns
    {pyannote id: name}, or {} when reading is off/unconfigured/unavailable so
    the pipeline degrades cleanly to voiceprint-only identity.

    In a dual-channel recording the local user is channel 0, resolved by the
    energy override, so their name is withheld from the reader — otherwise their
    self-view tile labels a remote speaker. A mixed file has no mic channel and
    no override, so withholding it would instead strand the local user's own
    cluster on Unknown forever. There the block is lifted and the reading is
    mapped back to the canonical label, so transcripts read alike in both modes.
    """
    nr = cfg.get("name_reader", {}) or {}
    backend = nr.get("backend", "gemini")
    if backend in (None, False, "off") or not unmatched or not frame_events:
        return {}
    model = nr.get("gemini_model") if backend == "gemini" else nr.get("ollama_model")
    try:
        reader = NameReader(backend=backend, model=model)
    except Exception as e:
        print(f"[name_reader] unavailable ({e}); using voiceprints only.")
        return {}
    local_names = (cfg.get("identity") or {}).get("local_names", []) or []
    try:
        names = name_clusters(segments, unmatched, frame_events, reader,
                              local_names=([] if mixed else local_names),
                              frames_per_cluster=nr.get("frames_per_cluster", 3),
                              window=cfg["visual"].get("match_window", 2.0),
                              min_agreement=nr.get("min_agreement", 2))
    except Exception as e:
        print(f"[name_reader] read failed ({e}); using voiceprints only.")
        return {}
    if mixed:
        names = {pid: ("Me (Local)" if is_local_name(n, local_names) else n)
                 for pid, n in names.items()}
    if names:
        print(f"[name_reader] named {len(names)} cluster(s) from frames: "
              f"{sorted(names.values())}")
    return names


def _canonicalize_local_label(resolved_segments, local_names):
    """Display the local user's own cluster as "Me (Local)" in a mixed file.

    In a mixed recording the local user is resolved like anyone else, by two
    tiers that answer with their real name: the VLM name-read (canonicalized in
    `_read_cluster_names`) and, once `tools/build_local_voiceprint.py` has
    enrolled them, a voiceprint match. Without this the label flips from
    "Me (Local)" to their real name the moment their print exists, for the same
    person in the same kind of meeting.

    Only the display label is rewritten. The cluster id keeps the DB key, which
    is what `ai/speaker_review.py` enrolls under and what
    `ai/speaker_directory.py` credits across meetings -- both already map
    "Me (Local)" back to `local_names[0]` in mixed mode.

    Dual-channel callers must not use this: there "Me (Local)" comes from the
    channel-energy override, and the local print is excluded from matching
    outright, so a local name reaching a cluster is bleed, not the local user.
    """
    for s in resolved_segments:
        if is_local_name(s["speaker"], local_names):
            s["speaker"] = "Me (Local)"
    return resolved_segments


def main():
    parser = argparse.ArgumentParser(description="Local Obsidian Meeting Notetaker")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--mic", type=int, help="Device ID for Physical Microphone.")
    parser.add_argument("--bh", type=int, help="Device ID for BlackHole 2ch.")
    parser.add_argument("--from-file", type=str, default=None,
                        help="Reprocess an existing WAV file (skips recording).")
    parser.add_argument("--mixed", action="store_true",
                        help="Treat --from-file input as one mixed track with no "
                             "mic channel (downmixes a foreign stereo wav). "
                             "Non-wav inputs are always treated this way.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output workspace directory (overrides config).")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Disable real-time audio monitoring (BlackHole passthrough).")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save captured Zoom window frames to workspace/frames/ for debugging.")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip interactive speaker naming prompt (for TUI mode).")
    parser.add_argument("--no-finalize", action="store_true",
                        help="Skip review AND summarization; the caller (TUI) finalizes via SpeakerReview.commit.")
    parser.add_argument("--transcription-backend", choices=["local", "groq"], default=None,
                        help="Override the transcription backend for this run (else config).")
    parser.add_argument("--name-reader-backend", choices=["gemini", "ollama", "off"], default=None,
                        help="Override the VLM name-reader backend for this run.")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Override the LLM summary model for this run.")
    parser.add_argument("--cleanup-frames", action="store_true",
                        help="Delete <wav>.frames/ after processing (the TUI sets this for "
                             "a fresh recording; a reprocess keeps them).")

    args = parser.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    cfg = load_config()
    # Per-run model overrides (from the TUI's confirm screen or the CLI flags)
    # take precedence over config.yaml without changing the sticky global.
    apply_overrides(cfg, args.transcription_backend, args.name_reader_backend,
                    args.llm_model)
    workspace = args.out or cfg["paths"]["workspace"]
    os.makedirs(workspace, exist_ok=True)
    denoiser = Denoiser(
        method=cfg["denoiser"]["method"],
        enabled=cfg["denoiser"]["enabled"],
    )

    if args.from_file:
        # ── Reprocess mode ──────────────────────────────────────────
        try:
            iplan = ingest.plan_for(args.from_file, workspace,
                                    mixed_flag=args.mixed)
            audio_path = ingest.run(
                iplan,
                frame_interval=(cfg["audio"].get("ingest") or {}).get(
                    "frame_interval", 4.0))
        except ingest.IngestError as e:
            print(f"Error: {e}")
            return

        print(f"\n--- Reprocessing {audio_path} ---")
        timestamp = iplan.timestamp
        start_time = datetime.datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
        # Reuse the frames captured while recording (or sampled from an imported
        # video) for the VLM name reader; [] when there are none, in which case
        # resolution degrades to voiceprint-only.
        frame_events = read_frames_manifest(audio_path)
        if frame_events:
            print(f"Loaded {len(frame_events)} captured frames for VLM naming.")
    else:
        # ── Record mode ─────────────────────────────────────────────
        backend = cfg["audio"].get("capture_backend", "sck")
        if backend == "blackhole":
            if args.mic is None or args.bh is None:
                detected_mic, detected_bh = _auto_detect_devices()
                args.mic = args.mic if args.mic is not None else detected_mic
                args.bh = args.bh if args.bh is not None else detected_bh
                if args.mic is None or args.bh is None:
                    print("Could not auto-detect mic or BlackHole. Use --list to see devices, then specify --mic / --bh manually.")
                    return
                print(f"Auto-detected: --mic {args.mic} --bh {args.bh}")
        else:  # sck: the mic is SCK's system default input (Phase 1), not a
               # sounddevice index — nothing to auto-detect. --mic is not honored on
               # this backend yet; the opened mic is printed at record start.
            if args.mic is not None:
                print("[note] --mic is ignored on the sck backend (Phase 1 uses the "
                      "system default input).")

        print("\n--- Starting Meeting Notetaker ---")
        start_time = datetime.datetime.now()
        timestamp = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        audio_path = os.path.join(workspace, f"meeting_{timestamp}.wav")

        vision = VisualIngestion(
            polling_interval=cfg["visual"]["polling_interval"],
            min_window_size=cfg["visual"].get("zoom_min_window_size", 300),
            frames_dir=frames_dir_for(audio_path),
            frame_interval=cfg["visual"].get("frame_interval", 4.0),
        )
        recorder = AudioRecorder(
            args.mic, args.bh, audio_path,
            samplerate=cfg["audio"]["sample_rate"],
            monitor_enabled=not args.no_monitor,
            capture_backend=backend,
            sck_binary_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "native", "sck_capture"),
        )

        vision.start()
        try:
            recorder.start()
        except CaptureStartupError as e:
            print(f"\n[error] Cannot start recording:\n{e}")
            vision.stop()
            return
        if getattr(recorder, "sck_mic_name", None):
            print(f"Microphone (ch0): {recorder.sck_mic_name}")
        if getattr(recorder, "mic_silent", False):
            print("[warning] No microphone audio detected — see the [sck] lines above for "
                  "whether the mic delivered nothing (Microphone permission) or delivered "
                  "audio that could not be decoded.")
        try:
            input("Recording... Press ENTER to stop meeting.\n")
        except (KeyboardInterrupt, EOFError):
            pass
        recorder.stop()
        vision.stop()

        # Frames were saved to disk during recording; rebuild the manifest from
        # there so the same path works for an in-process run and a reprocess.
        frame_events = read_frames_manifest(audio_path)

    # The layout is derived from the audio, not carried as a flag: this is the
    # only wav writer here (core/audio_recorder.py) and it hardcodes channels=2,
    # silence-padding a stalled producer, so a native recording is never mono.
    # Printed rather than assumed, so a wrong derivation is visible here instead
    # of surfacing as mislabeled speakers three stages later.
    mixed = sf.info(audio_path).channels == 1
    print(f"[ingest] layout: {'mixed (1ch, no mic channel)' if mixed else 'dual (2ch)'}")

    # A failed system capture rails instead of going silent, so the remote
    # channel is full-scale noise. Downmixing then buries the mic under it and
    # the whole transcript is hallucination. Detect it once here: diarization has
    # nothing to find on that channel, and transcription reads the mic alone.
    remote_channel = cfg["diarization"]["target_channel"]
    remote_dead = not mixed and channel_health.is_unusable(audio_path, remote_channel)
    if remote_dead:
        print(f"[audio] channel {remote_channel} (remote) is railed noise, not audio "
              f"— treating this as a mic-only recording.")

    print("\n--- Post-Processing Pipeline ---")
    _pipeline_start = time.perf_counter()
    stage_times = {}

    def _stage(label, start):
        stage_times[label] = time.perf_counter() - start
        print(f"[time] {label}: {stage_times[label]:.1f}s")

    # 1. Diarization
    _t = time.perf_counter()
    if remote_dead:
        # Nothing to diarize: the channel it runs on is noise. Skipping saves
        # minutes of pyannote on a signal that yields no speakers anyway.
        raw_segments = []
    else:
        diarizer = Diarizer(
            model=cfg["diarization"]["model"],
            min_segment_duration=cfg["diarization"].get("min_segment_duration", 0.5),
            device=cfg["diarization"].get("device", "auto"),
        )
        raw_segments = diarizer.diarize(
            audio_path, target_channel=cfg["diarization"]["target_channel"])
    _stage("diarization", _t)

    # 2. Identity resolution. Extract one embedding per turn (a single pass on the
    #    denoised channel), then merge over-clustered pyannote labels and resolve
    #    one identity PER CLUSTER — DB voiceprint match first (pre-filled), then a
    #    validated visual name, else Unknown. Grouping/naming is model-free
    #    (ai.identity_resolution); the DB stays read-only until the user confirms.
    bio_cfg = cfg["biometrics"]
    bio_manager = BiometricsManager(
        db_path=os.path.join(workspace, "speakers.json"),
        backend=bio_cfg["backend"],
        model_name=model_for(bio_cfg),
        device=bio_cfg.get("device", "auto"),
        target_channel=cfg["diarization"]["target_channel"],
    )

    # Denoising the channel before embedding is optional (off = faster; speaker
    # embeddings are noise-robust). Transcription always uses the original audio.
    _t = time.perf_counter()
    if bio_cfg.get("denoise_for_embeddings", False):
        denoised_path = denoiser.denoise_file(audio_path, target_channel=cfg["diarization"]["target_channel"])
    else:
        denoised_path = audio_path
    embeddings = bio_manager.embed_all(denoised_path, raw_segments)
    if denoised_path != audio_path:
        os.remove(denoised_path)
    _stage("embeddings", _t)

    # Fold this meeting's embeddings into the persisted running mean and use it to
    # center every cosine comparison below. Raw ECAPA embeddings share a dominant
    # common-mean direction that otherwise fuses distinct speakers at ~0.85. Keyed
    # on the recording name so reprocessing the same meeting (--from-file) does not
    # re-count its turns and bias the mean toward that meeting's speakers.
    # Centering rescues ECAPA, whose raw embeddings share a dominant common mean.
    # WeSpeaker separates speakers on raw cosine and needs none, and the stored
    # mean belongs to whichever model produced it — so skip both the subtraction
    # and the accumulation rather than mixing spaces across a backend swap.
    if bio_cfg.get("center_embeddings", True):
        meeting_key = os.path.splitext(os.path.basename(audio_path))[0]
        mean = embedding_mean.update(
            os.path.join(workspace, "embedding_mean.json"),
            embeddings, key=meeting_key)
    else:
        mean = None

    raw_segments = merge_clusters(
        raw_segments, embeddings, mean=mean, threshold=bio_cfg["merge_threshold"],
    )
    # Name the clusters the DB did NOT recognize with the vision-LLM, reading a
    # few of each unmatched cluster's own frames (no frames sent for people
    # already recognized by voiceprint).
    _t = time.perf_counter()
    local_names = (cfg.get("identity") or {}).get("local_names", []) or []
    # A dual-channel recording must never match the local user's enrolled print
    # (see _db_match): their voice is on ch0 and never embedded here, so a ch1
    # hit is bleed naming someone else.
    local_db_keys = () if mixed else tuple(
        k for k in bio_manager.speaker_db if is_local_name(k, local_names))
    db_names = cluster_db_names(raw_segments, embeddings, bio_manager.speaker_db,
                                mean=mean, db_threshold=bio_cfg["db_threshold"],
                                db_margin=bio_cfg.get("db_margin", 0.0),
                                exclude=local_db_keys)
    unmatched = [pid for pid, name in db_names.items() if name is None]
    cluster_visual_names = _read_cluster_names(cfg, raw_segments, unmatched,
                                               frame_events, mixed=mixed)
    resolved_segments = resolve_clusters(
        raw_segments, embeddings, bio_manager.speaker_db, cluster_visual_names,
        mean=mean, db_threshold=bio_cfg["db_threshold"], db_names=db_names,
        db_margin=bio_cfg.get("db_margin", 0.0),
        collapse_similarity=bio_cfg.get("collapse_similarity", 1.01),
        min_speech=bio_cfg.get("min_speech", 0.0),
        min_turn_seconds=bio_cfg.get("min_turn_seconds", 0.5),
        exclude=local_db_keys,
    )
    if mixed:
        _canonicalize_local_label(resolved_segments, local_names)
    _stage("identity (db+vlm)", _t)

    unique_speakers = list({
        s['speaker'] for s in resolved_segments
        if not s['speaker'].startswith("Unknown (")
    })
    if "Me (Local)" not in unique_speakers:
        unique_speakers.append("Me (Local)")

    # 3. Transcription with WhisperX
    t_cfg = cfg["transcription"]
    # Canonical spellings for the post-recognition correction pass: the curated
    # glossary, the terms learned from past meetings, and this meeting's own
    # resolved speaker names.
    correction_terms = (list(t_cfg.get("glossary") or [])
                        + glossary.active_terms(
                            os.path.join(workspace, "glossary_learned.json"))
                        + unique_speakers)
    _t = time.perf_counter()
    transcriber = Transcriber(
        backend=t_cfg.get("backend", "local"),
        model_name=t_cfg["model"],
        device=t_cfg["device"],
        compute_type=t_cfg.get("compute_type", "default"),
        beam_size=t_cfg.get("beam_size", 5),
        glossary_terms=correction_terms,
        style_prompt=t_cfg.get("style_prompt"),
        groq_model=t_cfg.get("groq_model", "whisper-large-v3-turbo"),
        chunk_seconds=t_cfg.get("chunk_seconds", 600),
    )
    formatted_transcript, transcript_segments = transcriber.transcribe(
        audio_path, resolved_segments, mixed=mixed,
        channel=(0 if remote_dead else None))
    _stage("transcription", _t)

    # 4. Obsidian export with a placeholder summary. The real summary is
    # generated later from the NAMED transcript at commit (name-first summary),
    # so action-item assignees reflect the confirmed speakers.
    placeholder_summary = {
        "executive_summary": [{"text": "_Summary pending speaker review._", "timestamp": None}],
        "action_items": [],
        "entities": [],
    }
    writer = ObsidianWriter(
        output_dir=os.path.join(workspace, cfg["paths"]["obsidian_vault"]),
    )
    # The note is named after the audio the pipeline consumed (like the sidecar
    # below), not after the meeting timestamp: two imports can derive the same
    # timestamp and would otherwise share one note. The timestamp still supplies
    # the date the note displays.
    note_path = writer.write_markdown(note_stem(audio_path), timestamp,
                                      formatted_transcript, placeholder_summary,
                                      unique_speakers)

    # 6. Review sidecar — per-turn embeddings + transcript segments, consumed by
    # the speaker review step (TUI/CLI) to play clips, merge/split/reassign, and
    # enroll voiceprints only on confirmation. workspace/ is gitignored.
    sidecar = {
        "wav": audio_path,
        "note": note_path,
        "mixed": mixed,
        "diarization": [
            {"idx": i, "start": r["start"], "end": r["end"],
             "label": r["speaker"], "cluster": r["cluster"],
             "source": r.get("source", "unknown"),
             "embedding": (embeddings[i].tolist() if embeddings[i].size else [])}
            for i, r in enumerate(resolved_segments)
        ],
        "transcript": transcript_segments,
    }
    sidecar_path = review_sidecar.path_for(workspace, audio_path)
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)
    # Announced, not merely logged: the TUI shells out to this script and reads
    # the path back from here, because an import's sidecar is named after
    # ingest's product (possibly suffix-bumped) and nothing outside this run
    # can derive it.
    # flush: stdout is block-buffered into the TUI's pipe, and this line has to
    # reach the reader rather than sit behind a later stage's output.
    print(review_sidecar.announce(sidecar_path), flush=True)

    # 7. Finalize: review speakers, then summarize the NAMED transcript at commit.
    #    --no-finalize (TUI): the caller reviews + commits in its own process, so
    #    do nothing here except leave ollama running for it.
    llm_model = cfg["llm"]["model"]
    llm_timeout = cfg["llm"].get("timeout", 120)
    placeholder_min_duration = cfg["biometrics"]["placeholder_min_duration"]
    db_path = os.path.join(workspace, "speakers.json")
    _ensure_ollama_running()  # commit (here or in the TUI) calls the LLM
    if not args.no_finalize:
        if not args.no_prompt:
            _review_speakers_cli(sidecar_path, db_path, llm_model=llm_model, llm_timeout=llm_timeout,
                                 placeholder_min_duration=placeholder_min_duration,
                                 local_names=local_names)
        else:
            # Non-interactive standalone: auto-finalize with current labels so a
            # summary is still produced.
            print("Generating summary...")
            SpeakerReview(sidecar_path, db_path, llm_model=llm_model, llm_timeout=llm_timeout,
                         placeholder_min_duration=placeholder_min_duration,
                         local_names=local_names).commit()

    print(f"[time] pipeline total: {time.perf_counter() - _pipeline_start:.1f}s "
          f"({', '.join(f'{k} {v:.0f}s' for k, v in stage_times.items())})")

    # Fresh recordings drop the transient frames now that clusters are named
    # (a CLI record, or a TUI record that passes --cleanup-frames), unless the
    # user asked to keep them. A reprocess of an existing WAV never deletes them.
    if (not args.from_file or args.cleanup_frames) and not args.save_frames:
        import shutil
        shutil.rmtree(frames_dir_for(audio_path), ignore_errors=True)

    print("\n--- All Done! ---")


if __name__ == "__main__":
    main()
