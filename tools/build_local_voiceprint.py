"""Build the local user's voiceprint from ch0 of existing recordings.

The local user has never been embedded. `diarization.target_channel: 1` means
every vector in the review sidecars came from the remote channel, so there is
nothing stored to reuse — the print has to be rebuilt by re-reading ch0 of the
original WAVs over the spans where the local user was speaking.

Those spans come from transcript turns labeled "Me (Local)", but the label is
not trusted: it is produced by a 1.5x channel-energy ratio that is right for
tagging a transcript turn and far too permissive for enrollment, and a mic-only
or dead-ch1 recording produces a transcript that is 100% "Me (Local)" while
proving nothing about who spoke. Every span is re-verified against the WAV,
and only WAVs that can prove they came from this tool (`ingest.is_native`) are
read at all -- a foreign stereo file has no mic channel for ch0 to mean.

Run:  .venv/bin/python -m tools.build_local_voiceprint --out /tmp/print.json
      .venv/bin/python -m tools.build_local_voiceprint --commit
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf
import yaml

from ai.identity_resolution import _centroid, _cos
from core import atomic_write, embedding_mean, ingest

LOCAL_LABEL = "Me (Local)"
DEFAULT_DOMINANCE = 8.0     # energy ratio; the pipeline's transcript label uses 1.5
DEFAULT_MIN_SPAN = 1.0      # seconds
QUIET = 1e-8                # below this a channel is dead, not merely quiet


def candidate_spans(sidecar, min_span):
    """(start, end) for every transcript turn labeled as the local user and at
    least `min_span` long. Sub-word fragments embed poorly and would only dilute
    the centroid."""
    out = []
    for t in sidecar.get("transcript", []):
        if t.get("label") != LOCAL_LABEL:
            continue
        start, end = float(t["start"]), float(t["end"])
        if end - start >= min_span:
            out.append((start, end))
    return out


def verify_span(data, sr, start, end, dominance):
    """True if this span is genuinely the local user alone on the mic channel.

    Requires real stereo, a non-empty window, a remote channel that is quiet but
    alive, and mic energy `dominance` times the remote channel's. A dead remote
    channel is rejected rather than accepted: it passes any ratio trivially and
    is the signature of a mic-only file, whose "Me (Local)" labels are an
    artifact of the override firing on everything.
    """
    if data.ndim != 2 or data.shape[1] < 2:
        return False
    s, e = int(start * sr), int(end * sr)
    window = data[s:e]
    if window.shape[0] == 0:
        return False
    ch0 = float(np.mean(np.asarray(window[:, 0], dtype=float) ** 2))
    ch1 = float(np.mean(np.asarray(window[:, 1], dtype=float) ** 2))
    if ch1 <= QUIET:
        return False
    return ch0 > ch1 * dominance


def collect(workspace, manager, dominance, min_span):
    """[(meeting, vectors, durations)] of verified local-user turns per meeting."""
    out = []
    for path in sorted(glob.glob(os.path.join(workspace, "*.segments.json"))):
        with open(path) as f:
            sidecar = json.load(f)
        wav = sidecar.get("wav", "")
        meeting = os.path.basename(path).replace(".segments.json", "")
        spans = candidate_spans(sidecar, min_span)
        if not wav or not os.path.exists(wav) or not spans:
            print(f"  {meeting}: skipped (no wav or no local turns)")
            continue
        # Channel count is not provenance. A foreign stereo recording is
        # 2-channel too, and one reprocessed before ingest existed carries
        # "Me (Local)" wherever the mix was panned left -- remote voices that
        # the 8x check cannot tell from the local user, because on ch0 they
        # genuinely dominate. Only a file that can prove it came from this tool
        # has a mic channel at all, so ask that first.
        if not ingest.is_native(wav):
            print(f"  {meeting}: skipped (unproven provenance; not a recording "
                  f"this tool made)")
            continue
        if sf.info(wav).channels != 2:
            print(f"  {meeting}: skipped (not a dual-channel recording)")
            continue
        data, sr = sf.read(wav)
        kept = [(s, e) for s, e in spans if verify_span(data, sr, s, e, dominance)]
        if not kept:
            print(f"  {meeting}: 0/{len(spans)} spans verified — skipped")
            continue
        signal, fs = manager._load_audio(wav)
        vecs, durs = [], []
        for s, e in kept:
            try:
                vecs.append(np.asarray(
                    manager.extract_embedding(wav, s, e, signal=signal, fs=fs),
                    dtype=float))
                durs.append(e - s)
            except Exception as err:
                print(f"  {meeting}: embedding failed at {s:.1f}s ({err})")
        if vecs:
            total = sum(durs)
            print(f"  {meeting}: {len(kept)}/{len(spans)} spans verified, "
                  f"{total:.0f}s embedded")
            out.append((meeting, vecs, durs))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None,
                        help="write the print here (default: print only)")
    parser.add_argument("--commit", action="store_true",
                        help="write into <workspace>/speakers.json (backs it up first)")
    parser.add_argument("--dominance", type=float, default=DEFAULT_DOMINANCE,
                        help="required ch0/ch1 energy ratio per span")
    parser.add_argument("--min-span", type=float, default=DEFAULT_MIN_SPAN,
                        help="shortest turn to embed, in seconds")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bio = cfg["biometrics"]
    local_names = (cfg.get("identity") or {}).get("local_names") or []
    if not local_names:
        print("Error: identity.local_names is empty; nothing to enroll under.")
        return 1
    key = local_names[0]

    from ai.biometrics import BiometricsManager, model_for
    manager = BiometricsManager(
        db_path=os.path.join(args.workspace, "speakers.json"),
        backend=bio["backend"], model_name=model_for(bio),
        device=bio.get("device", "auto"),
        target_channel=0)          # the local user lives on ch0, not ch1

    print(f"scanning {args.workspace} for verified local-user speech "
          f"(dominance {args.dominance}x, min span {args.min_span}s)")
    per_meeting = collect(args.workspace, manager, args.dominance, args.min_span)
    if not per_meeting:
        print("No verified local speech found; nothing written.")
        return 1

    if bio.get("center_embeddings", True):
        mean, _ = embedding_mean.load(
            os.path.join(args.workspace, "embedding_mean.json"))
    else:
        mean = None

    vecs = [v for _, vs, _ in per_meeting for v in vs]
    durs = [d for _, _, ds in per_meeting for d in ds]
    centroid = _centroid(vecs, mean, durations=durs,
                         min_seconds=bio.get("min_turn_seconds", 0.5))
    if centroid is None or not np.linalg.norm(centroid):
        print("Centroid is empty; nothing written.")
        return 1

    # A print pooled from meetings that disagree with each other is a blend, not
    # an identity. Show the per-meeting agreement before it can be committed.
    print(f"\npooled print: {len(vecs)} turns, {sum(durs):.0f}s, {centroid.size}d")
    print("per-meeting agreement with the pooled print:")
    for meeting, vs, ds in per_meeting:
        c = _centroid(vs, mean, durations=ds,
                      min_seconds=bio.get("min_turn_seconds", 0.5))
        if c is not None:
            print(f"  {_cos(centroid, c):+.3f}  {meeting}")

    payload = {key: centroid.tolist()}
    if args.out:
        atomic_write.write_text(args.out, json.dumps(payload, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")

    if args.commit:
        db_path = os.path.join(args.workspace, "speakers.json")
        if os.path.exists(db_path):
            backup = db_path + ".pre-local-print.bak"
            shutil.copy2(db_path, backup)
            print(f"backed up {db_path} -> {backup}")
            with open(db_path) as f:
                db = json.load(f)
        else:
            db = {}
        db[key] = centroid.tolist()
        atomic_write.write_text(db_path, json.dumps(db, indent=2, sort_keys=True))
        print(f"enrolled {key!r} into {db_path} ({len(db)} prints)")
    elif not args.out:
        print("\n(dry run — pass --out <path> or --commit to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
