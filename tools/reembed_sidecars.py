"""Re-extract every sidecar's turn embeddings with the configured backend.

Sidecars written before a backend change hold that model's vectors — 192-dim
ECAPA where the pipeline now produces 256-dim WeSpeaker — so the review screen
and the speaker directory end up comparing an old-space embedding against a
new-space voiceprint whenever an old meeting is revisited.

Only the `embedding` field is stale. Labels, clusters, sources and the
transcript carry the user's reviewed work and are left exactly as they are, so
this is not a reprocess: no diarization, no transcription, no summary, and
nothing in the Obsidian vault is touched.

Run:  .venv/bin/python -m tools.reembed_sidecars --dry-run
      .venv/bin/python -m tools.reembed_sidecars
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import yaml

from core import atomic_write


def reembed(sidecar_path, manager, dry_run=False):
    """Replace every turn's embedding in one sidecar.

    Returns a stats dict; `skipped` is set and nothing is written when the
    sidecar has no turns or its recording is gone.
    """
    with open(sidecar_path) as f:
        data = json.load(f)

    turns = data.get("diarization", [])
    stats = {"turns": len(turns), "old_dim": 0, "new_dim": 0, "empty": 0,
             "skipped": None}
    if not turns:
        stats["skipped"] = "no turns"
        return stats

    wav = data.get("wav", "")
    if not wav or not os.path.exists(wav):
        stats["skipped"] = "missing wav"
        return stats

    stats["old_dim"] = max((len(t.get("embedding") or []) for t in turns),
                           default=0)
    segments = [{"start": float(t["start"]), "end": float(t["end"])}
                for t in turns]
    vectors = manager.embed_all(wav, segments)

    for turn, vec in zip(turns, vectors):
        arr = np.asarray(vec, dtype=float)
        # An extraction failure leaves the turn without an embedding rather than
        # writing a zero vector, which would read as a real point in the space.
        turn["embedding"] = arr.tolist() if arr.size else []
        if not arr.size:
            stats["empty"] += 1
    stats["new_dim"] = max((len(t["embedding"]) for t in turns), default=0)

    if dry_run:
        return stats

    # The first backup holds the only copy of the pre-switch vectors, so a
    # second run must not overwrite it with already-re-embedded ones.
    backup = sidecar_path + ".pre-reembed.bak"
    if not os.path.exists(backup):
        shutil.copy2(sidecar_path, backup)
    atomic_write.write_text(sidecar_path, json.dumps(data))
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    parser.add_argument("--only", default=None,
                        help="only sidecars whose name contains this text")
    args = parser.parse_args(argv)

    from ai.biometrics import BiometricsManager, model_for

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bio = cfg["biometrics"]

    paths = sorted(glob.glob(os.path.join(args.workspace, "*.segments.json")))
    if args.only:
        paths = [p for p in paths if args.only.lower() in os.path.basename(p).lower()]
    if not paths:
        print("no sidecars found")
        return 1

    print(f"backend {bio['backend']} ({model_for(bio)})")
    print(f"{len(paths)} sidecars"
          + ("  [DRY RUN — nothing will be written]" if args.dry_run else ""))

    manager = BiometricsManager(
        db_path=os.path.join(args.workspace, "speakers.json"),
        backend=bio["backend"], model_name=model_for(bio),
        device=bio.get("device", "auto"),
        target_channel=cfg["diarization"]["target_channel"])

    total_turns = total_empty = 0
    for path in paths:
        name = os.path.basename(path)[:-len(".segments.json")]
        started = time.perf_counter()
        stats = reembed(path, manager, dry_run=args.dry_run)
        took = time.perf_counter() - started
        if stats["skipped"]:
            print(f"  {name[8:]:<22} skipped ({stats['skipped']})")
            continue
        total_turns += stats["turns"]
        total_empty += stats["empty"]
        empty = f", {stats['empty']} failed" if stats["empty"] else ""
        print(f"  {name[8:]:<22} {stats['turns']:>5} turns  "
              f"{stats['old_dim']}d -> {stats['new_dim']}d{empty}  [{took:.0f}s]")

    print(f"\n{total_turns} turns re-embedded"
          + (f", {total_empty} extractions failed" if total_empty else ""))
    if not args.dry_run:
        print("originals kept alongside as *.segments.json.pre-reembed.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
