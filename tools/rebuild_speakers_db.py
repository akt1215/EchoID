"""Regenerate the voiceprint DB from human-confirmed clusters only.

The live DB accumulated through an EMA that fired on auto-filled names nobody
confirmed, so a wrong match dragged a print toward the wrong person and made the
next wrong match easier. Rebuilding from ground truth gives every voiceprint a
known provenance.

Run:  .venv/bin/python -m tools.rebuild_speakers_db --out /tmp/scratch.json

Embedding space follows `config.yaml`. On a non-ECAPA backend the vectors are
re-extracted from the WAVs on the configured target channel, which also settles
the older mismatch where sidecar embeddings came from the ch0+ch1 downmix while
the pipeline had moved to a single channel — the rebuilt prints and future
recordings now come from the same place. The ECAPA path still reads the stored
sidecar vectors and so still carries that downmix caveat.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import yaml

from ai.identity_resolution import _centroid, normalize_name
from core import atomic_write, embedding_mean
from evaluation import clusters, ground_truth, trials

MIN_TURN_SECONDS = 0.5      # fallback when the config does not say


def build(groups, gt, mean, min_turn_seconds=MIN_TURN_SECONDS):
    """{display name: centered unit voiceprint} from confirmed clusters only.

    Turns from every meeting a person appears in are pooled before averaging, so
    one print represents them across sessions rather than favouring the meeting
    they spoke most in.

    Turns are weighted by length and the briefest are dropped, the same way the
    pipeline builds a query centroid. A plain mean over every turn is not merely
    less precise: a sub-second turn embeds so poorly it misses even its own
    speaker, yet counts as much as a thirty-second one, which is how a print
    ends up blended. Unweighted pooling measurably collapses speaker separation.
    """
    pooled, display = {}, {}
    for group, name in trials.labeled(groups, gt):
        key = normalize_name(name)
        vecs, durations = pooled.setdefault(key, ([], []))
        for (start, end), emb in zip(group.turns, group.embeddings):
            vecs.append(emb)
            durations.append(end - start)
        display.setdefault(key, name)

    out = {}
    for key, (vecs, durations) in pooled.items():
        centroid = _centroid(vecs, mean, durations=durations,
                             min_seconds=min_turn_seconds)
        if centroid is not None and np.linalg.norm(centroid) > 0:
            out[display[key]] = centroid.tolist()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--out", default=None,
                        help="destination DB (default <workspace>/speakers.json)")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--config", default="config.yaml",
                        help="read backend and centering from this config")
    args = parser.parse_args(argv)

    out = args.out or os.path.join(args.workspace, "speakers.json")
    gt_path = args.ground_truth or os.path.join(
        args.workspace, "eval", "ground_truth.json")

    gt = ground_truth.load(gt_path)
    groups = clusters.discover(args.workspace)

    config_path = args.config
    if not os.path.exists(config_path) and config_path == "config.yaml":
        config_path = "config.example.yaml"
    with open(config_path) as f:
        bio = yaml.safe_load(f)["biometrics"]
    backend = bio.get("backend", "speechbrain")

    if backend == "speechbrain":
        # Sidecars already hold ECAPA vectors, so no re-extraction is needed.
        source = "sidecar embeddings"
    else:
        # A different model needs its own vectors: the sidecars' 192-dim ECAPA
        # embeddings are not comparable with anything this backend produces.
        from ai.biometrics import BiometricsManager, model_for
        from evaluation import embeddings as emb

        print(f"re-extracting with {backend} (cached between runs)")
        path = emb.cache_path(args.workspace, backend)
        cache = emb.load_cache(path)
        manager = BiometricsManager(
            db_path=os.path.join(args.workspace, "speakers.json"),
            backend=backend, model_name=model_for(bio), target_channel=1)
        matrices = emb.extract(groups, manager, cache)
        emb.save_cache(path, cache)
        groups = emb.regroup(groups, matrices)
        source = f"{backend} re-extracted from ch1"

    if bio.get("center_embeddings", True):
        mean, _ = embedding_mean.load(
            os.path.join(args.workspace, "embedding_mean.json"))
    else:
        mean = None
    db = build(groups, gt, mean,
               min_turn_seconds=bio.get("min_turn_seconds", MIN_TURN_SECONDS))

    # The existing DB is irreplaceable user data: copy it aside before replacing.
    if os.path.exists(out):
        backup = out + ".pre-rebuild.bak"
        shutil.copy2(out, backup)
        print(f"backed up {out} -> {backup}")

    atomic_write.write_text(out, json.dumps(db, indent=2, sort_keys=True))
    dim = len(next(iter(db.values()))) if db else 0
    print(f"wrote {len(db)} voiceprints ({dim}d, from {source}) to {out}")
    for name in sorted(db):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
