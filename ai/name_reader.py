"""Provider-agnostic vision-LLM name reader.

Reads the display name of the actively-speaking participant from a meeting
screenshot. Replaces the brittle tesseract crop in visual_ingestion. The model
backends are thin and only touch the network at runtime; the cleaning/voting
logic is pure and unit-tested. A reading is kept only if it passes the shared
`looks_like_name` gate, so a bad read can never surface as a speaker.

Backends (selected in config, image frames only — never audio/transcript):
  - "gemini": google-genai, reads GEMINI_API_KEY from the environment.
  - "ollama": a local or Ollama-Cloud vision model via the ollama client.
"""

from collections import Counter
from difflib import SequenceMatcher

from ai.identity_resolution import _group_indices, looks_like_name, normalize_name

DEFAULT_PROMPT = (
    "This is a screenshot of a video call. Reply with ONLY the display name of "
    "the participant who is currently speaking — the one highlighted by a "
    "coloured/green border or shown as the enlarged active-speaker tile. Reply "
    "with the name and nothing else. If you cannot tell, reply exactly: NONE"
)


def is_local_name(name, local_names):
    """True if `name` is the local user's own display name.

    The local user is channel 0 ("Me (Local)"), resolved by the channel-energy
    override — never by visual naming — so their name must never be assigned to a
    diarization cluster. A multi-token local name matches fuzzily (ratio >= 0.85)
    so a truncated read like "Alex Morga" is still caught; a single-token name
    (e.g. "Alex") matches only on exact-normalized equality, so a genuine
    participant with a similar name is never over-excluded.
    """
    if not name or not local_names:
        return False
    n = normalize_name(name)
    for ln in local_names:
        l = normalize_name(ln)
        if not l:
            continue
        if n == l:
            return True
        if " " in l and SequenceMatcher(None, n, l).ratio() >= 0.85:
            return True
    return False


def clean_reading(raw, known=(), local_names=()):
    """Normalize a raw model reply to a name, or None. Takes the first line,
    strips wrapping quotes/punctuation, drops NONE/empty, requires the result to
    look like a name, and rejects the local user's own name (which belongs to
    channel 0, never to a visually-named cluster)."""
    s = (raw or "").strip()
    if not s:
        return None
    s = s.splitlines()[0].strip().strip('."\'').strip()
    if not s or s.upper() == "NONE":
        return None
    if not looks_like_name(s, known):
        return None
    return None if is_local_name(s, local_names) else s


def vote_cluster_name(readings, known=(), local_names=(), min_agreement=2):
    """Majority-vote a name over a cluster's frame readings.

    Returns None unless the winning name is backed by at least `min_agreement`
    reads. When fewer than `min_agreement` reads were even collected, a lone read
    is still accepted (nothing to corroborate against); the guard only fires once
    enough frames were read to expect agreement, turning a disagreeing set into a
    recoverable Unknown instead of a confident wrong guess.
    """
    valid = [c for c in (clean_reading(r, known, local_names) for r in readings) if c]
    if not valid:
        return None
    name, count = Counter(valid).most_common(1)[0]
    if len(valid) >= min_agreement and count < min_agreement:
        return None
    return name


def name_clusters(segments, unmatched_pids, frame_events, reader, known=(),
                  local_names=(), frames_per_cluster=3, window=2.0, min_agreement=2):
    """Read a name for each DB-unmatched cluster from its own frames.

    `frame_events` is [(elapsed_seconds, frame_path)]. For each unmatched cluster
    we take frames whose time falls within any of the cluster's turns (± window),
    read up to `frames_per_cluster` of them (spread out) with `reader`, and
    majority-vote (requiring `min_agreement`, and excluding `local_names`).
    Returns {pyannote id: name} only for clusters that got a corroborated read —
    so no VLM call is made for already-recognized speakers.
    """
    groups = _group_indices(segments)
    events = sorted(frame_events)
    out = {}
    for pid in unmatched_pids:
        cand, seen = [], set()
        for i in groups.get(pid, []):
            lo, hi = segments[i]["start"] - window, segments[i]["end"] + window
            for t, path in events:
                if lo <= t <= hi and path not in seen:
                    seen.add(path)
                    cand.append(path)
        if not cand:
            continue
        step = max(1, len(cand) // frames_per_cluster)
        picks = cand[::step][:frames_per_cluster]
        readings = [reader.read(p, known, local_names) for p in picks]
        name = vote_cluster_name(readings, known, local_names, min_agreement)
        if name:
            out[pid] = name
    return out


class NameReader:
    def __init__(self, backend="gemini", model=None, prompt=DEFAULT_PROMPT,
                 backend_fn=None):
        self.prompt = prompt
        if backend_fn is not None:            # dependency injection for tests
            self._call = backend_fn
        elif backend == "gemini":
            self._call = self._gemini_backend(model or "gemini-3.1-flash-lite")
        elif backend == "ollama":
            self._call = self._ollama_backend(model or "qwen2.5-vl:7b")
        else:
            raise ValueError(f"unknown name_reader backend: {backend!r}")

    def read(self, image_path, known=(), local_names=()):
        """Return the cleaned active-speaker name for a frame, or None (also on
        any backend/network error — the caller degrades to voiceprint-only).
        `local_names` are excluded so the local user's own tile is never read as
        a speaker."""
        try:
            raw = self._call(image_path)
        except Exception:
            return None
        return clean_reading(raw, known, local_names)

    def _gemini_backend(self, model):
        from google import genai
        from google.genai import types
        client = genai.Client()  # reads GEMINI_API_KEY from the environment
        prompt = self.prompt

        def call(image_path):
            with open(image_path, "rb") as f:
                data = f.read()
            resp = client.models.generate_content(
                model=model,
                contents=[prompt,
                          types.Part.from_bytes(data=data, mime_type="image/jpeg")],
            )
            return (resp.text or "").strip()
        return call

    def _ollama_backend(self, model):
        import ollama
        prompt = self.prompt

        def call(image_path):
            resp = ollama.chat(model=model, messages=[
                {"role": "user", "content": prompt, "images": [image_path]}])
            return (resp["message"]["content"] or "").strip()
        return call
