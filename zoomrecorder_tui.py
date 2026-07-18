#!/usr/bin/env python3
"""EchoID TUI — Terminal UI for meeting recording and processing."""

import os
import sys
import subprocess
import shlex
import threading
import time
import datetime
import re
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sounddevice as sd
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Button, Static, Header, Footer, RichLog, Input, Rule, Label, Digits, Checkbox,
    Select,
)
from textual.suggester import SuggestFromList
from textual.containers import Horizontal, Vertical, Container, ScrollableContainer
from textual.reactive import reactive
from textual import work

import yaml

from ai.speaker_review import SpeakerReview
from ai.speaker_directory import SpeakerDirectory
from core import review_sidecar
from core.frames_store import frames_dir_for
from core.config_store import save_config
from core.clip_player import ClipPlayer


ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_CACHE = None
_TUI_STATE_FILE = os.path.join(ROOT, ".tui_state.yaml")

SUPPORTED_RECORDING_EXTS = (
    ".wav", ".m4a", ".mp3", ".mp4", ".mov", ".aac", ".m4b", ".m4r", ".aif", ".aiff"
)
DEFAULT_TUI_PRESET = "balanced"

RUN_PRESETS = {
    "balanced": {
        "label": "Balanced",
        "run_transcription": "groq",
        "run_name_reader": "gemini",
        "run_llm": "minimax-m3:cloud",
    },
    "fast_audio": {
        "label": "Fast (audio, no visual read)",
        "run_transcription": "groq",
        "run_name_reader": "off",
        "run_llm": "gpt-oss:20b",
    },
    "local_quality": {
        "label": "Local Quality",
        "run_transcription": "local",
        "run_name_reader": "gemini",
        "run_llm": "minimax-m3:cloud",
    },
    "private_local": {
        "label": "Private Local",
        "run_transcription": "local",
        "run_name_reader": "off",
        "run_llm": "gpt-oss:20b",
    },
}

RUN_PRESET_OPTIONS = [
    ("Custom", "custom"),
    (RUN_PRESETS["balanced"]["label"], "balanced"),
    (RUN_PRESETS["fast_audio"]["label"], "fast_audio"),
    (RUN_PRESETS["local_quality"]["label"], "local_quality"),
    (RUN_PRESETS["private_local"]["label"], "private_local"),
]

PIPELINE_STAGE_MARKERS = {
    "layout": re.compile(r"\[ingest\]"),
    "diarization": re.compile(r"\[time\] diarization:", re.IGNORECASE),
    "embeddings": re.compile(r"\[time\] embeddings", re.IGNORECASE),
    "identity": re.compile(r"\[time\] identity", re.IGNORECASE),
    "transcription": re.compile(r"\[time\] transcription", re.IGNORECASE),
    "finalizing": re.compile(r"\[time\] pipeline total", re.IGNORECASE),
}

KNOWN_ERROR_HINTS = [
    (re.compile(r"error:\s*no such file|not found|does not exist", re.IGNORECASE),
     "Input file not found", "Re-check the path and pick from Finder."),
    (re.compile(r"unsupported|unknown format|codec", re.IGNORECASE),
     "Unsupported file format", "Choose a recording in WAV/MP4/M4A/MP3. Re-encode if needed."),
    (re.compile(r"could not read header|invalid data|broken pipe", re.IGNORECASE),
     "Corrupt or partially downloaded file", "Replace with a clean file copy before reprocessing."),
    (re.compile(r"permission denied|denied|could not open file", re.IGNORECASE),
     "Permission blocked", "Grant read access for the file and run again."),
]

BATCH_STATUS_DONE = "done"
BATCH_STATUS_FAILED = "failed"
BATCH_STATUS_CANCELED = "canceled"
RUN_STAGE_TO_LABEL = {
    "layout": "Ingest / Layout",
    "diarization": "Speaker ID (diarization)",
    "embeddings": "Embeddings",
    "identity": "Identity matching",
    "transcription": "Transcription",
    "finalizing": "Finalizing",
}


def _strip_invisible_text(value):
    if not isinstance(value, str):
        return str(value)
    # Remove control chars and zero-width noise often pasted with copied paths.
    return re.sub(r"[\u0000-\u001f\u007f-\u00a0\u200b-\u200f\u2060\ufeff]", "", value)


def _is_supported_recording_file(path):
    return os.path.splitext(path)[1].lower() in SUPPORTED_RECORDING_EXTS


def _normalize_file_path(raw_path):
    """Return a cleaned file path in filesystem form."""
    value = _strip_invisible_text(raw_path or "").strip()
    if not value:
        return value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    try:
        tokens = shlex.split(value)
    except ValueError:
        tokens = []
    if len(tokens) == 1:
        return tokens[0]
    return value.replace("\\ ", " ")


def _file_path_candidates(raw_path):
    """Build path candidates from pasted/quoted input.

    Accepts Windows-style backslashes, quoted shell paths, and relative paths.
    """
    normalized = _normalize_file_path(raw_path)
    if not normalized:
        return []
    normalized = os.path.expanduser(normalized)
    abs_path = normalized if os.path.isabs(normalized) else os.path.abspath(normalized)

    candidates = [normalized]
    if abs_path != normalized:
        candidates.append(abs_path)

    if "\\" in normalized:
        slash_path = normalized.replace("\\", "/")
        if slash_path not in candidates:
            candidates.append(slash_path)
        match = re.match(r"^([A-Za-z]):/(.*)", slash_path)
        if match:
            drive, rest = match.groups()
            candidates.append(f"/Volumes/{drive.upper()}/{rest}")

    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _resolve_file_path(raw_path):
    candidates = _file_path_candidates(raw_path)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else ""


def _load_tui_state():
    state = {
        "recent_files": [],
        "last_file_dir": "",
    }
    if not os.path.exists(_TUI_STATE_FILE):
        return state
    try:
        with open(_TUI_STATE_FILE) as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return state
    if not isinstance(data, dict):
        return state
    recent = data.get("recent_files")
    if isinstance(recent, list):
        state["recent_files"] = [str(x) for x in recent if isinstance(x, str)]
    last_dir = data.get("last_file_dir")
    if isinstance(last_dir, str):
        state["last_file_dir"] = last_dir
    default_preset = data.get("default_preset")
    if isinstance(default_preset, str):
        state["default_preset"] = default_preset
    return state


def _save_tui_state(state):
    try:
        with open(_TUI_STATE_FILE, "w") as f:
            yaml.safe_dump(state, f)
    except Exception:
        pass


def _update_tui_state(update):
    state = _load_tui_state()
    update(state)
    _save_tui_state(state)


def _record_recent_files(paths):
    normalized = []
    for p in paths:
        p = _resolve_file_path(p)
        if not p:
            continue
        p = os.path.abspath(p)
        if os.path.exists(p) and p not in normalized:
            normalized.append(p)
    if not normalized:
        return []
    def _save(state):
        recents = list(state.get("recent_files", []))
        for item in normalized:
            if item in recents:
                recents.remove(item)
            recents.insert(0, item)
        state["recent_files"] = recents[:10]
        state["last_file_dir"] = os.path.dirname(normalized[0])
        return state
    _update_tui_state(_save)
    return normalized


def _default_preset_for_queue(files):
    if not files:
        return DEFAULT_TUI_PRESET
    exts = {os.path.splitext(f)[1].lower() for f in files}
    if exts.issubset({".m4a", ".mp3", ".aac", ".m4b", ".m4r", ".aif", ".aiff"}):
        return "fast_audio"
    if exts.issubset({".mp4", ".mov"}):
        return "fast_audio"
    if len(exts) > 1:
        return DEFAULT_TUI_PRESET
    return DEFAULT_TUI_PRESET


def _apply_preset_to_selects(screen, preset_id):
    preset = RUN_PRESETS.get(preset_id)
    if not preset:
        return
    for key, value in preset.items():
        if key == "label":
            continue
        screen.query_one(f"#{key}", Select).value = value


def _escape_applescript_string(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _choose_file_with_finder(multiple=False, start_dir=""):
    """Open macOS Finder and return selected POSIX path(s), or empty list."""
    if not sys.platform.startswith("darwin"):
        return []
    use_start_dir = bool(start_dir and os.path.isdir(start_dir))

    def run_picker(include_start_dir):
        script = [r'set media_types to {"public.audio", "public.movie"}']
        if include_start_dir:
            script.append(
                f'set start_dir to POSIX file "{_escape_applescript_string(start_dir)}"'
            )
        choose_cmd = "choose file with prompt \"Select recording file\" of type media_types"
        if multiple:
            choose_cmd += " with multiple selections allowed"
        if include_start_dir:
            choose_cmd += " default location start_dir"
        script.append(f"set selected_file to {choose_cmd}")
        if multiple:
            script.extend([
                "set selected_paths to \"\"",
                "repeat with selected_item in selected_file",
                "  set selected_paths to selected_paths & (POSIX path of selected_item) & linefeed",
                "end repeat",
                "return selected_paths",
            ])
        else:
            script.append('return POSIX path of selected_file')
        return subprocess.run(
            ["osascript", "-e", "\n".join(script)],
            check=False,
            capture_output=True,
            text=True,
        )

    proc = run_picker(use_start_dir)
    if proc.returncode != 0 and use_start_dir:
        proc = run_picker(False)
    if proc.returncode != 0:
        if "-128" not in (proc.stderr or ""):
            raise RuntimeError((proc.stderr or "File picker failed to open.").strip())
        return []
    text = (proc.stdout or "").strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _extract_error_hint(lines):
    joined = " ".join(lines[-20:]).lower()
    for pattern, reason, hint in KNOWN_ERROR_HINTS:
        if pattern.search(joined):
            return reason, hint
    return "Pipeline exited with non-zero status", "Retry once; if it repeats, check codec/path and run on a short clip first."


def _note_path_from_sidecar(path):
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            sidecar = json.load(f)
    except Exception:
        return ""
    note = sidecar.get("note")
    if isinstance(note, str):
        return note
    return ""


def _open_path(path):
    if not path:
        return False
    if not os.path.exists(path):
        return False
    cmd = None
    if sys.platform.startswith("darwin"):
        cmd = ["open", path]
    elif sys.platform.startswith("win"):
        cmd = ["cmd", "/c", "start", "", path]
    else:
        cmd = ["xdg-open", path]
    try:
        subprocess.Popen(cmd)
        return True
    except Exception:
        return False


def load_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        path = os.path.join(ROOT, "config.yaml")
        if not os.path.exists(path):
            path = os.path.join(ROOT, "config.example.yaml")
        with open(path) as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def auto_detect_devices():
    devices = sd.query_devices()
    bh = mic = None
    for i, d in enumerate(devices):
        name = d['name'].lower()
        if 'blackhole' in name and d['max_input_channels'] > 0:
            bh = i
        elif d['max_input_channels'] > 0 and mic is None:
            mic = i
    return mic, bh


TRANSCRIPTION_OPTS = [("Local Whisper (CPU)", "local"), ("Groq (cloud, fast)", "groq")]
NAME_READER_OPTS = [("Gemini (cloud)", "gemini"), ("Ollama (local)", "ollama"), ("Off", "off")]
LLM_OPTS = [("minimax-m3:cloud", "minimax-m3:cloud"), ("gpt-oss:20b", "gpt-oss:20b")]

# Select id suffix -> (config section, key) for persisting/reading model choices.
MODEL_FIELDS = {
    "transcription": ("transcription", "backend"),
    "name_reader": ("name_reader", "backend"),
    "llm": ("llm", "model"),
}


def _invalidate_config_cache():
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def _select_value(value, options, default):
    """Coerce a config value to a valid Select option. YAML parses a bare
    `off`/`on`/`yes`/`no` as a bool, so map those back to strings; fall back to
    `default` if the value isn't a known option (so a stray config never crashes
    the Select)."""
    valid = {v for _, v in options}
    value = {True: "on", False: "off"}.get(value, value)
    return value if value in valid else default


def _model_selects(cfg, prefix):
    """Three model Selects (transcription / name-reader / LLM) seeded from cfg.
    `prefix` namespaces the ids ('home_' persists on change, 'run_' is per-run)."""
    tx = _select_value(cfg.get("transcription", {}).get("backend", "local"),
                       TRANSCRIPTION_OPTS, "local")
    nr = _select_value(cfg.get("name_reader", {}).get("backend", "gemini"),
                       NAME_READER_OPTS, "gemini")
    llm = cfg.get("llm", {}).get("model", "minimax-m3:cloud")
    llm_opts = list(LLM_OPTS)
    if llm not in [v for _, v in llm_opts]:
        llm_opts.insert(0, (llm, llm))
    return [
        Label("Transcription"),
        Select(TRANSCRIPTION_OPTS, value=tx, allow_blank=False, id=f"{prefix}_transcription"),
        Label("Name reader (Zoom name tags)"),
        Select(NAME_READER_OPTS, value=nr, allow_blank=False, id=f"{prefix}_name_reader"),
        Label("Summary LLM"),
        Select(llm_opts, value=llm, allow_blank=False, id=f"{prefix}_llm"),
    ]


class HomeScreen(Screen):
    def compose(self):
        mic, bh = auto_detect_devices()
        cfg = load_config()
        backend = cfg["audio"].get("capture_backend", "sck")
        devs = sd.query_devices()
        mic_name = devs[mic]['name'] if mic is not None and mic < len(devs) else "Not found"
        bh_name = devs[bh]['name'] if bh is not None and bh < len(devs) else "Not found"

        if backend == "sck":
            bh_line = "  Audio: [b]System (ScreenCaptureKit)[/]  (no BlackHole needed)"
        elif bh is not None:
            bh_line = f"  BH:  [b]{bh_name}[/]  (--bh [i]{bh}[/])"
        else:
            bh_line = "  BH:  [red]Not detected[/]"

        yield Header(show_clock=False)
        yield Container(
            Static("EchoID", classes="title"),
            Static("Meeting recording -> Obsidian notes", classes="subtitle"),
            Rule(),
            Label("Audio Devices"),
            Static(
                f"  Mic: [b]{mic_name}[/]  (--mic [i]{mic}[/])"
                if mic is not None else "  Mic: [red]Not detected[/]"
            ),
            Static(bh_line),
            Rule(),
            Label("Models (saved globally)"),
            *_model_selects(cfg, "home"),
            Rule(),
            Button("Start Recording", id="record", variant="primary"),
            Button("Process Existing File...", id="process"),
            Button("Review Past Meeting...", id="review_past"),
            Button("Manage Speakers...", id="manage_speakers"),
            Rule(),
            Button("List Audio Devices", id="list_devices"),
            Button("Quit", id="quit"),
            id="home",
        )
        yield Footer()

    def on_select_changed(self, event):
        # Persist a model choice to config.yaml (sticky global) when changed here.
        sid = event.select.id or ""
        if not sid.startswith("home_"):
            return
        suffix = sid.removeprefix("home_")
        if suffix not in MODEL_FIELDS:
            return
        section, key = MODEL_FIELDS[suffix]
        # Skip the Changed event Select fires on initial mount (and any no-op):
        # only write when the value actually differs from config. Coerce a bare
        # YAML `off`/`on` (loaded as a bool) so `off` vs "off" reads as a no-op.
        current = load_config().get(section, {}).get(key)
        current = {True: "on", False: "off"}.get(current, current)
        if current == event.value:
            return
        save_config(os.path.join(ROOT, "config.yaml"), section, key, event.value)
        _invalidate_config_cache()

    def on_button_pressed(self, event):
        if event.button.id == "record":
            self.app.mic, self.app.bh = auto_detect_devices()
            backend = load_config()["audio"].get("capture_backend", "sck")
            if self.app.mic is None:
                self.sub_title = "[red]Error: microphone not detected[/]"
                return
            # BlackHole is only required by the legacy blackhole backend; the sck
            # backend captures system audio via ScreenCaptureKit with no BlackHole.
            if backend == "blackhole" and self.app.bh is None:
                self.sub_title = "[red]Error: BlackHole not detected[/]"
                return
            self.app.push_screen("record")
        elif event.button.id == "process":
            self.app.push_screen("file_picker")
        elif event.button.id == "review_past":
            self.app.push_screen("review_picker")
        elif event.button.id == "manage_speakers":
            self.app.push_screen("manage_speakers")
        elif event.button.id == "list_devices":
            self.app.push_screen("device_list")
        elif event.button.id == "quit":
            self.app.exit()


class DeviceListScreen(Screen):
    def compose(self):
        yield Header(show_clock=False)
        yield Container(
            Static("Audio Devices", classes="title"),
            Rule(),
            ScrollableContainer(
                *[
                    Static(
                        f"  {i}: {d['name']}  "
                        f"(in={d['max_input_channels']}, out={d['max_output_channels']})"
                    )
                    for i, d in enumerate(sd.query_devices())
                ]
            ),
            Button("Back", id="back", variant="default"),
            id="devices",
        )
        yield Footer()

    def on_button_pressed(self, event):
        if event.button.id == "back":
            self.app.pop_screen()


class ReviewPickerScreen(Screen):
    """Re-open speaker review on a meeting that was already processed.

    The pipeline's names are not always right -- a cluster can carry a name the
    visual reader misread, and until now the only ways to change it were a full
    reprocess (which reassigns cluster ids) or a rename-everywhere (which is
    global by name and would rewrite the meetings where that name is correct).
    Re-reviewing one sidecar renames within that meeting alone, then rebuilds
    its transcript and summary from the corrected names.
    """

    BINDINGS = [("escape", "back")]

    def action_back(self):
        self.app.pop_screen()

    def compose(self):
        yield Header(show_clock=False)
        yield ScrollableContainer(
            Static("Review Past Meeting", classes="title"),
            Static("Pick a meeting to re-open speaker review: play a clip, "
                   "correct a name, and its transcript and summary are rebuilt. "
                   "Only that meeting changes."),
            Rule(),
            Container(id="review_rows"),
            Rule(),
            Button("Back", id="back"),
            id="reviewpicker",
        )
        yield Footer()

    async def on_mount(self):
        cfg = load_config()
        # Rows are addressed by index: a widget id must be a valid identifier,
        # and a meeting stem contains dots and dashes.
        self._meetings = review_sidecar.list_reviewable(cfg["paths"]["workspace"])
        rows = self.query_one("#review_rows", Container)
        if not self._meetings:
            await rows.mount(Static("No processed meetings found in the workspace."))
            return
        for i, m in enumerate(self._meetings):
            names = ", ".join(m["labels"][:3]) or "no speakers labeled"
            if len(m["labels"]) > 3:
                names += f", +{len(m['labels']) - 3} more"
            await rows.mount(Horizontal(
                Static(f"{m['meeting']}  ({m['turns']} turns)  {names}",
                       classes="review_info"),
                Button("Review", id=f"review_{i}", classes="rowbtn"),
            ))

    def on_button_pressed(self, event):
        bid = event.button.id or ""
        if bid == "back":
            self.app.pop_screen()
        elif bid.startswith("review_"):
            self.app.push_screen("namer", self._meetings[int(bid.split("_")[1])]["path"])


class FilePickerScreen(Screen):
    async def on_mount(self):
        await self._refresh_queue_view()
        await self._refresh_recent_view()

    def compose(self):
        yield Header(show_clock=False)
        yield Container(
            Static("Process Existing Recording", classes="title"),
            Rule(),
            Label("Enter path to a recording (.wav, .mp4, .m4a, .mp3, …):"),
            Input(placeholder="/path/to/meeting.mp4", id="file_path"),
            Horizontal(
                Button("Select Files...", id="select_files"),
                Button("Add to Queue", id="add_file"),
                Button("Process", id="process", variant="primary"),
                Button("Clear Queue", id="clear_queue", variant="default"),
                Button("Back", id="back"),
            ),
            Rule(),
            Static("Queued files:", id="queue_header"),
            ScrollableContainer(id="queue_list"),
            Rule(),
            Static("Recent files:", id="recent_header"),
            ScrollableContainer(id="recent_list"),
            id="filepicker",
        )
        yield Footer()

    async def on_button_pressed(self, event):
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "process":
            self._start_processing()
        elif event.button.id == "select_files":
            state = _load_tui_state()
            try:
                selected = _choose_file_with_finder(
                    multiple=True,
                    start_dir=state.get("last_file_dir", ""),
                )
            except (OSError, RuntimeError) as exc:
                self.sub_title = f"[red]File picker error: {exc}[/]"
                return
            if not selected:
                return
            for path in selected:
                self._add_path_to_queue(path)
            await self._refresh_queue_view()
        elif event.button.id == "add_file":
            if self._add_path_to_queue(
                    self.query_one("#file_path", Input).value):
                self.query_one("#file_path", Input).value = ""
                await self._refresh_queue_view()
        elif event.button.id == "clear_queue":
            self.app.processing_queue = []
            self.app.processing_cleanup_flags = []
            await self._refresh_queue_view()
        elif (event.button.id or "").startswith("remove_"):
            await self._remove_queued_file(event.button.id.removeprefix("remove_"))
        elif (event.button.id or "").startswith("recent_"):
            await self._start_recent_reprocess(event.button.id.removeprefix("recent_"))

    def _get_recent_files(self):
        state = _load_tui_state()
        recents = state.get("recent_files") or []
        if not isinstance(recents, list):
            return []
        return [str(p) for p in recents if isinstance(p, str)][:10]

    def _add_path_to_queue(self, raw_path):
        path = _resolve_file_path(raw_path)
        if path != (raw_path or "").strip():
            self.query_one("#file_path", Input).value = path
        if path and not _is_supported_recording_file(path):
            self.sub_title = "[red]Unsupported file type[/]"
            return False
        if not path or not os.path.exists(path):
            self.sub_title = "[red]File not found[/]"
            return False

        if not hasattr(self.app, "processing_queue") or self.app.processing_queue is None:
            self.app.processing_queue = []
        if path not in self.app.processing_queue:
            self.app.processing_queue.append(path)
            if not hasattr(self.app, "processing_cleanup_flags"):
                self.app.processing_cleanup_flags = []
            self.app.processing_cleanup_flags.append(False)
        self.sub_title = ""
        return True

    async def _start_recent_reprocess(self, raw_index):
        if not raw_index.isdigit():
            return
        idx = int(raw_index)
        recent = self._get_recent_files()
        if not (0 <= idx < len(recent)):
            return
        path = _resolve_file_path(recent[idx])
        if not path or not os.path.exists(path):
            self.sub_title = "[red]Recent file not found[/]"
            return
        self.app.start_batch([path], [False])
        self.app.push_screen("model_confirm")

    async def _refresh_recent_view(self):
        rows = self.query_one("#recent_list", ScrollableContainer)
        await rows.remove_children()
        recent = self._get_recent_files()
        if not recent:
            await rows.mount(Static("  (none)"))
            self.query_one("#recent_header", Static).update("Recent files: 0")
            return
        self.query_one("#recent_header", Static).update(f"Recent files: {len(recent)}")
        for i, item in enumerate(recent):
            short = item
            if len(item) > 90:
                short = item[:40] + "..." + item[-40:]
            await rows.mount(Horizontal(
                Static(f"{i + 1}. {short}", classes="recent_item"),
                Button("Reprocess", id=f"recent_{i}", variant="default"),
            ))

    async def _remove_queued_file(self, raw_index):
        if not (getattr(self.app, "processing_queue", None)
                and raw_index and raw_index.isdigit()):
            return
        idx = int(raw_index)
        if 0 <= idx < len(self.app.processing_queue):
            self.app.processing_queue.pop(idx)
            if getattr(self.app, "processing_cleanup_flags", None) is not None:
                try:
                    self.app.processing_cleanup_flags.pop(idx)
                except Exception:
                    pass
            await self._refresh_queue_view()
            self.sub_title = ""

    async def _refresh_queue_view(self):
        queue = getattr(self.app, "processing_queue", [])
        rows = self.query_one("#queue_list", ScrollableContainer)
        await rows.remove_children()
        if not queue:
            await rows.mount(Static("  (none)"))
            self.query_one("#queue_header", Static).update("Queued files: 0")
            return
        self.query_one("#queue_header", Static).update(f"Queued files: {len(queue)}")
        for i, item in enumerate(queue):
            # Keep text short, but preserve path identity.
            short = item
            if len(item) > 90:
                short = item[:40] + "..." + item[-40:]
            await rows.mount(Horizontal(
                Static(f"{i + 1}. {short}", classes="queue_item"),
                Button("Remove", id=f"remove_{i}", variant="error"),
            ))

    def _start_processing(self):
        if not getattr(self.app, "processing_queue", None):
            self._add_path_to_queue(self.query_one("#file_path", Input).value)
        if not self.app.processing_queue:
            self.sub_title = "[red]Add at least one file[/]"
            return
        cleaned_queue = []
        cleaned_flags = []
        for path, keep_frames in zip(
                self.app.processing_queue,
                self.app.processing_cleanup_flags or [False] * len(self.app.processing_queue)):
            if path and _is_supported_recording_file(path) and os.path.exists(path):
                cleaned_queue.append(path)
                cleaned_flags.append(keep_frames)
        self.app.processing_queue = cleaned_queue
        self.app.processing_cleanup_flags = cleaned_flags
        # Deduplicate while preserving order, then start a fresh batch run.
        deduped = []
        deduped_flags = []
        for path, keep_frames in zip(
                self.app.processing_queue,
                self.app.processing_cleanup_flags or [False] * len(self.app.processing_queue)):
            if path not in deduped:
                deduped.append(path)
                deduped_flags.append(keep_frames)
        self.app.processing_queue = deduped
        self.app.processing_cleanup_flags = deduped_flags
        if not self.app.processing_queue:
            self.sub_title = "[red]No valid files to process[/]"
            return
        self.app.start_batch(self.app.processing_queue, self.app.processing_cleanup_flags)
        self.app.push_screen("model_confirm")


class RecordingScreen(Screen):
    elapsed = reactive(0.0)
    file_size = reactive(0)
    active_speaker = reactive("")
    warning = reactive("")

    def compose(self):
        yield Header(show_clock=False)
        yield Container(
            Static("Recording", classes="title"),
            Rule(),
            Digits(id="timer", classes="timer"),
            Static("", id="file_info"),
            Static("", id="speaker_info"),
            Static("", id="warn"),
            Rule(),
            Button("Stop Recording", id="stop", variant="error"),
            id="record",
        )
        yield Footer()

    def on_mount(self):
        self.start_time = time.monotonic()
        self.recorder = None
        self.vision = None
        self.audio_path = None
        self._start_error = None
        self._finalized = False
        self._start_recording()
        self.set_interval(0.25, self._update)

    def _start_recording(self):
        cfg = load_config()
        workspace = cfg["paths"]["workspace"]
        os.makedirs(workspace, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.audio_path = os.path.join(workspace, f"meeting_{timestamp}.wav")

        from core.audio_recorder import AudioRecorder, CaptureStartupError
        from core.visual_ingestion import VisualIngestion

        self.vision = VisualIngestion(
            polling_interval=cfg["visual"]["polling_interval"],
            min_window_size=cfg["visual"].get("zoom_min_window_size", 300),
            frames_dir=frames_dir_for(self.audio_path),
            frame_interval=cfg["visual"].get("frame_interval", 4.0),
        )
        self.recorder = AudioRecorder(
            self.app.mic, self.app.bh, self.audio_path,
            samplerate=cfg["audio"]["sample_rate"],
            monitor_enabled=True,
            capture_backend=cfg["audio"].get("capture_backend", "sck"),
            sck_binary_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "native", "sck_capture"),
        )
        self.vision.start()
        try:
            self.recorder.start()
        except CaptureStartupError as e:
            # Startup failed (usually a denied Screen Recording permission) —
            # don't sit through a whole meeting recording nothing. Show the
            # error; the button becomes "Back".
            self._start_error = str(e)
            self.recorder = None
            try:
                self.vision.stop()
            except Exception:
                pass
            self.vision = None
            self.query_one("#warn", Static).update(f"[red]{e}[/]")
            self.query_one("#stop", Button).label = "Back"

    def _update(self):
        self.elapsed = time.monotonic() - self.start_time
        if self.audio_path and os.path.exists(self.audio_path):
            self.file_size = os.path.getsize(self.audio_path)
        if self.vision:
            self.active_speaker = str(self.vision.frame_count)
        if self.recorder:
            if self.recorder.mic_silent:
                self.warning = ("No microphone audio (ch0) — grant Microphone "
                                "permission and restart to capture your voice")
            elif self.recorder.ch1_silent:
                self.warning = ("No system audio (ch1) — grant Screen Recording "
                                "permission and restart to capture remote speech")
            elif self.recorder.stalled_side:
                self.warning = (f"The {self.recorder.stalled_side} channel stopped "
                                f"producing audio — check your devices")

    def watch_elapsed(self, val):
        mins = int(val // 60)
        secs = int(val % 60)
        self.query_one("#timer", Digits).update(f"{mins:02d}:{secs:02d}")

    def watch_file_size(self, val):
        kb = val / 1024
        self.query_one("#file_info", Static).update(f"File size: {kb:.0f} KB")

    def watch_active_speaker(self, val):
        self.query_one("#speaker_info", Static).update(
            f"Frames captured: {val}" if val else ""
        )

    def watch_warning(self, val):
        self.query_one("#warn", Static).update(f"[red]⚠ {val}[/]" if val else "")

    def on_button_pressed(self, event):
        if event.button.id == "stop":
            if self._start_error:
                self.app.pop_screen()  # recording never started — back to home
            else:
                self._stop_recording()

    def on_unmount(self):
        # Any exit (Stop, quit, Ctrl+C) finalizes: without this, quitting mid-
        # recording tore down the app without stopping the recorder, so the WAV
        # was never closed and the whole meeting was lost.
        self._finalize_recording()

    def _finalize_recording(self):
        if self._finalized:
            return
        self._finalized = True
        if self.recorder:
            try:
                self.recorder.stop()
            except Exception:
                pass
        if self.vision:
            try:
                self.vision.stop()
            except Exception:
                pass
            # Frames were saved to disk during recording; the --from-file
            # subprocess rebuilds the manifest from the frames dir next to the WAV.

    def _stop_recording(self):
        self.query_one("#stop", Button).disabled = True
        self._finalize_recording()
        self.app.selected_file = self.audio_path
        self.app.start_batch([self.audio_path], [True])  # fresh recording => cleanup
        self.app.push_screen("model_confirm")


class ModelConfirmScreen(Screen):
    """A last-minute chance to change models for THIS run before processing,
    without disturbing the saved global defaults."""

    BINDINGS = [("escape", "go")]

    def __init__(self):
        super().__init__()
        self._initial_preset = DEFAULT_TUI_PRESET

    def action_go(self):
        self._process()

    def compose(self):
        cfg = load_config()
        yield Header(show_clock=False)
        yield Container(
            Static("Models for this run", classes="title"),
            Static("Defaults are your saved settings; changes here apply to this run only."),
            Rule(),
            Label("Run preset"),
            Select(
                RUN_PRESET_OPTIONS,
                value=self._initial_preset
                if self._initial_preset in [x[1] for x in RUN_PRESET_OPTIONS]
                else "custom",
                id="run_preset",
            ),
            *_model_selects(cfg, "run"),
            Rule(),
            Button("Process", id="go", variant="primary"),
            id="modelconfirm",
        )
        yield Footer()

    def on_mount(self):
        queue = getattr(self.app, "processing_queue", [])
        state = _load_tui_state()
        if isinstance(queue, list) and queue:
            guessed = _default_preset_for_queue(queue)
            if isinstance(guessed, str):
                self._initial_preset = guessed
        if state.get("default_preset") in [x[1] for x in RUN_PRESET_OPTIONS]:
            self._initial_preset = state["default_preset"]
        if self._initial_preset not in [x[1] for x in RUN_PRESET_OPTIONS]:
            self._initial_preset = "custom"
        preset_widget = self.query_one("#run_preset", Select)
        preset_widget.value = self._initial_preset
        if self._initial_preset in RUN_PRESETS:
            _apply_preset_to_selects(self, self._initial_preset)

    def on_select_changed(self, event):
        if event.select.id != "run_preset":
            return
        if event.value in RUN_PRESETS:
            _apply_preset_to_selects(self, event.value)

    def on_button_pressed(self, event):
        if event.button.id == "go":
            self._process()

    def _process(self):
        preset = self.query_one("#run_preset", Select).value
        if preset in RUN_PRESETS:
            _update_tui_state(lambda state: state.update({
                "default_preset": preset
            }))
        self.app.model_overrides = {
            "transcription_backend": self.query_one("#run_transcription", Select).value,
            "name_reader_backend": self.query_one("#run_name_reader", Select).value,
            "llm_model": self.query_one("#run_llm", Select).value,
        }
        self.app.push_screen("processing")


class ProcessingScreen(Screen):
    BINDINGS = [("escape", "cancel_processing")]

    def action_cancel_processing(self):
        self._request_cancel()

    def compose(self):
        yield Header(show_clock=False)
        yield Container(
            Static("Processing Meeting...", classes="title"),
            Rule(),
            Static("Waiting for pipeline...", id="status"),
            Horizontal(
                Static("Stage: --", id="stage"),
                Static("Elapsed: 00:00", id="elapsed"),
            ),
            Static("No file selected", id="current_file"),
            Horizontal(
                Button("Cancel", id="cancel", variant="error"),
                Button("Retry", id="retry", variant="default", disabled=True),
                id="pipeline_actions",
            ),
            RichLog(id="log", highlight=True, markup=True, wrap=True),
            id="processing",
        )
        yield Footer()

    def on_mount(self):
        self._is_running = False
        self._cancel_requested = False
        self._proc = None
        self._log_lines = []
        self._last_announced = None
        self._start_time = None
        self._tick_handle = self.set_interval(0.5, self._tick)
        self._set_controls(disable_retry=True, disable_cancel=False)
        self.run_pipeline()

    def _set_controls(self, disable_retry=True, disable_cancel=False):
        self.query_one("#retry", Button).disabled = disable_retry
        self.query_one("#cancel", Button).disabled = disable_cancel

    def on_unmount(self):
        if self._tick_handle:
            self._tick_handle.stop()

    def on_button_pressed(self, event):
        if event.button.id == "cancel":
            self._request_cancel()
        elif event.button.id == "retry":
            self._set_controls(disable_retry=True, disable_cancel=False)
            self.query_one("#status", Static).update("Retrying...")
            self.run_pipeline()

    def _request_cancel(self):
        self._cancel_requested = True
        self._set_controls(disable_cancel=True)
        self.query_one("#status", Static).update("Cancel requested...")
        proc = getattr(self, "_proc", None)
        if proc is not None:
            self._terminate_process(proc)

    @work(thread=True)
    def run_pipeline(self):
        queue = list(getattr(self.app, "processing_queue", []) or [])
        if not queue and getattr(self.app, "selected_file", None):
            queue = [self.app.selected_file]
            self.app.processing_queue = queue
            self.app.batch_index = 0
        idx = getattr(self.app, "batch_index", 0)
        if not queue:
            self.app.call_from_thread(self._done, None, BATCH_STATUS_FAILED,
                                      "No files in queue.")
            return
        if idx < 0 or idx >= len(queue):
            self.app.call_from_thread(self._done, None, BATCH_STATUS_FAILED,
                                      "Batch index is out of range.")
            return

        wav = queue[idx]
        self.app.selected_file = wav
        self._set_current_file(wav)
        self._set_stage("")
        self._set_status(f"Processing {os.path.basename(wav)}")
        self._last_announced = None
        self._log_lines = []
        self._start_time = time.monotonic()
        self._cancel_requested = False
        self._is_running = True
        self._set_controls(disable_retry=True, disable_cancel=False)

        overrides = getattr(self.app, "model_overrides", {}) or {}
        cmd = [
            sys.executable, os.path.join(ROOT, "main.py"),
            "--from-file", wav,
            "--no-prompt",
            "--no-finalize",   # the TUI reviews + summarizes via SpeakerReview.commit
        ]
        for flag, key in (("--transcription-backend", "transcription_backend"),
                          ("--name-reader-backend", "name_reader_backend"),
                          ("--llm-model", "llm_model")):
            if overrides.get(key):
                cmd += [flag, overrides[key]]
        if (idx < len(getattr(self.app, "processing_cleanup_flags", []))
                and self.app.processing_cleanup_flags[idx]):
            cmd.append("--cleanup-frames")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=ROOT,
        )
        self._proc = proc

        announced = None
        for line in proc.stdout:
            announced = review_sidecar.parse_announcement(line) or announced
            self._log_lines.append(line)
            self._update_stage(line)
            self.app.call_from_thread(self._log, line.rstrip())
            if self._cancel_requested:
                self._terminate_process(proc)
                break
        code = proc.wait()
        self._is_running = False
        if self._cancel_requested:
            status = BATCH_STATUS_CANCELED
            message = "Canceled by user."
        elif code == 0:
            status = BATCH_STATUS_DONE
            message = None
        else:
            status = BATCH_STATUS_FAILED
            reason, hint = _extract_error_hint(self._log_lines)
            message = f"{reason}. {hint}"
        self.app.call_from_thread(self._done, announced, status, message)

    def _tick(self):
        if self._start_time is None:
            return
        self.query_one("#elapsed", Static).update(
            f"Elapsed: {_format_duration(time.monotonic() - self._start_time)}")

    def _terminate_process(self, proc):
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            return
        try:
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _set_status(self, msg):
        self.query_one("#status", Static).update(msg)

    def _set_stage(self, stage):
        self._current_stage = stage
        label = RUN_STAGE_TO_LABEL.get(stage, "processing" if stage else "ready")
        self.query_one("#stage", Static).update(f"Stage: {label}")

    def _set_current_file(self, path):
        if not path:
            self.query_one("#current_file", Static).update("No file selected")
            return
        display = os.path.basename(path)
        self.query_one("#current_file", Static).update(f"File: {display}")

    def _update_stage(self, line):
        for stage, marker in PIPELINE_STAGE_MARKERS.items():
            if marker.search(line):
                self._set_stage(stage)
                return

    def _log(self, msg):
        self.query_one("#log", RichLog).write(msg.rstrip())

    def _done(self, announced=None, status=BATCH_STATUS_DONE, message=None):
        cfg = load_config()
        wav = self.app.selected_file
        sidecar = review_sidecar.resolve(announced, cfg["paths"]["workspace"], wav) if wav else None
        result = {
            "file": wav,
            "sidecar": sidecar,
            "note": _note_path_from_sidecar(sidecar),
            "status": status,
            "message": message,
            "stage": getattr(self, "_current_stage", ""),
        }
        if self._start_time is not None:
            result["duration_seconds"] = int(time.monotonic() - self._start_time)
        self.app.record_batch_result(getattr(self.app, "batch_index", 0), result)

        if status == BATCH_STATUS_DONE:
            if wav:
                _record_recent_files([wav])
            self._set_status("Pipeline complete.")
            self.query_one("#log", RichLog).write("\n[green]Pipeline complete![/]")
            self._set_controls(disable_cancel=True)
            self.app.push_screen("namer", sidecar)
            return
        self._set_controls(disable_retry=False if status == BATCH_STATUS_FAILED else True,
                           disable_cancel=True)
        if status == BATCH_STATUS_CANCELED:
            self._set_status("Pipeline canceled.")
            self.query_one("#log", RichLog).write("[yellow]Pipeline canceled.[/]")
            queue = getattr(self.app, "processing_queue", [])
            idx = getattr(self.app, "batch_index", 0)
            if idx + 1 < len(queue):
                self.app.batch_index = idx + 1
                self.app.selected_file = queue[self.app.batch_index]
                self.run_pipeline()
            else:
                self.app.push_screen("summary")
            return
        self._set_status("Pipeline failed.")
        if message:
            self.query_one("#log", RichLog).write(f"[red]{message}[/]")
        if getattr(self, "_current_stage", ""):
            self.query_one("#log", RichLog).write(f"[yellow]Last stage: {RUN_STAGE_TO_LABEL.get(self._current_stage, self._current_stage)}[/]")



class NamerScreen(Screen):
    BINDINGS = [
        ("escape", "quit"),
    ]

    def action_quit(self):
        # Escape still finalizes, so the note never stays on the placeholder
        # summary (also covers the no-speakers case, which has no Save button).
        self._begin_finalize()

    def on_unmount(self):
        # Backstop: stop any clip if the screen is torn down (e.g. app exit).
        if self.review:
            try:
                self.review.stop_playback()
            except Exception:
                pass

    def __init__(self, sidecar_path: str = ""):
        super().__init__()
        self._sidecar = sidecar_path
        self.review = None
        self._error = ""
        self._finalizing = False

    def compose(self):
        yield Header(show_clock=False)
        yield ScrollableContainer(
            Static("Review Speakers", classes="title"),
            Static(
                "Play a clip (▶), rename, Split a mixed speaker, or check rows "
                "and Merge. Leave someone unnamed and keep 'remember' ticked to "
                "save them as a Speaker N you can identify later. Save enrolls "
                "voiceprints and updates the note.",
            ),
            Rule(),
            Container(id="rows"),
            Rule(),
            Horizontal(
                Button("Merge Checked", id="merge"),
                Button("Save & Enroll", id="save", variant="primary"),
                id="namer_actions",
            ),
            id="namer",
        )
        yield Footer()

    async def on_mount(self):
        cfg = load_config()
        db = os.path.join(cfg["paths"]["workspace"], "speakers.json")
        try:
            # The TUI summarizes here (the subprocess runs --no-finalize), so the
            # per-run Summary-LLM override must be applied at THIS construction,
            # not just passed to the subprocess.
            llm_model = self.app.model_overrides.get("llm_model") or cfg["llm"]["model"]
            self.review = SpeakerReview(
                self._sidecar, db,
                llm_model=llm_model,
                llm_timeout=cfg["llm"].get("timeout", 120),
                placeholder_min_duration=cfg["biometrics"]["placeholder_min_duration"],
                local_names=(cfg.get("identity") or {}).get("local_names", []) or [],
            )
        except Exception as e:  # sidecar missing/corrupt — degrade gracefully
            self.review = None
            self._error = str(e)
        await self._refresh_rows()

    async def _refresh_rows(self):
        rows = self.query_one("#rows", Container)
        actions = self.query_one("#namer_actions", Horizontal)
        await rows.remove_children()

        if self.review is None:
            actions.display = False
            await rows.mount(Static(f"[red]Could not load review data:[/] {self._error}"))
            return

        speakers = self.review.list_speakers()
        if not speakers:
            actions.display = False
            await rows.mount(Static("No speakers to review."))
            return

        actions.display = True
        # Typeahead source: enrolled voiceprints + glossary + names entered so far.
        suggester = SuggestFromList(self.review.suggestion_names(), case_sensitive=False)
        widgets = []
        for s in speakers:
            gid = s["id"]
            info = f"{s['name']}  ({s['total_dur']:.0f}s, {s['n_segments']} seg, {s['source']})"
            widgets.append(Horizontal(
                Static(info, classes="spk_info"),
                Button("▶", id=f"play_{gid}", classes="rowbtn"),
                Input(value=s["name"], id=f"name_{gid}", classes="rowname",
                      suggester=suggester),
                Button("Split", id=f"split_{gid}", classes="rowbtn"),
                Checkbox("remember", id=f"rem_{gid}", value=s["remember"]),
                Checkbox("merge", id=f"chk_{gid}"),
                classes="spkrow",
            ))
        await rows.mount(*widgets)

    async def on_button_pressed(self, event):
        bid = event.button.id or ""
        if bid == "save":
            self._begin_finalize()
        elif bid == "merge":
            await self._merge_checked()
        elif bid.startswith("play_"):
            self._play(int(bid[len("play_"):]))
        elif bid.startswith("split_"):
            self._apply_names()
            self.review.split(int(bid[len("split_"):]))
            await self._refresh_rows()

    def _apply_names(self):
        """Push the current text of each name Input, and each Remember choice,
        back into the review model."""
        for gid in list(self.review.groups.keys()):
            try:
                val = self.query_one(f"#name_{gid}", Input).value.strip()
            except Exception:
                continue
            if val:
                self.review.rename(gid, val)
            try:
                self.review.set_remember(
                    gid, self.query_one(f"#rem_{gid}", Checkbox).value)
            except Exception:
                pass

    async def _merge_checked(self):
        self._apply_names()
        checked = []
        for gid in list(self.review.groups.keys()):
            try:
                if self.query_one(f"#chk_{gid}", Checkbox).value:
                    checked.append(gid)
            except Exception:
                pass
        if len(checked) >= 2:
            self.review.merge(checked)
        await self._refresh_rows()

    @work(thread=True)
    def _play(self, gid):
        try:
            self.review.play_clip(gid)
        except Exception:
            pass

    def _begin_finalize(self):
        """Apply pending names, then commit (enroll + summarize) off the UI
        thread. Shared by Save and Escape so the note is always finalized."""
        if self.review is None or self._finalizing:
            self.app.complete_batch_or_summary()
            return
        self._finalizing = True
        self._apply_names()
        self.review.stop_playback()  # don't let a clip keep playing into summary
        for bid in ("#save", "#merge"):  # buttons absent in the no-speakers case
            try:
                self.query_one(bid, Button).disabled = True
            except Exception:
                pass
        try:
            self.query_one("#save", Button).label = "Summarizing..."
        except Exception:
            pass
        self._finalize()

    @work(thread=True)
    def _finalize(self):
        # Enroll + summarize the named transcript + rewrite the note (LLM call).
        try:
            self.review.commit()
        except Exception:
            pass
        self.app.call_from_thread(self.app.complete_batch_or_summary)


class SpeakerManagerScreen(Screen):
    """Identify and correct people across every recorded meeting.

    Standalone: reads only saved data (voiceprint DB, sidecars, notes) and never
    records or runs the pipeline. Play a clip to hear who a "Speaker N" actually
    is, then rename them once and have it corrected everywhere.
    """

    BINDINGS = [("escape", "back")]

    def action_back(self):
        self._player.stop()
        self.app.pop_screen()

    def on_unmount(self):
        self._player.stop()

    def __init__(self):
        super().__init__()
        self.directory = None
        self._player = ClipPlayer()
        self._pending = None      # a rename_plan awaiting a second confirming press
        self._error = ""
        # Rows are addressed by index, never by name: a Textual widget id must be a
        # valid identifier and real names contain spaces ("Speaker 1", "Jordan Lee").
        self._people = []

    def compose(self):
        yield Header(show_clock=False)
        yield ScrollableContainer(
            Static("Manage Speakers", classes="title"),
            Static("Play a clip (▶) to hear who someone is, type their real name, "
                   "then Rename — it is corrected in every note. Press Rename "
                   "again to confirm."),
            Rule(),
            Static("", id="mgr_status"),
            Container(id="mgr_rows"),
            id="manager",
        )
        yield Footer()

    async def on_mount(self):
        await self._reload()

    async def on_screen_resume(self):
        # Textual caches this screen (installed by name in App.SCREENS), so
        # popping it back to Home only suspends it -- on_mount never runs
        # again on a later push. Reload from disk on every revisit so a
        # voiceprint enrolled or updated by an intervening meeting shows up,
        # and so a rename never writes back a stale in-memory DB snapshot
        # over one that has since moved on.
        await self._reload()

    async def _reload(self):
        self._pending = None   # a preview computed against the old snapshot must not survive
        cfg = load_config()
        ws = cfg["paths"]["workspace"]
        try:
            bio = cfg["biometrics"]
            local_names = (cfg.get("identity") or {}).get("local_names", []) or []
            self.directory = SpeakerDirectory(
                ws, os.path.join(ws, cfg["paths"]["obsidian_vault"]),
                db_threshold=bio["db_threshold"],
                db_margin=bio.get("db_margin", 0.0),
                min_speech=bio.get("min_speech", 0.0),
                min_turn_seconds=bio.get("min_turn_seconds", 0.5),
                local_names=local_names)
            self._error = ""
        except Exception as e:
            self.directory = None
            self._error = str(e)
        await self._refresh_rows()

    async def _refresh_rows(self):
        rows = self.query_one("#mgr_rows", Container)
        await rows.remove_children()
        if self.directory is None:
            await rows.mount(Static(f"[red]Could not load speakers:[/] {self._error}"))
            return
        self._people = self.directory.identities()
        if not self._people:
            await rows.mount(Static("No enrolled speakers yet."))
            return
        widgets = []
        for i, e in enumerate(self._people):
            kind = "placeholder" if e["is_placeholder"] else "named"
            info = (f"{e['name']}  ({e['n_meetings']} meetings, "
                    f"{e['total_dur']:.0f}s, {kind})")
            widgets.append(Horizontal(
                Static(info, classes="spk_info"),
                Button("▶", id=f"mplay_{i}", classes="rowbtn"),
                Input(value=e["name"], id=f"mname_{i}", classes="rowname"),
                Button("Rename", id=f"mren_{i}", classes="rowbtn"),
                classes="spkrow",
            ))
        await rows.mount(*widgets)

    def _status(self, msg):
        self.query_one("#mgr_status", Static).update(msg)

    async def on_button_pressed(self, event):
        bid = event.button.id or ""
        if bid.startswith("mplay_"):
            name = self._people[int(bid[len("mplay_"):])]["name"]
            clip = self.directory.best_clip(name)
            if clip is None:
                self._status(f"[yellow]No recorded clip found for {name}.[/]")
            else:
                self._player.play(*clip)
        elif bid.startswith("mren_"):
            await self._rename(int(bid[len("mren_"):]))

    async def _rename(self, row):
        old = self._people[row]["name"]
        new = self.query_one(f"#mname_{row}", Input).value.strip()
        if not new or new == old:
            self._status("[yellow]Type the person's real name first.[/]")
            return
        # First press previews, second press applies.
        if self._pending and self._pending["old"] == old and self._pending["new"] == new:
            self.directory.apply_rename(self._pending)
            await self._reload()
            self._status(f"[green]Renamed to {new} everywhere.[/]")
            return
        plan = self.directory.rename_plan(old, new)
        self._pending = plan
        merge = " and merge into the existing person" if plan["merges_into_existing"] else ""
        self._status(
            f"[b]{old} → {new}[/]: {len(plan['notes'])} notes, "
            f"{plan['n_mentions']} mentions{merge}. Press Rename again to confirm.")


class SummaryScreen(Screen):
    BINDINGS = [
        ("escape", "quit_app"),
    ]

    def action_quit_app(self):
        self.app.exit()

    def compose(self):
        cfg = load_config()
        output_dir = os.path.join(cfg["paths"]["workspace"],
                                  cfg["paths"]["obsidian_vault"])
        notes = []
        if os.path.exists(output_dir):
            notes = sorted(os.listdir(output_dir), reverse=True)
        results = list(getattr(self.app, "batch_results", []))
        rows = []
        for i, item in enumerate(results):
            status = item.get("status", "done")
            status_txt = (
                "[green]Done[/]" if status == BATCH_STATUS_DONE else
                "[yellow]Canceled[/]" if status == BATCH_STATUS_CANCELED else
                "[red]Failed[/]"
            )
            source = item.get("file", "")
            note = item.get("note", "")
            if not note:
                note = _note_path_from_sidecar(item.get("sidecar", ""))
            label = os.path.basename(source) if source else f"item {i + 1}"
            rows.append(Horizontal(
                Static(f"{i + 1}. {label}", classes="summary_item"),
                Static(status_txt),
                Static(f"{os.path.basename(note) or 'no note'}", classes="summary_item"),
                Button("Open Note", id=f"open_note_{i}", disabled=not bool(note)),
                Button("Open Folder", id=f"open_folder_{i}", disabled=not bool(note)),
            ))
        summary_rows = (
            ScrollableContainer(*rows, id="summary_rows") if rows
            else Static(f"  {os.path.join(output_dir, notes[0]) if notes else 'N/A'}")
        )

        yield Header(show_clock=False)
        yield Container(
            Static("Batch Complete!", classes="title"),
            Rule(),
            Label(f"Processed {len(results)} file(s)."),
            Rule(),
            Static("Last notes:"),
            summary_rows,
            Rule(),
            Horizontal(
                Button("Open Output Folder", id="open_output", variant="default"),
                Button("Record Another", id="record", variant="primary"),
                Button("Quit", id="quit"),
            ),
            id="summary",
        )
        yield Footer()

    def on_button_pressed(self, event):
        bid = event.button.id or ""
        if event.button.id == "record":
            self.dismiss()
            self.app.pop_screen()
            self.app.pop_screen()
            self.app.push_screen("record")
        elif event.button.id == "open_output":
            cfg = load_config()
            output_dir = os.path.join(cfg["paths"]["workspace"],
                                      cfg["paths"]["obsidian_vault"])
            _open_path(output_dir)
        elif bid.startswith("open_note_"):
            try:
                idx = int(bid[len("open_note_"):])
            except ValueError:
                return
            if idx >= len(getattr(self.app, "batch_results", [])):
                return
            item = (getattr(self.app, "batch_results", []) or [])[idx]
            _open_path(item.get("note") or _note_path_from_sidecar(item.get("sidecar", "")))
        elif bid.startswith("open_folder_"):
            try:
                idx = int(bid[len("open_folder_"):])
            except ValueError:
                return
            if idx >= len(getattr(self.app, "batch_results", [])):
                return
            item = (getattr(self.app, "batch_results", []) or [])[idx]
            note = item.get("note") or _note_path_from_sidecar(item.get("sidecar", ""))
            if note:
                _open_path(os.path.dirname(note))
        elif event.button.id == "quit":
            self.app.exit()


class ZoomRecorderApp(App):
    CSS = """
    Screen {
        align: center top;
    }
    Container {
        width: 80%;
        margin: 1 2;
    }
    .title {
        text-style: bold;
        content-align: center top;
        height: 3;
    }
    .subtitle {
        content-align: center top;
        height: 1;
    }
    .timer {
        content-align: center top;
        height: 5;
    }
    Digits {
        text-style: bold;
    }
    Button {
        margin: 1 1;
    }
    Horizontal {
        height: auto;
    }
    /* A row container inside a ScrollableContainer MUST size to its content.
       Left at the default it takes the viewport's height and clips the rows
       that overflow, so the scroller sees nothing to scroll and every row past
       the first screenful is mounted, rendered and unreachable. */
    #rows, #review_rows, #mgr_rows, #queue_list, #recent_list {
        height: auto;
    }
    #queue_list, #recent_list {
        max-height: 12;
        border: solid $primary;
    }
    .queue_item {
        width: 1fr;
        content-align: left middle;
    }
    .recent_item {
        width: 1fr;
        content-align: left middle;
    }
    #summary_rows {
        height: auto;
        max-height: 10;
        border: solid $primary;
    }
    .spk_info {
        width: 34;
        content-align: left middle;
        height: 3;
    }
    /* The review picker's label is the row's whole point -- a meeting is
       identified by the speaker names it carries -- so it takes the width the
       terminal has rather than the fixed 34 of a speaker row, where the name is
       an Input beside it. At 34 the tail of a many-speaker meeting was clipped. */
    .review_info {
        width: 1fr;
        content-align: left middle;
        height: 3;
    }
    .rowname {
        width: 22;
    }
    .rowbtn {
        min-width: 8;
    }
    #log {
        height: 80%;
        border: solid $primary;
    }
    """

    # Only screens that survive being revisited are installed by name: Textual
    # instantiates an installed screen once and caches it, so `on_mount` never
    # fires again. SpeakerManagerScreen re-reads the DB in `on_screen_resume`,
    # and home is the base screen everything pops back to.
    SCREENS = {
        "home": HomeScreen,
        "manage_speakers": SpeakerManagerScreen,
    }

    # Every other screen does per-run work on mount -- starts a recorder, runs
    # the pipeline, reads config into widgets, lists devices -- so a cached one
    # makes the *second* meeting of a session a no-op. That failed silently in
    # the worst place: re-entering Recording rendered a live-looking screen
    # with no recorder behind it. These are constructed fresh on every push.
    PER_RUN_SCREENS = {
        "device_list": DeviceListScreen,
        "review_picker": ReviewPickerScreen,
        "file_picker": FilePickerScreen,
        "record": RecordingScreen,
        "model_confirm": ModelConfirmScreen,
        "processing": ProcessingScreen,
        "summary": SummaryScreen,
    }

    def __init__(self):
        super().__init__()
        self.mic = None
        self.bh = None
        self.selected_file = None
        self.processing_queue = []
        self.processing_cleanup_flags = []
        self.batch_index = 0
        self.batch_results = []
        self.speaker_renames = {}
        self.model_overrides = {}
        self.cleanup_frames = False  # True for a fresh recording, False for a reprocess

    def start_batch(self, files, cleanup_flags=None):
        self.processing_queue = list(files)
        self.processing_cleanup_flags = list(cleanup_flags or [False] * len(self.processing_queue))
        self.batch_index = 0
        self.batch_results = []
        if self.processing_queue:
            self.selected_file = self.processing_queue[0]

    def record_batch_result(self, index, result):
        while len(self.batch_results) <= index:
            self.batch_results.append({})
        if not isinstance(result, dict):
            result = {}
        self.batch_results[index] = result

    def complete_batch_or_summary(self):
        # Called after each Namer run when the run is committed (or skipped).
        idx = self.batch_index
        if idx + 1 < len(self.processing_queue):
            self.batch_index += 1
            if self.processing_queue:
                self.selected_file = self.processing_queue[self.batch_index]
            self.push_screen("processing")
            return
        self.push_screen("summary")

    def on_ready(self):
        self.push_screen("home")

    def push_screen(self, screen, data=None):
        if screen == "namer":
            super().push_screen(NamerScreen(data or ""))
        elif screen in self.PER_RUN_SCREENS:
            super().push_screen(self.PER_RUN_SCREENS[screen]())
        else:
            super().push_screen(screen)


def main():
    app = ZoomRecorderApp()
    app.run()


if __name__ == "__main__":
    main()
