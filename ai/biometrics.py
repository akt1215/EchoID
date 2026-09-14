import os
import json
import torch
import torchaudio
import numpy as np

from core.device import resolve_device


#: config key holding the model id for each backend. Kept beside the manager so
#: every caller resolves the name the same way — main.py once passed the
#: speechbrain id regardless of backend, which handed the ONNX loader an ECAPA
#: repo and killed the pipeline at model load.
_MODEL_KEYS = {
    "speechbrain": "speechbrain_model",
    "wespeaker_onnx": "wespeaker_model",
}


def model_for(bio_cfg):
    """The model id for the configured backend.

    Raises rather than falling back: a wrong-but-present model id fails deep
    inside a download with a confusing 404, while a missing one should stop the
    run where the mistake actually is.
    """
    backend = bio_cfg.get("backend")
    key = _MODEL_KEYS.get(backend)
    if key is None:
        raise ValueError(
            f"unknown biometrics backend {backend!r}; "
            f"expected one of {sorted(_MODEL_KEYS)}")
    name = bio_cfg.get(key)
    if not name:
        raise ValueError(
            f"biometrics.{key} is required for backend {backend!r}")
    return name


class BiometricsManager:
    """Loads the speaker-embedding model and the persistent voiceprint DB, and
    extracts one embedding per diarization turn. Naming/merging logic lives in
    the model-free `ai.identity_resolution` module; enrollment lives in
    `ai.speaker_review`. This class only produces embeddings and exposes the DB.
    """

    def __init__(self, db_path="speakers.json", backend="speechbrain",
                 model_name="speechbrain/spkrec-ecapa-voxceleb", device="auto",
                 target_channel=None):
        self.db_path = db_path
        self.backend = backend
        # Which WAV channel carries the speakers being embedded. Recordings are
        # stereo (ch0 = local mic, ch1 = remote); averaging them mixes the local
        # channel's noise into every remote speaker's voiceprint. None keeps the
        # old downmix, for mono or unknown inputs.
        self.target_channel = target_channel
        self.device = resolve_device(device)  # Metal (MPS) on Apple Silicon
        self.classifier = None
        self.speaker_db = self._load_db()

        if backend == "speechbrain":
            self._init_speechbrain(model_name)
        elif backend == "wespeaker_onnx":
            self._init_wespeaker_onnx(model_name)

    def _init_speechbrain(self, model_name):
        from speechbrain.inference.speaker import EncoderClassifier
        # speechbrain's run_opts device= path only sets device_type for cpu/cuda
        # (not mps), so it crashes on Apple Silicon GPU. Load on CPU, then move the
        # modules to the target device ourselves; encode_batch reads self.device
        # to place inputs.
        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            savedir="tmp_speechbrain",
        )
        if self.device != "cpu":
            self.classifier.device = self.device
            self.classifier.mods.to(self.device)

    def _init_wespeaker_onnx(self, model_name):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id=model_name,
            filename="voxceleb_resnet34.onnx"
        )
        self._ort_session = ort.InferenceSession(model_path)
        self._ort_input_name = self._ort_session.get_inputs()[0].name

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                return {k: np.array(v) for k, v in data.items()}
        return {}

    def _load_audio(self, audio_path):
        """Load a recording once and reduce it to the single channel to embed.

        Recordings are stereo (ch0 = mic, ch1 = remote) but speaker-embedding
        models expect a single channel. Selecting `target_channel` rather than
        averaging keeps the local mic out of every remote speaker's embedding.
        Call this once per batch and pass the result into extract_embedding to
        avoid reloading the whole WAV per turn.
        """
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            channel = self.target_channel
            if channel is not None and channel < signal.shape[0]:
                signal = signal[channel:channel + 1]
            else:
                signal = signal.mean(dim=0, keepdim=True)
        return signal, fs

    def _fbank(self, waveform, fs):
        """80-dim kaldi fbank features, shaped (1, frames, 80) for WeSpeaker.

        WeSpeaker's exported ONNX graphs take features, not audio, and are
        trained at 16 kHz with per-utterance mean normalization. Passing a raw
        waveform produces garbage rather than an error, so this conversion is
        what makes the backend usable at all.
        """
        import torchaudio.compliance.kaldi as kaldi

        if fs != 16000:
            waveform = torchaudio.functional.resample(waveform, fs, 16000)
        # Kaldi's frontend expects int16-scaled samples.
        feats = kaldi.fbank(waveform * (1 << 15), num_mel_bins=80,
                            frame_length=25, frame_shift=10,
                            dither=0.0, sample_frequency=16000,
                            energy_floor=0.0, window_type="hamming",
                            htk_compat=True, use_energy=False)
        feats = feats - feats.mean(dim=0, keepdim=True)     # per-utterance CMN
        return feats.unsqueeze(0).numpy().astype("float32")

    def extract_embedding(self, audio_path, start, end, signal=None, fs=None):
        # Load the file only when a preloaded (mono) signal is not supplied.
        if signal is None:
            signal, fs = self._load_audio(audio_path)
        segment = signal[:, int(start * fs):int(end * fs)]
        if segment.shape[0] > 1:  # guard for direct callers passing raw stereo
            segment = segment.mean(dim=0, keepdim=True)

        if self.backend == "speechbrain":
            embeddings = self.classifier.encode_batch(segment)
            # .cpu() is required before .numpy() when the model runs on MPS/CUDA.
            return embeddings.squeeze().cpu().numpy()
        elif self.backend == "wespeaker_onnx":
            feats = self._fbank(segment, fs)
            emb = self._ort_session.run(None, {self._ort_input_name: feats})[0]
            return emb.squeeze()

    def embed_all(self, audio_path, segments):
        """One embedding per segment, aligned to `segments` by index (empty array
        for any turn whose extraction fails). Loads the audio once and slices in
        memory, so this is the single embedding pass reused by merging, identity
        resolution, and the review sidecar.
        """
        signal, fs = self._load_audio(audio_path)
        out = []
        for seg in segments:
            try:
                emb = self.extract_embedding(audio_path, seg['start'], seg['end'],
                                             signal=signal, fs=fs)
                out.append(np.asarray(emb, dtype=float))
            except Exception:
                out.append(np.array([], dtype=float))
        return out
