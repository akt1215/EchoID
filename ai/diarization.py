import os
import torch
import soundfile as sf
from pyannote.audio import Pipeline

from core.device import resolve_device

class Diarizer:
    def __init__(self, model="pyannote/speaker-diarization-3.1",
                 use_auth_token=None, min_segment_duration=0.5, device="auto"):
        token = os.environ.get("HF_TOKEN", use_auth_token)
        kwargs = {}
        if token:
            kwargs["token"] = token
        self.pipeline = Pipeline.from_pretrained(model, **kwargs)
        # Metal (MPS) is ~7x faster than CPU here and gives the same speakers.
        self.device = resolve_device(device)
        self.pipeline.to(torch.device(self.device))
        print(f"Diarization pipeline on {self.device}.")
        self.min_segment_duration = min_segment_duration

    def diarize(self, audio_path, target_channel=1):
        print(f"Extracting Channel {target_channel + 1} for diarization...")

        data, samplerate = sf.read(audio_path)
        if len(data.shape) > 1 and data.shape[1] > target_channel:
            channel_data = data[:, target_channel]
        else:
            channel_data = data

        # Convert to float32 tensor and reshape to (1, num_samples)
        waveform = torch.from_numpy(channel_data.astype("float32")).unsqueeze(0)

        # Pass in-memory to bypass torchcodec's broken shared-library linking
        print("Running PyAnnote diarization (in-memory)...")
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": samplerate}
        )

        if hasattr(diarization, "speaker_diarization"):
            annotation = diarization.speaker_diarization
        else:
            annotation = diarization

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration >= self.min_segment_duration:
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })

        # Keep overlapping turns: downstream, WhisperX assign_word_speakers
        # resolves each transcript segment to the dominant speaker by summed
        # overlap, so dropping whole turns here only loses crosstalk audio.
        return segments
