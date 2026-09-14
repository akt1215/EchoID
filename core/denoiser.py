import warnings
import numpy as np
import soundfile as sf
import tempfile
import os

DF_SR = 48000  # DeepFilterNet operates at 48kHz


class Denoiser:
    def __init__(self, method="noisereduce", enabled=True):
        self.enabled = enabled
        self.method = method
        self._noisereduce = None
        self._df_model = None
        self._df_state = None

    def _lazy_import_noisereduce(self):
        if self._noisereduce is None:
            import noisereduce as nr
            self._noisereduce = nr

    def _lazy_import_deepfilter(self):
        if self._df_model is None:
            import torch
            from df import enhance, init_df
            model, df_state, _ = init_df()
            self._df_model = model
            self._df_state = df_state
            self._torch = torch
            self._df_enhance = enhance

    def _resample(self, data, orig_sr, target_sr):
        import scipy.signal as ss
        if orig_sr == target_sr:
            return data
        length = int(len(data) * target_sr / orig_sr)
        return ss.resample(data, length)

    def denoise_channel(self, audio_data, sample_rate):
        if not self.enabled:
            return audio_data
        peak = np.max(np.abs(audio_data))
        if peak < 1e-6:
            return audio_data

        if self.method == "deepfilter":
            self._lazy_import_deepfilter()
            # Resample to 48kHz if needed
            if sample_rate != DF_SR:
                resampled = self._resample(audio_data, sample_rate, DF_SR)
            else:
                resampled = audio_data
            inp = self._torch.from_numpy(resampled.astype(np.float32)).unsqueeze(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = self._df_enhance(self._df_model, self._df_state, inp)
            cleaned = out.squeeze(0).numpy().astype(np.float64)
            # Resample back to original sample rate
            if sample_rate != DF_SR:
                cleaned = self._resample(cleaned, DF_SR, sample_rate)
            return cleaned

        # Default: noisereduce
        self._lazy_import_noisereduce()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            reduced = self._noisereduce.reduce_noise(y=audio_data, sr=sample_rate, stationary=False)
        return reduced

    def denoise_file(self, audio_path, target_channel=None):
        if not self.enabled:
            return audio_path
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1 and target_channel is not None:
            channel_data = data[:, target_channel].copy()
        elif len(data.shape) > 1:
            channel_data = np.mean(data, axis=1)
        else:
            channel_data = data.copy()
        cleaned = self.denoise_channel(channel_data, sr)
        fd, tmp = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        sf.write(tmp, cleaned, sr)
        return tmp
