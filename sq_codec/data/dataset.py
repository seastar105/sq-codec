from typing import Optional

import pandas as pd
import torch
from audiotools import AudioSignal
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        tsv_path: str,
        sample_rate: int = 16000,
        excerpt_seconds: float = 3.0,
        audio_norm_scale: float = 0.95,
        normalize_db: Optional[float] = None,
        loudness_cutoff: Optional[float] = None,
    ):
        super().__init__()

        df = pd.read_csv(tsv_path, sep="\t")
        self.paths = df["audio_path"].tolist()
        self.sample_rate = sample_rate

        self.excerpt_seconds = excerpt_seconds
        self.segment_size = int(excerpt_seconds * self.sample_rate)
        self.audio_norm_scale = audio_norm_scale

        self.normalize_db = normalize_db
        self.loudness_cutoff = loudness_cutoff

    def __len__(self):
        return len(self.paths)

    def load_audio(self, path):
        # random crop
        signal = AudioSignal.salient_excerpt(path, loudness_cutoff=self.loudness_cutoff, duration=self.excerpt_seconds)
        if signal.num_channels > 1:
            signal = signal.to_mono()

        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)

        if self.normalize_db is not None:
            signal = signal.normalize(self.normalize_db).ensure_max_of_audio()

        length = signal.shape[-1]
        if length > self.segment_size:
            signal = signal[:, :, : self.segment_size]
        else:
            pad_len = self.segment_size - signal.shape[-1]
            signal = signal.zero_pad(0, pad_len)

        if self.audio_norm_scale < 1.0:
            signal = signal * self.audio_norm_scale

        return signal.audio_data.squeeze(0)

    def __getitem__(self, index):
        path = self.paths[index]
        audio = self.load_audio(path)
        assert audio.ndim == 2, "Dimension is not 2"
        assert audio.shape[-1] == self.segment_size, f"Expect audio has {self.segment_size}, but {audio.shape[-1]}"
        return audio

    def get_full_audio(self, index):
        path = self.paths[index]
        signal = AudioSignal(path)
        if signal.num_channels > 1:
            signal = signal.to_mono()

        if signal.sample_rate != self.sample_rate:
            signal = signal.resample(self.sample_rate)

        if self.normalize_db is not None:
            signal = signal.normalize(self.normalize_db).ensure_max_of_audio()
        return signal.audio_data.squeeze(0)
