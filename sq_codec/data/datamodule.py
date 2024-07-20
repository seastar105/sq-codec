import os
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from sq_codec.data.dataset import AudioDataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_tsv_path: str,
        val_tsv_path: str,
        sample_rate: int = 16000,
        excerpt_seconds: float = 3.0,
        audio_norm_scale: float = 0.95,
        normalize_db: Optional[float] = None,
        loudness_cutoff: Optional[float] = None,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()

        self.train_tsv_path = train_tsv_path
        self.val_tsv_path = val_tsv_path

        self.train_ds: Optional[AudioDataset] = None
        self.val_ds: Optional[AudioDataset] = None

        self.sample_rate = sample_rate
        self.excerpt_seconds = excerpt_seconds
        self.audio_norm_scale = audio_norm_scale
        self.normalize_db = normalize_db
        self.loudness_cutoff = loudness_cutoff

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        assert os.path.exists(self.train_tsv_path)
        assert os.path.exists(self.val_tsv_path)

    def setup(self, stage: str):
        if stage != "fit":
            raise ValueError(f"Stage {stage} not supported")

        self.train_ds = AudioDataset(
            self.train_tsv_path,
            self.sample_rate,
            self.excerpt_seconds,
            self.audio_norm_scale,
            self.normalize_db,
            self.loudness_cutoff,
        )
        self.val_ds = AudioDataset(
            self.val_tsv_path,
            self.sample_rate,
            self.excerpt_seconds,
            self.audio_norm_scale,
            self.normalize_db,
            self.loudness_cutoff,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
