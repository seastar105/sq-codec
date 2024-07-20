# from https://github.com/yangdongchao/UniAudio/blob/main/codec/losses/generator_loss.py
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from sq_codec.data.hifigan_mel import mel_spectrogram
from sq_codec.losses.basic_loss import (
    FeatureMatchLoss,
    MSEGLoss,
    MultiResolutionSTFTLoss,
)
from sq_codec.modules.pqmf import PQMF


class BasicGeneratorLoss(nn.Module):
    def __init__(self, fm_weight: float):
        super().__init__()
        self.fm_weight = fm_weight

        self.adv_criterion = MSEGLoss()
        self.feature_match_criterion = FeatureMatchLoss()

        self.use_mel_loss = False
        self.mel_loss_weight = 0

    def forward(
        self,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        output_real: Dict[str, torch.Tensor],
        output_fake: Dict[str, torch.Tensor],
        fmap_real: Optional[Dict[str, torch.Tensor]] = None,
        fmap_fake: Optional[Dict[str, torch.Tensor]] = None,
        use_adv_loss: bool = True,
    ):
        """
        Args:
            targets: ground-truth waveforms.
            outputs: generated waveforms.
            output_real: logits from discriminators on real waveforms.
            output_fake: logits from discriminators on generated/fake waveforms.
            fmap_real: feature mappings of real waveforms.
            fmap_fake: feature mappings of generated/fake waveforms.
        """
        g_loss = 0
        g_loss_items = {}

        if use_adv_loss:
            for key in output_fake.keys():
                adv_loss_item = self.adv_criterion(output_fake[key])
                g_loss += adv_loss_item
                g_loss_items[f"Train/G_adv_{key}"] = adv_loss_item.item()

                assert fmap_real is not None and fmap_fake is not None
                fmap_loss_item = self.feature_match_criterion(fmap_real[key], fmap_fake[key]) * self.fm_weight
                g_loss += fmap_loss_item
                g_loss_items[f"Train/G_fm_{key}"] = fmap_loss_item.item() / self.fm_weight

        # UniAudio config do not use here
        if self.use_mel_loss:
            # hps_mel_scale_loss = (
            #     self.config.mel_scale_loss
            #     if isinstance(self.config.mel_scale_loss, list)
            #     else [self.config.mel_scale_loss]
            # )
            hps_mel_scale_loss = []

            for i, _hps_mel_scale_loss in enumerate(hps_mel_scale_loss):
                outputs_mel = mel_spectrogram(outputs.squeeze(1), **_hps_mel_scale_loss)
                target_mel = mel_spectrogram(targets.squeeze(1), **_hps_mel_scale_loss)
                mel_loss = F.l1_loss(outputs_mel, target_mel.detach()) * self.mel_loss_weight
                g_loss += mel_loss
                g_loss_items[f"Train/G_mel_loss_{i}"] = mel_loss.item() / self.mel_loss_weight

        return g_loss, g_loss_items


class GeneratorSTFTLoss(BasicGeneratorLoss):
    def __init__(
        self,
        fm_weight: float,
        full_weight: float,
        sub_weight: float,
        full_fft_sizes: list[int] = [512, 1024, 2048],
        full_win_sizes: list[int] = [480, 960, 1200],
        full_hop_sizes: list[int] = [120, 240, 300],
        sub_num_bands: int = 6,
        sub_fft_sizes: list[int] = [128, 256, 256],
        sub_win_sizes: list[int] = [80, 120, 200],
        sub_hop_sizes: list[int] = [20, 40, 50],
    ):
        super().__init__(fm_weight)
        self.full_weight = full_weight
        self.stft_full_criterion = MultiResolutionSTFTLoss(full_fft_sizes, full_win_sizes, full_hop_sizes)

        self.sub_weight = sub_weight
        self.pqmf = PQMF(sub_num_bands)
        self.stft_sub_criterion = MultiResolutionSTFTLoss(sub_fft_sizes, sub_win_sizes, sub_hop_sizes)

    def forward(self, targets, outputs, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss: bool = True):
        g_loss, g_loss_items = super().forward(
            targets, outputs, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss=use_adv_loss
        )

        # Optional: full-band STFT Loss
        sc_full_loss, mg_full_loss = self.stft_full_criterion(outputs.squeeze(1), targets.squeeze(1))
        g_loss = g_loss + self.full_weight * (sc_full_loss + mg_full_loss)
        g_loss_items["Train/G_sc_full"] = sc_full_loss.item()
        g_loss_items["Train/G_mg_full"] = mg_full_loss.item()

        # Optional: sub-band STFT Loss
        targets_sub = self.pqmf.analysis(targets)
        outputs_sub = self.pqmf.analysis(outputs)
        size = outputs_sub.size(-1)
        outputs_sub_view = outputs_sub.view(-1, size)
        targets_sub_view = targets_sub.view(-1, size)

        sc_sub_loss, mg_sub_loss = self.stft_sub_criterion(outputs_sub_view, targets_sub_view)
        g_loss = g_loss + self.sub_weight * (sc_sub_loss + mg_sub_loss)
        g_loss_items["Train/G_sc_sub"] = sc_sub_loss.item()
        g_loss_items["Train/G_mg_sub"] = mg_sub_loss.item()

        return g_loss, g_loss_items
