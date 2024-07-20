import math
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from sq_codec.modules.quantization import ScalarQuantizer
from sq_codec.modules.seanet import SEANetDecoder, SEANetEncoder


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SQCodec(nn.Module):
    """
    Args:
        n_filters (int): n_filters (int): Base width for the model.
        D (int): Intermediate representation dimension.
        ratios (Sequence[int]): downsampling factors, whose multiplication is the hop size.
        sample_rate (int): wave sampling rate.
    """

    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        level: int = 9,
        codebook_dim: int = 32,
        ratios: Sequence[int] = [2, 2, 4, 4, 5],
        sample_rate: int = 16000,
        causal: bool = False,
        encoder_grad_mult: float = 1.0,
        fsq: bool = False,
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        self.sample_rate = sample_rate

        self.encoder = SEANetEncoder(n_filters=n_filters, dimension=D, ratios=ratios, causal=causal)
        self.quantizer = ScalarQuantizer(level=level, dim=D, codebook_dim=codebook_dim, fsq=fsq)
        self.decoder = SEANetDecoder(n_filters=n_filters, dimension=D, ratios=ratios, causal=causal)

        self.encoder_grad_mult = encoder_grad_mult

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def forward(self, x: torch.Tensor):
        e = self.encoder(x)
        if self.encoder_grad_mult != 1.0:
            e = GradMultiply.apply(e, self.encoder_grad_mult)
        quantized = self.quantizer(e)
        o = self.decoder(quantized)
        return o

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        latent = self.quantizer(e)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(latent)
        o = self.decoder(quantized)
        return o
