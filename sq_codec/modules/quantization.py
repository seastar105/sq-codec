# adapted from https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/finite_scalar_quantization.py
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.cuda.amp import autocast
from torch.nn import Module

# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class ScalarQuantizer(Module):
    def __init__(
        self,
        level: int,
        dim: int,
        codebook_dim: int,
        allowed_dtypes: Tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        fsq: bool = False,
    ):
        super().__init__()
        self.level = level
        self.dim = dim
        self.codebook_dim = codebook_dim

        self.project_in = nn.Linear(self.dim, self.codebook_dim)
        self.project_out = nn.Linear(self.codebook_dim, self.dim)
        self.allowed_dtypes = allowed_dtypes
        self.fsq = fsq

    def fsq_bound(self, z, eps=1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = self.level * (1 + eps) / 2
        offset = 0.5 if self.level % 2 == 0 else 0.0
        shift = torch.atanh(torch.tensor(offset / half_l))
        return (z + shift).tanh() * half_l - offset

    def sq_bound(self, z):
        """Bound `z`, an array of shape (..., d)."""
        return z.tanh() * self.level

    def bound(self, z):
        if self.fsq:
            half_w = self.level // 2
            return self.fsq_bound(z) / half_w
        else:
            return self.sq_bound(z)

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        return quantized / self.level

    def decode(self, z):
        out = self.project_out(z)
        out = rearrange(out, "b t c -> b c t")
        return out

    @autocast(enabled=False)
    def forward(self, z):
        """
        einstein notation
        b - batch
        c - channel
        t - sequence
        """
        assert z.shape[-2] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-2]}"
        orig_dtype = z.dtype
        # make sure allowed dtype before quantizing
        if z.dtype not in self.allowed_dtypes:
            z = z.float()

        z = rearrange(z, "b c t -> b t c")
        z = self.project_in(z)

        latent = self.quantize(z)
        # project out
        out = self.project_out(latent)
        if out.dtype != orig_dtype:
            out = out.type(orig_dtype)
        out = rearrange(out, "b t c -> b c t")
        return out
