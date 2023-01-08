# Copyright (c) Zhihao Liang. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn

import semneus.libaccvr as _C


class PositionalEncoding(nn.Module):
    def __init__(self, level: int = 10) -> None:
        super().__init__()
        self.level = level

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        pi = 1.0
        p_transformed = torch.cat(
            [
                torch.cat([torch.sin((2**i) * pi * p), torch.cos((2**i) * pi * p)], dim=-1)
                for i in range(self.level)
            ],
            dim=-1,
        )

        return torch.cat([p, p_transformed], dim=-1)

    def __repr__(self) -> str:
        return f"PositionalEncoding(level={self.level})"


class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dirs: torch.Tensor, sh_degree: int) -> torch.Tensor:
        ctx.save_for_backward(dirs)
        ctx.sh_degree = sh_degree
        outputs = _C.spherical_harmonic_forward(dirs, sh_degree)

        return outputs

    @staticmethod
    def backward(ctx, outputs_grad: torch.Tensor) -> Tuple:
        (dirs,) = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        inputs_grad = _C.spherical_harmonic_backward(outputs_grad, dirs, sh_degree)

        return inputs_grad, None


sphericalharmoic = _SphericalHarmonics.apply


class SphericalEncoding(nn.Module):
    def __init__(self, degree: int = 4) -> None:
        super().__init__()
        self.degree = degree

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        return sphericalharmoic((dirs + 1) / 2, self.degree)

    def __repr__(self) -> str:
        return f"SphericalEncoding(degree={self.degree})"
