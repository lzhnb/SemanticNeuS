# Copyright (c) Zhihao Liang. All rights reserved.
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .embedder import PositionalEncoding, SphericalEncoding


class SemanticNeuS(nn.Module):
    def __init__(
        self,
        level: int = 10,
        sh_degree: int = 4,
        sdf_n_layers: int = 8,
        sdf_skip_in: Sequence[int] = (4,),
        rgb_n_layers: int = 4,
        dim_geo: int = 64,
        init_val: float = 0.3,
        inside_outside: bool = True,
        **kwargs,
    ) -> None:
        """initialize the Semantic NeuS

        Args:
            level (int, optional): level of positional encoding. Defaults to 10.
            sdf_n_layers (int, optional): the numbers of layers in SDF network. Defaults to 8.
            sdf_skip_in (Sequence[int], optional): the ids of skip layers in SDF network. Defaults to (4,).
            rgb_n_layers (int, optional): the numbers of layers in rgb network. Defaults to 4.
            dim_geo (int, optional): dimension of the geometry feature. Defaults to 64.
            init_val (float, optional): init val of single variance. Defaults to 0.3.
            inside_outside (bool, optional): inverse the SDF value for the scene. Defaults to True.
        """
        super().__init__()

        self.sdf_net: SDFNetwork
        self.rgb_net: RGBNetwork
        self.deviation_net = SingleVarianceNetwork(init_val)

        self.sdf_dim_output = dim_geo + 1
        self.rgb_dim_input = 3 + 3 + sh_degree**2 + dim_geo

        self.init_sdf_net(
            dim_hidden=dim_geo,
            n_layers=sdf_n_layers,
            skip_in=sdf_skip_in,
            level=level,
            inside_outside=inside_outside,
            bias=1.0,
        )
        self.init_rgb_net(dim_hidden=dim_geo, n_layers=rgb_n_layers, sh_degree=sh_degree)

    def forward(
        self,
        pos_input: torch.Tensor,
        dir_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """network forward

        Args:
            pos_input (torch.Tensor, [n, 3]): positions of input sample points
            dir_input (torch.Tensor, [n, 3]): directions of input sample points

        Returns:
            colors (torch.Tensor, [n, 3]): predicted colors
            sdf (torch.Tensor, [n]): predicted sdf
            normals (torch.Tensor, [n, 3]): predicted normals
            inv_s (torch.Tensor, []): optimized global inv_s
        """
        with torch.enable_grad():
            x, normals = self.sdf_net(pos_input, with_normals=True)  # [num_samples, 1 + geo_dim]
        sdf = x[..., :1]
        geo_feat = x[..., 1:]

        # Note that tcnn SH encoding requires inputs to be in [0, 1]
        colors = self.rgb_net(pos_input, dir_input, normals, geo_feat)  # [B, output_dim]

        inv_s = self.deviation_net(torch.zeros([1, 3])).clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(len(pos_input), 1)

        if colors.shape[-1] == 3:  # for rgb output, use sigmoid function
            colors = torch.sigmoid(colors)

        return colors, sdf, normals, inv_s

    def init_sdf_net(self, **kwargs) -> None:
        kwargs.update(dict(dim_output=self.sdf_dim_output))
        self.sdf_net = SDFNetwork(**kwargs)

    def init_rgb_net(self, **kwargs) -> None:
        kwargs.update(dict(dim_input=self.rgb_dim_input))
        self.rgb_net = RGBNetwork(**kwargs)


class SDFNetwork(nn.Module):
    def __init__(
        self,
        dim_output: int,
        dim_hidden: int = 256,
        n_layers: int = 8,
        skip_in: Sequence[int] = (4,),
        level: int = 10,
        geometric_init: bool = True,
        weight_norm: bool = True,
        inside_outside: bool = False,
        bias: float = 1.0,
    ) -> None:
        super().__init__()

        self.frequency_encoder = PositionalEncoding(level)

        input_ch = level * 6 + 3

        dims = [input_ch] + [dim_hidden for _ in range(n_layers)] + [dim_output]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif level > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif level > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if l < self.num_layers - 2:
                setattr(self, "act" + str(l), nn.Softplus(beta=100))
                # setattr(self, "act" + str(l), nn.ReLU(inplace=True))

    def forward(self, inputs: torch.Tensor, with_normals: bool = True) -> torch.Tensor:
        if with_normals:
            inputs.requires_grad_(True)

        x = self.frequency_encoder(inputs)
        x_enc = x
        for l in range(0, self.num_layers - 1):
            lin: nn.Linear = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, x_enc], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = getattr(self, "act" + str(l))(x)

        if not with_normals:
            return x
        else:
            sdf = x[:, :1]
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            normals = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return x, normals

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, with_normals=False)[:, :1]


class RGBNetwork(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int = 256,
        n_layers: int = 4,
        sh_degree: int = 4,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        self.spherical_encoder = SphericalEncoding(sh_degree)
        dims = [dim_input] + [dim_hidden for _ in range(n_layers)] + [3]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if l < self.num_layers - 2:
                setattr(self, "act" + str(l), nn.ReLU(inplace=True))

    def forward(
        self,
        points: torch.Tensor,
        dirs: torch.Tensor,
        normals: torch.Tensor,
        geo_feature: torch.Tensor,
    ) -> torch.Tensor:
        dirs_encoded = self.spherical_encoder(dirs)
        rendering_input = torch.cat([points, dirs_encoded, normals, geo_feature], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = getattr(self, "act" + str(l))(x)

        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val: float) -> None:
        super().__init__()
        self.variance: nn.Parameter
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return self.variance.new_ones([len(x)]) * torch.exp(self.variance * 10.0)

    def __repr__(self) -> str:
        return f"SingleVarianceNetwork(variance: {self.variance.item()})"
