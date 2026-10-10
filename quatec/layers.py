"""Quaternion hyperbolic layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold import PoincareBall


def hamilton_product(
    left: torch.Tensor, right: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    lr, li, lj, lk = torch.chunk(left, 4, dim=dim)
    rr, ri, rj, rk = torch.chunk(right, 4, dim=dim)
    real = lr * rr - li * ri - lj * rj - lk * rk
    imag_i = lr * ri + li * rr + lj * rk - lk * rj
    imag_j = lr * rj - li * rk + lj * rr + lk * ri
    imag_k = lr * rk + li * rj - lj * ri + lk * rr
    return torch.cat([real, imag_i, imag_j, imag_k], dim=dim)


def normalize_quaternion(
    quaternion: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> torch.Tensor:
    real, imag_i, imag_j, imag_k = torch.chunk(quaternion, 4, dim=dim)
    norm = torch.sqrt(
        real.square() + imag_i.square() + imag_j.square() + imag_k.square()
    ).clamp_min(eps)
    return torch.cat(
        [real / norm, imag_i / norm, imag_j / norm, imag_k / norm], dim=dim
    )


def assemble_quaternion_weight(
    wr: torch.Tensor, wi: torch.Tensor, wj: torch.Tensor, wk: torch.Tensor
) -> torch.Tensor:
    row_1 = torch.cat([wr, -wi, -wj, -wk], dim=1)
    row_2 = torch.cat([wi, wr, -wk, wj], dim=1)
    row_3 = torch.cat([wj, wk, wr, -wi], dim=1)
    row_4 = torch.cat([wk, -wj, wi, wr], dim=1)
    return torch.cat([row_1, row_2, row_3, row_4], dim=0)


class QuaternionWeight(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if in_features % 4 or out_features % 4:
            raise ValueError("Quaternion feature dimensions must be divisible by four.")
        in_quaternions = in_features // 4
        out_quaternions = out_features // 4
        shape = (out_quaternions, in_quaternions)
        self.wr = nn.Parameter(torch.empty(shape))
        self.wi = nn.Parameter(torch.empty(shape))
        self.wj = nn.Parameter(torch.empty(shape))
        self.wk = nn.Parameter(torch.empty(shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_out, fan_in = self.wr.shape
        scale = 1.0 / math.sqrt(2.0 * (fan_in + fan_out))
        modulus = torch.sqrt(
            -2.0
            * scale**2
            * torch.log(torch.rand_like(self.wr).clamp_min(1e-12))
        )
        phase = torch.empty_like(self.wr).uniform_(-math.pi, math.pi)
        axis = torch.empty(
            (3, *self.wr.shape), device=self.wr.device, dtype=self.wr.dtype
        ).uniform_(-1.0, 1.0)
        axis = axis / axis.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
        with torch.no_grad():
            self.wr.copy_(modulus * phase.cos())
            self.wi.copy_(modulus * axis[0] * phase.sin())
            self.wj.copy_(modulus * axis[1] * phase.sin())
            self.wk.copy_(modulus * axis[2] * phase.sin())

    def forward(self) -> torch.Tensor:
        return assemble_quaternion_weight(self.wr, self.wi, self.wj, self.wk)


class QuaternionMobiusLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        curvature: float,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = QuaternionWeight(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.curvature = curvature
        self.dropout = dropout
        self.manifold = PoincareBall()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.dropout(self.weight(), self.dropout, training=self.training)
        output = self.manifold.mobius_matvec(weight, x, self.curvature)
        output = self.manifold.proj(output, self.curvature)
        if self.bias is not None:
            tangent_bias = self.manifold.proj_tan0(
                self.bias.view(1, -1), self.curvature
            )
            hyperbolic_bias = self.manifold.expmap0(tangent_bias, self.curvature)
            output = self.manifold.mobius_add(
                output, hyperbolic_bias, self.curvature
            )
            output = self.manifold.proj(output, self.curvature)
        return output


class QuaternionHyperbolicGraphConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        curvature: float,
        dropout: float,
        activation,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = QuaternionMobiusLinear(
            in_features, out_features, curvature, dropout, bias
        )
        self.curvature = curvature
        self.activation = activation
        self.manifold = PoincareBall()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        tangent = self.manifold.logmap0(x, self.curvature)
        if adjacency.is_sparse:
            tangent = torch.sparse.mm(adjacency, tangent)
        else:
            tangent = adjacency @ tangent
        x = self.manifold.proj(
            self.manifold.expmap0(tangent, self.curvature), self.curvature
        )
        tangent = self.activation(self.manifold.logmap0(x, self.curvature))
        return self.manifold.proj(
            self.manifold.expmap0(tangent, self.curvature), self.curvature
        )


class QuaternionRotation(nn.Module):
    def __init__(self, dim: int, init_scale: float = 0.1) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError("Quaternion feature dimensions must be divisible by four.")
        rotation = torch.empty(dim).uniform_(-init_scale, init_scale)
        rotation[: dim // 4] += 1.0
        self.rotation = nn.Parameter(rotation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hamilton_product(x, normalize_quaternion(self.rotation))

