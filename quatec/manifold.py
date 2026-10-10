"""Poincare-ball operations used by QuatEC."""

import torch


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class PoincareBall:
    def __init__(self) -> None:
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def proj(self, x: torch.Tensor, c: float) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        max_norm = (1 - self.eps[x.dtype]) / (c**0.5)
        projected = x / norm * max_norm
        return torch.where(norm > max_norm, projected, x)

    def proj_tan0(self, u: torch.Tensor, c: float) -> torch.Tensor:
        del c
        return u

    def expmap0(self, u: torch.Tensor, c: float) -> torch.Tensor:
        sqrt_c = c**0.5
        norm = u.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        return torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)

    def logmap0(self, x: torch.Tensor, c: float) -> torch.Tensor:
        sqrt_c = c**0.5
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        return artanh(sqrt_c * norm) * x / (sqrt_c * norm)

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, c: float
    ) -> torch.Tensor:
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denominator = 1 + 2 * c * xy + c**2 * x2 * y2
        return numerator / denominator.clamp_min(self.min_norm)

    def mobius_matvec(
        self, matrix: torch.Tensor, x: torch.Tensor, c: float
    ) -> torch.Tensor:
        sqrt_c = c**0.5
        x_norm = x.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        mx = x @ matrix.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True).clamp_min(self.min_norm)
        scale = torch.tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm))
        result = scale * mx / (mx_norm * sqrt_c)
        return torch.where(mx_norm > self.min_norm, result, torch.zeros_like(result))

