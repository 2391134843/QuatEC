"""QuatEC dual-graph classifier."""

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import QuaternionHyperbolicGraphConv, QuaternionRotation
from .manifold import PoincareBall


class QuatECOutput(TypedDict):
    structure_embedding: torch.Tensor
    active_embedding: torch.Tensor
    structure_logits: torch.Tensor
    active_logits: torch.Tensor
    logits: torch.Tensor


class QuatECEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        curvature: float,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_rotation: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least two.")
        self.curvature = curvature
        self.manifold = PoincareBall()
        self.input_rotation = (
            QuaternionRotation(input_dim) if input_rotation else None
        )
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            QuaternionHyperbolicGraphConv(
                dimensions[index],
                dimensions[index + 1],
                curvature,
                dropout,
                F.relu,
            )
            for index in range(len(dimensions) - 1)
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if self.input_rotation is not None:
            features = self.input_rotation(features)
        x = self.manifold.proj(
            self.manifold.expmap0(features, self.curvature), self.curvature
        )
        for layer in self.layers:
            x = layer(x, adjacency)
        return x


class TangentDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, curvature: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.curvature = curvature
        self.manifold = PoincareBall()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        tangent = self.manifold.logmap0(embedding, self.curvature)
        return self.linear(tangent)


class QuatEC(nn.Module):
    """Dual-graph quaternion hyperbolic classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        curvature: float,
        structure_weight: float,
        active_weight: float,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_rotation: bool = False,
        alignment_rotation: bool = True,
    ) -> None:
        super().__init__()
        if abs(structure_weight + active_weight - 1.0) > 1e-8:
            raise ValueError("The two prediction weights must sum to one.")
        encoder_options = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            curvature=curvature,
            num_layers=num_layers,
            dropout=dropout,
            input_rotation=input_rotation,
        )
        self.structure_encoder = QuatECEncoder(**encoder_options)
        self.active_encoder = QuatECEncoder(**encoder_options)
        self.structure_decoder = TangentDecoder(
            hidden_dim, num_classes, curvature
        )
        self.active_decoder = TangentDecoder(hidden_dim, num_classes, curvature)
        self.alignment_rotation = (
            QuaternionRotation(hidden_dim) if alignment_rotation else None
        )
        self.structure_weight = structure_weight
        self.active_weight = active_weight
        self.hidden_dim = hidden_dim

    def forward(
        self,
        features: torch.Tensor,
        structure_adjacency: torch.Tensor,
        active_adjacency: torch.Tensor,
    ) -> QuatECOutput:
        structure_embedding = self.structure_encoder(
            features, structure_adjacency
        )
        active_embedding = self.active_encoder(features, active_adjacency)
        structure_logits = self.structure_decoder(structure_embedding)
        active_logits = self.active_decoder(active_embedding)
        logits = (
            self.structure_weight * structure_logits
            + self.active_weight * active_logits
        )
        return {
            "structure_embedding": structure_embedding,
            "active_embedding": active_embedding,
            "structure_logits": structure_logits,
            "active_logits": active_logits,
            "logits": logits,
        }

    def loss(
        self,
        output: QuatECOutput,
        labels: torch.Tensor,
        indices: torch.Tensor,
        alignment_weight: float,
        decorrelation_weight: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structure_loss = F.binary_cross_entropy_with_logits(
            output["structure_logits"][indices], labels[indices]
        )
        active_loss = F.binary_cross_entropy_with_logits(
            output["active_logits"][indices], labels[indices]
        )
        structure_embedding = output["structure_embedding"]
        active_embedding = output["active_embedding"]
        aligned_structure = (
            self.alignment_rotation(structure_embedding)
            if self.alignment_rotation is not None
            else structure_embedding
        )
        alignment_loss = (
            aligned_structure - active_embedding
        ).norm(p=2, dim=-1).mean()
        decorrelation_loss = torch.zeros(
            (), device=structure_embedding.device, dtype=structure_embedding.dtype
        )
        if decorrelation_weight:
            identity = torch.eye(
                self.hidden_dim,
                device=structure_embedding.device,
                dtype=structure_embedding.dtype,
            )
            decorrelation_loss = (
                torch.linalg.matrix_norm(
                    structure_embedding.T @ structure_embedding - identity
                )
                + torch.linalg.matrix_norm(
                    active_embedding.T @ active_embedding - identity
                )
            ) / self.hidden_dim**2
        total = (
            structure_loss
            + active_loss
            + alignment_weight * alignment_loss
            + decorrelation_weight * decorrelation_loss
        )
        return total, {
            "structure": structure_loss,
            "active": active_loss,
            "alignment": alignment_loss,
            "decorrelation": decorrelation_loss,
        }

