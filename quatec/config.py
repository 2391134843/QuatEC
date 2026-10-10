"""Configuration for the three QuatEC variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VariantConfig:
    name: str
    hidden_dim: int
    curvature: float
    structure_weight: float
    active_weight: float
    alignment_weight: float
    learning_rate: float
    epochs: int
    min_epochs: int
    patience: int
    structure_ppr_alpha: float | None
    structure_ppr_steps: int
    active_ppr_alpha: float | None
    active_ppr_steps: int
    input_rotation: bool
    alignment_rotation: bool

    def to_dict(self) -> dict:
        return asdict(self)


VARIANTS = {
    "S": VariantConfig(
        name="QuatEC-S",
        hidden_dim=1792,
        curvature=4.75,
        structure_weight=0.80,
        active_weight=0.20,
        alignment_weight=1e-3,
        learning_rate=1e-2,
        epochs=20_000,
        min_epochs=8_000,
        patience=2_000,
        structure_ppr_alpha=None,
        structure_ppr_steps=0,
        active_ppr_alpha=0.95,
        active_ppr_steps=4,
        input_rotation=False,
        alignment_rotation=True,
    ),
    "D": VariantConfig(
        name="QuatEC-D",
        hidden_dim=2048,
        curvature=3.0,
        structure_weight=0.20,
        active_weight=0.80,
        alignment_weight=1e-4,
        learning_rate=1e-2,
        epochs=10_000,
        min_epochs=3_000,
        patience=1_000,
        structure_ppr_alpha=None,
        structure_ppr_steps=0,
        active_ppr_alpha=0.80,
        active_ppr_steps=4,
        input_rotation=False,
        alignment_rotation=True,
    ),
    "M": VariantConfig(
        name="QuatEC-M",
        hidden_dim=1024,
        curvature=2.5,
        structure_weight=0.35,
        active_weight=0.65,
        alignment_weight=1e-4,
        learning_rate=5e-3,
        epochs=20_000,
        min_epochs=8_000,
        patience=2_000,
        structure_ppr_alpha=0.80,
        structure_ppr_steps=1,
        active_ppr_alpha=0.80,
        active_ppr_steps=2,
        input_rotation=True,
        alignment_rotation=True,
    ),
}


def get_variant(name: str) -> VariantConfig:
    key = name.upper()
    if key not in VARIANTS:
        choices = ", ".join(VARIANTS)
        raise ValueError(f"Unknown QuatEC variant {name!r}; choose one of {choices}.")
    return VARIANTS[key]
