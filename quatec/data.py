"""Dataset loading for the CARE graph benchmark."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class QuatECData:
    features: torch.Tensor
    labels: torch.Tensor
    structure_adjacency: torch.Tensor
    active_adjacency: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    test_indices: dict[str, torch.Tensor]
    index_to_ec: dict[int, str]

    def to(self, device: torch.device) -> "QuatECData":
        return QuatECData(
            features=self.features.to(device),
            labels=self.labels.to(device),
            structure_adjacency=self.structure_adjacency.to(device),
            active_adjacency=self.active_adjacency.to(device),
            train_indices=self.train_indices.to(device),
            validation_indices=self.validation_indices.to(device),
            test_indices={
                name: indices.to(device) for name, indices in self.test_indices.items()
            },
            index_to_ec=self.index_to_ec,
        )


def _load_mapping(path: Path) -> dict:
    if path.suffix.lower() != ".json":
        raise ValueError(
            "Mapping files must use JSON (.json); pickle mappings are not supported."
        )
    with path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    if not isinstance(mapping, dict):
        raise ValueError("A mapping file must contain a JSON object.")
    return mapping


def _load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        return sp.load_npz(path).toarray()
    return np.load(path, allow_pickle=False)


def _read_split(path: Path, entry_to_index: dict[str, int]) -> np.ndarray:
    entries: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row_number, row in enumerate(reader):
            if not row:
                continue
            entry = row[0].strip()
            if row_number == 0 and entry not in entry_to_index:
                continue
            entries.append(entry)
    missing = [entry for entry in entries if entry not in entry_to_index]
    if missing:
        raise KeyError(f"{len(missing)} split entries are absent from the index map.")
    return np.asarray([entry_to_index[entry] for entry in entries], dtype=np.int64)


def _row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float64)
    nonzero = row_sum != 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    return sp.diags(inverse) @ matrix


def _ppr(matrix: sp.spmatrix, alpha: float, steps: int) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    identity = sp.eye(matrix.shape[0], format="csr")
    diffusion = alpha * identity
    for _ in range(steps):
        diffusion = (1 - alpha) * diffusion @ matrix + alpha * identity
    return diffusion.tocsr()


def _prepare_adjacency(
    matrix: sp.spmatrix, ppr_alpha: float | None, ppr_steps: int
) -> torch.Tensor:
    if ppr_alpha is not None:
        matrix = _ppr(matrix, ppr_alpha, ppr_steps)
    matrix = _row_normalize(matrix + sp.eye(matrix.shape[0], format="csr"))
    matrix = matrix.tocoo()
    indices = torch.from_numpy(
        np.vstack([matrix.row, matrix.col]).astype(np.int64)
    )
    values = torch.from_numpy(matrix.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def load_data(
    data_dir: str | Path,
    *,
    features_file: str,
    labels_file: str,
    structure_graph_file: str,
    active_graph_file: str,
    entry_map_file: str,
    ec_map_file: str,
    train_split_file: str,
    test_split_files: dict[str, str],
    structure_ppr_alpha: float | None,
    structure_ppr_steps: int,
    active_ppr_alpha: float | None,
    active_ppr_steps: int,
    validation_fraction: float = 0.05,
    split_seed: int = 1234,
) -> QuatECData:
    root = Path(data_dir).expanduser()
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    features = _load_matrix(root / features_file).astype(np.float32)
    labels = _load_matrix(root / labels_file).astype(np.float32)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels must contain the same number of nodes.")

    feature_row_sum = features.sum(axis=1, keepdims=True)
    feature_row_sum[feature_row_sum == 0] = 1
    features = features / feature_row_sum

    entry_to_index = _load_mapping(root / entry_map_file)
    ec_to_index = _load_mapping(root / ec_map_file)
    index_to_ec = {int(index): ec for ec, index in ec_to_index.items()}

    train_indices = _read_split(root / train_split_file, entry_to_index)
    generator = np.random.default_rng(split_seed)
    shuffled = generator.permutation(train_indices)
    validation_size = max(1, round(validation_fraction * len(shuffled)))
    validation_indices = shuffled[:validation_size]
    train_indices = shuffled[validation_size:]

    test_indices = {
        name: torch.from_numpy(_read_split(root / filename, entry_to_index))
        for name, filename in test_split_files.items()
    }

    structure_graph = sp.load_npz(root / structure_graph_file)
    active_graph = sp.load_npz(root / active_graph_file)
    expected_shape = (features.shape[0], features.shape[0])
    if structure_graph.shape != expected_shape or active_graph.shape != expected_shape:
        raise ValueError("Each graph must be square and aligned with the feature rows.")

    return QuatECData(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
        structure_adjacency=_prepare_adjacency(
            structure_graph, structure_ppr_alpha, structure_ppr_steps
        ),
        active_adjacency=_prepare_adjacency(
            active_graph, active_ppr_alpha, active_ppr_steps
        ),
        train_indices=torch.from_numpy(train_indices),
        validation_indices=torch.from_numpy(validation_indices),
        test_indices=test_indices,
        index_to_ec=index_to_ec,
    )
