"""Train and evaluate QuatEC."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from quatec import QuatEC
from quatec.config import get_variant
from quatec.data import QuatECData, load_data
from quatec.metrics import hierarchical_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a QuatEC model.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--variant", choices=["S", "D", "M"], default="S")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--min-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--decorrelation-weight", type=float, default=0.0)

    parser.add_argument("--features-file", default="features.npy")
    parser.add_argument("--labels-file", default="labels.npz")
    parser.add_argument("--structure-graph", default="structure_graph.npz")
    parser.add_argument("--active-graph", default="active_site_graph.npz")
    parser.add_argument("--entry-map", default="entry2index.json")
    parser.add_argument("--ec-map", default="ec2idx.json")
    parser.add_argument("--train-split", default="train.tsv")
    parser.add_argument("--lt30-split", default="test_lt30.tsv")
    parser.add_argument("--mid-split", default="test_30_50.tsv")
    parser.add_argument("--difficult-split", default="test_difficult.tsv")
    parser.add_argument("--multifunction-split", default="test_multifunction.tsv")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(data: QuatECData, variant, args) -> QuatEC:
    return QuatEC(
        input_dim=data.features.shape[1],
        num_classes=data.labels.shape[1],
        hidden_dim=variant.hidden_dim,
        curvature=variant.curvature,
        structure_weight=variant.structure_weight,
        active_weight=variant.active_weight,
        num_layers=args.num_layers,
        dropout=args.dropout,
        input_rotation=variant.input_rotation,
        alignment_rotation=variant.alignment_rotation,
    )


@torch.no_grad()
def validation_score(model: QuatEC, data: QuatECData) -> float:
    model.eval()
    output = model(
        data.features, data.structure_adjacency, data.active_adjacency
    )
    logits = output["logits"][data.validation_indices]
    labels = data.labels[data.validation_indices]
    single_label = labels.sum(dim=1) == 1
    if single_label.any():
        predictions = logits[single_label].argmax(dim=1)
        targets = labels[single_label].argmax(dim=1)
        return float((predictions == targets).float().mean())
    return -float(F.binary_cross_entropy_with_logits(logits, labels))


@torch.no_grad()
def evaluate(model: QuatEC, data: QuatECData) -> dict[str, dict[str, float]]:
    model.eval()
    output = model(
        data.features, data.structure_adjacency, data.active_adjacency
    )
    return {
        split: hierarchical_accuracy(
            output["logits"][indices], data.labels[indices], data.index_to_ec
        )
        for split, indices in data.test_indices.items()
    }


def train_once(
    data: QuatECData,
    variant,
    args,
    seed: int,
    device: torch.device,
) -> tuple[QuatEC, dict[str, dict[str, float]]]:
    set_seed(seed)
    model = build_model(data, variant, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=variant.learning_rate)
    epochs = args.epochs if args.epochs is not None else variant.epochs
    min_epochs = (
        args.min_epochs if args.min_epochs is not None else variant.min_epochs
    )
    patience = args.patience if args.patience is not None else variant.patience

    best_score = -float("inf")
    best_state = None
    stale_evaluations = 0
    progress = trange(epochs, desc=f"{variant.name} seed={seed}", dynamic_ncols=True)
    for epoch in progress:
        model.train()
        optimizer.zero_grad()
        output = model(
            data.features, data.structure_adjacency, data.active_adjacency
        )
        loss, _ = model.loss(
            output,
            data.labels,
            data.train_indices,
            alignment_weight=variant.alignment_weight,
            decorrelation_weight=args.decorrelation_weight,
        )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % args.eval_every == 0:
            current_score = validation_score(model, data)
            progress.set_postfix(
                train=f"{float(loss):.4f}", validation=f"{current_score:.4f}"
            )
            if current_score > best_score:
                best_score = current_score
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if epoch + 1 >= min_epochs and stale_evaluations >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, evaluate(model, data)


def summarize(
    runs: list[dict[str, dict[str, float]]]
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    if not runs:
        return summary
    for split in runs[0]:
        summary[split] = {}
        for level in runs[0][split]:
            values = np.asarray([run[split][level] for run in runs])
            summary[split][level] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be at least one.")
    if args.eval_every < 1:
        raise ValueError("eval_every must be at least one.")
    variant = get_variant(args.variant)
    device = resolve_device(args.device)
    test_files = {
        "low_similarity": args.lt30_split,
        "medium_similarity": args.mid_split,
        "difficult": args.difficult_split,
        "multifunction": args.multifunction_split,
    }
    data = load_data(
        args.data_dir,
        features_file=args.features_file,
        labels_file=args.labels_file,
        structure_graph_file=args.structure_graph,
        active_graph_file=args.active_graph,
        entry_map_file=args.entry_map,
        ec_map_file=args.ec_map,
        train_split_file=args.train_split,
        test_split_files=test_files,
        structure_ppr_alpha=variant.structure_ppr_alpha,
        structure_ppr_steps=variant.structure_ppr_steps,
        active_ppr_alpha=variant.active_ppr_alpha,
        active_ppr_steps=variant.active_ppr_steps,
        validation_fraction=args.validation_fraction,
        split_seed=args.seed,
    ).to(device)
    if data.features.shape[1] % 4:
        raise ValueError("The feature dimension must be divisible by four.")

    output_dir = Path(args.output_dir) / variant.name
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        model, metrics = train_once(data, variant, args, seed, device)
        run_metrics.append(metrics)
        torch.save(
            {
                "model": model.state_dict(),
                "variant": variant.to_dict(),
                "input_dim": data.features.shape[1],
                "num_classes": data.labels.shape[1],
                "num_layers": args.num_layers,
            },
            output_dir / f"model_seed_{seed}.pt",
        )
        with (output_dir / f"metrics_seed_{seed}.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(metrics, stream, indent=2)

    result = {
        "model": variant.name,
        "seeds": [args.seed + index for index in range(args.repeats)],
        "metrics": summarize(run_metrics),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
