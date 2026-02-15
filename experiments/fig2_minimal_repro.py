import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: torch. Install PyTorch first, then rerun.\n"
        "CPU example: pip install torch torchvision torchaudio\n"
        "CUDA example: see https://pytorch.org/get-started/locally/"
    ) from exc


@dataclass
class ExperimentConfig:
    n_samples: int
    n_clusters: int
    overlaps: List[float]
    repeats: int
    epochs: int
    embedding_dim: int
    fusing_iteration: int
    neighbor_size: int
    clustering: str
    random_state: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_views(
    n_samples: int,
    n_clusters: int,
    feature_dims: Tuple[int, int, int] = (600, 500, 700),
    latent_dim: int = 20,
    effect: float = 0.1,
    noise: float = 0.8,
    random_state: int = 42,
) -> Tuple[List[pd.DataFrame], pd.Series]:
    rng = np.random.default_rng(random_state)

    labels = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
    if labels.shape[0] < n_samples:
        labels = np.concatenate(
            [labels, rng.choice(np.arange(n_clusters), size=n_samples - labels.shape[0])]
        )
    rng.shuffle(labels)

    centers = rng.normal(0.0, effect, size=(n_clusters, latent_dim))
    latent = centers[labels] + rng.normal(0.0, 1.0, size=(n_samples, latent_dim))

    sample_ids = [f"s{i:04d}" for i in range(n_samples)]
    view_names = ["omics1", "omics2", "omics3"]

    views: List[pd.DataFrame] = []
    for name, out_dim in zip(view_names, feature_dims):
        proj = rng.normal(0.0, 1.0, size=(latent_dim, out_dim))
        x = latent @ proj + rng.normal(0.0, noise, size=(n_samples, out_dim))
        cols = [f"{name}_f{j:04d}" for j in range(out_dim)]
        views.append(pd.DataFrame(x, index=sample_ids, columns=cols))

    y = pd.Series(labels, index=sample_ids, name="label")
    return views, y


def load_intersim_csv(input_dir: str) -> Tuple[List[pd.DataFrame], pd.Series]:
    view1 = pd.read_csv(os.path.join(input_dir, "view1.csv"), index_col=0)
    view2 = pd.read_csv(os.path.join(input_dir, "view2.csv"), index_col=0)
    view3 = pd.read_csv(os.path.join(input_dir, "view3.csv"), index_col=0)
    labels = pd.read_csv(os.path.join(input_dir, "labels.csv"), index_col=0).iloc[:, 0]
    labels.name = "label"
    return [view1, view2, view3], labels


def apply_missing_one_complete(
    views: List[pd.DataFrame], overlap: float, rng: np.random.Generator
) -> List[pd.DataFrame]:
    sample_ids = views[0].index.to_numpy()
    n = len(sample_ids)
    keep_n = max(2, int(round(overlap * n)))

    kept_1 = set(rng.choice(sample_ids, size=keep_n, replace=False).tolist())
    kept_2 = set(rng.choice(sample_ids, size=keep_n, replace=False).tolist())

    out = [
        views[0].copy(),
        views[1].loc[sorted(kept_1)].copy(),
        views[2].loc[sorted(kept_2)].copy(),
    ]
    return out


def apply_missing_all_incomplete(
    views: List[pd.DataFrame], overlap: float, rng: np.random.Generator
) -> List[pd.DataFrame]:
    sample_ids = views[0].index.to_numpy()
    n = len(sample_ids)
    n_common = max(2, int(round(overlap * n)))

    common = set(rng.choice(sample_ids, size=n_common, replace=False).tolist())
    remaining = [sid for sid in sample_ids if sid not in common]
    rng.shuffle(remaining)

    chunks = [remaining[i::3] for i in range(3)]
    out = []
    for i in range(3):
        ids = sorted(list(common) + chunks[i])
        out.append(views[i].loc[ids].copy())
    return out


def cluster_and_score(
    embeddings: pd.DataFrame,
    graph: np.ndarray,
    labels: pd.Series,
    n_clusters: int,
    mode: str,
    seed: int,
) -> float:
    labels = labels.loc[embeddings.index]

    if mode == "spectral":
        pred = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=seed,
        ).fit_predict(graph)
    else:
        pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(
            embeddings.values
        )

    return normalized_mutual_info_score(labels.values, pred)


def run_single_case(
    views_full: List[pd.DataFrame],
    labels: pd.Series,
    scenario: str,
    overlap: float,
    cfg: ExperimentConfig,
    seed: int,
) -> float:
    from integrao.integrater import integrao_integrater

    rng = np.random.default_rng(seed)
    if scenario == "one_complete":
        views = apply_missing_one_complete(views_full, overlap, rng)
    elif scenario == "all_incomplete":
        views = apply_missing_all_incomplete(views_full, overlap, rng)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    model = integrao_integrater(
        datasets=views,
        modalities_name_list=["omics1", "omics2", "omics3"],
        neighbor_size=cfg.neighbor_size,
        embedding_dims=cfg.embedding_dim,
        fusing_iteration=cfg.fusing_iteration,
        alighment_epochs=cfg.epochs,
        random_state=seed,
    )

    model.network_diffusion()
    embeddings, graph, _ = model.unsupervised_alignment()
    nmi = cluster_and_score(
        embeddings=embeddings,
        graph=graph,
        labels=labels,
        n_clusters=cfg.n_clusters,
        mode=cfg.clustering,
        seed=seed,
    )
    return nmi


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["scenario", "overlap"], as_index=False)["nmi"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["scenario", "overlap", "nmi_mean", "nmi_std"]
    return agg.sort_values(["scenario", "overlap"]).reset_index(drop=True)


def plot_curve(summary: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(8, 5))
    for scenario, grp in summary.groupby("scenario"):
        x = grp["overlap"].values
        y = grp["nmi_mean"].values
        e = grp["nmi_std"].values
        plt.plot(x, y, marker="o", label=scenario)
        plt.fill_between(x, y - e, y + e, alpha=0.2)
    plt.xlabel("Overlap Ratio")
    plt.ylabel("NMI")
    plt.title("IntegrAO NMI vs Overlap (Minimal Reproduction)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Fig.2-like IntegrAO reproduction on InterSim/synthetic data."
    )
    parser.add_argument("--data-source", choices=["synthetic", "intersim_csv"], default="synthetic")
    parser.add_argument(
        "--intersim-dir",
        type=str,
        default="",
        help="Directory containing view1.csv, view2.csv, view3.csv, labels.csv.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/fig2_minimal")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--fusing-iteration", type=int, default=20)
    parser.add_argument("--neighbor-size", type=int, default=20)
    parser.add_argument("--clustering", choices=["spectral", "kmeans"], default="spectral")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = ExperimentConfig(
        n_samples=args.n_samples,
        n_clusters=args.n_clusters,
        overlaps=list(args.overlaps),
        repeats=args.repeats,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        fusing_iteration=args.fusing_iteration,
        neighbor_size=args.neighbor_size,
        clustering=args.clustering,
        random_state=args.seed,
    )

    if args.data_source == "synthetic":
        views_full, labels = make_synthetic_views(
            n_samples=cfg.n_samples,
            n_clusters=cfg.n_clusters,
            random_state=args.seed,
        )
    else:
        if not args.intersim_dir:
            raise ValueError("--intersim-dir is required when --data-source intersim_csv")
        views_full, labels = load_intersim_csv(args.intersim_dir)
        cfg.n_clusters = int(labels.nunique())

    records = []
    scenarios = ["one_complete", "all_incomplete"]

    for scenario in scenarios:
        for overlap in cfg.overlaps:
            for rep in range(cfg.repeats):
                run_seed = cfg.random_state + rep + int(overlap * 1000)
                nmi = run_single_case(
                    views_full=views_full,
                    labels=labels,
                    scenario=scenario,
                    overlap=overlap,
                    cfg=cfg,
                    seed=run_seed,
                )
                records.append(
                    {
                        "scenario": scenario,
                        "overlap": overlap,
                        "repeat": rep,
                        "nmi": nmi,
                    }
                )
                print(
                    f"[{scenario}] overlap={overlap:.1f} rep={rep + 1}/{cfg.repeats} NMI={nmi:.4f}"
                )

    result_df = pd.DataFrame(records)
    summary_df = summarize_results(result_df)

    raw_path = os.path.join(args.output_dir, "nmi_raw.csv")
    summary_path = os.path.join(args.output_dir, "nmi_summary.csv")
    fig_path = os.path.join(args.output_dir, "nmi_curve.png")

    result_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot_curve(summary_df, fig_path)

    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
