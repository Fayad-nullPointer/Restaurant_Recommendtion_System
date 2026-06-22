"""
benchmark.py  (optimised — parallel + cached)
==============================================
Key performance improvements over the original:

1. PARALLEL MODEL EVALUATION  — joblib.Parallel trains and evaluates all
   models simultaneously across CPU cores (n_jobs = -1 = all cores).

2. BULK RECOMMENDATIONS  — model.batch_recommend() predicts all users in one
   vectorised call instead of looping user-by-user.

3. TRAIN/TEST SPLIT instead of full K-Fold cross validation (much faster).
   Use cv_folds=1 (default) for a single split, or set higher for CV.

4. DATASET SAMPLING  — only `sample_users` users are evaluated for
   expensive ranking & beyond-accuracy metrics (default 200).

5. RATING MATRIX CACHE  — data_loader saves the matrix as Parquet so the
   preprocessing step only runs once.

Usage
-----
    # from project root:
    python run_benchmark.py

    # or programmatically:
    from src.benchmark import run_benchmark
    df = run_benchmark(fast_mode=True)   # quick demo, ~2-5 min
    df = run_benchmark(fast_mode=False)  # all models, more folds
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — prevents display freeze
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.preprocessing import MultiLabelBinarizer
from surprise.model_selection import train_test_split as surprise_split
from surprise.model_selection import KFold
from joblib import Parallel, delayed

try:
    from .data_loader import load_raw, build_rating_matrix, to_surprise_dataset, get_all_item_ids
    from .models import build_fast_models, build_all_models, RecommenderModel
    from .metrics import evaluate_all
except ImportError:
    from src.data_loader import load_raw, build_rating_matrix, to_surprise_dataset, get_all_item_ids
    from src.models import build_fast_models, build_all_models, RecommenderModel
    from src.metrics import evaluate_all

warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Auxiliary builders
# ══════════════════════════════════════════════════════════════════════════════

def _compute_item_popularity(rating_df: pd.DataFrame) -> Dict[str, float]:
    counts = rating_df.groupby("item_id")["rating"].count()
    return (counts / counts.max()).to_dict()


def _build_item_features(data: dict) -> Optional[Dict[str, np.ndarray]]:
    vendors = data["vendors"].copy()
    if "vendor_tag_name" not in vendors.columns:
        return None
    vendors["tags_list"] = (
        vendors["vendor_tag_name"].fillna("")
        .str.split(",")
        .apply(lambda lst: [t.strip() for t in lst if t.strip()])
    )
    mlb    = MultiLabelBinarizer()
    matrix = mlb.fit_transform(vendors["tags_list"])
    ids    = vendors["id"].astype(str).str.strip().tolist()
    return {vid: vec.astype(float) for vid, vec in zip(ids, matrix)}


# ══════════════════════════════════════════════════════════════════════════════
# Single-model worker  (called in parallel)
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_one_model(
    model: RecommenderModel,
    trainset,
    testset: list,
    all_items: List[str],
    item_pop: Dict[str, float],
    item_features: Optional[Dict[str, np.ndarray]],
    k: int,
    sample_users: Optional[int],
    rating_threshold: float,
) -> Dict[str, float]:
    """Train one model and compute all metrics. Designed to run in a worker."""
    t0 = time.time()
    model.fit(trainset)
    t_train = time.time() - t0

    # ── sub-sample test users for ranking metrics ─────────────────────────────
    raw_testset = list(testset)
    unique_users = list({uid for uid, _, _ in raw_testset})
    if sample_users and len(unique_users) > sample_users:
        chosen = set(np.random.choice(unique_users, size=sample_users, replace=False))
        sampled_testset = [(u, i, r) for u, i, r in raw_testset if u in chosen]
    else:
        sampled_testset = raw_testset
        chosen = set(unique_users)

    # ── bulk recommend for all sampled users at once ──────────────────────────
    user_recs = model.batch_recommend(list(chosen), k, all_items)

    metrics = evaluate_all(
        model            = model,
        testset          = sampled_testset,
        all_items        = all_items,
        item_popularity  = item_pop,
        item_features    = item_features,
        k                = k,
        rating_threshold = rating_threshold,
        precomputed_recs = user_recs,   # skip re-computing inside evaluate_all
    )
    metrics["TrainTime_s"] = t_train
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main benchmark runner
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    fast_mode: bool      = True,
    k: int               = 10,
    cv_folds: int        = 1,
    sample_users: int    = 200,
    rating_threshold: float = 3.5,
    n_jobs: int          = -1,         # -1 = use all CPU cores
    force_rebuild: bool  = False,
) -> pd.DataFrame:
    """
    Run the full benchmarking pipeline.

    Parameters
    ----------
    fast_mode     : True  → 11 fast models (default, ~2-5 min on laptop)
                    False → full catalogue incl. SVD++ and NMF (~20-40 min)
    k             : top-K cut-off for ranking metrics
    cv_folds      : 1 = single train/test split (fastest)
                    >1 = k-fold cross-validation
    sample_users  : number of test users for ranking/beyond metrics
    rating_threshold: min rating to count an item as 'relevant'
    n_jobs        : parallel workers (-1 = all cores)
    force_rebuild : ignore Parquet cache and rebuild rating matrix
    """
    print("=" * 70)
    print("  RESTAURANT RECOMMENDATION SYSTEM — BENCHMARK (PARALLEL MODE)")
    print("=" * 70)
    print(f"  Mode: {'FAST (11 models)' if fast_mode else 'FULL (15 models)'} | "
          f"K={k} | folds={cv_folds} | sample_users={sample_users} | "
          f"n_jobs={n_jobs}")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    data        = load_raw()
    rating_df   = build_rating_matrix(data, force_rebuild=force_rebuild)
    surprise_ds = to_surprise_dataset(rating_df)
    all_items   = get_all_item_ids(data)

    # ── 2. Auxiliaries ────────────────────────────────────────────────────────
    print("\n[2/5] Building item popularity & features …")
    item_pop      = _compute_item_popularity(rating_df)
    item_features = _build_item_features(data)
    print(f"  ✔ item_features = {'tag one-hot' if item_features else 'disabled'}")

    # ── 3. Splits ─────────────────────────────────────────────────────────────
    print(f"\n[3/5] Preparing {'train/test split' if cv_folds == 1 else f'{cv_folds}-fold CV'} …")
    if cv_folds == 1:
        splits = [surprise_split(surprise_ds, test_size=0.20, random_state=42)]
    else:
        kf     = KFold(n_splits=cv_folds, random_state=42, shuffle=True)
        splits = list(kf.split(surprise_ds))

    models_factory = build_fast_models if fast_mode else build_all_models
    model_names    = [m.name for m in models_factory()]
    print(f"  ✔ {len(model_names)} models | {len(splits)} split(s)")

    # ── 4. Parallel training & evaluation ────────────────────────────────────
    print(f"\n[4/5] Training & evaluating (parallel, n_jobs={n_jobs}) …")

    aggregated: Dict[str, Dict[str, List[float]]] = {n: {} for n in model_names}

    for fold_idx, (trainset, testset) in enumerate(splits, 1):
        print(f"\n  ── Fold {fold_idx}/{len(splits)} ──────────────────────────")
        fold_models = models_factory()   # fresh instances per fold

        # run all models in parallel
        fold_results: List[Dict] = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_evaluate_one_model)(
                model, trainset, testset,
                all_items, item_pop, item_features,
                k, sample_users, rating_threshold,
            )
            for model in fold_models
        )

        for model, metrics in zip(fold_models, fold_results):
            print(f"    {model.name[:60]:<60} "
                  f"RMSE={metrics.get('RMSE', float('nan')):.4f}  "
                  f"HR@{k}={metrics.get(f'HitRate@{k}', float('nan')):.4f}  "
                  f"({metrics.get('TrainTime_s', 0):.1f}s)")
            for metric, value in metrics.items():
                aggregated[model.name].setdefault(metric, []).append(value)

    # ── 5. Aggregate ─────────────────────────────────────────────────────────
    print("\n[5/5] Aggregating results …")
    rows = []
    for name, metric_dict in aggregated.items():
        row = {"Model": name}
        for metric, values in metric_dict.items():
            row[metric] = float(np.nanmean(values))
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("Model")
    csv_path   = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    results_df.to_csv(csv_path)
    print(f"  ✔ Saved → {csv_path}")
    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results_df: pd.DataFrame, k: int = 1000) -> None:
    cols = [
        "RMSE", "MSE", "MAE",
        f"Precision@{k}", f"Recall@{k}", f"F1@{k}",
        f"NDCG@{k}", f"MAP@{k}", f"HitRate@{k}", f"AvgHitRate@{k}", 
        f"Novelty@{k}", f"Serendipity@{k}", f"Coverage@{k}", f"Diversity@{k}",
        "TrainTime_s",
    ]
    display = [c for c in cols if c in results_df.columns]
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(tabulate(results_df[display].round(4), headers="keys",
                   tablefmt="fancy_grid", floatfmt=".4f"))


def plot_results(results_df: pd.DataFrame, k: int = 10) -> None:
    _plot_group(results_df, ["RMSE", "MSE", "MAE"],
                "Rating Accuracy (lower is better)",
                os.path.join(RESULTS_DIR, "benchmark_accuracy.png"), True)
    _plot_group(results_df,
                [f"Precision@{k}", f"Recall@{k}", f"F1@{k}",
                 f"NDCG@{k}", f"MAP@{k}", f"HitRate@{k}", f"AvgHitRate@{k}"],
                f"Ranking Metrics @ K={k} (higher is better)",
                os.path.join(RESULTS_DIR, "benchmark_ranking.png"), False)
    _plot_group(results_df,
                [f"Novelty@{k}", f"Serendipity@{k}", f"Coverage@{k}", f"Diversity@{k}"],
                f"Beyond-Accuracy Metrics @ K={k}",
                os.path.join(RESULTS_DIR, "benchmark_beyond.png"), False)


def _plot_group(df, metrics, title, filename, lower_is_better):
    present = [m for m in metrics if m in df.columns and not df[m].isna().all()]
    if not present:
        return
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 3.5 * n), 7),
                             constrained_layout=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold")
    names   = [s[:40] for s in df.index.tolist()]
    palette = plt.cm.tab20(np.linspace(0, 1, len(names)))
    for ax, metric in zip(axes, present):
        vals  = df[metric].fillna(0).values
        order = np.argsort(vals) if lower_is_better else np.argsort(vals)[::-1]
        sv, sn = vals[order], [names[i] for i in order]
        bars = ax.barh(range(len(sn)), sv, color=[palette[i] for i in order],
                       edgecolor="white", height=0.7)
        ax.set_yticks(range(len(sn)))
        ax.set_yticklabels(sn, fontsize=7.5)
        ax.set_title(metric, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, sv):
            ax.text(bar.get_width() * 1.01 + 1e-6,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Chart → {filename}")


# ── entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    K = 10
    df = run_benchmark(fast_mode=True, k=K, cv_folds=1, sample_users=1000)
    print_results_table(df, k=K)
    plot_results(df, k=K)
    print("\n✅  Done — check results/ folder.")
