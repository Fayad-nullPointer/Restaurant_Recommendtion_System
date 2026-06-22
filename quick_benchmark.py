"""
quick_benchmark.py
==================
STANDALONE script — works on any low-spec device.
No package imports, no parallel processing, minimal RAM usage.

Run:
    python quick_benchmark.py

Models benchmarked
------------------
Baseline Estimates  : Random, ALS Biased, SGD Biased
User-Based CF       : KNNBasic (Cosine), KNNWithMeans (Pearson)
Item-Based CF       : KNNBasic (Cosine)
Matrix Factorization: SVD, SlopeOne
Content-Based (CB)  : Cosine similarity between user preference
                      profile and vendor feature vectors
                      (food tags + rating + price + distance + discount)

Metrics evaluated
-----------------
Accuracy  : RMSE, MSE, MAE
Ranking   : Precision@K, Recall@K, F1@K, NDCG@K, MAP@K
Hit-based : HitRate@K, AvgHitRate@K
Beyond    : Novelty@K, Serendipity@K, Coverage@K, Diversity@K

Expected runtime: 2-5 minutes on a weak/old laptop.
"""

import os
import sys
import math
import time
import warnings
import random

warnings.filterwarnings("ignore")

# ── check dependencies ────────────────────────────────────────────────────────
def _require(pkg, install_name=None):
    try:
        __import__(pkg)
    except ImportError:
        name = install_name or pkg
        print(f"\n❌  Missing package: {name}")
        print(f"    Fix: pip install {name}\n")
        sys.exit(1)

_require("surprise",    "scikit-surprise")
_require("sklearn",     "scikit-learn")
_require("pandas",      "pandas")
_require("numpy",       "numpy")
_require("matplotlib",  "matplotlib")
_require("tabulate",    "tabulate")

# ── imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # MUST be before pyplot — prevents GUI freeze
import matplotlib.pyplot as plt
from tabulate import tabulate

from surprise import (
    Dataset, Reader,
    NormalPredictor, BaselineOnly,
    KNNBasic, KNNWithMeans,
    SVD, SlopeOne,
)
from surprise.model_selection import train_test_split as surprise_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Representative & Fast
# ══════════════════════════════════════════════════════════════════════════════
MAX_USERS    = 5000     # sample this many of the most active users
MAX_RATINGS  = 50000    # hard cap on rows in rating matrix
MIN_ORDERS   = 2      # drop users with fewer orders than this
K            = 10     # top-K for ranking metrics
SAMPLE_EVAL  = 1000    # users evaluated for ranking/beyond metrics
RATING_MIN   = 1.0
RATING_MAX   = 5.0
TEST_SIZE    = 0.20   # fraction of data used for testing
RANDOM_SEED  = 42
# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
ARCHIVE     = os.path.join(ROOT, "archive")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & RATING MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("\n[1/5] Loading data …")

    orders_path  = os.path.join(ARCHIVE, "orders.csv")
    vendors_path = os.path.join(ARCHIVE, "vendors.csv")

    for p in [orders_path, vendors_path]:
        if not os.path.exists(p):
            print(f"❌  File not found: {p}")
            sys.exit(1)

    # load only the columns we need — saves RAM
    orders = pd.read_csv(orders_path,
                         usecols=["customer_id", "vendor_id",
                                  "grand_total", "vendor_rating"],
                         low_memory=False)
    vendors = pd.read_csv(vendors_path, low_memory=False)

    orders["customer_id"] = orders["customer_id"].astype(str).str.strip()
    orders["vendor_id"]   = orders["vendor_id"].astype(str).str.strip()
    orders["grand_total"] = pd.to_numeric(orders["grand_total"], errors="coerce").fillna(0)

    # ── keep only top MAX_USERS most active users ────────────────────────────
    user_counts  = orders["customer_id"].value_counts()
    active       = user_counts[user_counts >= MIN_ORDERS].head(MAX_USERS).index
    orders       = orders[orders["customer_id"].isin(active)]

    print(f"  Orders after sampling: {len(orders):,}  |  Users: {orders['customer_id'].nunique():,}")

    # ── aggregate to (user, item) level ──────────────────────────────────────
    agg = orders.groupby(["customer_id", "vendor_id"]).agg(
        order_count      = ("grand_total", "count"),
        total_spend      = ("grand_total", "sum"),
        explicit_rating  = ("vendor_rating", "mean"),
    ).reset_index()

    # ── implicit rating ───────────────────────────────────────────────────────
    def norm_per_user(col):
        mn = agg.groupby("customer_id")[col].transform("min")
        mx = agg.groupby("customer_id")[col].transform("max")
        return (agg[col] - mn) / (mx - mn).replace(0, 1)

    freq_norm  = norm_per_user("order_count")
    spend_norm = norm_per_user("total_spend")
    rated_flag = agg["explicit_rating"].notna().astype(float)

    implicit = RATING_MIN + (0.50 * freq_norm + 0.35 * spend_norm + 0.15 * rated_flag) \
               * (RATING_MAX - RATING_MIN)

    # blend with explicit if available
    has_exp = agg["explicit_rating"].notna()
    agg["rating"] = implicit
    agg.loc[has_exp, "rating"] = (
        0.6 * agg.loc[has_exp, "explicit_rating"] + 0.4 * implicit[has_exp]
    )
    agg["rating"] = agg["rating"].clip(RATING_MIN, RATING_MAX)

    # hard cap on matrix size
    if len(agg) > MAX_RATINGS:
        agg = agg.sample(MAX_RATINGS, random_state=RANDOM_SEED)

    rating_df = agg[["customer_id", "vendor_id", "rating"]].copy()
    rating_df.columns = ["user_id", "item_id", "rating"]

    print(f"  Rating matrix: {len(rating_df):,} pairs  |  "
          f"users={rating_df['user_id'].nunique()}  |  "
          f"items={rating_df['item_id'].nunique()}")

    return rating_df, vendors


# ══════════════════════════════════════════════════════════════════════════════
# 2. COLLABORATIVE FILTERING MODELS  (Surprise library)
# ══════════════════════════════════════════════════════════════════════════════

CF_MODELS = [
    # ── Baseline Estimates ───────────────────────────────────────────────────
    ("Baseline: Random",           NormalPredictor()),
    ("Baseline: ALS Biased",       BaselineOnly(bsl_options={"method": "als", "n_epochs": 10})),
    ("Baseline: SGD Biased",       BaselineOnly(bsl_options={"method": "sgd", "n_epochs": 10})),
    # ── User-Based Collaborative Filtering ──────────────────────────────────
    ("UserKNN: Cosine (k=20)",     KNNBasic(k=20,    sim_options={"name": "cosine",  "user_based": True})),
    ("UserKNN: WithMeans Pearson", KNNWithMeans(k=20, sim_options={"name": "pearson", "user_based": True})),
    # ── Item-Based Collaborative Filtering ──────────────────────────────────
    ("ItemKNN: Cosine (k=20)",     KNNBasic(k=20,    sim_options={"name": "cosine",  "user_based": False})),
    ("ItemKNN: Pearson-BL (k=20)", KNNWithMeans(k=20, sim_options={"name": "pearson_baseline", "user_based": False})),
    # ── Matrix Factorisation ─────────────────────────────────────────────────
    ("MF: SVD (30 factors)",       SVD(n_factors=30, n_epochs=15, lr_all=0.005, reg_all=0.02)),
    ("MF: SlopeOne",               SlopeOne()),
]


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTENT-BASED FILTERING MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ContentBasedRecommender:
    """
    Content-Based Filtering Recommender.

    How it works
    ------------
    Step 1 — Build vendor (item) feature vectors
        Each restaurant is represented as a numeric vector combining:
        • Food tags one-hot encoded  (e.g. Burgers=1, Pizza=1, Sushi=0 …)
        • Normalised vendor_rating   (0-1)
        • Normalised delivery_charge (0-1, inverted so lower=better)
        • Normalised serving_distance(0-1)
        • Normalised discount_percentage (0-1)
        • is_open binary flag

    Step 2 — Build user preference profile
        For each user in the training set:
            profile = Σ (rating_ui × feature_vector_i)  /  Σ rating_ui
        This is the rating-weighted centroid of all vendors the user ordered.
        Interpretation: if Ahmed loves Burgers restaurants and rates them high,
        his profile vector will have a large Burgers component.

    Step 3 — Score unseen vendors
        score(user, vendor) = cosine_similarity(user_profile, vendor_features)
        Rescaled to [1, 5] so it is comparable with CF ratings.

    Cold-start fallback
        If a user has no training history, fall back to the global mean.
    """

    def __init__(self, vendors_df: pd.DataFrame):
        self.vendors_df   = vendors_df.copy()
        self.item_vectors : dict = {}   # item_id  -> np.ndarray
        self.user_profiles: dict = {}   # user_id  -> np.ndarray
        self.global_mean  : float = (RATING_MIN + RATING_MAX) / 2
        self._trainset    = None

    # ── Step 1: build vendor feature vectors ─────────────────────────────────
    def _build_item_vectors(self):
        v = self.vendors_df.copy()
        v["id"] = v["id"].astype(str).str.strip()

        # --- food tags (one-hot) ---
        v["tags_list"] = (
            v["vendor_tag_name"].fillna("")
             .str.split(",")
             .apply(lambda lst: [t.strip() for t in lst if t.strip()])
        ) if "vendor_tag_name" in v.columns else [[] for _ in range(len(v))]

        mlb      = MultiLabelBinarizer()
        tag_mat  = mlb.fit_transform(v["tags_list"]).astype(float)

        # --- numeric features (normalised to [0,1]) ---
        def norm_col(col, invert=False):
            if col not in v.columns:
                return np.zeros(len(v))
            vals = pd.to_numeric(v[col], errors="coerce").fillna(0).values.astype(float)
            rng  = vals.max() - vals.min()
            normed = (vals - vals.min()) / rng if rng > 0 else np.zeros_like(vals)
            return 1.0 - normed if invert else normed

        rating_feat   = norm_col("vendor_rating").reshape(-1, 1)
        charge_feat   = norm_col("delivery_charge", invert=True).reshape(-1, 1)  # lower = better
        dist_feat     = norm_col("serving_distance").reshape(-1, 1)
        discount_feat = norm_col("discount_percentage").reshape(-1, 1)
        is_open_feat  = (pd.to_numeric(v.get("is_open", pd.Series([1]*len(v))),
                                       errors="coerce").fillna(1).values
                          .astype(float).reshape(-1, 1))

        # --- concatenate all features ---
        feature_mat = np.hstack([
            tag_mat * 2.0,     # weight tags more (most informative)
            rating_feat,
            charge_feat,
            dist_feat,
            discount_feat,
            is_open_feat,
        ])

        self.item_vectors = {
            vid: vec for vid, vec in zip(v["id"].tolist(), feature_mat)
        }
        self._feature_dim = feature_mat.shape[1]

    # ── Step 2: build user preference profiles ────────────────────────────────
    def _build_user_profiles(self, trainset):
        self._trainset = trainset
        for inner_uid in trainset.all_users():
            raw_uid = trainset.to_raw_uid(inner_uid)
            profile     = np.zeros(self._feature_dim)
            total_weight = 0.0
            for inner_iid, rating in trainset.ur[inner_uid]:
                raw_iid = trainset.to_raw_iid(inner_iid)
                if raw_iid in self.item_vectors:
                    profile      += rating * self.item_vectors[raw_iid]
                    total_weight += rating
            if total_weight > 0:
                self.user_profiles[raw_uid] = profile / total_weight
        # global mean from training ratings
        all_ratings = [r for _, _, r in trainset.all_ratings()]
        self.global_mean = float(np.mean(all_ratings)) if all_ratings else 3.0

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, trainset):
        self._build_item_vectors()
        self._build_user_profiles(trainset)
        return self

    # ── score one (user, item) pair ───────────────────────────────────────────
    def _score(self, uid: str, iid: str) -> float:
        if uid not in self.user_profiles or iid not in self.item_vectors:
            return self.global_mean
        u_vec = self.user_profiles[uid]
        i_vec = self.item_vectors[iid]
        norm_u = np.linalg.norm(u_vec)
        norm_i = np.linalg.norm(i_vec)
        if norm_u == 0 or norm_i == 0:
            return self.global_mean
        # cosine similarity in [-1, 1]  →  rescale to [RATING_MIN, RATING_MAX]
        cos_sim = float(np.dot(u_vec, i_vec) / (norm_u * norm_i))
        return RATING_MIN + (cos_sim + 1) / 2 * (RATING_MAX - RATING_MIN)

    # ── Surprise-compatible .test() ───────────────────────────────────────────
    def test(self, testset):
        """Accept list of (uid, iid, r_ui) and return Surprise-like Predictions."""
        from collections import namedtuple
        Pred = namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])
        return [
            Pred(uid=str(uid), iid=str(iid), r_ui=r_ui,
                 est=self._score(str(uid), str(iid)), details={})
            for uid, iid, r_ui in testset
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRICS (all implemented from scratch)
# ══════════════════════════════════════════════════════════════════════════════

def rmse(preds):
    return math.sqrt(sum((p.r_ui - p.est)**2 for p in preds) / len(preds))

def mse(preds):
    return sum((p.r_ui - p.est)**2 for p in preds) / len(preds)

def mae(preds):
    return sum(abs(p.r_ui - p.est) for p in preds) / len(preds)

def precision_at_k(recs, relevant, k):
    return sum(1 for r in recs[:k] if r in relevant) / k if k else 0.0

def recall_at_k(recs, relevant, k):
    return sum(1 for r in recs[:k] if r in relevant) / len(relevant) if relevant else 0.0

def f1_at_k(recs, relevant, k):
    p = precision_at_k(recs, relevant, k)
    r = recall_at_k(recs, relevant, k)
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0

def ndcg_at_k(recs, relevant, k):
    dcg  = sum(1/math.log2(i+2) for i, r in enumerate(recs[:k]) if r in relevant)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg > 0 else 0.0

def ap_at_k(recs, relevant, k):
    if not relevant: return 0.0
    hits = score = 0
    for i, r in enumerate(recs[:k], 1):
        if r in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)

def hit_rate(user_recs, user_rel, k):
    hits = [any(r in user_rel.get(u, set()) for r in recs[:k])
            for u, recs in user_recs.items()]
    return float(np.mean(hits)) if hits else 0.0

def avg_hit_rate(user_recs, user_rel, k):
    scores = [sum(1 for r in recs[:k] if r in user_rel.get(u, set()))/k
              for u, recs in user_recs.items()]
    return float(np.mean(scores)) if scores else 0.0

def novelty(user_recs, item_pop, k):
    scores = []
    for _, recs in user_recs.items():
        top = recs[:k]
        if top:
            scores.append(np.mean([-math.log2(max(item_pop.get(i, 1e-9), 1e-9))
                                   for i in top]))
    return float(np.mean(scores)) if scores else 0.0

def serendipity(user_recs, user_rel, item_pop, k):
    scores = []
    for u, recs in user_recs.items():
        rel = user_rel.get(u, set())
        top = recs[:k]
        if top:
            scores.append(np.mean([(1.0 if i in rel else 0.0) *
                                   (1.0 - item_pop.get(i, 0.0)) for i in top]))
    return float(np.mean(scores)) if scores else 0.0

def coverage(user_recs, all_items, k):
    seen = set()
    for recs in user_recs.values():
        seen.update(recs[:k])
    return len(seen) / len(all_items) if all_items else 0.0

def diversity(user_recs, item_feat, k):
    scores = []
    for _, recs in user_recs.items():
        top = [r for r in recs[:k] if r in item_feat]
        if len(top) < 2: continue
        mat  = np.vstack([item_feat[i] for i in top])
        sim  = cosine_similarity(mat)
        n    = len(top)
        vals = [sim[i,j] for i in range(n) for j in range(i+1,n)]
        scores.append(1.0 - float(np.mean(vals)))   # diversity = 1 - similarity
    return float(np.mean(scores)) if scores else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(algo, trainset, testset_list, all_items, item_pop, item_feat,
             name, k, n_eval_users):
    # bulk predict testset
    preds = algo.test(testset_list)

    # relevant items per user (rating >= 3.5)
    user_rel = defaultdict(set)
    for p in preds:
        if p.r_ui >= 3.5:
            user_rel[p.uid].add(p.iid)

    # pick users to evaluate for top-N metrics
    eval_users = list({p.uid for p in preds})
    if len(eval_users) > n_eval_users:
        eval_users = random.sample(eval_users, n_eval_users)

    # seen items per user (from training)
    user_seen = defaultdict(set)
    for uid, iid, _ in trainset.all_ratings():
        user_seen[trainset.to_raw_uid(uid)].add(trainset.to_raw_iid(iid))

    # generate top-K recommendations for eval users
    # build one big testset for all (user, item) pairs in batch
    rows = [(u, i, 0) for u in eval_users
                      for i in all_items if i not in user_seen[u]]
    batch_preds = algo.test(rows)

    user_recs = defaultdict(list)
    for p in batch_preds:
        user_recs[p.uid].append((p.iid, p.est))
    user_recs_sorted = {u: [i for i, _ in sorted(v, key=lambda x: -x[1])]
                        for u, v in user_recs.items()}

    # compute metrics
    prec, rec, f1s, ndcgs, aps = [], [], [], [], []
    for u in eval_users:
        recs = user_recs_sorted.get(u, [])
        rel  = user_rel.get(u, set())
        prec.append(precision_at_k(recs, rel, k))
        rec.append(recall_at_k(recs, rel, k))
        f1s.append(f1_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))
        aps.append(ap_at_k(recs, rel, k))

    return {
        "Model"             : name,
        "RMSE"              : rmse(preds),
        "MSE"               : mse(preds),
        "MAE"               : mae(preds),
        f"Precision@{k}"    : float(np.mean(prec)),
        f"Recall@{k}"       : float(np.mean(rec)),
        f"F1@{k}"           : float(np.mean(f1s)),
        f"NDCG@{k}"         : float(np.mean(ndcgs)),
        f"MAP@{k}"          : float(np.mean(aps)),
        f"HitRate@{k}"      : hit_rate(user_recs_sorted, user_rel, k),
        f"AvgHitRate@{k}"   : avg_hit_rate(user_recs_sorted, user_rel, k),
        f"Novelty@{k}"      : novelty(user_recs_sorted, item_pop, k),
        f"Serendipity@{k}"  : serendipity(user_recs_sorted, user_rel, item_pop, k),
        f"Coverage@{k}"     : coverage(user_recs_sorted, all_items, k),
        f"Diversity@{k}"    : diversity(user_recs_sorted, item_feat, k) if item_feat else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _bar_chart(df, cols, title, path, lower_better=False):
    present = [c for c in cols if c in df.columns and not df[c].isna().all()]
    if not present: return
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(max(5, 3*n), 7), constrained_layout=True)
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=11, fontweight="bold")
    names   = [s[:38] for s in df["Model"].tolist()]
    palette = plt.cm.tab20(np.linspace(0, 1, len(names)))
    for ax, col in zip(axes, present):
        vals  = df[col].fillna(0).values
        order = np.argsort(vals) if lower_better else np.argsort(vals)[::-1]
        sv, sn = vals[order], [names[i] for i in order]
        bars = ax.barh(range(len(sn)), sv, color=[palette[i] for i in order],
                       edgecolor="white", height=0.7)
        ax.set_yticks(range(len(sn)))
        ax.set_yticklabels(sn, fontsize=7)
        ax.set_title(col, fontsize=8, fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, sv):
            ax.text(bar.get_width()*1.01 + 1e-6,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", ha="left", fontsize=6.5)
        ax.grid(axis="x", alpha=0.3)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RESTAURANT RECOMMENDER — LIGHTWEIGHT BENCHMARK")
    print(f"  MAX_USERS={MAX_USERS}  K={K}  EVAL_USERS={SAMPLE_EVAL}")
    print("=" * 60)

    # ── load ─────────────────────────────────────────────────────────────────
    rating_df, vendors = load_data()

    # ── all item ids ─────────────────────────────────────────────────────────
    all_items = vendors["id"].astype(str).str.strip().tolist()

    # ── item popularity (order count normalised) ──────────────────────────────
    counts   = rating_df.groupby("item_id")["rating"].count()
    item_pop = (counts / counts.max()).to_dict()

    # ── item features (tag one-hot for diversity metric) ─────────────────────
    print("\n[2/5] Building item features …")
    item_feat = None
    if "vendor_tag_name" in vendors.columns:
        vendors["tags"] = (vendors["vendor_tag_name"].fillna("")
                           .str.split(",")
                           .apply(lambda x: [t.strip() for t in x if t.strip()]))
        mlb       = MultiLabelBinarizer()
        mat       = mlb.fit_transform(vendors["tags"])
        ids       = vendors["id"].astype(str).str.strip().tolist()
        item_feat = {vid: vec.astype(float) for vid, vec in zip(ids, mat)}
        print(f"  ✔ Item features: {mat.shape[1]} tag dimensions")

    # ── train/test split ──────────────────────────────────────────────────────
    print("\n[3/5] Splitting data …")
    reader  = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    dataset = Dataset.load_from_df(rating_df, reader)
    trainset_sur, testset_sur = surprise_split(dataset, test_size=TEST_SIZE,
                                               random_state=RANDOM_SEED)
    testset_list = list(testset_sur)
    print(f"  Train: {trainset_sur.n_ratings:,}  |  Test: {len(testset_list):,}")

    # ── build content-based model ─────────────────────────────────────────────
    print("\n  Building ContentBasedRecommender …", end=" ", flush=True)
    cb_model = ContentBasedRecommender(vendors)
    cb_model.fit(trainset_sur)
    print(f"✔ ({len(cb_model.item_vectors)} vendor vectors, "
          f"{len(cb_model.user_profiles)} user profiles)")

    # ── combine all models for benchmarking ───────────────────────────────────
    all_models = CF_MODELS + [("CB: Content-Based (Tag+Features)", cb_model)]

    # ── benchmark loop ────────────────────────────────────────────────────────
    print(f"\n[4/5] Evaluating {len(all_models)} models …\n")
    results = []
    for name, algo in all_models:
        print(f"  ▶  {name} …", end=" ", flush=True)
        t0 = time.time()
        # CF models need fit(); CB model is already fitted above
        if algo is not cb_model:
            algo.fit(trainset_sur)
        t_train = time.time() - t0
        row = evaluate(algo, trainset_sur, testset_list,
                       all_items, item_pop, item_feat,
                       name, K, SAMPLE_EVAL)
        row["TrainTime_s"] = round(t_train, 2)
        results.append(row)
        print(f"RMSE={row['RMSE']:.4f}  HitRate@{K}={row[f'HitRate@{K}']:.4f}  "
              f"({t_train:.1f}s)")

    # ── results table ─────────────────────────────────────────────────────────
    print("\n[5/5] Results\n")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "benchmark_results.csv"), index=False)

    display_cols = ["Model", "RMSE", "MSE", "MAE",
                    f"Precision@{K}", f"Recall@{K}", f"F1@{K}",
                    f"NDCG@{K}", f"MAP@{K}",
                    f"HitRate@{K}", f"AvgHitRate@{K}",
                    f"Novelty@{K}", f"Serendipity@{K}",
                    f"Coverage@{K}", f"Diversity@{K}", "TrainTime_s"]
    display_cols = [c for c in display_cols if c in df.columns]

    print(tabulate(df[display_cols].round(4), headers="keys",
                   tablefmt="fancy_grid", floatfmt=".4f", showindex=False))

    # ── charts ────────────────────────────────────────────────────────────────
    _bar_chart(df, ["RMSE", "MSE", "MAE"],
               "Accuracy (lower = better)",
               os.path.join(RESULTS_DIR, "accuracy.png"), lower_better=True)
    _bar_chart(df, [f"Precision@{K}", f"Recall@{K}", f"F1@{K}",
                    f"NDCG@{K}", f"MAP@{K}", f"HitRate@{K}", f"AvgHitRate@{K}"],
               f"Ranking Metrics @ K={K}  (higher = better)",
               os.path.join(RESULTS_DIR, "ranking.png"))
    _bar_chart(df, [f"Novelty@{K}", f"Serendipity@{K}",
                    f"Coverage@{K}", f"Diversity@{K}"],
               f"Beyond-Accuracy @ K={K}",
               os.path.join(RESULTS_DIR, "beyond.png"))

    # ── best model per metric ─────────────────────────────────────────────────
    print("\n  Best model per metric:")
    lower = {"RMSE", "MSE", "MAE", "TrainTime_s"}
    for col in display_cols[1:]:
        try:
            if col in lower:
                idx = df[col].idxmin()
            else:
                idx = df[col].idxmax()
            print(f"    {col:<22}: {df.loc[idx, 'Model']}")
        except Exception:
            pass

    print(f"\n✅  Done!  Results saved in:  {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
