"""
metrics.py
==========
Custom evaluation-metrics library for recommender systems.

All metrics are implemented as pure functions operating on standard Python /
NumPy structures so they can be used independently of any framework.

Metric groups
-------------
Rating-accuracy metrics (require ground-truth ratings):
    rmse(predictions)
    mse(predictions)
    mae(predictions)

Ranking / top-N metrics (require recommendation lists + ground-truth sets):
    precision_at_k(recommended, relevant, k)
    recall_at_k(recommended, relevant, k)
    f1_at_k(recommended, relevant, k)
    average_precision_at_k(recommended, relevant, k)      -> scalar
    mean_average_precision(user_recs, user_relevant, k)   -> scalar
    ndcg_at_k(recommended, relevant, k)
    hit_rate(user_recs, user_relevant, k)                 -> scalar
    average_hit_rate(user_recs, user_relevant, k)         -> scalar

Beyond-accuracy metrics (require extra catalogue information):
    novelty(user_recs, item_popularity, k)
    serendipity(user_recs, user_relevant, item_popularity, k)
    intra_list_similarity(user_recs, item_features, k)    -> uses cosine similarity
    catalog_coverage(all_recs, all_items)
    diversity(user_recs, item_features, k)

Helper
    evaluate_all(model, testset, all_items, item_popularity, item_features, k)
    -> returns a dict with every metric above
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Rating-Accuracy Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def rmse(predictions) -> float:
    """
    Root Mean Squared Error between estimated and actual ratings.

    Parameters
    ----------
    predictions : list of Surprise Prediction namedtuples
        (or any iterable with .r_ui and .est attributes)

    Returns
    -------
    float
    """
    errors = [(p.r_ui - p.est) ** 2 for p in predictions]
    return math.sqrt(sum(errors) / len(errors))


def mse(predictions) -> float:
    """Mean Squared Error."""
    errors = [(p.r_ui - p.est) ** 2 for p in predictions]
    return sum(errors) / len(errors)


def mae(predictions) -> float:
    """Mean Absolute Error."""
    errors = [abs(p.r_ui - p.est) for p in predictions]
    return sum(errors) / len(errors)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ranking / Top-N Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _top_k(recommended: List[str], k: int) -> List[str]:
    return recommended[:k]


def precision_at_k(recommended: List[str],
                   relevant: Set[str],
                   k: int) -> float:
    """
    Fraction of the top-K recommended items that are relevant.

        Precision@K = |{top-K} ∩ {relevant}| / K
    """
    if k == 0:
        return 0.0
    hits = sum(1 for item in _top_k(recommended, k) if item in relevant)
    return hits / k


def recall_at_k(recommended: List[str],
                relevant: Set[str],
                k: int) -> float:
    """
    Fraction of all relevant items that appear in the top-K recommendations.

        Recall@K = |{top-K} ∩ {relevant}| / |{relevant}|
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in _top_k(recommended, k) if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: List[str],
            relevant: Set[str],
            k: int) -> float:
    """
    Harmonic mean of Precision@K and Recall@K.
    """
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def average_precision_at_k(recommended: List[str],
                            relevant: Set[str],
                            k: int) -> float:
    """
    Average Precision (AP) for a single user.

    AP@K = (1/|relevant|) * Σ [P@i * rel(i)]  for i in 1..K
    """
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(_top_k(recommended, k), start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def mean_average_precision(user_recs: Dict[str, List[str]],
                           user_relevant: Dict[str, Set[str]],
                           k: int) -> float:
    """
    Mean Average Precision across all users (MAP@K).
    """
    aps = [
        average_precision_at_k(user_recs.get(uid, []),
                                user_relevant.get(uid, set()), k)
        for uid in user_relevant
    ]
    return float(np.mean(aps)) if aps else 0.0


def ndcg_at_k(recommended: List[str],
              relevant: Set[str],
              k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    DCG@K  = Σ rel_i / log2(i+1)   for i in 1..K   (rel_i ∈ {0,1})
    IDCG@K = Σ 1 / log2(i+1)       for i in 1..min(|relevant|, K)
    NDCG@K = DCG@K / IDCG@K
    """
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(_top_k(recommended, k))
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ── Hit-Rate helpers ──────────────────────────────────────────────────────────

def _user_hit(recommended: List[str], relevant: Set[str], k: int) -> bool:
    """True if at least one of the top-K items is relevant."""
    return any(item in relevant for item in _top_k(recommended, k))


def hit_rate(user_recs: Dict[str, List[str]],
             user_relevant: Dict[str, Set[str]],
             k: int) -> float:
    """
    Fraction of users for whom at least one relevant item appears in top-K.

        HR@K = |{users with hit}| / |{all users}|
    """
    hits = [
        _user_hit(user_recs.get(uid, []), user_relevant.get(uid, set()), k)
        for uid in user_relevant
    ]
    return float(np.mean(hits)) if hits else 0.0


def average_hit_rate(user_recs: Dict[str, List[str]],
                     user_relevant: Dict[str, Set[str]],
                     k: int) -> float:
    """
    Mean number of relevant items found per user, normalised by K.

        AHR@K = (1/N) * Σ |{top-K_u} ∩ {relevant_u}| / K
    """
    scores = [
        sum(1 for item in _top_k(user_recs.get(uid, []), k)
            if item in user_relevant.get(uid, set())) / k
        for uid in user_relevant
    ]
    return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Beyond-Accuracy Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def novelty(user_recs: Dict[str, List[str]],
            item_popularity: Dict[str, float],
            k: int) -> float:
    """
    Mean self-information (surprise) of recommended items.

    Novelty = (1/N) * Σ_u (1/K) * Σ_{i in top-K_u} -log2( pop(i) )

    A higher value means less popular (more novel) items are recommended.
    item_popularity[item_id] should be in (0, 1] (normalised order count).
    """
    scores = []
    for uid, recs in user_recs.items():
        top = _top_k(recs, k)
        if not top:
            continue
        user_score = np.mean([
            -math.log2(max(item_popularity.get(iid, 1e-9), 1e-9))
            for iid in top
        ])
        scores.append(user_score)
    return float(np.mean(scores)) if scores else 0.0


def serendipity(user_recs: Dict[str, List[str]],
                user_relevant: Dict[str, Set[str]],
                item_popularity: Dict[str, float],
                k: int) -> float:
    """
    Average unexpected relevance of recommendations.

    Serendipity = (1/N) * Σ_u (1/K) * Σ_{i in top-K_u}
                      relevant(i,u) * (1 - popularity(i))

    Items that are both relevant AND unpopular score highest.
    """
    scores = []
    for uid, recs in user_recs.items():
        relevant = user_relevant.get(uid, set())
        top = _top_k(recs, k)
        if not top:
            continue
        user_score = np.mean([
            (1.0 if iid in relevant else 0.0)
            * (1.0 - item_popularity.get(iid, 0.0))
            for iid in top
        ])
        scores.append(user_score)
    return float(np.mean(scores)) if scores else 0.0


def intra_list_similarity(user_recs: Dict[str, List[str]],
                          item_features: Dict[str, np.ndarray],
                          k: int) -> float:
    """
    Mean pairwise cosine similarity within each user's top-K list.

    Lower value = more diverse recommendations.
    item_features[item_id] is a 1-D numpy feature vector.
    """
    scores = []
    for uid, recs in user_recs.items():
        top = [iid for iid in _top_k(recs, k) if iid in item_features]
        if len(top) < 2:
            continue
        mat = np.vstack([item_features[iid] for iid in top])
        sim_matrix = cosine_similarity(mat)
        # upper triangle (excluding diagonal)
        n = len(top)
        pairs = [(sim_matrix[i, j])
                 for i in range(n) for j in range(i + 1, n)]
        scores.append(float(np.mean(pairs)))
    return float(np.mean(scores)) if scores else 0.0


def diversity(user_recs: Dict[str, List[str]],
              item_features: Dict[str, np.ndarray],
              k: int) -> float:
    """
    1 - intra_list_similarity  (higher = more diverse).
    """
    return 1.0 - intra_list_similarity(user_recs, item_features, k)


def catalog_coverage(all_recs: Dict[str, List[str]],
                     all_items: List[str],
                     k: int) -> float:
    """
    Fraction of the item catalogue that appears in at least one recommendation.

        Coverage = |⋃_u top-K_u| / |catalogue|
    """
    if not all_items:
        return 0.0
    recommended_set: set = set()
    for recs in all_recs.values():
        recommended_set.update(_top_k(recs, k))
    return len(recommended_set) / len(all_items)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Omnibus Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    model,
    testset,
    all_items: List[str],
    item_popularity: Dict[str, float],
    item_features: Optional[Dict[str, np.ndarray]],
    k: int = 10,
    rating_threshold: float = 3.5,
    precomputed_recs: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> Dict[str, float]:
    """
    Run every implemented metric against `model` on `testset`.

    Parameters
    ----------
    model           : RecommenderModel (already fitted)
    testset         : list of (uid, iid, r_ui) triples from Surprise
    all_items       : list of all item id strings in the catalogue
    item_popularity : {item_id: normalised popularity score ∈ (0,1]}
    item_features   : {item_id: feature vector} — pass None to skip ILS metrics
    k               : cut-off for top-N metrics
    rating_threshold: minimum estimated rating to count as 'relevant'

    Returns
    -------
    dict mapping metric_name -> scalar value
    """
    # ── bulk predict on testset (fast .test() call) ───────────────────────────
    predictions = model.algo.test(list(testset))

    # ── build ground-truth relevant sets per user ────────────────────────────
    user_relevant: Dict[str, Set[str]] = {}
    for uid, iid, r in testset:
        uid = str(uid)
        if r >= rating_threshold:
            user_relevant.setdefault(uid, set()).add(str(iid))

    # ── use precomputed recs if provided, else compute now ────────────────────
    if precomputed_recs is not None:
        user_recs: Dict[str, List[str]] = {
            uid: [iid for iid, _ in recs]
            for uid, recs in precomputed_recs.items()
        }
    else:
        test_users = list({str(uid) for uid, _, _ in testset})
        user_recs = {}
        for uid in test_users:
            recs = model.recommend(uid, k, all_items, exclude_seen=True)
            user_recs[uid] = [iid for iid, _ in recs]

    # ── rating-accuracy ───────────────────────────────────────────────────────
    results: Dict[str, float] = {
        "RMSE": rmse(predictions),
        "MSE" : mse(predictions),
        "MAE" : mae(predictions),
    }

    # ── ranking metrics ───────────────────────────────────────────────────────
    prec_scores, rec_scores, f1_scores, ndcg_scores, ap_scores = [], [], [], [], []
    for uid in user_relevant:
        rec_list = user_recs.get(uid, [])
        rel = user_relevant[uid]
        prec_scores.append(precision_at_k(rec_list, rel, k))
        rec_scores.append(recall_at_k(rec_list, rel, k))
        f1_scores.append(f1_at_k(rec_list, rel, k))
        ndcg_scores.append(ndcg_at_k(rec_list, rel, k))
        ap_scores.append(average_precision_at_k(rec_list, rel, k))

    results[f"Precision@{k}"]  = float(np.mean(prec_scores))  if prec_scores  else 0.0
    results[f"Recall@{k}"]     = float(np.mean(rec_scores))   if rec_scores   else 0.0
    results[f"F1@{k}"]         = float(np.mean(f1_scores))    if f1_scores    else 0.0
    results[f"NDCG@{k}"]       = float(np.mean(ndcg_scores))  if ndcg_scores  else 0.0
    results[f"MAP@{k}"]        = float(np.mean(ap_scores))    if ap_scores    else 0.0
    results[f"HitRate@{k}"]    = hit_rate(user_recs, user_relevant, k)
    results[f"AvgHitRate@{k}"] = average_hit_rate(user_recs, user_relevant, k)

    # ── beyond-accuracy ───────────────────────────────────────────────────────
    results[f"Novelty@{k}"]     = novelty(user_recs, item_popularity, k)
    results[f"Serendipity@{k}"] = serendipity(user_recs, user_relevant,
                                              item_popularity, k)
    results[f"Coverage@{k}"]    = catalog_coverage(user_recs, all_items, k)
    results[f"Diversity@{k}"]   = diversity(user_recs, item_features, k) \
                                   if item_features else float("nan")
    results[f"ILS@{k}"]         = intra_list_similarity(user_recs, item_features, k) \
                                   if item_features else float("nan")
    return results
