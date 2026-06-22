"""
models.py  (optimised)
======================
Same unified wrapper as before, but with two catalogues:

  build_fast_models()   ← default; 11 fast models, no SVD++
  build_all_models()    ← full catalogue; includes SVD++ and NMF (slow)

The wrapper now also uses Surprise's bulk `.test()` call instead of
one-by-one `.predict()`, which is 10-50× faster for large test sets.
"""

from __future__ import annotations
from typing import List, Tuple

from surprise import (
    AlgoBase, BaselineOnly, CoClustering,
    KNNBasic, KNNWithMeans, KNNWithZScore,
    NMF, NormalPredictor, SlopeOne, SVD, SVDpp,
)


class RecommenderModel:
    """Thin unified wrapper around any Surprise AlgoBase."""

    def __init__(self, algo: AlgoBase, name: str):
        self.algo = algo
        self.name = name
        self._trainset = None
        # cache: uid → list of (iid, score)
        self._rec_cache: dict = {}

    def fit(self, trainset) -> "RecommenderModel":
        self._trainset = trainset
        self._rec_cache.clear()
        self.algo.fit(trainset)
        return self

    def predict(self, uid: str, iid: str) -> float:
        return self.algo.predict(uid, iid, verbose=False).est

    def test_bulk(self, testset) -> list:
        """
        Run predictions on an entire testset in one call.
        Returns a list of Surprise Prediction namedtuples.
        Much faster than calling predict() in a loop.
        """
        return self.algo.test(testset)

    def recommend(self, uid: str, n: int, all_items: List[str],
                  exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Top-N recommendations, with per-user caching."""
        cache_key = (uid, n, exclude_seen)
        if cache_key in self._rec_cache:
            return self._rec_cache[cache_key]

        seen: set = set()
        if exclude_seen and self._trainset is not None:
            try:
                inner_uid = self._trainset.to_inner_uid(uid)
                seen = {
                    self._trainset.to_raw_iid(iid)
                    for iid, _ in self._trainset.ur[inner_uid]
                }
            except ValueError:
                pass

        candidates = [iid for iid in all_items if iid not in seen]
        # bulk predict all candidates for this user
        testset = [(uid, iid, 0) for iid in candidates]
        preds   = self.algo.test(testset)
        scored  = sorted([(p.iid, p.est) for p in preds],
                         key=lambda x: x[1], reverse=True)[:n]

        self._rec_cache[cache_key] = scored
        return scored

    def batch_recommend(self, uids: List[str], n: int,
                        all_items: List[str]) -> dict:
        """
        Generate top-N recommendations for ALL users in one bulk predict call.
        This is dramatically faster than calling recommend() per user.

        Returns  {uid: [(iid, score), ...]}
        """
        # Build the full (uid × item) testset — skip items seen in training
        rows = []
        user_seen: dict = {}
        for uid in uids:
            try:
                inner_uid = self._trainset.to_inner_uid(uid)
                user_seen[uid] = {
                    self._trainset.to_raw_iid(iid)
                    for iid, _ in self._trainset.ur[inner_uid]
                }
            except ValueError:
                user_seen[uid] = set()

            for iid in all_items:
                if iid not in user_seen[uid]:
                    rows.append((uid, iid, 0))

        # one bulk predict call
        preds = self.algo.test(rows)

        # group by user and sort
        from collections import defaultdict
        grouped = defaultdict(list)
        for p in preds:
            grouped[p.uid].append((p.iid, p.est))

        result = {}
        for uid in uids:
            result[uid] = sorted(grouped[uid], key=lambda x: x[1], reverse=True)[:n]

        return result

    def __repr__(self):
        return f"RecommenderModel(name={self.name!r})"


# ── model catalogues ──────────────────────────────────────────────────────────

def build_fast_models() -> List[RecommenderModel]:
    """
    11 models that run quickly (no SVD++, no NMF).
    Recommended for the default benchmark run.
    """
    return [
        # Baselines
        RecommenderModel(NormalPredictor(),
                         "Baseline: Random (Normal)"),
        RecommenderModel(BaselineOnly(bsl_options={"method": "als", "n_epochs": 20}),
                         "Baseline: ALS Biased (b_u + b_i)"),
        RecommenderModel(BaselineOnly(bsl_options={"method": "sgd", "n_epochs": 20,
                                                   "learning_rate": 0.005, "reg": 0.1}),
                         "Baseline: SGD Biased (b_u + b_i)"),

        # User-Based KNN
        RecommenderModel(KNNBasic(k=40,    sim_options={"name": "cosine",  "user_based": True}),
                         "UserKNN: Basic Cosine (k=40)"),
        RecommenderModel(KNNWithMeans(k=40, sim_options={"name": "pearson", "user_based": True}),
                         "UserKNN: WithMeans Pearson (k=40)"),
        RecommenderModel(KNNWithZScore(k=40, sim_options={"name": "pearson", "user_based": True}),
                         "UserKNN: WithZScore Pearson (k=40)"),

        # Item-Based KNN
        RecommenderModel(KNNBasic(k=40,    sim_options={"name": "cosine",           "user_based": False}),
                         "ItemKNN: Basic Cosine (k=40)"),
        RecommenderModel(KNNWithMeans(k=40, sim_options={"name": "pearson_baseline", "user_based": False}),
                         "ItemKNN: WithMeans Pearson-Baseline (k=40)"),

        # Matrix Factorisation
        RecommenderModel(SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02),
                         "MF: SVD (50 factors)"),
        RecommenderModel(SlopeOne(),
                         "MF: SlopeOne"),
        RecommenderModel(CoClustering(n_cltr_u=3, n_cltr_i=3, n_epochs=20),
                         "MF: Co-Clustering (3×3)"),
    ]


def build_all_models() -> List[RecommenderModel]:
    """Full catalogue including slow models (SVD++, NMF)."""
    models = build_fast_models()
    models += [
        RecommenderModel(SVDpp(n_factors=20, n_epochs=15, lr_all=0.005, reg_all=0.02),
                         "MF: SVD++ (20 factors)  [slow]"),
        RecommenderModel(NMF(n_factors=15, n_epochs=30),
                         "MF: NMF (15 factors)  [slow]"),
        RecommenderModel(KNNBasic(k=40, sim_options={"name": "pearson", "user_based": True}),
                         "UserKNN: Basic Pearson (k=40)"),
        RecommenderModel(KNNWithZScore(k=40, sim_options={"name": "pearson_baseline", "user_based": False}),
                         "ItemKNN: WithZScore Pearson-Baseline (k=40)"),
    ]
    return models
