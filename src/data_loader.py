"""
data_loader.py  (optimised)
===========================
Loads Akeed dataset and builds a compact rating matrix.

Key optimisations vs the original version
------------------------------------------
* Only `orders.csv` is loaded — we never touch the 3 GB train_full.csv.
* Active-user filtering: users with fewer than MIN_ORDERS interactions are
  dropped, drastically shrinking the matrix without losing signal.
* The aggregated rating matrix is cached to disk as a Parquet file so the
  expensive build step runs only once.

Implicit Rating Formula  (unchanged, documented here for reference)
--------------------------------------------------------------------
    R_implicit = w1 * freq_norm + w2 * spend_norm + w3 * rated_bonus
                                                    rescaled → [1, 5]

Explicit Rating blending (when vendor_rating is present)
--------------------------------------------------------
    R_final = 0.6 * R_explicit + 0.4 * R_implicit
"""

import os
import pandas as pd
import numpy as np
from surprise import Dataset, Reader

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ARCHIVE      = os.path.join(_PROJECT_ROOT, "archive")
_CACHE_DIR    = os.path.join(_PROJECT_ROOT, ".cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# ── tunable constants ─────────────────────────────────────────────────────────
W_FREQ     = 0.50
W_SPEND    = 0.35
W_RATED    = 0.15
RATING_MIN = 1.0
RATING_MAX = 5.0
MIN_ORDERS = 2          # drop users with fewer than this many orders


# ── helpers ───────────────────────────────────────────────────────────────────

def _minmax_per_user(series: pd.Series, group_col: pd.Series) -> pd.Series:
    df   = pd.DataFrame({"val": series, "uid": group_col})
    mins = df.groupby("uid")["val"].transform("min")
    maxs = df.groupby("uid")["val"].transform("max")
    return (series - mins) / (maxs - mins).replace(0, 1)

def _rescale(x: pd.Series, lo=1.0, hi=5.0) -> pd.Series:
    return lo + x * (hi - lo)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_raw() -> dict:
    """Load only the small files we actually need (skip train_full.csv)."""
    files = {
        "orders"          : "orders.csv",
        "train_customers" : "train_customers.csv",
        "train_locations" : "train_locations.csv",
        "vendors"         : "vendors.csv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(_ARCHIVE, fname)
        print(f"  Loading {fname} ...", end=" ", flush=True)
        data[key] = pd.read_csv(path, low_memory=False)
        print(f"({len(data[key]):,} rows)")
    return data


def build_rating_matrix(data: dict, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build (user_id, item_id, rating) matrix with disk caching.

    Parameters
    ----------
    force_rebuild : ignore cache and rebuild from scratch.
    """
    cache_path = os.path.join(_CACHE_DIR, "rating_matrix.parquet")

    if not force_rebuild and os.path.exists(cache_path):
        print(f"  ✔ Loading cached rating matrix from {cache_path}")
        return pd.read_parquet(cache_path)

    orders = data["orders"][["customer_id", "vendor_id",
                              "grand_total", "vendor_rating"]].copy()
    orders.columns = ["user_id", "item_id", "spend", "explicit_rating"]
    orders["user_id"] = orders["user_id"].astype(str).str.strip()
    orders["item_id"] = orders["item_id"].astype(str).str.strip()
    orders["spend"]   = pd.to_numeric(orders["spend"], errors="coerce").fillna(0.0)

    # ── filter to active users only ───────────────────────────────────────────
    user_counts = orders["user_id"].value_counts()
    active_users = user_counts[user_counts >= MIN_ORDERS].index
    orders = orders[orders["user_id"].isin(active_users)]
    print(f"  Active users (≥{MIN_ORDERS} orders): {len(active_users):,}")

    # ── aggregate to (user, item) ─────────────────────────────────────────────
    agg = orders.groupby(["user_id", "item_id"]).agg(
        order_count     = ("spend", "count"),
        total_spend     = ("spend", "sum"),
        explicit_rating = ("explicit_rating", "mean"),
    ).reset_index()

    agg["freq_norm"]   = _minmax_per_user(agg["order_count"].astype(float), agg["user_id"])
    agg["spend_norm"]  = _minmax_per_user(agg["total_spend"], agg["user_id"])
    agg["rated_bonus"] = agg["explicit_rating"].notna().astype(float)
    agg["implicit_rating"] = _rescale(
        W_FREQ * agg["freq_norm"] + W_SPEND * agg["spend_norm"] + W_RATED * agg["rated_bonus"]
    )

    has_exp = agg["explicit_rating"].notna()
    agg.loc[ has_exp, "rating"] = (0.6 * agg.loc[has_exp, "explicit_rating"]
                                 + 0.4 * agg.loc[has_exp, "implicit_rating"])
    agg.loc[~has_exp, "rating"] = agg.loc[~has_exp, "implicit_rating"]
    agg["rating"] = agg["rating"].clip(RATING_MIN, RATING_MAX)

    result = agg[["user_id", "item_id", "rating"]].copy()

    result.to_parquet(cache_path, index=False)
    print(f"  ✔ Rating matrix: {len(result):,} pairs | "
          f"users={result['user_id'].nunique():,} | "
          f"vendors={result['item_id'].nunique():,} | cached → {cache_path}")
    return result


def to_surprise_dataset(rating_df: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    return Dataset.load_from_df(rating_df[["user_id", "item_id", "rating"]], reader)


def get_all_item_ids(data: dict) -> list:
    return data["vendors"]["id"].astype(str).str.strip().tolist()
