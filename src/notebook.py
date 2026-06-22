"""
notebook.py
===========
Interactive entry-point intended to be run cell-by-cell in Jupyter or
as a standalone script.  Demonstrates the full pipeline end-to-end.

Run:
    python src/notebook.py
or in Jupyter:
    %run src/notebook.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cell 1 – Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark import run_benchmark, print_results_table, plot_results
from src.data_loader import load_raw, build_rating_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 – Quick data exploration
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Data Exploration ===")
data = load_raw()

print(f"\nVendors       : {len(data['vendors'])} restaurants")
print(f"Train customers: {len(data['train_customers'])} users")
print(f"Train locations: {len(data['train_locations'])} locations")
print(f"Orders         : {len(data['orders'])} transactions")

rating_df = build_rating_matrix(data)
print(f"\nRating matrix shape : {rating_df.shape}")
print(f"Rating statistics:\n{rating_df['rating'].describe().round(3)}")

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 – Run benchmark (fast mode: 3-fold CV, 300 sampled users)
# ─────────────────────────────────────────────────────────────────────────────
K = 10
results_df = run_benchmark(k=K, cv_folds=3, sample_users=300)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 – Print table & save charts
# ─────────────────────────────────────────────────────────────────────────────
print_results_table(results_df, k=K)
plot_results(results_df, k=K)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 – Highlight best model per metric
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Best Model Per Metric ===")

lower_is_better = {"RMSE", "MSE", "MAE", "TrainTime_s"}

for col in results_df.columns:
    if results_df[col].isna().all():
        continue
    if col in lower_is_better:
        best_idx = results_df[col].idxmin()
        best_val = results_df[col].min()
    else:
        best_idx = results_df[col].idxmax()
        best_val = results_df[col].max()
    print(f"  {col:<22}: {best_idx[:55]:<55}  ({best_val:.4f})")
