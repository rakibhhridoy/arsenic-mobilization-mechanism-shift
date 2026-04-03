"""
A10 — Sensitivity Analysis: Outlier Robustness + CBE Filtering Impact
======================================================================
Tests whether key findings are robust to:
  1. Removal of extreme outliers (>3×IQR) in As, Mn, Fe, PO4
  2. Exclusion of samples with CBE > ±10%
  3. Combined (both filters)

Re-runs:
  - A1 Wilcoxon paired tests (non-parametric, should be robust)
  - A5 SEM path coefficients (parametric, may be sensitive)
  - A3 key correlations (PO4-As, Fe-As, Fe/Mn ratio)

Outputs:
  - Table: T11_sensitivity_analysis.csv
  - Finding: A10_sensitivity.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import semopy

from config import (
    TABLE_DIR, FIGURE_DIR, KEY_CONTAMINANTS, RANDOM_SEED,
    DEPTH_BINS, DEPTH_LABELS, ALPHA
)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────

paired = pd.read_csv(TABLE_DIR / "matched_wells.csv")
old = paired[paired["Period"] == "2012-2013"].sort_values("pair_key").reset_index(drop=True)
new = paired[paired["Period"] == "2020-2021"].sort_values("pair_key").reset_index(drop=True)

old_full = pd.read_csv(TABLE_DIR / "old_harmonized.csv")
new_full_all = pd.read_csv(TABLE_DIR / "new_harmonized.csv")

coastal_districts = set(old_full["District"].dropna().str.strip().str.title().unique())
new_full_all["District_norm"] = new_full_all["District"].str.strip().str.title()
new_full = new_full_all[new_full_all["District_norm"].isin(coastal_districts)].copy()

print("=" * 80)
print("A10 — SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"Paired wells: {len(old)}")
print(f"Full old: {len(old_full)}, Full new (coastal): {len(new_full)}")

# ─────────────────────────────────────────────────────────
# 2. DEFINE FILTERING FUNCTIONS
# ─────────────────────────────────────────────────────────

def identify_outliers(old_df, new_df, params=KEY_CONTAMINANTS, multiplier=3.0):
    """Identify rows with extreme outliers (>multiplier×IQR) in any key param.

    Returns boolean mask (True = keep, False = outlier to remove).
    Uses the combined distribution to set thresholds.
    """
    keep = pd.Series(True, index=old_df.index)
    outlier_counts = {}

    for param in params:
        if param not in old_df.columns or param not in new_df.columns:
            continue

        # Compute IQR from combined data
        all_vals = pd.concat([old_df[param], new_df[param]]).dropna()
        q1, q3 = all_vals.quantile(0.25), all_vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        # Flag in both old and new
        old_outlier = (old_df[param] < lower) | (old_df[param] > upper)
        new_outlier = (new_df[param] < lower) | (new_df[param] > upper)
        pair_outlier = old_outlier | new_outlier

        n_flagged = pair_outlier.sum()
        outlier_counts[param] = n_flagged
        keep &= ~pair_outlier

    return keep, outlier_counts


def cbe_filter(old_df, new_df, threshold=10.0):
    """Filter paired wells where either period has CBE > threshold.

    Returns boolean mask (True = keep).
    """
    if "CBE" not in old_df.columns or "CBE" not in new_df.columns:
        print("  WARNING: CBE column not found — run a0 first")
        return pd.Series(True, index=old_df.index)

    old_pass = old_df["CBE"].isna() | (old_df["CBE"].abs() <= threshold)
    new_pass = new_df["CBE"].isna() | (new_df["CBE"].abs() <= threshold)
    return old_pass & new_pass


# ─────────────────────────────────────────────────────────
# 3. DEFINE ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────────────────

def run_wilcoxon_tests(old_df, new_df, params=KEY_CONTAMINANTS):
    """Run paired Wilcoxon tests, return results dict."""
    results = {}
    for param in params:
        mask = old_df[param].notna() & new_df[param].notna()
        old_v = old_df.loc[mask, param].values
        new_v = new_df.loc[mask, param].values
        n = len(old_v)

        if n < 10:
            results[param] = {"n": n, "W": np.nan, "p": np.nan, "r": np.nan,
                              "median_old": np.nan, "median_new": np.nan, "pct_change": np.nan}
            continue

        w, p = stats.wilcoxon(new_v - old_v, alternative="two-sided")
        # Rank-biserial r
        diff = new_v - old_v
        ranks = stats.rankdata(np.abs(diff))
        pos_rank_sum = ranks[diff > 0].sum()
        neg_rank_sum = ranks[diff < 0].sum()
        total_rank = pos_rank_sum + neg_rank_sum
        r_rb = (pos_rank_sum - neg_rank_sum) / total_rank if total_rank > 0 else 0

        med_old = np.median(old_v)
        med_new = np.median(new_v)
        pct = (med_new - med_old) / abs(med_old) * 100 if med_old != 0 else np.nan

        results[param] = {"n": n, "W": w, "p": p, "r": r_rb,
                          "median_old": med_old, "median_new": med_new, "pct_change": pct}
    return results


def run_sem(df, period_label):
    """Fit SEM and return path coefficients dict."""
    sem_spec = """
    Fe ~ Eh
    Mn ~ Eh
    As ~ Fe + PO4 + Eh
    """
    sem_vars = ["As", "Fe", "Mn", "PO4", "Eh"]
    sem_data = df[sem_vars].dropna()

    if len(sem_data) < 50:
        return {}

    sem_z = (sem_data - sem_data.mean()) / sem_data.std()
    model = semopy.Model(sem_spec)
    model.fit(sem_z)
    estimates = model.inspect()

    paths = {}
    for _, row in estimates.iterrows():
        if row["op"] == "~":
            path_label = f"{row['lval']} ← {row['rval']}"
            paths[path_label] = {
                "beta": row["Estimate"],
                "se": row["Std. Err"],
                "p": row["p-value"],
            }
    return paths


def run_correlations(old_df, new_df):
    """Run key Spearman correlations for both periods."""
    results = {}
    pairs = [("PO4", "As"), ("Fe", "As"), ("Eh", "As")]
    for x_var, y_var in pairs:
        for label, df in [("old", old_df), ("new", new_df)]:
            mask = df[x_var].notna() & df[y_var].notna()
            if mask.sum() < 10:
                continue
            rho, p = stats.spearmanr(df.loc[mask, x_var], df.loc[mask, y_var])
            results[f"{x_var}-{y_var}_{label}"] = {"rho": rho, "p": p, "n": mask.sum()}
    return results


# ─────────────────────────────────────────────────────────
# 4. RUN SCENARIOS
# ─────────────────────────────────────────────────────────

# Build filter masks
outlier_keep, outlier_counts = identify_outliers(old, new)
cbe_keep = cbe_filter(old, new)
combined_keep = outlier_keep & cbe_keep

scenarios = {
    "Full (no filter)":     pd.Series(True, index=old.index),
    "Outlier removed":      outlier_keep,
    "CBE filtered":         cbe_keep,
    "Combined (both)":      combined_keep,
}

print(f"\n{'=' * 60}")
print("FILTER SUMMARY")
print(f"{'=' * 60}")
print(f"  Total paired wells: {len(old)}")
print(f"\n  Outliers flagged per parameter (>3×IQR):")
for param, count in outlier_counts.items():
    print(f"    {param}: {count} pairs")
print(f"  Pairs remaining after outlier filter: {outlier_keep.sum()}")
print(f"  Pairs remaining after CBE filter: {cbe_keep.sum()}")
print(f"  Pairs remaining after combined: {combined_keep.sum()}")

# Run all analyses for each scenario
all_results = []
findings_lines = []
findings_lines.append("A10 — SENSITIVITY ANALYSIS")
findings_lines.append("=" * 60)

for scenario_name, mask in scenarios.items():
    print(f"\n{'─' * 60}")
    print(f"SCENARIO: {scenario_name} (n={mask.sum()} pairs)")
    print(f"{'─' * 60}")

    old_s = old[mask].reset_index(drop=True)
    new_s = new[mask].reset_index(drop=True)
    n_pairs = len(old_s)

    # --- Wilcoxon tests ---
    wilcox = run_wilcoxon_tests(old_s, new_s)
    print(f"\n  Wilcoxon paired tests:")
    for param, res in wilcox.items():
        sig = "*" if res["p"] < 0.05 else "ns"
        print(f"    {param:>4}: median {res['median_old']:.3f}→{res['median_new']:.3f} "
              f"({res['pct_change']:+.1f}%), p={res['p']:.4f} {sig}, r={res['r']:.3f}")
        all_results.append({
            "Scenario": scenario_name, "n_pairs": n_pairs,
            "Analysis": "Wilcoxon", "Parameter": param,
            "Value": res["p"], "Effect_size": res["r"],
            "Pct_change": res["pct_change"],
        })

    # --- Correlations ---
    corrs = run_correlations(old_s, new_s)
    print(f"\n  Key correlations:")
    for key, res in corrs.items():
        sig = "*" if res["p"] < 0.05 else "ns"
        print(f"    {key}: ρ={res['rho']:.3f}, p={res['p']:.4f} {sig}")
        all_results.append({
            "Scenario": scenario_name, "n_pairs": n_pairs,
            "Analysis": "Correlation", "Parameter": key,
            "Value": res["p"], "Effect_size": res["rho"],
            "Pct_change": np.nan,
        })

# --- SEM (on full datasets with filters) ---
# SEM uses full datasets, not just paired wells
print(f"\n{'─' * 60}")
print("SEM SENSITIVITY (full datasets)")
print(f"{'─' * 60}")

sem_scenarios = {}

# Full SEM (no filter)
for period_label, df in [("2012-2013", old_full), ("2020-2021", new_full)]:
    sem_scenarios[f"Full_{period_label}"] = run_sem(df, period_label)

# SEM with CBE filter on full datasets
for period_label, df in [("2012-2013", old_full), ("2020-2021", new_full)]:
    if "CBE" in df.columns:
        df_f = df[(df["CBE"].isna()) | (df["CBE"].abs() <= 10.0)]
    else:
        df_f = df
    sem_scenarios[f"CBE_{period_label}"] = run_sem(df_f, period_label)

# SEM with outlier removal on full datasets
for period_label, df in [("2012-2013", old_full), ("2020-2021", new_full)]:
    keep = pd.Series(True, index=df.index)
    for param in KEY_CONTAMINANTS:
        if param in df.columns:
            vals = df[param].dropna()
            if len(vals) == 0:
                continue
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            keep &= ~((df[param] < q1 - 3 * iqr) | (df[param] > q3 + 3 * iqr)).fillna(False)
    sem_scenarios[f"Outlier_{period_label}"] = run_sem(df[keep], period_label)

print(f"\n  SEM path coefficients across scenarios:")
print(f"  {'Scenario':<25} {'Path':<20} {'Beta':>8} {'SE':>8} {'p':>10}")
print(f"  {'-' * 73}")

for scenario_key, paths in sem_scenarios.items():
    for path, vals in paths.items():
        sig = "*" if vals["p"] < 0.05 else "ns"
        print(f"  {scenario_key:<25} {path:<20} {vals['beta']:>+8.4f} {vals['se']:>8.4f} {vals['p']:>10.4f} {sig}")
        all_results.append({
            "Scenario": scenario_key, "n_pairs": np.nan,
            "Analysis": "SEM", "Parameter": path,
            "Value": vals["p"], "Effect_size": vals["beta"],
            "Pct_change": np.nan,
        })

# ─────────────────────────────────────────────────────────
# 5. ROBUSTNESS ASSESSMENT
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("ROBUSTNESS ASSESSMENT")
print(f"{'=' * 60}")

# Compare key findings across scenarios
results_df = pd.DataFrame(all_results)

# Check if Wilcoxon significance changes
print("\n  Wilcoxon significance stability:")
for param in KEY_CONTAMINANTS:
    param_rows = results_df[(results_df["Analysis"] == "Wilcoxon") &
                            (results_df["Parameter"] == param)]
    p_vals = param_rows["Value"].values
    all_sig = all(p < 0.05 for p in p_vals if not np.isnan(p))
    all_ns = all(p >= 0.05 for p in p_vals if not np.isnan(p))
    status = "STABLE sig" if all_sig else ("STABLE ns" if all_ns else "UNSTABLE")
    p_range = f"[{min(p_vals):.4f}, {max(p_vals):.4f}]"
    print(f"    {param:>4}: {status} — p range {p_range}")

# Check SEM key path stability
print("\n  SEM PO4→As path stability:")
po4_as_rows = results_df[(results_df["Analysis"] == "SEM") &
                          (results_df["Parameter"] == "As ← PO4")]
if len(po4_as_rows) > 0:
    for _, row in po4_as_rows.iterrows():
        sig = "*" if row["Value"] < 0.05 else "ns"
        print(f"    {row['Scenario']:<25}: β={row['Effect_size']:+.4f}, p={row['Value']:.4f} {sig}")

print("\n  SEM Eh→Fe path stability:")
eh_fe_rows = results_df[(results_df["Analysis"] == "SEM") &
                         (results_df["Parameter"] == "Fe ← Eh")]
if len(eh_fe_rows) > 0:
    for _, row in eh_fe_rows.iterrows():
        sig = "*" if row["Value"] < 0.05 else "ns"
        print(f"    {row['Scenario']:<25}: β={row['Effect_size']:+.4f}, p={row['Value']:.4f} {sig}")

# ─────────────────────────────────────────────────────────
# 6. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────

results_df.to_csv(TABLE_DIR / "T11_sensitivity_analysis.csv", index=False)
print(f"\nSaved: {TABLE_DIR / 'T11_sensitivity_analysis.csv'}")

# Write findings file
findings_path = TABLE_DIR.parent / "findings" / "A10_sensitivity.txt"
findings_path.parent.mkdir(parents=True, exist_ok=True)

with open(findings_path, "w") as f:
    f.write("A10 — SENSITIVITY ANALYSIS: OUTLIER & CBE ROBUSTNESS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total paired wells: {len(old)}\n")
    f.write(f"After outlier removal (>3×IQR): {outlier_keep.sum()}\n")
    f.write(f"After CBE filter (±10%): {cbe_keep.sum()}\n")
    f.write(f"After combined: {combined_keep.sum()}\n\n")

    f.write("OUTLIERS REMOVED PER PARAMETER:\n")
    for param, count in outlier_counts.items():
        f.write(f"  {param}: {count} pairs\n")

    f.write("\nWILCOXON ROBUSTNESS:\n")
    for param in KEY_CONTAMINANTS:
        param_rows = results_df[(results_df["Analysis"] == "Wilcoxon") &
                                (results_df["Parameter"] == param)]
        p_vals = param_rows["Value"].values
        all_sig = all(p < 0.05 for p in p_vals if not np.isnan(p))
        all_ns = all(p >= 0.05 for p in p_vals if not np.isnan(p))
        status = "STABLE significant" if all_sig else ("STABLE non-significant" if all_ns else "UNSTABLE")
        f.write(f"  {param}: {status} (p range [{min(p_vals):.4f}, {max(p_vals):.4f}])\n")

    f.write("\nSEM PATH ROBUSTNESS:\n")
    for path_name in ["As ← PO4", "As ← Fe", "Fe ← Eh"]:
        path_rows = results_df[(results_df["Analysis"] == "SEM") &
                               (results_df["Parameter"] == path_name)]
        if len(path_rows) > 0:
            betas = path_rows["Effect_size"].values
            f.write(f"  {path_name}: β range [{min(betas):+.4f}, {max(betas):+.4f}]\n")

    f.write("\nCONCLUSION:\n")
    f.write("Results reported in the findings file above. Check T11_sensitivity_analysis.csv\n")
    f.write("for the full table of results across all scenarios.\n")

print(f"Saved: {findings_path}")
print("\nDone.")
