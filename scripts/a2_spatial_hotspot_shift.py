"""
A2 — Spatial Hotspot Shift Analysis
====================================
Maps how arsenic and manganese exceedance zones shifted between 2012-13 and 2020-21.

Methods:
  - District-level exceedance rate comparison (WHO thresholds)
  - McNemar's test for paired exceedance status change (matched wells)
  - Chi-squared test for district-level exceedance rate change
  - Exceedance transition matrix (safe→unsafe, unsafe→safe, etc.)
  - Choropleth maps of exceedance rates by district and period

Outputs:
  - Table: T02_exceedance_transition.csv (well-level transition matrix)
  - Table: T02b_district_exceedance.csv (district-level rates)
  - Figure: F02_exceedance_maps.png (side-by-side maps)
  - Figure: F02b_transition_sankey.png (transition flows)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

from config import (
    TABLE_DIR, FIGURE_DIR, WHO_LIMITS, BD_LIMITS,
    KEY_CONTAMINANTS, set_est_style
)

set_est_style()

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────

paired = pd.read_csv(TABLE_DIR / "matched_wells.csv")
sort_key = "pair_key" if "pair_key" in paired.columns else "ID_norm"
old = paired[paired["Period"] == "2012-2013"].sort_values(sort_key).reset_index(drop=True)
new = paired[paired["Period"] == "2020-2021"].sort_values(sort_key).reset_index(drop=True)

old_full = pd.read_csv(TABLE_DIR / "old_harmonized.csv")
new_full_all = pd.read_csv(TABLE_DIR / "new_harmonized.csv")
district_summary = pd.read_csv(TABLE_DIR / "district_summary.csv")

# Restrict new dataset to coastal districts present in old dataset
coastal_districts = set(old_full["District"].dropna().str.strip().str.title().unique())
new_full_all["District_norm"] = new_full_all["District"].str.strip().str.title()
new_full = new_full_all[new_full_all["District_norm"].isin(coastal_districts)].copy()

n_wells = len(old)
print(f"Paired wells: {n_wells}")
print(f"Old full (coastal): {len(old_full)}, New full (coastal only): {len(new_full)}/{len(new_full_all)}")

# ─────────────────────────────────────────────────────────
# 2. PAIRED WELL EXCEEDANCE TRANSITIONS
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("PAIRED WELL EXCEEDANCE TRANSITIONS")
print(f"{'=' * 70}")

contaminant_thresholds = {
    "As": ("WHO", WHO_LIMITS["As"]),    # 10 µg/L
    "Mn": ("WHO", WHO_LIMITS["Mn"]),    # 0.4 mg/L
    "Fe": ("WHO", WHO_LIMITS["Fe"]),    # 0.3 mg/L
}

transition_results = []

for param, (std_name, threshold) in contaminant_thresholds.items():
    if param not in old.columns or param not in new.columns:
        continue

    mask = old[param].notna() & new[param].notna()
    old_vals = old.loc[mask, param].values
    new_vals = new.loc[mask, param].values
    n = mask.sum()

    old_exceed = old_vals > threshold
    new_exceed = new_vals > threshold

    # Transition matrix (2×2)
    # a: safe→safe, b: safe→unsafe, c: unsafe→safe, d: unsafe→unsafe
    a = np.sum(~old_exceed & ~new_exceed)  # remained safe
    b = np.sum(~old_exceed & new_exceed)   # became unsafe (deteriorated)
    c = np.sum(old_exceed & ~new_exceed)   # became safe (improved)
    d = np.sum(old_exceed & new_exceed)    # remained unsafe

    old_rate = np.sum(old_exceed) / n * 100
    new_rate = np.sum(new_exceed) / n * 100

    # McNemar's test: tests whether the number of transitions in each
    # direction (safe→unsafe vs unsafe→safe) are significantly different.
    # Uses exact binomial test when b+c < 25, chi-squared approximation otherwise.
    # H0: P(safe→unsafe) = P(unsafe→safe)
    n_discordant = b + c
    if n_discordant > 0:
        if n_discordant < 25:
            # Exact binomial test (more accurate for small samples)
            mcnemar_p = stats.binomtest(b, n_discordant, 0.5).pvalue
        else:
            # McNemar's chi-squared with continuity correction
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            mcnemar_p = stats.chi2.sf(chi2, df=1)
    else:
        mcnemar_p = 1.0

    print(f"\n  {param} ({std_name} limit: {threshold}):")
    print(f"    2012-13 exceedance: {np.sum(old_exceed)}/{n} ({old_rate:.1f}%)")
    print(f"    2020-21 exceedance: {np.sum(new_exceed)}/{n} ({new_rate:.1f}%)")
    print(f"    Transition matrix:")
    print(f"      Remained safe:    {a} ({a/n*100:.1f}%)")
    print(f"      Deteriorated:     {b} ({b/n*100:.1f}%) [safe → unsafe]")
    print(f"      Improved:         {c} ({c/n*100:.1f}%) [unsafe → safe]")
    print(f"      Remained unsafe:  {d} ({d/n*100:.1f}%)")
    print(f"    McNemar's test: p = {mcnemar_p:.4f} {'***' if mcnemar_p < 0.001 else '**' if mcnemar_p < 0.01 else '*' if mcnemar_p < 0.05 else 'ns'}")

    transition_results.append({
        "Parameter": param,
        "Threshold": threshold,
        "Standard": std_name,
        "n_paired": n,
        "Old_exceedance_n": int(np.sum(old_exceed)),
        "Old_exceedance_pct": old_rate,
        "New_exceedance_n": int(np.sum(new_exceed)),
        "New_exceedance_pct": new_rate,
        "Remained_safe": int(a),
        "Deteriorated": int(b),
        "Improved": int(c),
        "Remained_unsafe": int(d),
        "McNemar_p": mcnemar_p,
        "Net_change": int(b - c),
        "Net_change_pct": (b - c) / n * 100,
    })

transition_df = pd.DataFrame(transition_results)
transition_df.to_csv(TABLE_DIR / "T02_exceedance_transition.csv", index=False)
print(f"\nSaved: {TABLE_DIR / 'T02_exceedance_transition.csv'}")

# ─────────────────────────────────────────────────────────
# 3. DISTRICT-LEVEL EXCEEDANCE RATES
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("DISTRICT-LEVEL EXCEEDANCE RATES")
print(f"{'=' * 70}")

# Normalize district names in full datasets
old_full["District_norm"] = old_full["District"].str.strip().str.title()
new_full["District_norm"] = new_full["District"].str.strip().str.title()

common_districts = sorted(
    set(old_full["District_norm"].dropna().unique()) &
    set(new_full["District_norm"].dropna().unique())
)

district_exceed_rows = []

for district in common_districts:
    old_d = old_full[old_full["District_norm"] == district]
    new_d = new_full[new_full["District_norm"] == district]

    for param, (std_name, threshold) in contaminant_thresholds.items():
        if param not in old_d.columns or param not in new_d.columns:
            continue

        old_vals = old_d[param].dropna()
        new_vals = new_d[param].dropna()

        if len(old_vals) < 3 or len(new_vals) < 3:
            continue

        old_exceed = (old_vals > threshold).sum()
        new_exceed = (new_vals > threshold).sum()
        old_rate = old_exceed / len(old_vals) * 100
        new_rate = new_exceed / len(new_vals) * 100

        # Fisher's exact test for 2×2 contingency table
        # (exceed vs not-exceed) × (old vs new)
        # More appropriate than chi-squared for small cell counts
        table = np.array([
            [old_exceed, len(old_vals) - old_exceed],
            [new_exceed, len(new_vals) - new_exceed]
        ])
        odds_ratio, fisher_p = stats.fisher_exact(table)

        district_exceed_rows.append({
            "District": district,
            "Parameter": param,
            "Threshold": threshold,
            "Old_n": len(old_vals),
            "Old_exceed": old_exceed,
            "Old_rate_pct": old_rate,
            "New_n": len(new_vals),
            "New_exceed": new_exceed,
            "New_rate_pct": new_rate,
            "Rate_change_pp": new_rate - old_rate,
            "Odds_ratio": odds_ratio,
            "Fisher_p": fisher_p,
        })

district_exceed_df = pd.DataFrame(district_exceed_rows)

# Apply Benjamini-Hochberg FDR correction across all district-level Fisher's tests
from statsmodels.stats.multitest import multipletests
if len(district_exceed_df) > 0:
    reject, pvals_fdr, _, _ = multipletests(
        district_exceed_df["Fisher_p"].values, alpha=0.05, method="fdr_bh"
    )
    district_exceed_df["Fisher_p_FDR"] = pvals_fdr
    district_exceed_df["FDR_sig"] = reject

district_exceed_df.to_csv(TABLE_DIR / "T02b_district_exceedance.csv", index=False)

# Print summary
for param in ["As", "Mn"]:
    pdf = district_exceed_df[district_exceed_df["Parameter"] == param]
    if len(pdf) == 0:
        continue
    print(f"\n  {param} (WHO {contaminant_thresholds[param][1]}):")
    print(f"  {'District':<15} {'Old%':>6} {'New%':>6} {'Δpp':>7} {'OR':>6} {'p_raw':>8} {'p_FDR':>8}")
    print(f"  {'-' * 62}")
    for _, r in pdf.sort_values("Rate_change_pp", ascending=False).iterrows():
        sig = "*" if r.get("FDR_sig", r["Fisher_p"] < 0.05) else ""
        print(f"  {r['District']:<15} {r['Old_rate_pct']:>5.1f}% {r['New_rate_pct']:>5.1f}% "
              f"{r['Rate_change_pp']:>+6.1f} {r['Odds_ratio']:>6.2f} {r['Fisher_p']:>7.3f} {r.get('Fisher_p_FDR', r['Fisher_p']):>7.3f}{sig}")

print(f"\nSaved: {TABLE_DIR / 'T02b_district_exceedance.csv'}")

# ─────────────────────────────────────────────────────────
# 4. FIGURE 2: SPATIAL EXCEEDANCE MAPS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(8, 8.5))

# Use scatter plots on lat/lon (no shapefile needed)
param_plot = {"As": 0, "Mn": 1}
period_labels = {"2012-2013": 0, "2020-2021": 1}

for row_idx, (param, threshold_info) in enumerate(
    [("As", WHO_LIMITS["As"]), ("Mn", WHO_LIMITS["Mn"])]
):
    for col_idx, (period, df_full) in enumerate(
        [("2012-2013", old_full), ("2020-2021", new_full)]
    ):
        ax = axes[row_idx, col_idx]

        if param not in df_full.columns:
            ax.set_visible(False)
            continue

        mask = df_full[param].notna() & df_full["Latitude"].notna() & df_full["Longitude"].notna()
        lats = df_full.loc[mask, "Latitude"].values
        lons = df_full.loc[mask, "Longitude"].values
        vals = df_full.loc[mask, param].values

        exceed = vals > threshold_info
        safe = ~exceed

        # Plot safe wells (gray, smaller)
        ax.scatter(lons[safe], lats[safe], c="steelblue", s=8, alpha=0.3,
                   edgecolors="none", label="Below threshold")
        # Plot exceedance wells (red, larger)
        ax.scatter(lons[exceed], lats[exceed], c="crimson", s=15, alpha=0.6,
                   edgecolors="none", label="Above threshold")

        exceed_pct = np.sum(exceed) / len(vals) * 100
        ax.set_title(f"{param} — {period}\n({np.sum(exceed)}/{len(vals)} = {exceed_pct:.1f}% exceed)",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=8)
        ax.set_ylabel("Latitude (°N)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Bangladesh approximate bounds
        ax.set_xlim(88, 93)
        ax.set_ylim(20.5, 26.8)
        ax.set_aspect("equal")

        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.8)

plt.suptitle("Spatial distribution of WHO exceedances", fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F02_exceedance_maps.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {FIGURE_DIR / 'F02_exceedance_maps.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 5. FIGURE 2b: TRANSITION BAR CHARTS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(8, 3.5))

colors = {
    "Remained safe": "#2ecc71",
    "Deteriorated": "#e74c3c",
    "Improved": "#3498db",
    "Remained unsafe": "#95a5a6",
}

for ax, (_, row) in zip(axes, transition_df.iterrows()):
    param = row["Parameter"]
    vals = [row["Remained_safe"], row["Deteriorated"],
            row["Improved"], row["Remained_unsafe"]]
    labels = list(colors.keys())
    cols = [colors[l] for l in labels]

    bars = ax.bar(range(4), vals, color=cols, edgecolor="white", width=0.7)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Safe→\nSafe", "Safe→\nUnsafe", "Unsafe→\nSafe", "Unsafe→\nUnsafe"],
                       fontsize=7)
    ax.set_ylabel("Number of wells", fontsize=8)

    threshold = row["Threshold"]
    unit = "µg/L" if param == "As" else "mg/L"
    sig_str = "***" if row["McNemar_p"] < 0.001 else ("**" if row["McNemar_p"] < 0.01 else
              ("*" if row["McNemar_p"] < 0.05 else "ns"))
    ax.set_title(f"{param} (>{threshold} {unit})\nMcNemar {sig_str}", fontsize=9, fontweight="bold")

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(int(val)), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig.savefig(FIGURE_DIR / "F02b_transition_bars.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F02b_transition_bars.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 6. MANUSCRIPT SUMMARY
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("MANUSCRIPT-READY SUMMARY — SPATIAL HOTSPOT SHIFT")
print(f"{'=' * 70}")

for _, r in transition_df.iterrows():
    param = r["Parameter"]
    unit = "µg/L" if param == "As" else "mg/L"
    print(f"\n  {param} (WHO {r['Threshold']} {unit}, n={r['n_paired']:.0f} paired wells):")
    print(f"    Exceedance: {r['Old_exceedance_pct']:.1f}% → {r['New_exceedance_pct']:.1f}%")
    print(f"    Net deterioration: {r['Net_change']:.0f} wells ({r['Net_change_pct']:+.1f}pp)")
    print(f"    {r['Deteriorated']:.0f} wells deteriorated, {r['Improved']:.0f} improved")
    mcn = f"p = {r['McNemar_p']:.4f}" if r["McNemar_p"] >= 0.001 else "p < 0.001"
    print(f"    McNemar's test: {mcn}")

# Multi-contaminant co-exceedance
print(f"\n  Multi-contaminant co-exceedance (paired wells):")
for period, df in [("2012-13", old), ("2020-21", new)]:
    mask_as = df["As"].notna() & (df["As"] > WHO_LIMITS["As"])
    mask_mn = df["Mn"].notna() & (df["Mn"] > WHO_LIMITS["Mn"])
    both = (mask_as & mask_mn).sum()
    either = (mask_as | mask_mn).sum()
    n_valid = (df["As"].notna() & df["Mn"].notna()).sum()
    print(f"    {period}: As+Mn co-exceedance = {both}/{n_valid} ({both/n_valid*100:.1f}%), "
          f"either = {either}/{n_valid} ({either/n_valid*100:.1f}%)")

print("\nDone.")
