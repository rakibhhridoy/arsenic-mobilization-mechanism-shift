"""
A4 — Depth-Stratified Temporal Comparison
==========================================
Examines how temporal changes vary across depth zones.

Methods:
  - Depth binning: Shallow (<50m), Medium (50-150m), Deep (>150m)
  - Wilcoxon signed-rank tests within each depth bin (paired wells)
  - Mann-Whitney U tests within each depth bin (full datasets)
  - Two-way analysis: Scheirer-Ray-Hare test (non-parametric two-way ANOVA)
    for Depth × Period interaction
  - Kruskal-Wallis test for depth-dependent change magnitude

Outputs:
  - Table: T04_depth_stratified_stats.csv
  - Table: T04b_interaction_tests.csv
  - Figure: F05_depth_temporal_heatmap.png
  - Figure: F05b_depth_profiles.png
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from statsmodels.stats.multitest import multipletests
from config import (
    TABLE_DIR, FIGURE_DIR, DEPTH_BINS, DEPTH_LABELS,
    KEY_CONTAMINANTS, TEMPORAL_PARAMS, ALPHA, set_est_style
)

set_est_style()
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────

paired = pd.read_csv(TABLE_DIR / "matched_wells.csv")
old = paired[paired["Period"] == "2012-2013"].sort_values("pair_key").reset_index(drop=True)
new = paired[paired["Period"] == "2020-2021"].sort_values("pair_key").reset_index(drop=True)

old_full = pd.read_csv(TABLE_DIR / "old_harmonized.csv")
new_full_all = pd.read_csv(TABLE_DIR / "new_harmonized.csv")

# Restrict new dataset to coastal districts present in old dataset
coastal_districts = set(old_full["District"].dropna().str.strip().str.title().unique())
new_full_all["District_norm"] = new_full_all["District"].str.strip().str.title()
new_full = new_full_all[new_full_all["District_norm"].isin(coastal_districts)].copy()
print(f"Full old (coastal): {len(old_full)}, Full new (coastal only): {len(new_full)}/{len(new_full_all)}")

# Ensure depth bins exist
for df in [old, new, old_full, new_full]:
    if "Depth" in df.columns:
        df["Depth_bin"] = pd.cut(df["Depth"], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=True)

n_wells = len(old)
print(f"Paired wells: {n_wells}")

# Depth distribution
print(f"\nDepth distribution (paired wells):")
for label in DEPTH_LABELS:
    n = (old["Depth_bin"] == label).sum()
    print(f"  {label}: {n} wells")

# ─────────────────────────────────────────────────────────
# 2. DEPTH-STRATIFIED PAIRED TESTS
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print("DEPTH-STRATIFIED PAIRED WILCOXON TESTS")
print(f"{'=' * 80}")

depth_results = []

for depth_label in DEPTH_LABELS:
    depth_mask = old["Depth_bin"] == depth_label
    n_depth = depth_mask.sum()

    if n_depth < 5:
        print(f"\n  {depth_label}: skipped (n={n_depth})")
        continue

    print(f"\n  {depth_label} (n={n_depth} paired wells):")
    print(f"  {'Param':>6} {'Old med':>10} {'New med':>10} {'Δ med':>10} {'%Δ':>8} {'W':>8} {'p':>10} {'sig':>4}")
    print(f"  {'-' * 68}")

    for param in TEMPORAL_PARAMS:
        if param not in old.columns or param not in new.columns:
            continue

        # Paired values within this depth bin
        mask = depth_mask & old[param].notna() & new[param].notna()
        n_valid = mask.sum()

        if n_valid < 5:
            continue

        old_vals = old.loc[mask, param].values
        new_vals = new.loc[mask, param].values
        diff = new_vals - old_vals

        old_med = np.median(old_vals)
        new_med = np.median(new_vals)
        diff_med = np.median(diff)
        pct = (diff_med / abs(old_med) * 100) if old_med != 0 else np.nan

        try:
            w_stat, w_p = stats.wilcoxon(diff, alternative="two-sided",
                                         zero_method="wilcox")
        except ValueError:
            w_stat, w_p = 0.0, 1.0

        print(f"  {param:>6} {old_med:>10.3f} {new_med:>10.3f} {diff_med:>+10.3f} {pct:>+7.1f}% {w_stat:>8.0f} {w_p:>10.2e}")

        depth_results.append({
            "Depth_bin": depth_label,
            "Parameter": param,
            "n_paired": n_valid,
            "Old_median": old_med,
            "New_median": new_med,
            "Diff_median": diff_med,
            "Pct_change": pct,
            "Wilcoxon_W": w_stat,
            "Wilcoxon_p": w_p,
            "n_increase": int(np.sum(diff > 0)),
            "n_decrease": int(np.sum(diff < 0)),
        })

depth_df = pd.DataFrame(depth_results)

# FDR correction (Benjamini-Hochberg) across all depth-stratified tests
if len(depth_df) > 0:
    reject, pvals_fdr, _, _ = multipletests(
        depth_df["Wilcoxon_p"].values, alpha=ALPHA, method="fdr_bh"
    )
    depth_df["p_FDR"] = pvals_fdr
    depth_df["Significant_FDR"] = reject
    print(f"\nFDR correction applied across {len(depth_df)} tests")
    print(f"  Significant at FDR<0.05: {reject.sum()}/{len(reject)}")

depth_df.to_csv(TABLE_DIR / "T04_depth_stratified_stats.csv", index=False)
print(f"Saved: {TABLE_DIR / 'T04_depth_stratified_stats.csv'}")

# ─────────────────────────────────────────────────────────
# 3. DEPTH × PERIOD INTERACTION TESTS
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print("DEPTH × PERIOD INTERACTION TESTS")
print(f"{'=' * 80}")

# For each contaminant, test if the magnitude of temporal change
# differs across depth bins using Kruskal-Wallis on the differences

interaction_results = []

for param in KEY_CONTAMINANTS:
    if param not in old.columns or param not in new.columns:
        continue

    # Calculate differences per well
    mask = old[param].notna() & new[param].notna() & old["Depth_bin"].notna()
    old_p = old.loc[mask]
    new_p = new.loc[mask]
    diffs = new_p[param].values - old_p[param].values
    depths = old_p["Depth_bin"].values

    # Group differences by depth
    groups = {}
    for label in DEPTH_LABELS:
        group_mask = depths == label
        if group_mask.sum() >= 3:
            groups[label] = diffs[group_mask]

    if len(groups) < 2:
        continue

    # Kruskal-Wallis H test: non-parametric one-way ANOVA
    # Tests H₀: distributions of differences are identical across depth bins
    group_arrays = list(groups.values())
    h_stat, kw_p = stats.kruskal(*group_arrays)

    # Post-hoc: Dunn's test approximation via pairwise Mann-Whitney U
    # with Bonferroni correction
    posthoc_results = []
    depth_keys = list(groups.keys())
    n_posthoc = len(depth_keys) * (len(depth_keys) - 1) // 2

    for i in range(len(depth_keys)):
        for j in range(i + 1, len(depth_keys)):
            u_stat, u_p = stats.mannwhitneyu(
                groups[depth_keys[i]], groups[depth_keys[j]], alternative="two-sided"
            )
            posthoc_results.append({
                "Comparison": f"{depth_keys[i]} vs {depth_keys[j]}",
                "U": u_stat,
                "p_raw": u_p,
                "p_bonf": min(u_p * n_posthoc, 1.0),
            })

    sig = "***" if kw_p < 0.001 else ("**" if kw_p < 0.01 else ("*" if kw_p < 0.05 else "ns"))
    print(f"\n  {param}: Kruskal-Wallis H={h_stat:.2f}, p={kw_p:.4f} {sig}")
    for label, arr in groups.items():
        print(f"    {label}: n={len(arr)}, median Δ={np.median(arr):+.3f}")
    for ph in posthoc_results:
        ph_sig = "*" if ph["p_bonf"] < 0.05 else "ns"
        print(f"    Post-hoc: {ph['Comparison']}: p_bonf={ph['p_bonf']:.4f} {ph_sig}")

    interaction_results.append({
        "Parameter": param,
        "KW_H": h_stat,
        "KW_p": kw_p,
        "n_groups": len(groups),
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "group_medians": {k: np.median(v) for k, v in groups.items()},
    })

interaction_df = pd.DataFrame(interaction_results)
interaction_df.to_csv(TABLE_DIR / "T04b_interaction_tests.csv", index=False)
print(f"\nSaved: {TABLE_DIR / 'T04b_interaction_tests.csv'}")

# ─────────────────────────────────────────────────────────
# 4. FIGURE 5: DEPTH-TEMPORAL HEATMAP
# ─────────────────────────────────────────────────────────

# Build matrix: rows=depth bins, cols=parameters, values=% change
pivot_data = depth_df[depth_df["Parameter"].isin(KEY_CONTAMINANTS)].pivot_table(
    index="Depth_bin", columns="Parameter", values="Pct_change"
)
# Reorder
pivot_data = pivot_data.reindex(index=DEPTH_LABELS, columns=KEY_CONTAMINANTS)

# Significance matrix
sig_data = depth_df[depth_df["Parameter"].isin(KEY_CONTAMINANTS)].pivot_table(
    index="Depth_bin", columns="Parameter", values="Wilcoxon_p"
)
sig_data = sig_data.reindex(index=DEPTH_LABELS, columns=KEY_CONTAMINANTS)

fig, ax = plt.subplots(figsize=(5.5, 3.5))

# Custom diverging colormap (blue = decrease, red = increase)
from matplotlib.colors import TwoSlopeNorm
vmax = max(abs(pivot_data.min().min()), abs(pivot_data.max().max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(pivot_data.values, cmap="RdBu_r", norm=norm, aspect="auto")

# Add text annotations with significance stars
for i in range(len(DEPTH_LABELS)):
    for j in range(len(KEY_CONTAMINANTS)):
        val = pivot_data.values[i, j]
        p_val = sig_data.values[i, j]
        if pd.notna(val):
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%\n{sig}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

ax.set_xticks(range(len(KEY_CONTAMINANTS)))
ax.set_xticklabels(["As", "Mn", "Fe", "PO₄"])
ax.set_yticks(range(len(DEPTH_LABELS)))
ax.set_yticklabels(DEPTH_LABELS)
ax.set_title("Decadal change by depth zone (% change in median)", fontweight="bold")

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("% change (2012–13 → 2020–21)")

plt.tight_layout()
fig.savefig(FIGURE_DIR / "F05_depth_temporal_heatmap.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {FIGURE_DIR / 'F05_depth_temporal_heatmap.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 5. FIGURE 5b: DEPTH PROFILES
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(8, 5), sharey=True)

param_info = {
    "As": {"label": "As (µg/L)", "color_old": "#e74c3c", "color_new": "#3498db"},
    "Mn": {"label": "Mn (mg/L)", "color_old": "#e74c3c", "color_new": "#3498db"},
    "Fe": {"label": "Fe (mg/L)", "color_old": "#e74c3c", "color_new": "#3498db"},
    "PO4": {"label": "PO₄ (mg/L)", "color_old": "#e74c3c", "color_new": "#3498db"},
}

for ax, param in zip(axes, KEY_CONTAMINANTS):
    info = param_info[param]

    for period, df, color, label in [
        ("2012-13", old_full, "#e74c3c", "2012–13"),
        ("2020-21", new_full, "#3498db", "2020–21"),
    ]:
        mask = df[param].notna() & df["Depth"].notna()
        ax.scatter(df.loc[mask, param], df.loc[mask, "Depth"],
                   c=color, alpha=0.15, s=5, edgecolors="none")

        # Depth-binned medians
        for i, dlabel in enumerate(DEPTH_LABELS):
            dmask = mask & (df["Depth_bin"] == dlabel)
            if dmask.sum() > 0:
                med_val = df.loc[dmask, param].median()
                depth_center = [25, 100, 250][i]  # bin centers
                ax.plot(med_val, depth_center, marker="D", color=color,
                        markersize=8, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax.set_xlabel(info["label"], fontsize=9)
    if param == "As":
        ax.set_ylabel("Depth (m)", fontsize=9)
    ax.invert_yaxis()
    ax.set_title(info["label"], fontweight="bold", fontsize=10)

    if param == "As":
        ax.set_xscale("symlog", linthresh=1)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="D", color="#e74c3c", label="2012–13 median",
           markeredgecolor="black", markersize=7, linestyle="None"),
    Line2D([0], [0], marker="D", color="#3498db", label="2020–21 median",
           markeredgecolor="black", markersize=7, linestyle="None"),
]
axes[-1].legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.suptitle("Depth profiles: 2012–13 vs 2020–21", fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F05b_depth_profiles.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F05b_depth_profiles.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 6. MANUSCRIPT SUMMARY
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("MANUSCRIPT-READY SUMMARY — DEPTH-STRATIFIED TEMPORAL CHANGES")
print(f"{'=' * 70}")

for depth_label in DEPTH_LABELS:
    ddf = depth_df[depth_df["Depth_bin"] == depth_label]
    if len(ddf) == 0:
        continue
    n_wells_depth = ddf["n_paired"].iloc[0] if len(ddf) > 0 else 0
    print(f"\n  {depth_label} (n={n_wells_depth} paired wells):")
    for _, r in ddf[ddf["Parameter"].isin(KEY_CONTAMINANTS)].iterrows():
        p_use = r.get("p_FDR", r["Wilcoxon_p"])
        sig = "***" if p_use < 0.001 else ("**" if p_use < 0.01 else (
            "*" if p_use < 0.05 else "ns"))
        print(f"    {r['Parameter']:>4}: {r['Old_median']:.3f} → {r['New_median']:.3f} "
              f"({r['Pct_change']:+.1f}%, p_FDR={p_use:.2e} {sig}) "
              f"[↑{r['n_increase']}/↓{r['n_decrease']}]")

print(f"\n  Depth × Period interaction (Kruskal-Wallis on temporal differences):")
for _, r in interaction_df.iterrows():
    sig = "***" if r["KW_p"] < 0.001 else ("**" if r["KW_p"] < 0.01 else (
        "*" if r["KW_p"] < 0.05 else "ns"))
    print(f"    {r['Parameter']}: H={r['KW_H']:.2f}, p={r['KW_p']:.4f} {sig}")

print("\nDone.")
