"""
A1 — Paired Well Temporal Statistics
=====================================
Formal statistical tests on 235 matched wells comparing 2012-13 vs 2020-21.

Methods:
  - Wilcoxon signed-rank test (non-parametric paired test for non-normal data)
  - Hodges-Lehmann estimator (robust estimate of median difference)
  - Cliff's delta (non-parametric effect size, robust to outliers)
  - Bootstrap 95% CIs for median difference (5000 iterations, BCa method)
  - Bonferroni correction for multiple comparisons
  - Shapiro-Wilk normality test on differences (justifies non-parametric choice)

Outputs:
  - Table: T01_paired_wilcoxon.csv (full results table)
  - Figure: F01_paired_before_after.png (connected dot plots)
  - Figure: F01b_paired_change_distributions.png (violin/box of differences)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from config import (
    TABLE_DIR, FIGURE_DIR, TEMPORAL_PARAMS, KEY_CONTAMINANTS,
    ALPHA, N_BOOTSTRAP, RANDOM_SEED, set_est_style
)

set_est_style()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 1. LOAD PAIRED DATA
# ─────────────────────────────────────────────────────────

paired = pd.read_csv(TABLE_DIR / "matched_wells.csv")
sort_key = "pair_key" if "pair_key" in paired.columns else "ID_norm"
old = paired[paired["Period"] == "2012-2013"].sort_values(sort_key).reset_index(drop=True)
new = paired[paired["Period"] == "2020-2021"].sort_values(sort_key).reset_index(drop=True)

assert len(old) == len(new), "Paired datasets must have equal length"
assert (old[sort_key].values == new[sort_key].values).all(), "Well keys must match"

n_wells = len(old)
print(f"Paired wells: {n_wells}")
print(f"Parameters to test: {TEMPORAL_PARAMS}")

# ─────────────────────────────────────────────────────────
# 2. STATISTICAL FUNCTIONS
# ─────────────────────────────────────────────────────────

def rank_biserial_r(W, n):
    """
    Matched-pairs rank-biserial correlation from Wilcoxon signed-rank test.

    The appropriate effect size for paired non-parametric data.
    r = 1 - (2T / (n(n+1)/2))
    where T = Wilcoxon statistic (sum of ranks of less frequent sign).

    Interpretation (Cohen-like thresholds adapted for r):
      |r| < 0.10: negligible
      |r| < 0.30: small
      |r| < 0.50: medium
      |r| ≥ 0.50: large

    Reference: Kerby (2014), Simple Differences and Effect Sizes,
    doi:10.9734/BJMCS/2014/7571
    """
    max_W = n * (n + 1) / 2
    r = 1 - (2 * W / max_W)
    return r


def effect_size_interpret(r):
    """Interpret rank-biserial r magnitude."""
    r = abs(r)
    if r < 0.10:
        return "negligible"
    elif r < 0.30:
        return "small"
    elif r < 0.50:
        return "medium"
    else:
        return "large"


def hodges_lehmann(diff):
    """
    Hodges-Lehmann estimator: median of all pairwise averages of differences.
    Robust estimate of the "typical" shift, associated with the Wilcoxon test.

    HL = median{ (d_i + d_j) / 2 : i ≤ j }
    """
    n = len(diff)
    # Walsh averages: all (d_i + d_j)/2 for i <= j
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append((diff[i] + diff[j]) / 2.0)
    return np.median(walsh)


def bootstrap_ci_bca(diff, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """
    BCa (bias-corrected and accelerated) bootstrap confidence interval
    for the median difference.

    More accurate than percentile bootstrap for skewed distributions.
    Reference: Efron & Tibshirani (1993), Chapter 14.
    """
    n = len(diff)
    theta_hat = np.median(diff)

    # Generate bootstrap replicates
    boot_medians = np.array([
        np.median(np.random.choice(diff, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Bias correction factor (z0)
    prop_below = np.mean(boot_medians < theta_hat)
    # Clip to avoid inf from ppf(0) or ppf(1)
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_below)

    # Acceleration factor (a) via jackknife
    jackknife_medians = np.array([
        np.median(np.delete(diff, i)) for i in range(n)
    ])
    jack_mean = jackknife_medians.mean()
    numerator = np.sum((jack_mean - jackknife_medians) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_medians) ** 2)) ** 1.5
    a = numerator / denominator if denominator != 0 else 0.0

    # BCa percentiles
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    # Adjusted percentiles
    p_lo = stats.norm.cdf(z0 + (z0 + z_alpha_lo) / (1 - a * (z0 + z_alpha_lo)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_alpha_hi) / (1 - a * (z0 + z_alpha_hi)))

    # Clip and get quantiles
    p_lo = np.clip(p_lo, 0.5 / n_boot, 1 - 0.5 / n_boot)
    p_hi = np.clip(p_hi, 0.5 / n_boot, 1 - 0.5 / n_boot)

    ci_lo = np.quantile(boot_medians, p_lo)
    ci_hi = np.quantile(boot_medians, p_hi)

    return ci_lo, ci_hi


# ─────────────────────────────────────────────────────────
# 3. RUN TESTS FOR ALL PARAMETERS
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 80}")
print(f"PAIRED WILCOXON SIGNED-RANK TESTS (n={n_wells} wells, α={ALPHA})")
print(f"{'=' * 80}")

n_tests = len(TEMPORAL_PARAMS)
bonferroni_alpha = ALPHA / n_tests
print(f"Bonferroni-corrected α = {ALPHA}/{n_tests} = {bonferroni_alpha:.4f}\n")

results = []

for param in TEMPORAL_PARAMS:
    if param not in old.columns or param not in new.columns:
        continue

    # Get paired values where both are available
    mask = old[param].notna() & new[param].notna()
    x_old = old.loc[mask, param].values
    x_new = new.loc[mask, param].values
    diff = x_new - x_old
    n_paired = len(diff)

    if n_paired < 10:
        print(f"  {param}: skipped (only {n_paired} paired values)")
        continue

    # Shapiro-Wilk normality test on differences
    if n_paired <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Wilcoxon signed-rank test (two-sided)
    # zero_method='wilcox': discard zero differences (Wilcoxon 1945 convention)
    try:
        wilcox_stat, wilcox_p = stats.wilcoxon(
            diff, alternative="two-sided", zero_method="wilcox"
        )
    except ValueError:
        # All differences are zero
        wilcox_stat, wilcox_p = 0.0, 1.0

    # Effect size: matched-pairs rank-biserial correlation
    # Appropriate for paired data (unlike Cliff's delta which is for unpaired)
    n_nonzero = np.sum(diff != 0)
    rbc = rank_biserial_r(wilcox_stat, n_nonzero) if n_nonzero > 0 else 0.0
    rbc_interp = effect_size_interpret(rbc)

    # Hodges-Lehmann estimator of median shift
    # O(n²) — tractable for n up to ~1000
    hl_est = hodges_lehmann(diff)

    # Bootstrap BCa 95% CI for median difference
    ci_lo, ci_hi = bootstrap_ci_bca(diff)

    # Summary statistics
    old_median = np.median(x_old)
    new_median = np.median(x_new)
    diff_median = np.median(diff)
    pct_change = (diff_median / abs(old_median) * 100) if old_median != 0 else np.nan

    # Direction counts
    n_increase = np.sum(diff > 0)
    n_decrease = np.sum(diff < 0)
    n_zero = np.sum(diff == 0)

    # Significance after Bonferroni
    sig_bonf = wilcox_p < bonferroni_alpha

    results.append({
        "Parameter": param,
        "n_paired": n_paired,
        "Old_median": old_median,
        "New_median": new_median,
        "Diff_median": diff_median,
        "Pct_change_median": pct_change,
        "HL_estimate": hl_est,
        "Bootstrap_CI_lo": ci_lo,
        "Bootstrap_CI_hi": ci_hi,
        "n_increase": n_increase,
        "n_decrease": n_decrease,
        "n_zero": n_zero,
        "Wilcoxon_stat": wilcox_stat,
        "Wilcoxon_p": wilcox_p,
        "Bonferroni_sig": sig_bonf,
        "Rank_biserial_r": rbc,
        "Effect_size_interp": rbc_interp,
        "Shapiro_W": shapiro_stat,
        "Shapiro_p": shapiro_p,
        "Normal_dist": shapiro_p > 0.05 if pd.notna(shapiro_p) else None,
    })

    # Print
    sig_str = "***" if wilcox_p < 0.001 else ("**" if wilcox_p < 0.01 else ("*" if wilcox_p < 0.05 else "ns"))
    bonf_str = " [Bonf]" if sig_bonf else ""
    print(f"  {param:>5}: median {old_median:.3f} → {new_median:.3f} "
          f"(Δ={diff_median:+.3f}, {pct_change:+.1f}%) "
          f"| W={wilcox_stat:.0f}, p={wilcox_p:.2e} {sig_str}{bonf_str} "
          f"| r={rbc:+.3f} ({rbc_interp}) "
          f"| HL={hl_est:+.3f} "
          f"| 95%CI [{ci_lo:.3f}, {ci_hi:.3f}] "
          f"| ↑{n_increase}/↓{n_decrease}")

# Save results table
results_df = pd.DataFrame(results)
results_df.to_csv(TABLE_DIR / "T01_paired_wilcoxon.csv", index=False)
print(f"\nSaved: {TABLE_DIR / 'T01_paired_wilcoxon.csv'}")

# ─────────────────────────────────────────────────────────
# 4. FIGURE 1: PAIRED BEFORE-AFTER PLOTS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))

param_info = {
    "As": {"label": "As (µg/L)", "color": "#c0392b"},
    "Mn": {"label": "Mn (mg/L)", "color": "#2980b9"},
    "Fe": {"label": "Fe (mg/L)", "color": "#27ae60"},
    "PO4": {"label": "PO₄ (mg/L)", "color": "#8e44ad"},
}

for ax, param in zip(axes.flat, KEY_CONTAMINANTS):
    mask = old[param].notna() & new[param].notna()
    x_old_vals = old.loc[mask, param].values
    x_new_vals = new.loc[mask, param].values

    info = param_info[param]
    res = results_df[results_df["Parameter"] == param].iloc[0]

    # Connected dot plot (spaghetti plot) — subsample if too many lines
    n_plot = len(x_old_vals)
    if n_plot > 100:
        idx = np.random.choice(n_plot, size=100, replace=False)
    else:
        idx = np.arange(n_plot)

    for i in idx:
        ax.plot([0, 1], [x_old_vals[i], x_new_vals[i]],
                color=info["color"], alpha=0.12, linewidth=0.5)

    # Box plots at each timepoint
    bp1 = ax.boxplot([x_old_vals], positions=[0], widths=0.25,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="white", edgecolor="gray"),
                     medianprops=dict(color=info["color"], linewidth=2))
    bp2 = ax.boxplot([x_new_vals], positions=[1], widths=0.25,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor="white", edgecolor="gray"),
                     medianprops=dict(color=info["color"], linewidth=2))

    # Median line
    ax.plot([0, 1], [np.median(x_old_vals), np.median(x_new_vals)],
            color=info["color"], linewidth=2.5, zorder=5, marker="o", markersize=6)

    # Annotation
    sig_str = ""
    if res["Wilcoxon_p"] < 0.001:
        sig_str = "***"
    elif res["Wilcoxon_p"] < 0.01:
        sig_str = "**"
    elif res["Wilcoxon_p"] < 0.05:
        sig_str = "*"
    else:
        sig_str = "ns"

    pct = res["Pct_change_median"]
    pct_str = f"{pct:+.0f}%" if pd.notna(pct) else ""

    ax.set_title(f"{info['label']}  ({pct_str}, {sig_str})", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2012–13", "2020–21"])
    ax.set_ylabel(info["label"])

    # Log scale for As (wide range)
    if param == "As":
        ax.set_yscale("symlog", linthresh=1)

fig.suptitle(f"Paired temporal changes (n={n_wells} wells)", fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F01_paired_before_after.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F01_paired_before_after.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 5. FIGURE 1b: CHANGE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))

for ax, param in zip(axes.flat, KEY_CONTAMINANTS):
    mask = old[param].notna() & new[param].notna()
    diff = new.loc[mask, param].values - old.loc[mask, param].values
    info = param_info[param]
    res = results_df[results_df["Parameter"] == param].iloc[0]

    # Histogram + KDE
    ax.hist(diff, bins=30, color=info["color"], alpha=0.4, edgecolor="white", density=True)

    # KDE overlay
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(diff)
        x_kde = np.linspace(np.percentile(diff, 1), np.percentile(diff, 99), 200)
        ax.plot(x_kde, kde(x_kde), color=info["color"], linewidth=1.5)
    except Exception:
        pass

    # Zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Median line
    ax.axvline(np.median(diff), color=info["color"], linestyle="-", linewidth=1.5,
               label=f"median={np.median(diff):+.3f}")

    # CI shading
    ci_lo, ci_hi = res["Bootstrap_CI_lo"], res["Bootstrap_CI_hi"]
    ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=info["color"],
               label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    ax.set_xlabel(f"Δ{info['label']} (2021 − 2013)")
    ax.set_ylabel("Density")
    ax.set_title(f"{info['label']} change distribution", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
fig.savefig(FIGURE_DIR / "F01b_paired_change_distributions.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F01b_paired_change_distributions.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 6. PRINT SUMMARY FOR MANUSCRIPT
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("MANUSCRIPT-READY SUMMARY")
print(f"{'=' * 70}")
print(f"\nAcross {n_wells} paired wells monitored in both 2012–2013 and 2020–2021:")
for _, r in results_df[results_df["Parameter"].isin(KEY_CONTAMINANTS)].iterrows():
    p = r["Parameter"]
    info = param_info.get(p, {"label": p})
    sig = "p < 0.001" if r["Wilcoxon_p"] < 0.001 else f"p = {r['Wilcoxon_p']:.3f}"
    print(f"\n  {info['label']}:")
    print(f"    Median: {r['Old_median']:.3f} → {r['New_median']:.3f} "
          f"(Δ = {r['Diff_median']:+.3f}, {r['Pct_change_median']:+.1f}%)")
    print(f"    Wilcoxon: {sig}, rank-biserial r = {r['Rank_biserial_r']:+.3f} ({r['Effect_size_interp']})")
    print(f"    HL estimate: {r['HL_estimate']:+.3f}, 95% BCa CI [{r['Bootstrap_CI_lo']:.3f}, {r['Bootstrap_CI_hi']:.3f}]")
    print(f"    Direction: ↑{r['n_increase']:.0f} wells, ↓{r['n_decrease']:.0f} wells")

print("\nDone.")
