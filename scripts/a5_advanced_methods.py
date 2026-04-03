"""
A5 — Advanced Methods: SEM Path Analysis + Bayesian Paired Estimation
======================================================================
Two targeted additions to strengthen the manuscript for ES&T:

1. Structural Equation Model (SEM) — Two-pathway model:
   Eh → Fe → As  (reductive dissolution pathway)
   PO4 → As      (competitive desorption pathway)
   Eh → Mn       (redox indicator)
   Fit separately for 2012-13 and 2020-21 to quantify mechanism shift.

2. Bayesian Estimation Supersedes the t-test (BEST, Kruschke 2013):
   Posterior distributions of paired change for As, Mn, Fe, PO4
   with Region of Practical Equivalence (ROPE) analysis.
   Implemented via MCMC-free closed-form conjugate approach for speed.

Outputs:
  - Table: T05_sem_path_coefficients.csv
  - Table: T05b_sem_fit_indices.csv
  - Table: T06_bayesian_paired_estimation.csv
  - Figure: F06_sem_path_diagram.png       (main text)
  - Figure: F07_bayesian_posteriors.png     (main text)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

import semopy

from config import (
    TABLE_DIR, FIGURE_DIR, TEMPORAL_PARAMS, KEY_CONTAMINANTS,
    ALPHA, N_BOOTSTRAP, RANDOM_SEED, set_est_style
)

set_est_style()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────

paired = pd.read_csv(TABLE_DIR / "matched_wells.csv")
old = paired[paired["Period"] == "2012-2013"].sort_values("pair_key").reset_index(drop=True)
new = paired[paired["Period"] == "2020-2021"].sort_values("pair_key").reset_index(drop=True)

old_full = pd.read_csv(TABLE_DIR / "old_harmonized.csv")
new_full_all = pd.read_csv(TABLE_DIR / "new_harmonized.csv")

# CRITICAL: Restrict new dataset to coastal districts present in old dataset
# Old data covers 20 coastal districts; new data covers 66 national districts.
# Comparing coastal vs national would be invalid for SEM path comparison.
coastal_districts = set(old_full["District"].dropna().str.strip().str.title().unique())
new_full_all["District_norm"] = new_full_all["District"].str.strip().str.title()
new_full = new_full_all[new_full_all["District_norm"].isin(coastal_districts)].copy()

print(f"Paired wells: {len(old)}")
print(f"Full old (coastal): {len(old_full)}")
print(f"Full new (coastal only): {len(new_full)} / {len(new_full_all)} "
      f"(excluded {len(new_full_all) - len(new_full)} non-coastal wells)")


# ═════════════════════════════════════════════════════════
# PART 1: STRUCTURAL EQUATION MODEL (SEM)
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 1: STRUCTURAL EQUATION MODEL — TWO-PATHWAY As MOBILIZATION")
print(f"{'=' * 80}")

# ─────────────────────────────────────────────────────────
# 1a. SEM SPECIFICATION
# ─────────────────────────────────────────────────────────

# Two-pathway causal model:
#   Eh → Fe  (reductive dissolution releases Fe from oxides)
#   Eh → Mn  (Mn-oxide reduction, redox indicator)
#   Fe → As  (reductive dissolution: Fe-oxide dissolution releases sorbed As)
#   PO4 → As (competitive desorption: PO4 competes with As for sorption sites)
#   Eh → As  (direct redox effect, residual)
#
# This is a recursive (acyclic) path model — no latent variables needed.

sem_spec = """
# Structural equations
Fe ~ Eh
Mn ~ Eh
As ~ Fe + PO4 + Eh
"""

# ─────────────────────────────────────────────────────────
# 1b. FIT SEM FOR EACH PERIOD (full datasets for power)
# ─────────────────────────────────────────────────────────

sem_vars = ["As", "Fe", "Mn", "PO4", "Eh"]

sem_results = []
fit_indices = []

for period_label, df in [("2012-2013", old_full), ("2020-2021", new_full)]:
    # Prepare data: drop NaN, standardize (z-scores) for comparable coefficients
    sem_data = df[sem_vars].dropna().copy()
    n_obs = len(sem_data)

    if n_obs < 50:
        print(f"\n  {period_label}: skipped (n={n_obs})")
        continue

    # Standardize for comparable path coefficients
    sem_z = (sem_data - sem_data.mean()) / sem_data.std()

    print(f"\n  {period_label} (n={n_obs}):")

    # Fit SEM
    model = semopy.Model(sem_spec)
    result = model.fit(sem_z)

    # Extract path coefficients
    estimates = model.inspect()
    print(f"\n  Path coefficients (standardized):")
    print(f"  {'Path':<20} {'Estimate':>10} {'Std.Err':>10} {'z':>8} {'p':>10}")
    print(f"  {'-' * 60}")

    for _, row in estimates.iterrows():
        if row["op"] == "~":
            path_label = f"{row['lval']} ← {row['rval']}"
            est = row["Estimate"]
            se = row["Std. Err"]
            z_val = row["z-value"]
            p_val = row["p-value"]
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else (
                "*" if p_val < 0.05 else "ns"))
            print(f"  {path_label:<20} {est:>+10.4f} {se:>10.4f} {z_val:>8.3f} {p_val:>10.2e} {sig}")

            sem_results.append({
                "Period": period_label,
                "Path": path_label,
                "DV": row["lval"],
                "IV": row["rval"],
                "Estimate": est,
                "Std_Err": se,
                "z_value": z_val,
                "p_value": p_val,
            })

    # Fit indices
    stats_dict = semopy.calc_stats(model)
    fit_row = {"Period": period_label, "n": n_obs}
    for col in stats_dict.columns:
        fit_row[col] = stats_dict[col].values[0]
    fit_indices.append(fit_row)
    print(f"\n  Fit indices:")
    for k, v in fit_row.items():
        if k not in ["Period", "n"]:
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

# ─────────────────────────────────────────────────────────
# 1c. COMPARE PATH COEFFICIENTS ACROSS PERIODS
# ─────────────────────────────────────────────────────────

sem_df = pd.DataFrame(sem_results)

print(f"\n{'=' * 60}")
print("PATH COEFFICIENT COMPARISON (2012-13 vs 2020-21)")
print(f"{'=' * 60}")

# Get unique paths
paths = sem_df["Path"].unique()
comparison_rows = []

for path in paths:
    old_row = sem_df[(sem_df["Period"] == "2012-2013") & (sem_df["Path"] == path)]
    new_row = sem_df[(sem_df["Period"] == "2020-2021") & (sem_df["Path"] == path)]

    if len(old_row) == 0 or len(new_row) == 0:
        continue

    b_old = old_row["Estimate"].values[0]
    b_new = new_row["Estimate"].values[0]
    se_old = old_row["Std_Err"].values[0]
    se_new = new_row["Std_Err"].values[0]

    # Wald test for difference: z = (b1 - b2) / sqrt(se1² + se2²)
    # Valid for independent samples (different datasets)
    delta_b = b_new - b_old
    se_diff = np.sqrt(se_old**2 + se_new**2)
    z_diff = delta_b / se_diff if se_diff > 0 else 0
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    sig = "***" if p_diff < 0.001 else ("**" if p_diff < 0.01 else (
        "*" if p_diff < 0.05 else "ns"))
    print(f"  {path:<20}: {b_old:+.4f} → {b_new:+.4f} (Δ={delta_b:+.4f}, z={z_diff:.3f}, p={p_diff:.4f} {sig})")

    comparison_rows.append({
        "Path": path,
        "Beta_old": b_old,
        "Beta_new": b_new,
        "Delta_beta": delta_b,
        "SE_diff": se_diff,
        "z_diff": z_diff,
        "p_diff": p_diff,
    })

# Indirect effects
print(f"\n  Indirect effects (mediated through Fe):")
for period in ["2012-2013", "2020-2021"]:
    pdf = sem_df[sem_df["Period"] == period]
    eh_fe = pdf[pdf["Path"] == "Fe ← Eh"]["Estimate"].values
    fe_as = pdf[pdf["Path"] == "As ← Fe"]["Estimate"].values
    po4_as = pdf[pdf["Path"] == "As ← PO4"]["Estimate"].values
    eh_as_direct = pdf[pdf["Path"] == "As ← Eh"]["Estimate"].values

    if len(eh_fe) > 0 and len(fe_as) > 0:
        indirect_reductive = eh_fe[0] * fe_as[0]
        total_eh = indirect_reductive + (eh_as_direct[0] if len(eh_as_direct) > 0 else 0)
        print(f"    {period}:")
        print(f"      Reductive pathway (Eh→Fe→As): {eh_fe[0]:.4f} × {fe_as[0]:.4f} = {indirect_reductive:.4f}")
        print(f"      Desorptive pathway (PO4→As): {po4_as[0]:.4f}" if len(po4_as) > 0 else "")
        print(f"      Direct Eh→As: {eh_as_direct[0]:.4f}" if len(eh_as_direct) > 0 else "")
        print(f"      Total Eh effect on As: {total_eh:.4f}")

# Save tables
sem_df.to_csv(TABLE_DIR / "T05_sem_path_coefficients.csv", index=False)
pd.DataFrame(fit_indices).to_csv(TABLE_DIR / "T05b_sem_fit_indices.csv", index=False)
pd.DataFrame(comparison_rows).to_csv(TABLE_DIR / "T05c_sem_path_comparison.csv", index=False)
print(f"\nSaved: T05_sem_path_coefficients.csv, T05b_sem_fit_indices.csv, T05c_sem_path_comparison.csv")

# ─────────────────────────────────────────────────────────
# 1d. FIGURE 6: SEM PATH DIAGRAM
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, period in zip(axes, ["2012-2013", "2020-2021"]):
    pdf = sem_df[sem_df["Period"] == period]

    # Node positions (x, y)
    positions = {
        "Eh":  (0.15, 0.5),
        "Fe":  (0.50, 0.78),
        "Mn":  (0.50, 0.22),
        "PO4": (0.15, 0.85),
        "As":  (0.85, 0.5),
    }

    # Draw nodes
    for var, (x, y) in positions.items():
        color = "#e74c3c" if var == "As" else (
            "#3498db" if var in ["Fe", "Mn"] else (
            "#2ecc71" if var == "PO4" else "#f39c12"))
        circle = plt.Circle((x, y), 0.08, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, var, ha="center", va="center", fontsize=11,
                fontweight="bold", color="white", zorder=4)

    # Draw arrows with path coefficients
    arrow_paths = [
        ("Eh", "Fe",  "Fe ← Eh"),
        ("Eh", "Mn",  "Mn ← Eh"),
        ("Fe", "As",  "As ← Fe"),
        ("PO4", "As", "As ← PO4"),
        ("Eh", "As",  "As ← Eh"),
    ]

    for src, dst, path_name in arrow_paths:
        row = pdf[pdf["Path"] == path_name]
        if len(row) == 0:
            continue

        beta = row["Estimate"].values[0]
        p_val = row["p_value"].values[0]

        x1, y1 = positions[src]
        x2, y2 = positions[dst]

        # Shorten arrows to not overlap circles
        dx, dy = x2 - x1, y2 - y1
        dist = np.sqrt(dx**2 + dy**2)
        ux, uy = dx/dist, dy/dist
        x1s, y1s = x1 + ux * 0.09, y1 + uy * 0.09
        x2s, y2s = x2 - ux * 0.09, y2 - uy * 0.09

        # Arrow color: red for positive, blue for negative
        color = "#c0392b" if beta > 0 else "#2980b9"
        lw = max(0.5, min(4, abs(beta) * 6))
        alpha = 0.9 if p_val < 0.05 else 0.35

        ax.annotate("", xy=(x2s, y2s), xytext=(x1s, y1s),
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=lw, alpha=alpha,
                                     connectionstyle="arc3,rad=0.0"))

        # Label
        mx, my = (x1s + x2s) / 2, (y1s + y2s) / 2
        # Offset label perpendicular to arrow
        offset_x, offset_y = -uy * 0.06, ux * 0.06
        sig_mark = "*" if p_val < 0.05 else ""
        ax.text(mx + offset_x, my + offset_y, f"{beta:+.3f}{sig_mark}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color=color, alpha=max(alpha, 0.6),
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(period, fontweight="bold", fontsize=12)

plt.suptitle("SEM Path Analysis: Two-Pathway As Mobilization Model",
             fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F06_sem_path_diagram.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F06_sem_path_diagram.png'}")
plt.close()


# ═════════════════════════════════════════════════════════
# PART 2: BAYESIAN PAIRED ESTIMATION (BEST)
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 2: BAYESIAN PAIRED ESTIMATION (BEST — Kruschke 2013)")
print(f"{'=' * 80}")

# ─────────────────────────────────────────────────────────
# 2a. BAYESIAN APPROACH
# ─────────────────────────────────────────────────────────
#
# For paired differences d_i = new_i - old_i:
#   Model: d_i ~ Normal(mu, sigma²)
#   Prior: mu ~ Normal(0, tau²)  [weakly informative]
#          sigma² ~ InvGamma(a, b)  [weakly informative]
#
# With conjugate Normal-InvGamma prior, the posterior is analytically tractable:
#   mu | d ~ t-distribution (posterior predictive)
#
# We use the classical Bayesian conjugate update which gives:
#   posterior mean ~ sample mean (with large n, prior washes out)
#   posterior credible intervals from the t-distribution
#
# ROPE (Region of Practical Equivalence):
#   Defined as [-0.1 * sd_pooled, +0.1 * sd_pooled] for each parameter
#   This is a standardized effect of 0.1 — negligible by Cohen's conventions.

def bayesian_paired_estimation(old_vals, new_vals, param_name, n_mcmc=50000):
    """
    Bayesian estimation of paired differences using conjugate Normal model.

    Uses Gibbs sampling for a t-distributed likelihood (robust to outliers)
    following Kruschke (2013) BEST approach:
      d_i ~ t(mu, sigma, nu)  [nu = degrees of freedom for heavy tails]

    Simplified here to Normal likelihood with Monte Carlo posterior sampling.
    """
    diff = new_vals - old_vals
    n = len(diff)

    # Sufficient statistics
    d_bar = np.mean(diff)
    d_var = np.var(diff, ddof=1)
    d_std = np.std(diff, ddof=1)

    # Weakly informative conjugate prior: Normal-InvGamma
    # mu ~ Normal(mu0, sigma² / kappa0)
    # sigma² ~ InvGamma(nu0/2, nu0*sigma0²/2)
    mu0 = 0         # prior mean centered at zero change
    kappa0 = 0.01   # very weak prior (n >> kappa0)
    nu0 = 1         # weak prior on variance
    sigma0_sq = d_var  # prior variance = sample variance (weakly informative)

    # Posterior hyperparameters (conjugate update)
    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * d_bar) / kappa_n
    nu_n = nu0 + n
    nu_n_sigma_n_sq = (nu0 * sigma0_sq +
                       (n - 1) * d_var +
                       (kappa0 * n / kappa_n) * (d_bar - mu0)**2)
    sigma_n_sq = nu_n_sigma_n_sq / nu_n

    # Posterior draws via composition:
    # sigma² | data ~ InvGamma(nu_n/2, nu_n*sigma_n²/2)
    # mu | sigma², data ~ Normal(mu_n, sigma²/kappa_n)
    sigma2_draws = 1.0 / np.random.gamma(
        shape=nu_n / 2, scale=2.0 / (nu_n * sigma_n_sq), size=n_mcmc
    )
    mu_draws = np.random.normal(
        loc=mu_n, scale=np.sqrt(sigma2_draws / kappa_n)
    )

    # Effect size (Cohen's d posterior)
    sigma_draws = np.sqrt(sigma2_draws)
    cohens_d_draws = mu_draws / sigma_draws

    # ROPE: +/- 0.1 * pooled SD (negligible effect)
    pooled_sd = np.sqrt((np.var(old_vals, ddof=1) + np.var(new_vals, ddof=1)) / 2)
    rope_low = -0.1 * pooled_sd
    rope_high = 0.1 * pooled_sd

    # Posterior summaries
    mu_mean = np.mean(mu_draws)
    mu_median = np.median(mu_draws)
    hdi_95 = np.percentile(mu_draws, [2.5, 97.5])
    hdi_89 = np.percentile(mu_draws, [5.5, 94.5])

    # Probability of direction (pd)
    if mu_mean >= 0:
        p_direction = np.mean(mu_draws > 0)
        direction = "increase"
    else:
        p_direction = np.mean(mu_draws < 0)
        direction = "decrease"

    # ROPE analysis
    p_in_rope = np.mean((mu_draws >= rope_low) & (mu_draws <= rope_high))
    p_above_rope = np.mean(mu_draws > rope_high)
    p_below_rope = np.mean(mu_draws < rope_low)

    # Cohen's d posterior
    d_mean = np.mean(cohens_d_draws)
    d_hdi_95 = np.percentile(cohens_d_draws, [2.5, 97.5])

    return {
        "Parameter": param_name,
        "n_pairs": n,
        "Diff_mean": d_bar,
        "Posterior_mean": mu_mean,
        "Posterior_median": mu_median,
        "HDI_95_low": hdi_95[0],
        "HDI_95_high": hdi_95[1],
        "HDI_89_low": hdi_89[0],
        "HDI_89_high": hdi_89[1],
        "P_direction": p_direction,
        "Direction": direction,
        "ROPE_low": rope_low,
        "ROPE_high": rope_high,
        "P_in_ROPE": p_in_rope,
        "P_above_ROPE": p_above_rope,
        "P_below_ROPE": p_below_rope,
        "Cohen_d_mean": d_mean,
        "Cohen_d_HDI95_low": d_hdi_95[0],
        "Cohen_d_HDI95_high": d_hdi_95[1],
        "mu_draws": mu_draws,
        "d_draws": cohens_d_draws,
    }

# ─────────────────────────────────────────────────────────
# 2b. RUN BAYESIAN ESTIMATION FOR KEY CONTAMINANTS
# ─────────────────────────────────────────────────────────

bayesian_results = []
draw_storage = {}

param_units = {"As": "ug/L", "Mn": "mg/L", "Fe": "mg/L", "PO4": "mg/L"}

for param in KEY_CONTAMINANTS:
    mask = old[param].notna() & new[param].notna()
    old_v = old.loc[mask, param].values
    new_v = new.loc[mask, param].values

    result = bayesian_paired_estimation(old_v, new_v, param, n_mcmc=100000)
    draw_storage[param] = {
        "mu_draws": result.pop("mu_draws"),
        "d_draws": result.pop("d_draws"),
    }
    bayesian_results.append(result)

    print(f"\n  {param} ({param_units[param]}, n={result['n_pairs']} pairs):")
    print(f"    Posterior mean: {result['Posterior_mean']:+.4f}")
    print(f"    95% HDI: [{result['HDI_95_low']:+.4f}, {result['HDI_95_high']:+.4f}]")
    print(f"    P(direction={result['Direction']}): {result['P_direction']:.4f}")
    print(f"    ROPE [{result['ROPE_low']:+.4f}, {result['ROPE_high']:+.4f}]:")
    print(f"      P(in ROPE) = {result['P_in_ROPE']:.4f}")
    print(f"      P(above ROPE) = {result['P_above_ROPE']:.4f}")
    print(f"      P(below ROPE) = {result['P_below_ROPE']:.4f}")
    print(f"    Cohen's d: {result['Cohen_d_mean']:+.4f} [{result['Cohen_d_HDI95_low']:+.4f}, {result['Cohen_d_HDI95_high']:+.4f}]")

    # Decision rule (Kruschke 2018)
    if result['P_in_ROPE'] > 0.95:
        decision = "ACCEPT null (practically equivalent to zero)"
    elif result['P_in_ROPE'] < 0.05:
        decision = "REJECT null (practically meaningful change)"
    else:
        decision = "UNDECIDED (need more data)"
    print(f"    Decision: {decision}")

bayesian_df = pd.DataFrame(bayesian_results)
bayesian_df.to_csv(TABLE_DIR / "T06_bayesian_paired_estimation.csv", index=False)
print(f"\nSaved: {TABLE_DIR / 'T06_bayesian_paired_estimation.csv'}")

# ─────────────────────────────────────────────────────────
# 2c. FIGURE 7: BAYESIAN POSTERIOR DISTRIBUTIONS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.ravel()

colors = {"As": "#e74c3c", "Mn": "#3498db", "Fe": "#2ecc71", "PO4": "#9b59b6"}

for ax, param in zip(axes, KEY_CONTAMINANTS):
    mu_draws = draw_storage[param]["mu_draws"]
    result = bayesian_df[bayesian_df["Parameter"] == param].iloc[0]

    # Histogram of posterior
    ax.hist(mu_draws, bins=150, density=True, color=colors[param],
            alpha=0.6, edgecolor="none")

    # HDI
    hdi_low, hdi_high = result["HDI_95_low"], result["HDI_95_high"]
    ax.axvline(hdi_low, color=colors[param], ls="--", lw=1.2, alpha=0.8)
    ax.axvline(hdi_high, color=colors[param], ls="--", lw=1.2, alpha=0.8)

    # ROPE region
    rope_low, rope_high = result["ROPE_low"], result["ROPE_high"]
    ymax = ax.get_ylim()[1]
    ax.axvspan(rope_low, rope_high, color="gray", alpha=0.2, zorder=0)

    # Zero line
    ax.axvline(0, color="black", ls="-", lw=0.8, alpha=0.5)

    # Posterior mean
    ax.axvline(result["Posterior_mean"], color=colors[param], ls="-", lw=2)

    # Annotations
    unit = param_units[param]
    pd_val = result["P_direction"]
    p_rope = result["P_in_ROPE"]

    ax.set_title(f"{param} ({unit})", fontweight="bold", fontsize=11)
    ax.set_xlabel(f"$\\Delta${param} ({unit})", fontsize=9)
    ax.set_ylabel("Posterior density", fontsize=9)

    # Text box with key stats
    text = (f"$\\mu$ = {result['Posterior_mean']:+.3f}\n"
            f"95% HDI [{hdi_low:+.3f}, {hdi_high:+.3f}]\n"
            f"P(dir) = {pd_val:.3f}\n"
            f"P(ROPE) = {p_rope:.3f}")
    ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

plt.suptitle("Bayesian Posterior Distributions of Decadal Change",
             fontweight="bold", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F07_bayesian_posteriors.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F07_bayesian_posteriors.png'}")
plt.close()

# ─────────────────────────────────────────────────────────
# 2d. SUPPLEMENTARY: EFFECT SIZE POSTERIOR
# ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 3.5))

for param in KEY_CONTAMINANTS:
    d_draws = draw_storage[param]["d_draws"]
    ax.hist(d_draws, bins=100, density=True, alpha=0.4,
            color=colors[param], label=param, edgecolor="none")

# Reference lines for Cohen's conventions
for d_val, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
    ax.axvline(d_val, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(-d_val, color="gray", ls=":", lw=0.8, alpha=0.6)

ax.axvline(0, color="black", ls="-", lw=1)
ax.set_xlabel("Cohen's d (effect size)", fontsize=10)
ax.set_ylabel("Posterior density", fontsize=10)
ax.set_title("Posterior Effect Size Distributions", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F07b_effect_size_posterior.png", dpi=300, bbox_inches="tight")
print(f"Saved: {FIGURE_DIR / 'F07b_effect_size_posterior.png'}")
plt.close()


# ═════════════════════════════════════════════════════════
# MANUSCRIPT SUMMARY
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("MANUSCRIPT-READY SUMMARY — ADVANCED METHODS")
print(f"{'=' * 80}")

print("\n--- SEM PATH ANALYSIS ---")
for period in ["2012-2013", "2020-2021"]:
    pdf = sem_df[sem_df["Period"] == period]
    print(f"\n  {period}:")
    for _, r in pdf.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else (
            "*" if r["p_value"] < 0.05 else "ns"))
        print(f"    {r['Path']:<20}: beta={r['Estimate']:+.4f} (p={r['p_value']:.2e} {sig})")

if len(comparison_rows) > 0:
    print(f"\n  Path coefficient changes (Wald test):")
    for r in comparison_rows:
        sig = "***" if r["p_diff"] < 0.001 else ("**" if r["p_diff"] < 0.01 else (
            "*" if r["p_diff"] < 0.05 else "ns"))
        print(f"    {r['Path']:<20}: {r['Beta_old']:+.4f} → {r['Beta_new']:+.4f} "
              f"(Δ={r['Delta_beta']:+.4f}, p={r['p_diff']:.4f} {sig})")

print("\n--- BAYESIAN PAIRED ESTIMATION ---")
for _, r in bayesian_df.iterrows():
    print(f"\n  {r['Parameter']} (n={r['n_pairs']} pairs):")
    print(f"    Posterior mean: {r['Posterior_mean']:+.4f}, "
          f"95% HDI [{r['HDI_95_low']:+.4f}, {r['HDI_95_high']:+.4f}]")
    print(f"    P({r['Direction']}) = {r['P_direction']:.4f}")
    print(f"    P(in ROPE) = {r['P_in_ROPE']:.4f} | "
          f"Cohen's d = {r['Cohen_d_mean']:+.3f} [{r['Cohen_d_HDI95_low']:+.3f}, "
          f"{r['Cohen_d_HDI95_high']:+.3f}]")

print("\nDone.")
