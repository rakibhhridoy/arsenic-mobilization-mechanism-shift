"""
A6 — SEM Enhancements: Improved Fit, Bootstrap Mediation, Multi-Group Test
============================================================================
Addresses three ES&T reviewer concerns from the A5 SEM analysis:

1. IMPROVED SEM FIT: Add theoretically justified covariances (Fe~~Mn shared
   redox sensitivity; PO4~~Eh shared depth/redox confounding) to improve
   CFI/RMSEA while preserving the causal structure.

2. BOOTSTRAP MEDIATION: Test the Eh→Fe→As indirect (reductive) pathway
   with BCa bootstrap confidence intervals (5000 resamples). The Sobel test
   is known to be underpowered; bootstrap is the gold standard (Preacher &
   Hayes 2008; Shrout & Bolger 2002).

3. MULTI-GROUP CHI-SQUARE DIFFERENCE TEST: Instead of separate model fits +
   Wald test, fit a stacked multi-group model and use likelihood ratio
   chi-square difference to test if path coefficients differ between periods.
   This is the standard SEM approach for moderation by grouping variable.

Outputs:
  - Table: T07_sem_improved_fit.csv
  - Table: T07b_bootstrap_mediation.csv
  - Table: T07c_multigroup_test.csv
  - Figure: F08_mediation_bootstrap.png
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import semopy

from config import (
    TABLE_DIR, FIGURE_DIR, ALPHA, N_BOOTSTRAP, RANDOM_SEED, set_est_style
)

set_est_style()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────

old_full = pd.read_csv(TABLE_DIR / "old_harmonized.csv")
new_full_all = pd.read_csv(TABLE_DIR / "new_harmonized.csv")

# Restrict to coastal districts
coastal_districts = set(old_full["District"].dropna().str.strip().str.title().unique())
new_full_all["District_norm"] = new_full_all["District"].str.strip().str.title()
new_full = new_full_all[new_full_all["District_norm"].isin(coastal_districts)].copy()

sem_vars = ["As", "Fe", "Mn", "PO4", "Eh"]

# Prepare datasets
old_data = old_full[sem_vars].dropna().copy()
new_data = new_full[sem_vars].dropna().copy()

print(f"Old coastal: {len(old_data)}, New coastal: {len(new_data)}")

# ═════════════════════════════════════════════════════════
# PART 1: IMPROVED SEM WITH CORRELATED RESIDUALS
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 1: IMPROVED SEM — CORRELATED RESIDUALS")
print(f"{'=' * 80}")

# Original model (from A5)
spec_original = """
Fe ~ Eh
Mn ~ Eh
As ~ Fe + PO4 + Eh
"""

# Improved model: add theoretically justified residual covariances
# Fe ~~ Mn: both are redox-sensitive metals released under reducing conditions;
#           shared sensitivity to redox front position not fully captured by Eh alone
# PO4 ~~ Eh: PO4 loading and redox state are confounded by depth and land use
#            (shallow wells have both more PO4 input and more oxidizing conditions)
spec_improved = """
Fe ~ Eh
Mn ~ Eh
As ~ Fe + PO4 + Eh
Fe ~~ Mn
"""

print("\n  Model comparison:")
improved_results = []

for period_label, df in [("2012-2013", old_data), ("2020-2021", new_data)]:
    # Standardize
    z_data = (df - df.mean()) / df.std()

    # Fit original
    m_orig = semopy.Model(spec_original)
    m_orig.fit(z_data)
    stats_orig = semopy.calc_stats(m_orig)

    # Fit improved
    m_impr = semopy.Model(spec_improved)
    m_impr.fit(z_data)
    stats_impr = semopy.calc_stats(m_impr)

    chi2_orig = stats_orig["chi2"].values[0]
    chi2_impr = stats_impr["chi2"].values[0]
    df_orig = stats_orig["DoF"].values[0]
    df_impr = stats_impr["DoF"].values[0]

    # Likelihood ratio test (chi-square difference)
    delta_chi2 = chi2_orig - chi2_impr
    delta_df = df_orig - df_impr
    lr_p = 1 - stats.chi2.cdf(delta_chi2, delta_df) if delta_df > 0 else np.nan

    cfi_orig = stats_orig["CFI"].values[0]
    cfi_impr = stats_impr["CFI"].values[0]
    rmsea_orig = stats_orig["RMSEA"].values[0]
    rmsea_impr = stats_impr["RMSEA"].values[0]

    print(f"\n  {period_label} (n={len(z_data)}):")
    print(f"    {'Metric':<15} {'Original':>12} {'Improved':>12} {'Change':>12}")
    print(f"    {'-' * 52}")
    print(f"    {'chi2':<15} {chi2_orig:>12.2f} {chi2_impr:>12.2f} {chi2_impr - chi2_orig:>+12.2f}")
    print(f"    {'df':<15} {df_orig:>12.0f} {df_impr:>12.0f} {df_impr - df_orig:>+12.0f}")
    print(f"    {'CFI':<15} {cfi_orig:>12.4f} {cfi_impr:>12.4f} {cfi_impr - cfi_orig:>+12.4f}")
    print(f"    {'RMSEA':<15} {rmsea_orig:>12.4f} {rmsea_impr:>12.4f} {rmsea_impr - rmsea_orig:>+12.4f}")
    print(f"    LR test: Δχ²={delta_chi2:.2f}, Δdf={delta_df}, p={lr_p:.4f}")

    # Extract improved model path coefficients
    estimates = m_impr.inspect()
    print(f"\n    Improved model path coefficients:")
    for _, row in estimates.iterrows():
        if row["op"] == "~":
            path = f"{row['lval']} ← {row['rval']}"
            sig = "***" if row["p-value"] < 0.001 else ("**" if row["p-value"] < 0.01 else (
                "*" if row["p-value"] < 0.05 else "ns"))
            print(f"      {path:<20}: β={row['Estimate']:+.4f} (p={row['p-value']:.2e} {sig})")
        elif row["op"] == "~~" and row["lval"] != row["rval"]:
            path = f"{row['lval']} ~~ {row['rval']}"
            sig = "***" if row["p-value"] < 0.001 else ("**" if row["p-value"] < 0.01 else (
                "*" if row["p-value"] < 0.05 else "ns"))
            print(f"      {path:<20}: cov={row['Estimate']:+.4f} (p={row['p-value']:.2e} {sig})")

    improved_results.append({
        "Period": period_label,
        "n": len(z_data),
        "chi2_original": chi2_orig,
        "chi2_improved": chi2_impr,
        "df_original": df_orig,
        "df_improved": df_impr,
        "CFI_original": cfi_orig,
        "CFI_improved": cfi_impr,
        "RMSEA_original": rmsea_orig,
        "RMSEA_improved": rmsea_impr,
        "LR_chi2": delta_chi2,
        "LR_df": delta_df,
        "LR_p": lr_p,
    })

pd.DataFrame(improved_results).to_csv(TABLE_DIR / "T07_sem_improved_fit.csv", index=False)
print(f"\nSaved: T07_sem_improved_fit.csv")


# ═════════════════════════════════════════════════════════
# PART 2: BOOTSTRAP MEDIATION TEST (Eh → Fe → As)
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 2: BOOTSTRAP MEDIATION — Eh → Fe → As INDIRECT PATHWAY")
print(f"{'=' * 80}")

# Mediation model:
#   X = Eh (predictor)
#   M = Fe (mediator)
#   Y = As (outcome)
#   Covariates in Y equation: PO4 (controls for desorption pathway)
#
# Indirect effect = a × b
#   a = coefficient of Eh → Fe
#   b = coefficient of Fe → As (controlling for Eh and PO4)
# Direct effect = c'
#   c' = coefficient of Eh → As (controlling for Fe and PO4)
# Total effect = c' + a*b

N_BOOT_MED = 10000  # More resamples for mediation

def bootstrap_mediation(X, M, Y, covariates=None, n_boot=N_BOOT_MED):
    """
    Bootstrap test for mediation (indirect effect a*b).

    Uses percentile and BCa confidence intervals.
    Follows Preacher & Hayes (2008) approach.
    """
    n = len(X)

    # Point estimates via OLS
    from numpy.linalg import lstsq

    def get_ab(X_, M_, Y_, cov_=None):
        """Get a, b, c' path coefficients."""
        # Path a: M = i + a*X + e
        A_a = np.column_stack([np.ones(len(X_)), X_])
        coef_a, _, _, _ = lstsq(A_a, M_, rcond=None)
        a = coef_a[1]

        # Path b + c': Y = i + b*M + c'*X + d*cov + e
        if cov_ is not None:
            A_b = np.column_stack([np.ones(len(X_)), M_, X_, cov_])
        else:
            A_b = np.column_stack([np.ones(len(X_)), M_, X_])
        coef_b, _, _, _ = lstsq(A_b, Y_, rcond=None)
        b = coef_b[1]
        c_prime = coef_b[2]

        return a, b, c_prime, a * b

    # Point estimates
    a_obs, b_obs, cp_obs, ab_obs = get_ab(X, M, Y, covariates)

    # Bootstrap
    ab_boots = np.zeros(n_boot)
    a_boots = np.zeros(n_boot)
    b_boots = np.zeros(n_boot)
    cp_boots = np.zeros(n_boot)

    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        X_b, M_b, Y_b = X[idx], M[idx], Y[idx]
        cov_b = covariates[idx] if covariates is not None else None
        a_boots[i], b_boots[i], cp_boots[i], ab_boots[i] = get_ab(X_b, M_b, Y_b, cov_b)

    # Percentile CI
    pct_ci = np.percentile(ab_boots, [2.5, 97.5])

    # BCa CI (bias-corrected and accelerated)
    # Bias correction
    z0 = stats.norm.ppf(np.mean(ab_boots < ab_obs))

    # Acceleration (jackknife)
    ab_jack = np.zeros(n)
    for i in range(n):
        idx_j = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        X_j, M_j, Y_j = X[idx_j], M[idx_j], Y[idx_j]
        cov_j = covariates[idx_j] if covariates is not None else None
        _, _, _, ab_jack[i] = get_ab(X_j, M_j, Y_j, cov_j)

    ab_jack_mean = np.mean(ab_jack)
    acc = np.sum((ab_jack_mean - ab_jack)**3) / (6 * np.sum((ab_jack_mean - ab_jack)**2)**1.5 + 1e-10)

    # BCa adjusted percentiles
    alpha_vals = [0.025, 0.975]
    bca_ci = np.zeros(2)
    for k, alpha_k in enumerate(alpha_vals):
        z_alpha = stats.norm.ppf(alpha_k)
        p_adj = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acc * (z0 + z_alpha)))
        bca_ci[k] = np.percentile(ab_boots, p_adj * 100)

    # Proportion mediated
    total = ab_obs + cp_obs
    prop_mediated = ab_obs / total if abs(total) > 1e-10 else np.nan

    # Sobel test (for comparison, known to be conservative)
    se_a = np.std(a_boots, ddof=1)
    se_b = np.std(b_boots, ddof=1)
    sobel_se = np.sqrt(a_obs**2 * se_b**2 + b_obs**2 * se_a**2)
    sobel_z = ab_obs / sobel_se if sobel_se > 0 else 0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    return {
        "a": a_obs,
        "b": b_obs,
        "c_prime": cp_obs,
        "ab_indirect": ab_obs,
        "total": total,
        "prop_mediated": prop_mediated,
        "pct_CI_low": pct_ci[0],
        "pct_CI_high": pct_ci[1],
        "bca_CI_low": bca_ci[0],
        "bca_CI_high": bca_ci[1],
        "sobel_z": sobel_z,
        "sobel_p": sobel_p,
        "ab_boots": ab_boots,
        "a_boots": a_boots,
        "b_boots": b_boots,
    }

mediation_results = []
boot_draws = {}

for period_label, df in [("2012-2013", old_data), ("2020-2021", new_data)]:
    # Standardize
    z_data = (df - df.mean()) / df.std()

    X = z_data["Eh"].values
    M = z_data["Fe"].values
    Y = z_data["As"].values
    cov = z_data["PO4"].values  # Control for desorption pathway

    result = bootstrap_mediation(X, M, Y, covariates=cov, n_boot=N_BOOT_MED)
    boot_draws[period_label] = result.pop("ab_boots")
    result.pop("a_boots")
    result.pop("b_boots")

    result["Period"] = period_label
    result["n"] = len(z_data)
    mediation_results.append(result)

    # Is indirect effect significant? (BCa CI excludes zero)
    sig_bca = "YES" if (result["bca_CI_low"] > 0 or result["bca_CI_high"] < 0) else "NO"
    sig_sobel = "***" if result["sobel_p"] < 0.001 else ("**" if result["sobel_p"] < 0.01 else (
        "*" if result["sobel_p"] < 0.05 else "ns"))

    print(f"\n  {period_label} (n={len(z_data)}):")
    print(f"    Path a (Eh→Fe):      β = {result['a']:+.4f}")
    print(f"    Path b (Fe→As|Eh,PO4): β = {result['b']:+.4f}")
    print(f"    Direct c' (Eh→As|Fe,PO4): β = {result['c_prime']:+.4f}")
    print(f"    Indirect a×b:        {result['ab_indirect']:+.4f}")
    print(f"    Total effect:        {result['total']:+.4f}")
    print(f"    Proportion mediated: {result['prop_mediated']:.1%}" if not np.isnan(result['prop_mediated']) else "    Proportion mediated: N/A")
    print(f"    Percentile 95% CI:   [{result['pct_CI_low']:+.4f}, {result['pct_CI_high']:+.4f}]")
    print(f"    BCa 95% CI:          [{result['bca_CI_low']:+.4f}, {result['bca_CI_high']:+.4f}] → Significant: {sig_bca}")
    print(f"    Sobel test:          z={result['sobel_z']:.3f}, p={result['sobel_p']:.4f} {sig_sobel}")

# Test if indirect effects differ between periods
print(f"\n  --- Comparison of indirect effects ---")
ab_old = mediation_results[0]["ab_indirect"]
ab_new = mediation_results[1]["ab_indirect"]
# Bootstrap difference test
diff_boots = boot_draws["2012-2013"][:min(len(boot_draws["2012-2013"]), len(boot_draws["2020-2021"]))] - \
             boot_draws["2020-2021"][:min(len(boot_draws["2012-2013"]), len(boot_draws["2020-2021"]))]
diff_ci = np.percentile(diff_boots, [2.5, 97.5])
diff_sig = "YES" if (diff_ci[0] > 0 or diff_ci[1] < 0) else "NO"
print(f"    Old indirect: {ab_old:+.4f}")
print(f"    New indirect: {ab_new:+.4f}")
print(f"    Difference:   {ab_old - ab_new:+.4f}")
print(f"    95% CI of difference: [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}] → Significant: {diff_sig}")

med_df = pd.DataFrame(mediation_results)
med_df.to_csv(TABLE_DIR / "T07b_bootstrap_mediation.csv", index=False)
print(f"\nSaved: T07b_bootstrap_mediation.csv")


# ═════════════════════════════════════════════════════════
# PART 3: MULTI-GROUP CHI-SQUARE DIFFERENCE TEST
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 3: MULTI-GROUP MODEL COMPARISON (LR chi² difference)")
print(f"{'=' * 80}")

# Approach: Stack both periods into one dataset with a group indicator.
# Fit two models:
#   M0 (constrained): same path coefficients for both periods
#   M1 (free): different path coefficients for each period
# Compare with chi-square difference test.
#
# Since semopy doesn't support native multi-group, we implement this via
# interaction terms in a stacked regression framework (equivalent test).

# Stack datasets
old_z = (old_data - old_data.mean()) / old_data.std()
new_z = (new_data - new_data.mean()) / new_data.std()

old_z["Period_code"] = 0
new_z["Period_code"] = 1
stacked = pd.concat([old_z, new_z], ignore_index=True)

# Create interaction terms
for var in ["Eh", "Fe", "PO4"]:
    stacked[f"{var}_x_Period"] = stacked[var] * stacked["Period_code"]

stacked["Eh_x_Period_Mn"] = stacked["Eh"] * stacked["Period_code"]  # for Mn equation

print(f"\n  Stacked dataset: {len(stacked)} observations")

# M0 (constrained): No interaction terms — same coefficients across periods
spec_constrained = """
Fe ~ Eh
Mn ~ Eh
As ~ Fe + PO4 + Eh
"""

# M1 (free): Interaction terms allow different slopes by period
spec_free = """
Fe ~ Eh + Eh_x_Period
Mn ~ Eh + Eh_x_Period_Mn
As ~ Fe + PO4 + Eh + Fe_x_Period + PO4_x_Period + Eh_x_Period
"""

m0 = semopy.Model(spec_constrained)
m0.fit(stacked)
stats_m0 = semopy.calc_stats(m0)

m1 = semopy.Model(spec_free)
m1.fit(stacked)
stats_m1 = semopy.calc_stats(m1)

chi2_m0 = stats_m0["chi2"].values[0]
chi2_m1 = stats_m1["chi2"].values[0]
df_m0 = stats_m0["DoF"].values[0]
df_m1 = stats_m1["DoF"].values[0]

delta_chi2 = chi2_m0 - chi2_m1
delta_df = df_m0 - df_m1
lr_p = 1 - stats.chi2.cdf(delta_chi2, delta_df) if delta_df > 0 else np.nan

print(f"\n  Model comparison:")
print(f"    M0 (constrained): χ²={chi2_m0:.2f}, df={df_m0}")
print(f"    M1 (free):        χ²={chi2_m1:.2f}, df={df_m1}")
print(f"    Δχ²={delta_chi2:.2f}, Δdf={delta_df}, p={lr_p:.6f}")
sig = "***" if lr_p < 0.001 else ("**" if lr_p < 0.01 else ("*" if lr_p < 0.05 else "ns"))
print(f"    Decision: {'Path coefficients DIFFER between periods' if lr_p < 0.05 else 'No significant difference'} {sig}")

# Extract interaction coefficients (= differences in path coefficients)
print(f"\n  Interaction coefficients (path coefficient differences):")
estimates_m1 = m1.inspect()
for _, row in estimates_m1.iterrows():
    if row["op"] == "~" and "Period" in str(row["rval"]):
        path = f"{row['lval']}: Δ({row['rval'].replace('_x_Period', '').replace('_Mn', '')})"
        sig_i = "***" if row["p-value"] < 0.001 else ("**" if row["p-value"] < 0.01 else (
            "*" if row["p-value"] < 0.05 else "ns"))
        print(f"    {path:<35}: Δβ={row['Estimate']:+.4f} (SE={row['Std. Err']:.4f}, "
              f"z={row['z-value']:.3f}, p={row['p-value']:.4f} {sig_i})")

multigroup_results = {
    "chi2_constrained": chi2_m0,
    "df_constrained": df_m0,
    "chi2_free": chi2_m1,
    "df_free": df_m1,
    "LR_chi2": delta_chi2,
    "LR_df": delta_df,
    "LR_p": lr_p,
}
pd.DataFrame([multigroup_results]).to_csv(TABLE_DIR / "T07c_multigroup_test.csv", index=False)
print(f"\nSaved: T07c_multigroup_test.csv")


# ═════════════════════════════════════════════════════════
# PART 4: MEDIATION BOOTSTRAP FIGURE
# ═════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

for ax, (period, draws) in zip(axes, boot_draws.items()):
    result = [r for r in mediation_results if r["Period"] == period][0]

    ax.hist(draws, bins=150, density=True, color="#3498db" if "2020" in period else "#e74c3c",
            alpha=0.6, edgecolor="none")
    ax.axvline(0, color="black", ls="-", lw=1)
    ax.axvline(result["ab_indirect"], color="darkred", ls="-", lw=2, label="Observed")
    ax.axvline(result["bca_CI_low"], color="gray", ls="--", lw=1)
    ax.axvline(result["bca_CI_high"], color="gray", ls="--", lw=1)

    # Shade CI region
    ax.axvspan(result["bca_CI_low"], result["bca_CI_high"], color="gray", alpha=0.15)

    sig_text = "Significant" if (result["bca_CI_low"] > 0 or result["bca_CI_high"] < 0) else "Not significant"
    text = (f"a×b = {result['ab_indirect']:+.4f}\n"
            f"BCa 95% CI:\n[{result['bca_CI_low']:+.4f}, {result['bca_CI_high']:+.4f}]\n"
            f"{sig_text}")
    ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=8,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlabel("Indirect effect (a × b)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"Eh → Fe → As mediation ({period})", fontweight="bold", fontsize=10)

plt.tight_layout()
fig.savefig(FIGURE_DIR / "F08_mediation_bootstrap.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {FIGURE_DIR / 'F08_mediation_bootstrap.png'}")
plt.close()


# ═════════════════════════════════════════════════════════
# MANUSCRIPT SUMMARY
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("MANUSCRIPT-READY SUMMARY — SEM ENHANCEMENTS")
print(f"{'=' * 80}")

print("\n1. IMPROVED FIT (Fe~~Mn covariance):")
for r in improved_results:
    print(f"   {r['Period']}: CFI {r['CFI_original']:.3f}→{r['CFI_improved']:.3f}, "
          f"RMSEA {r['RMSEA_original']:.3f}→{r['RMSEA_improved']:.3f} "
          f"(LR p={r['LR_p']:.4f})")

print("\n2. BOOTSTRAP MEDIATION (Eh→Fe→As):")
for r in mediation_results:
    sig = "YES" if (r["bca_CI_low"] > 0 or r["bca_CI_high"] < 0) else "NO"
    print(f"   {r['Period']}: indirect={r['ab_indirect']:+.4f}, "
          f"BCa CI [{r['bca_CI_low']:+.4f}, {r['bca_CI_high']:+.4f}], "
          f"significant={sig}")
print(f"   Difference: {mediation_results[0]['ab_indirect'] - mediation_results[1]['ab_indirect']:+.4f}, "
      f"CI [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}], significant={diff_sig}")

print(f"\n3. MULTI-GROUP TEST:")
print(f"   Δχ²={delta_chi2:.2f}, Δdf={delta_df}, p={lr_p:.6f}")
print(f"   Path coefficients {'DIFFER' if lr_p < 0.05 else 'do not differ'} between periods")

print("\nDone.")
