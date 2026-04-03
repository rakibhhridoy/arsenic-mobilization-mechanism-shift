"""
A7 — Satellite & Land Use Linkage: Why Did PO₄ Increase?
==========================================================
Links Paper3 temporal changes to Paper1 satellite-derived land use proxies
to explain the mechanistic driver behind PO₄ increase.

Hypothesis: Anthropogenic phosphate loading (agriculture + sewage) drove the
82% PO₄ increase, which in turn activated the competitive desorption pathway
(As ← PO₄) documented in A5 SEM.

Method:
  1. Join Paper3 matched wells to Paper1 satellite features via Sample ID
  2. Test: Do wells with higher NDVI (agriculture intensity) show larger ΔPO₄?
  3. Test: Do wells with higher NDWI (wetland/irrigation) show larger ΔPO₄?
  4. Partial correlation: NDVI → ΔPO₄ controlling for depth and baseline PO₄
  5. Quantile comparison: top vs bottom NDVI tercile wells

Outputs:
  - Table: T08_satellite_linkage.csv
  - Table: T08b_landuse_regression.csv
  - Figure: F09_satellite_landuse_linkage.png
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

from config import (
    TABLE_DIR, FIGURE_DIR, ALPHA, RANDOM_SEED, set_est_style
)

set_est_style()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 1. LOAD & JOIN DATASETS
# ─────────────────────────────────────────────────────────

print(f"{'=' * 80}")
print("A7 — SATELLITE & LAND USE LINKAGE")
print(f"{'=' * 80}")

# Paper3 matched wells
paired = pd.read_csv(TABLE_DIR / "matched_wells.csv")
old = paired[paired["Period"] == "2012-2013"].sort_values("pair_key").reset_index(drop=True)
new = paired[paired["Period"] == "2020-2021"].sort_values("pair_key").reset_index(drop=True)

# Paper1 integrated data (has Sample IDs matching Paper3 new dataset)
p1_data = pd.read_csv("/Users/rakibhhridoy/AsGW/GroundWater/Paper1/hypo/hypo1/output/data_phase1_integrated.csv")

# Paper1 satellite features (indexed by Location_ID = row order in p1_data)
sat_features = pd.read_csv("/Users/rakibhhridoy/AsGW/GroundWater/Paper1/hypo/hypo1/output/tables/Table_12_satellite_features.csv")

# Add Location_ID to p1_data (1-indexed, matching satellite features)
p1_data["Location_ID"] = range(1, len(p1_data) + 1)

# Normalize IDs for matching
def normalize_id(sid):
    s = str(sid).strip().upper()
    s = re.sub(r"/\d+$", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip()

p1_data["ID_norm"] = p1_data["Sample ID"].apply(normalize_id)

# Merge satellite features onto p1_data
p1_sat = p1_data.merge(sat_features, on="Location_ID", how="left")

print(f"Paper1 data with satellite: {len(p1_sat)}")
print(f"Satellite features: {[c for c in sat_features.columns if c != 'Location_ID']}")

# ─────────────────────────────────────────────────────────
# 2. MATCH PAPER3 WELLS TO SATELLITE DATA
# ─────────────────────────────────────────────────────────

# Strategy: For each Paper3 new (2020-21) well, find the matching Paper1 row
# by normalized ID, then pull satellite features.

# Group Paper1 by ID_norm (average if multiple samples per well)
sat_cols = [c for c in sat_features.columns if c != "Location_ID"]
p1_by_well = p1_sat.groupby("ID_norm")[sat_cols].mean().reset_index()

# Merge onto Paper3 new data
new_with_sat = new.merge(p1_by_well, on="ID_norm", how="left")
sat_matched = new_with_sat[new_with_sat["NDVI_mean"].notna()]

print(f"\nPaper3 new wells with satellite data: {len(sat_matched)} / {len(new)}")

# Calculate temporal changes (new - old)
# Align old and new by pair_key
changes = pd.DataFrame()
changes["pair_key"] = new["pair_key"].values
changes["ID_norm"] = new["ID_norm"].values

for param in ["As", "Mn", "Fe", "PO4", "pH", "Eh"]:
    if param in old.columns and param in new.columns:
        changes[f"Delta_{param}"] = new[param].values - old[param].values
        changes[f"Old_{param}"] = old[param].values
        changes[f"New_{param}"] = new[param].values

# Add depth
if "Depth" in new.columns:
    changes["Depth"] = new["Depth"].values

# Merge satellite features onto changes
changes_sat = changes.merge(p1_by_well, on="ID_norm", how="left")
changes_sat = changes_sat[changes_sat["NDVI_mean"].notna()].copy()

print(f"Wells with both temporal changes and satellite data: {len(changes_sat)}")

# ─────────────────────────────────────────────────────────
# 3. SATELLITE PROXY → ΔPO₄ CORRELATIONS
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SATELLITE PROXY CORRELATIONS WITH TEMPORAL CHANGES")
print(f"{'=' * 60}")

# Key satellite proxies
proxies = {
    "NDVI_mean": "Vegetation intensity (agriculture proxy)",
    "NDVI_max": "Peak vegetation (crop season proxy)",
    "NDWI_mean": "Water index (wetland/irrigation proxy)",
    "EVI_mean": "Enhanced vegetation index",
    "NDMI_mean": "Moisture index",
}

# Key temporal changes
targets = {
    "Delta_PO4": "ΔPO₄ (anthropogenic loading indicator)",
    "Delta_As": "ΔAs (arsenic mobilization)",
    "Delta_Fe": "ΔFe (Fe-oxide dissolution)",
    "Delta_Mn": "ΔMn (Mn-oxide reduction)",
}

corr_results = []

for proxy_col, proxy_desc in proxies.items():
    if proxy_col not in changes_sat.columns:
        continue

    print(f"\n  {proxy_col} ({proxy_desc}):")
    for target_col, target_desc in targets.items():
        if target_col not in changes_sat.columns:
            continue

        mask = changes_sat[proxy_col].notna() & changes_sat[target_col].notna()
        x = changes_sat.loc[mask, proxy_col].values
        y = changes_sat.loc[mask, target_col].values

        if len(x) < 10:
            continue

        rho, p_val = stats.spearmanr(x, y)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else (
            "*" if p_val < 0.05 else "ns"))
        print(f"    vs {target_col:>10}: ρ={rho:+.3f}, p={p_val:.4f} {sig} (n={len(x)})")

        corr_results.append({
            "Satellite_proxy": proxy_col,
            "Proxy_description": proxy_desc,
            "Target": target_col,
            "Target_description": target_desc,
            "n": len(x),
            "Spearman_rho": rho,
            "p_value": p_val,
        })

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(TABLE_DIR / "T08_satellite_linkage.csv", index=False)
print(f"\nSaved: T08_satellite_linkage.csv")

# ─────────────────────────────────────────────────────────
# 4. PARTIAL CORRELATION: NDVI → ΔPO₄ | Depth, baseline PO₄
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("PARTIAL CORRELATIONS (controlling for confounders)")
print(f"{'=' * 60}")

def partial_spearman(x, y, covariates):
    """Partial Spearman correlation via rank residualization."""
    from scipy.stats import rankdata
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    cov_ranks = np.column_stack([rankdata(c) for c in covariates.T])

    # Residualize x and y on covariates
    from numpy.linalg import lstsq
    A = np.column_stack([np.ones(len(x_rank)), cov_ranks])
    coef_x, _, _, _ = lstsq(A, x_rank, rcond=None)
    coef_y, _, _, _ = lstsq(A, y_rank, rcond=None)
    res_x = x_rank - A @ coef_x
    res_y = y_rank - A @ coef_y

    r, p = stats.pearsonr(res_x, res_y)
    return r, p

partial_results = []

for proxy in ["NDVI_mean", "NDWI_mean", "EVI_mean"]:
    for target in ["Delta_PO4", "Delta_As", "Delta_Fe"]:
        mask = (changes_sat[proxy].notna() &
                changes_sat[target].notna() &
                changes_sat["Depth"].notna() &
                changes_sat["Old_PO4"].notna())
        if mask.sum() < 20:
            continue

        x = changes_sat.loc[mask, proxy].values
        y = changes_sat.loc[mask, target].values
        covs = changes_sat.loc[mask, ["Depth", "Old_PO4"]].values

        r_partial, p_partial = partial_spearman(x, y, covs)
        r_zero, p_zero = stats.spearmanr(x, y)

        sig = "***" if p_partial < 0.001 else ("**" if p_partial < 0.01 else (
            "*" if p_partial < 0.05 else "ns"))
        print(f"  {proxy:>10} → {target:>10} | Depth, baseline PO₄:")
        print(f"    Zero-order ρ={r_zero:+.3f} (p={p_zero:.4f})")
        print(f"    Partial ρ   ={r_partial:+.3f} (p={p_partial:.4f}) {sig}")

        partial_results.append({
            "Proxy": proxy,
            "Target": target,
            "n": mask.sum(),
            "r_zero_order": r_zero,
            "p_zero_order": p_zero,
            "r_partial": r_partial,
            "p_partial": p_partial,
            "Controls": "Depth, baseline PO4",
        })

partial_df = pd.DataFrame(partial_results)

# ─────────────────────────────────────────────────────────
# 5. TERCILE COMPARISON: HIGH vs LOW NDVI WELLS
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("NDVI TERCILE COMPARISON")
print(f"{'=' * 60}")

ndvi_valid = changes_sat[changes_sat["NDVI_mean"].notna() & changes_sat["Delta_PO4"].notna()].copy()

if len(ndvi_valid) >= 30:
    terciles = pd.qcut(ndvi_valid["NDVI_mean"], q=3, labels=["Low NDVI", "Medium NDVI", "High NDVI"])
    ndvi_valid["NDVI_tercile"] = terciles

    print(f"\n  NDVI tercile thresholds: {ndvi_valid.groupby('NDVI_tercile')['NDVI_mean'].agg(['min','max']).to_string()}")

    tercile_results = []
    for param in ["Delta_PO4", "Delta_As", "Delta_Fe", "Delta_Mn"]:
        if param not in ndvi_valid.columns:
            continue

        print(f"\n  {param}:")
        groups = {}
        for t in ["Low NDVI", "Medium NDVI", "High NDVI"]:
            vals = ndvi_valid.loc[ndvi_valid["NDVI_tercile"] == t, param].dropna()
            groups[t] = vals
            print(f"    {t}: n={len(vals)}, median={vals.median():+.3f}, mean={vals.mean():+.3f}")

        # Kruskal-Wallis across terciles
        if all(len(v) >= 5 for v in groups.values()):
            h_stat, kw_p = stats.kruskal(*groups.values())
            sig = "***" if kw_p < 0.001 else ("**" if kw_p < 0.01 else (
                "*" if kw_p < 0.05 else "ns"))
            print(f"    Kruskal-Wallis: H={h_stat:.2f}, p={kw_p:.4f} {sig}")

            # Mann-Whitney: High vs Low
            u_stat, mw_p = stats.mannwhitneyu(
                groups["High NDVI"], groups["Low NDVI"], alternative="two-sided"
            )
            sig_mw = "***" if mw_p < 0.001 else ("**" if mw_p < 0.01 else (
                "*" if mw_p < 0.05 else "ns"))
            print(f"    High vs Low: U={u_stat:.0f}, p={mw_p:.4f} {sig_mw}")

            tercile_results.append({
                "Parameter": param,
                "Low_NDVI_median": groups["Low NDVI"].median(),
                "High_NDVI_median": groups["High NDVI"].median(),
                "KW_H": h_stat,
                "KW_p": kw_p,
                "HighVsLow_U": u_stat,
                "HighVsLow_p": mw_p,
            })

# ─────────────────────────────────────────────────────────
# 6. MULTIVARIATE REGRESSION: ΔPO₄ ~ NDVI + NDWI + Depth
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("MULTIVARIATE REGRESSION: ΔPO₄ ~ satellite proxies")
print(f"{'=' * 60}")

reg_vars = ["NDVI_mean", "NDWI_mean", "EVI_mean", "Depth"]
reg_target = "Delta_PO4"

mask = changes_sat[reg_target].notna()
for v in reg_vars:
    if v in changes_sat.columns:
        mask = mask & changes_sat[v].notna()

reg_data = changes_sat[mask].copy()
print(f"\n  Regression n = {len(reg_data)}")

if len(reg_data) >= 30:
    X = reg_data[reg_vars].values
    y = reg_data[reg_target].values

    # Standardize predictors for comparable coefficients
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_z = (X - X_mean) / (X_std + 1e-10)

    reg = LinearRegression()
    reg.fit(X_z, y)

    y_pred = reg.predict(X_z)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(reg_vars) - 1)

    # F-test
    n, p = len(y), len(reg_vars)
    f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
    f_p = 1 - stats.f.cdf(f_stat, p, n - p - 1)

    print(f"  R² = {r2:.4f}, Adj R² = {r2_adj:.4f}")
    print(f"  F({p}, {n-p-1}) = {f_stat:.2f}, p = {f_p:.4f}")

    # Coefficient significance via bootstrap
    n_boot = 5000
    coef_boots = np.zeros((n_boot, len(reg_vars)))
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        reg_b = LinearRegression()
        reg_b.fit(X_z[idx], y[idx])
        coef_boots[i] = reg_b.coef_

    print(f"\n  {'Predictor':<15} {'Std. β':>10} {'95% CI':>25} {'p (boot)':>10}")
    print(f"  {'-' * 62}")

    reg_rows = []
    for j, var in enumerate(reg_vars):
        beta = reg.coef_[j]
        ci = np.percentile(coef_boots[:, j], [2.5, 97.5])
        # Bootstrap p-value (two-sided)
        if beta >= 0:
            p_boot = 2 * np.mean(coef_boots[:, j] <= 0)
        else:
            p_boot = 2 * np.mean(coef_boots[:, j] >= 0)
        p_boot = min(p_boot, 1.0)

        sig = "***" if p_boot < 0.001 else ("**" if p_boot < 0.01 else (
            "*" if p_boot < 0.05 else "ns"))
        print(f"  {var:<15} {beta:>+10.4f} [{ci[0]:>+10.4f}, {ci[1]:>+10.4f}] {p_boot:>10.4f} {sig}")

        reg_rows.append({
            "Predictor": var,
            "Std_beta": beta,
            "CI_low": ci[0],
            "CI_high": ci[1],
            "p_bootstrap": p_boot,
        })

    reg_df = pd.DataFrame(reg_rows)
    reg_df["R2"] = r2
    reg_df["R2_adj"] = r2_adj
    reg_df["F_stat"] = f_stat
    reg_df["F_p"] = f_p
    reg_df.to_csv(TABLE_DIR / "T08b_landuse_regression.csv", index=False)
    print(f"\n  Saved: T08b_landuse_regression.csv")

# ─────────────────────────────────────────────────────────
# 7. FIGURE: SATELLITE-GEOCHEMISTRY LINKAGE
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(8, 7))

plot_data = changes_sat.copy()

# Panel A: NDVI vs ΔPO₄
ax = axes[0, 0]
mask = plot_data["NDVI_mean"].notna() & plot_data["Delta_PO4"].notna()
if mask.sum() > 5:
    x, y = plot_data.loc[mask, "NDVI_mean"], plot_data.loc[mask, "Delta_PO4"]
    ax.scatter(x, y, c="#9b59b6", alpha=0.4, s=15, edgecolors="none")
    # Trend line
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1.5, alpha=0.7)
    rho, p = stats.spearmanr(x, y)
    ax.set_title(f"NDVI vs ΔPO₄ (ρ={rho:+.3f})", fontweight="bold", fontsize=10)
ax.set_xlabel("NDVI mean", fontsize=9)
ax.set_ylabel("ΔPO₄ (mg/L)", fontsize=9)
ax.axhline(0, color="gray", ls=":", lw=0.8)

# Panel B: NDVI vs ΔAs
ax = axes[0, 1]
mask = plot_data["NDVI_mean"].notna() & plot_data["Delta_As"].notna()
if mask.sum() > 5:
    x, y = plot_data.loc[mask, "NDVI_mean"], plot_data.loc[mask, "Delta_As"]
    ax.scatter(x, y, c="#e74c3c", alpha=0.4, s=15, edgecolors="none")
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1.5, alpha=0.7)
    rho, p = stats.spearmanr(x, y)
    ax.set_title(f"NDVI vs ΔAs (ρ={rho:+.3f})", fontweight="bold", fontsize=10)
ax.set_xlabel("NDVI mean", fontsize=9)
ax.set_ylabel("ΔAs (µg/L)", fontsize=9)
ax.axhline(0, color="gray", ls=":", lw=0.8)

# Panel C: NDWI vs ΔPO₄
ax = axes[1, 0]
mask = plot_data["NDWI_mean"].notna() & plot_data["Delta_PO4"].notna()
if mask.sum() > 5:
    x, y = plot_data.loc[mask, "NDWI_mean"], plot_data.loc[mask, "Delta_PO4"]
    ax.scatter(x, y, c="#3498db", alpha=0.4, s=15, edgecolors="none")
    z = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", lw=1.5, alpha=0.7)
    rho, p = stats.spearmanr(x, y)
    ax.set_title(f"NDWI vs ΔPO₄ (ρ={rho:+.3f})", fontweight="bold", fontsize=10)
ax.set_xlabel("NDWI mean", fontsize=9)
ax.set_ylabel("ΔPO₄ (mg/L)", fontsize=9)
ax.axhline(0, color="gray", ls=":", lw=0.8)

# Panel D: NDVI tercile boxplot for ΔPO₄
ax = axes[1, 1]
if "NDVI_tercile" in ndvi_valid.columns:
    tercile_data = [
        ndvi_valid.loc[ndvi_valid["NDVI_tercile"] == t, "Delta_PO4"].dropna().values
        for t in ["Low NDVI", "Medium NDVI", "High NDVI"]
    ]
    bp = ax.boxplot(tercile_data, labels=["Low\nNDVI", "Med\nNDVI", "High\nNDVI"],
                    patch_artist=True, widths=0.5)
    colors_bp = ["#2ecc71", "#f1c40f", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("ΔPO₄ (mg/L)", fontsize=9)
    ax.set_title("ΔPO₄ by vegetation intensity", fontweight="bold", fontsize=10)

plt.suptitle("Satellite-Derived Land Use Proxies vs Geochemical Changes",
             fontweight="bold", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "F09_satellite_landuse_linkage.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {FIGURE_DIR / 'F09_satellite_landuse_linkage.png'}")
plt.close()


# ═════════════════════════════════════════════════════════
# MANUSCRIPT SUMMARY
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("MANUSCRIPT-READY SUMMARY — SATELLITE LAND USE LINKAGE")
print(f"{'=' * 80}")

print(f"\n  Wells matched to satellite data: {len(changes_sat)}")
print(f"\n  Top correlations (Spearman ρ, satellite proxy → temporal change):")
if len(corr_df) > 0:
    sig_corrs = corr_df[corr_df["p_value"] < 0.1].sort_values("p_value")
    if len(sig_corrs) > 0:
        for _, r in sig_corrs.head(10).iterrows():
            sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else (
                "*" if r["p_value"] < 0.05 else "."))
            print(f"    {r['Satellite_proxy']:>12} → {r['Target']:>12}: ρ={r['Spearman_rho']:+.3f} (p={r['p_value']:.4f} {sig})")
    else:
        print("    No significant correlations at p<0.1")
        print("    Top 5 by effect size:")
        top5 = corr_df.sort_values("Spearman_rho", key=abs, ascending=False).head(5)
        for _, r in top5.iterrows():
            print(f"    {r['Satellite_proxy']:>12} → {r['Target']:>12}: ρ={r['Spearman_rho']:+.3f} (p={r['p_value']:.4f})")

print("\nDone.")
