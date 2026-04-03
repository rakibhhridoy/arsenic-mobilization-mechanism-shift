"""
A8 — Tipping Point Analysis: Saturation Index + PO₄-As Threshold
==================================================================
Two analyses to establish urgency for ES&T:

1. FERRIHYDRITE SATURATION INDEX (SI):
   SI = log10(IAP/Ksp) where IAP = [Fe³⁺][OH⁻]³
   Using Eh to partition Fe²⁺/Fe³⁺ via Nernst equation.
   Tests whether the Fe-oxide mineral buffer is approaching exhaustion.

2. PO₄-As NONLINEAR THRESHOLD:
   Segmented regression to find the PO₄ breakpoint above which
   As concentration accelerates. Compare fraction of wells above
   threshold between periods.

Outputs:
  - Table: T09_saturation_index.csv
  - Table: T09b_threshold_analysis.csv
  - Figure: F10_tipping_point.png (MAIN TEXT — composite)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from config import (
    TABLE_DIR, FIGURE_DIR, ALPHA, RANDOM_SEED, DEPTH_BINS, DEPTH_LABELS,
    set_est_style
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

# Coastal filter
coastal_districts = set(old_full["District"].dropna().str.strip().str.title().unique())
new_full_all["District_norm"] = new_full_all["District"].str.strip().str.title()
new_full = new_full_all[new_full_all["District_norm"].isin(coastal_districts)].copy()

print(f"Paired: {len(old)} wells")
print(f"Full old: {len(old_full)}, Full new (coastal): {len(new_full)}")


# ═════════════════════════════════════════════════════════
# PART 1: FERRIHYDRITE SATURATION INDEX
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 1: FERRIHYDRITE SATURATION INDEX (SI)")
print(f"{'=' * 80}")

# Ferrihydrite: Fe(OH)₃(am)
# Dissolution: Fe(OH)₃ + 3H⁺ → Fe³⁺ + 3H₂O
# log Ksp = 4.891 (ferrihydrite, Stumm & Morgan 1996; Dzombak & Morel 1990)
# At 25°C, commonly used values: log Ksp = 3.0 to 5.0 depending on crystallinity
# We use 4.891 for amorphous ferrihydrite (most reactive form)

LOG_KSP_FERRIHYDRITE = 4.891  # log Ksp at 25°C

# Fe²⁺/Fe³⁺ partitioning via Nernst equation:
# Fe³⁺ + e⁻ → Fe²⁺   E° = 0.771 V
# Eh = E° + (RT/nF) * ln([Fe³⁺]/[Fe²⁺])
# At 25°C: Eh = 0.771 + 0.05916 * log10([Fe³⁺]/[Fe²⁺])
# → log10([Fe³⁺]/[Fe²⁺]) = (Eh - 0.771) / 0.05916
# → [Fe³⁺] = [Fe_total] * (10^((Eh-0.771)/0.05916)) / (1 + 10^((Eh-0.771)/0.05916))

E0_FE = 0.771  # V, standard reduction potential Fe³⁺/Fe²⁺
R_NERNST = 0.05916  # RT/nF at 25°C in V (for n=1)


def calc_si_ferrihydrite(fe_total_mg_l, ph, eh_mv, temp_c=25.0):
    """
    Calculate Saturation Index for ferrihydrite Fe(OH)₃(am).

    Parameters
    ----------
    fe_total_mg_l : float — Total dissolved Fe in mg/L
    ph : float — pH
    eh_mv : float — Eh in mV
    temp_c : float — Temperature in °C (default 25)

    Returns
    -------
    SI : float — Saturation Index (>0 supersaturated, <0 undersaturated)
    log_fe3 : float — log10([Fe³⁺] in mol/L)
    """
    # Convert Fe to mol/L (MW Fe = 55.845 g/mol)
    fe_total_mol = fe_total_mg_l / 55845.0  # mg/L → mol/L

    if fe_total_mol <= 0 or np.isnan(fe_total_mol):
        return np.nan, np.nan

    # Eh in volts
    eh_v = eh_mv / 1000.0

    # Fe³⁺/Fe²⁺ ratio from Nernst
    log_ratio = (eh_v - E0_FE) / R_NERNST
    ratio = 10**log_ratio

    # Fe³⁺ concentration
    fe3_mol = fe_total_mol * ratio / (1.0 + ratio)

    # Clamp to avoid log of zero
    fe3_mol = max(fe3_mol, 1e-30)

    log_fe3 = np.log10(fe3_mol)

    # SI = log10(IAP) - log10(Ksp)
    # IAP for Fe(OH)₃ dissolution: Fe(OH)₃ + 3H⁺ → Fe³⁺ + 3H₂O
    # Ksp = [Fe³⁺] / [H⁺]³ = [Fe³⁺] * 10^(3*pH)
    # SI = log10([Fe³⁺]) + 3*pH - log10(Ksp)
    log_iap = log_fe3 + 3.0 * ph
    si = log_iap - LOG_KSP_FERRIHYDRITE

    return si, log_fe3


# Calculate SI for all datasets
si_results = []

for label, df in [("2012-2013 (paired)", old),
                   ("2020-2021 (paired)", new),
                   ("2012-2013 (full)", old_full),
                   ("2020-2021 (full)", new_full)]:
    required = ["Fe", "pH", "Eh"]
    mask = df[required].notna().all(axis=1)
    # Exclude zero/negative Fe
    mask = mask & (df["Fe"] > 0)

    si_vals = []
    for idx in df[mask].index:
        si, _ = calc_si_ferrihydrite(
            df.loc[idx, "Fe"],
            df.loc[idx, "pH"],
            df.loc[idx, "Eh"],
            df.loc[idx, "Temperature"] if "Temperature" in df.columns and pd.notna(df.loc[idx, "Temperature"]) else 25.0
        )
        si_vals.append(si)

    si_arr = np.array(si_vals)
    si_arr = si_arr[np.isfinite(si_arr)]

    n_super = np.sum(si_arr > 0)
    n_under = np.sum(si_arr < 0)

    print(f"\n  {label} (n={len(si_arr)} valid):")
    print(f"    SI median: {np.median(si_arr):.2f}")
    print(f"    SI mean:   {np.mean(si_arr):.2f} ± {np.std(si_arr):.2f}")
    print(f"    Supersaturated (SI>0): {n_super} ({n_super/len(si_arr)*100:.1f}%)")
    print(f"    Undersaturated (SI<0): {n_under} ({n_under/len(si_arr)*100:.1f}%)")
    print(f"    Range: [{np.min(si_arr):.2f}, {np.max(si_arr):.2f}]")

    si_results.append({
        "Dataset": label,
        "n": len(si_arr),
        "SI_median": np.median(si_arr),
        "SI_mean": np.mean(si_arr),
        "SI_std": np.std(si_arr),
        "Pct_supersaturated": n_super / len(si_arr) * 100,
        "Pct_undersaturated": n_under / len(si_arr) * 100,
    })

# Add SI to paired datasets for comparison
for df in [old, new]:
    si_col = []
    for idx in df.index:
        if pd.notna(df.loc[idx, "Fe"]) and pd.notna(df.loc[idx, "pH"]) and pd.notna(df.loc[idx, "Eh"]) and df.loc[idx, "Fe"] > 0:
            si, _ = calc_si_ferrihydrite(df.loc[idx, "Fe"], df.loc[idx, "pH"], df.loc[idx, "Eh"])
        else:
            si = np.nan
        si_col.append(si)
    df["SI_ferrihydrite"] = si_col

# Paired SI comparison
mask_si = old["SI_ferrihydrite"].notna() & new["SI_ferrihydrite"].notna() & \
          np.isfinite(old["SI_ferrihydrite"]) & np.isfinite(new["SI_ferrihydrite"])
old_si = old.loc[mask_si, "SI_ferrihydrite"].values
new_si = new.loc[mask_si, "SI_ferrihydrite"].values
diff_si = new_si - old_si

print(f"\n  PAIRED SI COMPARISON (n={mask_si.sum()}):")
print(f"    Old SI median: {np.median(old_si):.2f}")
print(f"    New SI median: {np.median(new_si):.2f}")
print(f"    ΔSI median:    {np.median(diff_si):+.2f}")

w_stat, w_p = stats.wilcoxon(diff_si, zero_method="wilcox")
print(f"    Wilcoxon: W={w_stat:.0f}, p={w_p:.2e}")

# How many wells shifted from supersaturated to undersaturated?
was_super = old_si > 0
now_under = new_si < 0
shifted = was_super & now_under
print(f"    Wells shifted super→under: {shifted.sum()} ({shifted.sum()/len(old_si)*100:.1f}%)")
print(f"    Wells remained super:      {(was_super & ~now_under).sum()}")
print(f"    Wells that were under both: {(~was_super & now_under).sum()}")

si_df = pd.DataFrame(si_results)
si_df.to_csv(TABLE_DIR / "T09_saturation_index.csv", index=False)
print(f"\nSaved: T09_saturation_index.csv")


# ═════════════════════════════════════════════════════════
# PART 2: PO₄-As NONLINEAR THRESHOLD
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("PART 2: PO₄-As NONLINEAR THRESHOLD (SEGMENTED REGRESSION)")
print(f"{'=' * 80}")

# Combine both periods for threshold detection
all_data = pd.concat([
    old_full.assign(Period="2012-2013"),
    new_full.assign(Period="2020-2021")
], ignore_index=True)

# Filter valid PO4 and As
mask_valid = all_data["PO4"].notna() & all_data["As"].notna() & (all_data["PO4"] > 0)
threshold_data = all_data[mask_valid].copy()

print(f"  Total samples for threshold analysis: {len(threshold_data)}")


def segmented_regression(x, y, breakpoint):
    """Fit piecewise linear regression with one breakpoint."""
    n = len(x)
    x1 = np.minimum(x, breakpoint)
    x2 = np.maximum(x - breakpoint, 0)
    X = np.column_stack([np.ones(n), x1, x2])
    try:
        coef, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coef
        ss_res = np.sum((y - y_pred)**2)
        return coef, ss_res, y_pred
    except:
        return None, np.inf, None


def find_optimal_breakpoint(x, y, search_range=None):
    """Find breakpoint that minimizes RSS."""
    if search_range is None:
        search_range = np.percentile(x, [10, 90])

    candidates = np.linspace(search_range[0], search_range[1], 200)
    best_bp, best_ss = None, np.inf

    for bp in candidates:
        coef, ss, _ = segmented_regression(x, y, bp)
        if ss < best_ss:
            best_ss = ss
            best_bp = bp

    return best_bp, best_ss


# Log-transform As for better behavior (many zeros/low values)
threshold_data["As_log"] = np.log10(threshold_data["As"].clip(lower=0.1))

# Find threshold for each period
threshold_results = []

for period in ["2012-2013", "2020-2021"]:
    pdf = threshold_data[threshold_data["Period"] == period]
    x = pdf["PO4"].values
    y = pdf["As_log"].values

    # Linear model (null)
    slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(x, y)
    y_pred_lin = intercept_lin + slope_lin * x
    ss_lin = np.sum((y - y_pred_lin)**2)

    # Segmented regression
    bp_opt, ss_seg = find_optimal_breakpoint(x, y)
    coef_seg, _, y_pred_seg = segmented_regression(x, y, bp_opt)

    # F-test: segmented vs linear (2 extra parameters: breakpoint + slope change)
    n = len(x)
    df_lin = n - 2  # linear: intercept + slope
    df_seg = n - 4  # segmented: intercept + slope1 + slope2 + breakpoint
    f_stat = ((ss_lin - ss_seg) / 2) / (ss_seg / df_seg) if ss_seg > 0 else 0
    f_p = 1 - stats.f.cdf(f_stat, 2, df_seg) if f_stat > 0 else 1.0

    slope_below = coef_seg[1] if coef_seg is not None else np.nan
    slope_above = coef_seg[1] + coef_seg[2] if coef_seg is not None else np.nan

    print(f"\n  {period} (n={n}):")
    print(f"    Linear: slope={slope_lin:.4f}, R²={r_lin**2:.4f}")
    print(f"    Segmented breakpoint: PO₄ = {bp_opt:.3f} mg/L")
    print(f"    Slope below BP: {slope_below:.4f} (log10 As per mg/L PO₄)")
    print(f"    Slope above BP: {slope_above:.4f}")
    print(f"    Slope ratio (above/below): {slope_above/slope_below:.1f}x" if slope_below != 0 else "")
    print(f"    F-test (segmented vs linear): F={f_stat:.2f}, p={f_p:.4f}")
    print(f"    SS linear: {ss_lin:.2f}, SS segmented: {ss_seg:.2f}")

    # What fraction of wells exceed the threshold?
    n_above = (pdf["PO4"] > bp_opt).sum()
    print(f"    Wells above threshold: {n_above}/{n} ({n_above/n*100:.1f}%)")

    threshold_results.append({
        "Period": period,
        "n": n,
        "Breakpoint_PO4": bp_opt,
        "Slope_below": slope_below,
        "Slope_above": slope_above,
        "Slope_ratio": slope_above / slope_below if slope_below != 0 else np.nan,
        "F_stat": f_stat,
        "F_p": f_p,
        "Pct_above_threshold": n_above / n * 100,
        "Linear_R2": r_lin**2,
    })

# Paired comparison: wells crossing the threshold
# Use the average breakpoint
avg_bp = np.mean([r["Breakpoint_PO4"] for r in threshold_results])
print(f"\n  Average breakpoint: PO₄ = {avg_bp:.3f} mg/L")

mask_po4 = old["PO4"].notna() & new["PO4"].notna()
old_above = (old.loc[mask_po4, "PO4"] > avg_bp).sum()
new_above = (new.loc[mask_po4, "PO4"] > avg_bp).sum()
total = mask_po4.sum()

print(f"  Paired wells above threshold:")
print(f"    2012-13: {old_above}/{total} ({old_above/total*100:.1f}%)")
print(f"    2020-21: {new_above}/{total} ({new_above/total*100:.1f}%)")
print(f"    Increase: +{new_above - old_above} wells ({(new_above-old_above)/total*100:.1f}pp)")

# McNemar test on threshold crossing
was_below = old.loc[mask_po4, "PO4"].values <= avg_bp
now_above = new.loc[mask_po4, "PO4"].values > avg_bp
crossed_up = was_below & now_above
crossed_down = (~was_below) & (~now_above)

# Forward projection
delta_po4_rate = new.loc[mask_po4, "PO4"].median() - old.loc[mask_po4, "PO4"].median()
years_elapsed = 8  # 2012-2020
po4_rate_per_year = delta_po4_rate / years_elapsed
current_median = new.loc[mask_po4, "PO4"].median()

if po4_rate_per_year > 0:
    years_to_all_above = (avg_bp - current_median) / po4_rate_per_year if current_median < avg_bp else 0
    print(f"\n  FORWARD PROJECTION (linear extrapolation):")
    print(f"    Current median PO₄: {current_median:.3f} mg/L")
    print(f"    Rate of increase: {po4_rate_per_year:.4f} mg/L per year")
    print(f"    Threshold: {avg_bp:.3f} mg/L")
    if current_median < avg_bp:
        print(f"    Years until median crosses threshold: ~{years_to_all_above:.0f}")
    else:
        print(f"    Median ALREADY above threshold")

thresh_df = pd.DataFrame(threshold_results)
thresh_df.to_csv(TABLE_DIR / "T09b_threshold_analysis.csv", index=False)
print(f"\nSaved: T09b_threshold_analysis.csv")


# ═════════════════════════════════════════════════════════
# PART 3: COMPOSITE FIGURE
# ═════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# --- Panel A: SI distributions ---
ax_a = fig.add_subplot(gs[0, 0])

# Compute SI for full datasets
for label, df, color, ls in [("2012-13", old_full, "#e74c3c", "-"),
                               ("2020-21", new_full, "#3498db", "-")]:
    si_vals = []
    mask = df["Fe"].notna() & df["pH"].notna() & df["Eh"].notna() & (df["Fe"] > 0)
    for idx in df[mask].index:
        si, _ = calc_si_ferrihydrite(df.loc[idx, "Fe"], df.loc[idx, "pH"], df.loc[idx, "Eh"])
        if np.isfinite(si):
            si_vals.append(si)

    si_arr = np.array(si_vals)
    ax_a.hist(si_arr, bins=50, density=True, alpha=0.4, color=color, edgecolor="none", label=label)
    ax_a.axvline(np.median(si_arr), color=color, ls="--", lw=1.5)

ax_a.axvline(0, color="black", ls="-", lw=2, alpha=0.8)
ax_a.text(0.05, 0.95, "← undersaturated | supersaturated →", transform=ax_a.transAxes,
          fontsize=7, va="top", style="italic", alpha=0.6)
ax_a.set_xlabel("SI (ferrihydrite)", fontsize=10)
ax_a.set_ylabel("Density", fontsize=10)
ax_a.set_title("(a) Ferrihydrite saturation index", fontweight="bold")
ax_a.legend(fontsize=9)

# --- Panel B: Paired SI change ---
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(old_si, new_si, c=diff_si, cmap="RdBu_r", alpha=0.5, s=15, edgecolors="none",
             vmin=-5, vmax=5)
lim = [min(old_si.min(), new_si.min()) - 1, max(old_si.max(), new_si.max()) + 1]
ax_b.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
ax_b.axhline(0, color="gray", ls=":", lw=0.8)
ax_b.axvline(0, color="gray", ls=":", lw=0.8)
ax_b.set_xlabel("SI 2012-13", fontsize=10)
ax_b.set_ylabel("SI 2020-21", fontsize=10)
ax_b.set_title(f"(b) Paired SI change (n={mask_si.sum()})", fontweight="bold")
# Annotate quadrants
ax_b.text(0.05, 0.05, f"super→under\n{shifted.sum()} wells", transform=ax_b.transAxes,
          fontsize=7, color="red", fontweight="bold")

# --- Panel C: PO₄-As segmented regression ---
ax_c = fig.add_subplot(gs[1, 0])

for period, color in [("2012-2013", "#e74c3c"), ("2020-2021", "#3498db")]:
    pdf = threshold_data[threshold_data["Period"] == period]
    x = pdf["PO4"].values
    y = pdf["As"].values
    ax_c.scatter(x, y, c=color, alpha=0.15, s=8, edgecolors="none",
                 label=f"{period[:7]}")

    # Segmented fit line
    r = [r for r in threshold_results if r["Period"] == period][0]
    bp = r["Breakpoint_PO4"]
    x_line = np.linspace(0, x.max(), 200)
    coef, _, _ = segmented_regression(x, np.log10(pdf["As"].clip(lower=0.1).values), bp)
    if coef is not None:
        x1 = np.minimum(x_line, bp)
        x2 = np.maximum(x_line - bp, 0)
        y_line = coef[0] + coef[1] * x1 + coef[2] * x2
        ax_c.plot(x_line, 10**y_line, color=color, lw=2, alpha=0.8)

ax_c.axvline(avg_bp, color="black", ls="--", lw=1.5, alpha=0.7, label=f"Threshold={avg_bp:.2f}")
ax_c.set_xlabel("PO₄ (mg/L)", fontsize=10)
ax_c.set_ylabel("As (µg/L)", fontsize=10)
ax_c.set_yscale("symlog", linthresh=1)
ax_c.set_title("(c) PO₄-As threshold", fontweight="bold")
ax_c.legend(fontsize=8)

# --- Panel D: Wells above threshold over time ---
ax_d = fig.add_subplot(gs[1, 1])

# Bar chart: wells above threshold
labels = ["2012-13", "2020-21"]
above_pcts = [old_above / total * 100, new_above / total * 100]
bars = ax_d.bar(labels, above_pcts, color=["#e74c3c", "#3498db"], alpha=0.7, width=0.5)
ax_d.set_ylabel("Wells above PO₄ threshold (%)", fontsize=10)
ax_d.set_title(f"(d) Threshold crossing ({avg_bp:.2f} mg/L)", fontweight="bold")

for bar, pct in zip(bars, above_pcts):
    ax_d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
              f"{pct:.1f}%", ha="center", fontsize=11, fontweight="bold")

# Add arrow showing change
ax_d.annotate(f"+{above_pcts[1]-above_pcts[0]:.1f}pp",
              xy=(1, above_pcts[1]), xytext=(0.5, max(above_pcts) + 8),
              fontsize=10, fontweight="bold", color="darkred",
              arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
              ha="center")

plt.suptitle("Geochemical Tipping Point Analysis: Fe-Oxide Buffer & PO₄ Threshold",
             fontweight="bold", fontsize=13, y=1.01)

fig.savefig(FIGURE_DIR / "F10_tipping_point.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {FIGURE_DIR / 'F10_tipping_point.png'}")
plt.close()


# ═════════════════════════════════════════════════════════
# MANUSCRIPT SUMMARY
# ═════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print("MANUSCRIPT-READY SUMMARY — TIPPING POINT ANALYSIS")
print(f"{'=' * 80}")

print("\n1. FERRIHYDRITE SATURATION INDEX:")
for r in si_results:
    print(f"   {r['Dataset']}: SI median={r['SI_median']:.2f}, "
          f"supersaturated={r['Pct_supersaturated']:.1f}%, "
          f"undersaturated={r['Pct_undersaturated']:.1f}%")

print(f"\n   Paired ΔSI: {np.median(diff_si):+.2f} (Wilcoxon p={w_p:.2e})")
print(f"   Wells shifted super→under: {shifted.sum()} ({shifted.sum()/len(old_si)*100:.1f}%)")

print(f"\n2. PO₄-As THRESHOLD:")
for r in threshold_results:
    print(f"   {r['Period']}: breakpoint={r['Breakpoint_PO4']:.3f} mg/L, "
          f"slope ratio={r['Slope_ratio']:.1f}x, "
          f"F={r['F_stat']:.2f} (p={r['F_p']:.4f}), "
          f"above threshold={r['Pct_above_threshold']:.1f}%")

print(f"\n   Paired wells above threshold: {old_above/total*100:.1f}% → {new_above/total*100:.1f}%")
print(f"   (+{(new_above-old_above)/total*100:.1f} percentage points)")

print("\nDone.")
