"""
A9 — PHREEQC Geochemical Speciation & Inverse Modeling
=======================================================
Uses phreeqpy IPhreeqc DLL interface with wateq4f.dat (has As species).

1. SPECIATION: Calculate mineral saturation indices (SI) for all wells
   - Fe(OH)3(a) [ferrihydrite], Goethite, Calcite, Siderite, Vivianite,
     Rhodochrosite, Gypsum, Halite, Dolomite, MnO2, Hydroxyapatite
2. PAIRED SI COMPARISON: Wilcoxon tests on mineral SI changes
3. INVERSE MODELING: What mass transfers explain 2012→2021 composition?
4. FIGURES: SI distributions, paired ΔSI, inverse model results

Outputs:
  - Table: T10_phreeqc_speciation.csv (all wells, all SIs)
  - Table: T10b_si_paired_stats.csv (paired SI change stats)
  - Table: T10c_inverse_models.csv (inverse modeling results)
  - Figure: F11_phreeqc_speciation.png (4-panel composite)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import re
import sys
warnings.filterwarnings("ignore")

from config import (
    TABLE_DIR, FIGURE_DIR, ALPHA, RANDOM_SEED, DEPTH_BINS, DEPTH_LABELS,
    set_est_style
)

set_est_style()
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────
# 0. PHREEQC SETUP
# ─────────────────────────────────────────────────────────

from pathlib import Path
DB_PATH = str(Path(__file__).parent / "wateq4f.dat")

from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

# Target minerals (names in wateq4f.dat)
TARGET_MINERALS = [
    "Fe(OH)3(a)",      # Ferrihydrite (amorphous)
    "Goethite",        # Crystalline Fe-oxide
    "Calcite",         # CaCO3
    "Siderite",        # FeCO3
    "Vivianite",       # Fe3(PO4)2·8H2O
    "Rhodochrosite",   # MnCO3
    "Gypsum",          # CaSO4·2H2O
    "Dolomite",        # CaMg(CO3)2
    "Halite",          # NaCl
    "Pyrolusite",      # MnO2
    "Hydroxyapatite",  # Ca5(PO4)3OH
    "Manganite",       # MnOOH
    "Aragonite",       # CaCO3 polymorph
]

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

print(f"Paired: {len(old)} old + {len(new)} new = {len(old)+len(new)}")
print(f"Full: {len(old_full)} old, {len(new_full)} new (coastal)")

# ─────────────────────────────────────────────────────────
# 2. PHREEQC SPECIATION FUNCTION
# ─────────────────────────────────────────────────────────

def build_solution_block(row, sol_num=1):
    """Build a PHREEQC SOLUTION block from a data row."""
    lines = [f"SOLUTION {sol_num}"]
    temp = row.get('Temperature', 25.0)
    if pd.isna(temp) or temp <= 0:
        temp = 25.0
    lines.append(f"    temp    {temp:.1f}")
    lines.append(f"    pH      {row['pH']:.2f}")

    # pe from Eh: pe = Eh(V) / (2.303 * RT/F)  -- Eh is in mV, convert to V
    eh = row.get("Eh", np.nan)
    if pd.notna(eh):
        T_K = (row.get("Temperature", 25.0) or 25.0) + 273.15
        pe = (eh / 1000.0) / (2.303 * 8.314 * T_K / 96485.0)  # mV → V
        lines.append(f"    pe      {pe:.3f}")
    else:
        lines.append("    pe      4.0")  # default

    # Redox: pe from Eh
    lines.append("    redox   pe")

    # Units: mg/L (PHREEQC default for ppm)
    lines.append("    units   mg/l")

    # Major ions (convert to element concentrations)
    # Note: Cl gets "charge" keyword to force charge balance (standard practice
    # for real-world analyses where analytical charge balance is imperfect)
    ion_map = {
        "Ca": ("Ca", 1.0, False),
        "Mg": ("Mg", 1.0, False),
        "Na": ("Na", 1.0, False),
        "K":  ("K", 1.0, False),
        "Cl": ("Cl", 1.0, True),     # charge balance on Cl
        "SO4": ("S(6)", 1.0 * 32.06 / 96.06, False),   # SO4 mg/L → S mg/L
        "HCO3": ("Alkalinity", 1.0, False),  # as HCO3
        "Fe": ("Fe", 1.0, False),
        "Mn": ("Mn", 1.0, False),
        "NO3": ("N(5)", 1.0 * 14.01 / 62.00, False),   # NO3 mg/L → N mg/L
        "PO4": ("P", 1.0 * 30.97 / 94.97, False),      # PO4 mg/L → P mg/L
    }

    for col, (phreeqc_name, factor, charge_bal) in ion_map.items():
        val = row.get(col, np.nan)
        if pd.notna(val) and val > 0:
            conc = val * factor
            if phreeqc_name == "Alkalinity":
                lines.append(f"    {phreeqc_name}    {conc:.3f} as HCO3")
            elif charge_bal:
                lines.append(f"    {phreeqc_name}    {conc:.4f}    charge")
            else:
                lines.append(f"    {phreeqc_name}    {conc:.4f}")

    # As in µg/L → mg/L
    as_val = row.get("As", np.nan)
    if pd.notna(as_val) and as_val > 0:
        as_mg = as_val / 1000.0  # µg/L → mg/L
        lines.append(f"    As      {as_mg:.6f}")

    # SiO2 if available
    si_val = row.get("SiO2", np.nan)
    if pd.notna(si_val) and si_val > 0:
        si_mg = si_val * 28.09 / 60.08  # SiO2 mg/L → Si mg/L
        lines.append(f"    Si      {si_mg:.4f}")

    lines.append("END")
    return "\n".join(lines)


def run_speciation(df, label=""):
    """Run PHREEQC speciation on all rows in df. Returns DataFrame with SI values."""
    results = []
    n_total = len(df)
    n_success = 0
    n_fail = 0

    # Required columns
    required = ["pH", "Ca", "Na", "Cl", "HCO3", "Fe"]
    df_valid = df.dropna(subset=["pH"]).copy()

    print(f"\n{'='*50}")
    print(f"PHREEQC speciation: {label} ({len(df_valid)}/{n_total} with pH)")
    print(f"{'='*50}")

    for idx, (_, row) in enumerate(df_valid.iterrows()):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df_valid)}...")

        try:
            phreeqc = IPhreeqc()
            phreeqc.load_database(DB_PATH)

            input_str = build_solution_block(row, sol_num=1)

            # SELECTED_OUTPUT must be in same block as SOLUTION (before END)
            # Remove trailing END from solution block, add SELECTED_OUTPUT, then END
            input_str = input_str.rstrip()
            if input_str.endswith("END"):
                input_str = input_str[:-3].rstrip()

            sel_lines = ["\nSELECTED_OUTPUT 1"]
            sel_lines.append("    -reset false")
            for mineral in TARGET_MINERALS:
                sel_lines.append(f"    -saturation_indices {mineral}")
            sel_lines.append("END")

            full_input = input_str + "\n".join(sel_lines)
            phreeqc.run_string(full_input)

            # Parse selected output
            output = phreeqc.get_selected_output_array()

            if len(output) >= 2:
                headers = output[0]
                values = output[1]
                si_dict = {}
                for h, v in zip(headers, values):
                    si_name = h.replace("si_", "SI_")
                    si_dict[si_name] = v if v != -999.999 else np.nan

                # Add metadata
                si_dict["pair_key"] = row.get("pair_key", "")
                si_dict["Period"] = row.get("Period", "")
                si_dict["pH"] = row["pH"]
                si_dict["Eh"] = row.get("Eh", np.nan)
                si_dict["As"] = row.get("As", np.nan)
                si_dict["Fe"] = row.get("Fe", np.nan)
                si_dict["Mn"] = row.get("Mn", np.nan)
                si_dict["PO4"] = row.get("PO4", np.nan)
                si_dict["Depth"] = row.get("Depth", np.nan)

                results.append(si_dict)
                n_success += 1
            else:
                n_fail += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAIL row {idx}: {e}")

    print(f"\n  Success: {n_success}, Failed: {n_fail}")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# 3. RUN SPECIATION ON ALL DATA
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RUNNING PHREEQC SPECIATION ON ALL WELLS")
print("="*60)

# Paired wells
si_old = run_speciation(old, label="Paired 2012-13")
si_new = run_speciation(new, label="Paired 2020-21")

# Full datasets
si_old_full = run_speciation(old_full, label="Full 2012-13")
si_new_full = run_speciation(new_full, label="Full 2020-21 (coastal)")

# Combine and save
si_all = pd.concat([si_old, si_new, si_old_full, si_new_full], ignore_index=True)
si_all.to_csv(TABLE_DIR / "T10_phreeqc_speciation.csv", index=False)
print(f"\nSaved T10_phreeqc_speciation.csv ({len(si_all)} rows)")

# ─────────────────────────────────────────────────────────
# 4. PAIRED SI COMPARISON (Wilcoxon signed-rank)
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PAIRED MINERAL SI COMPARISON")
print("="*60)

# Merge paired SI on pair_key
si_old_pk = si_old.set_index("pair_key")
si_new_pk = si_new.set_index("pair_key")
common_keys = si_old_pk.index.intersection(si_new_pk.index)
print(f"Common paired keys with PHREEQC results: {len(common_keys)}")

si_cols = [c for c in si_old_pk.columns if c.startswith("SI_") or c.startswith("si_")]
# Standardize to SI_ prefix
si_cols_clean = []
for c in si_old_pk.columns:
    if c.startswith("si_") or c.startswith("SI_"):
        si_cols_clean.append(c)

paired_stats = []
for col in si_cols_clean:
    old_vals = si_old_pk.loc[common_keys, col].values.astype(float)
    new_vals = si_new_pk.loc[common_keys, col].values.astype(float)

    # Drop NaN pairs
    mask = np.isfinite(old_vals) & np.isfinite(new_vals)
    o = old_vals[mask]
    n_vals = new_vals[mask]

    if len(o) < 10:
        continue

    delta = n_vals - o
    median_old = np.median(o)
    median_new = np.median(n_vals)
    median_delta = np.median(delta)

    # Wilcoxon signed-rank
    try:
        stat, p = stats.wilcoxon(delta, zero_method="wilcox")
        # Rank-biserial r
        n_d = len(delta[delta != 0])
        r_rb = 1 - (2 * stat) / (n_d * (n_d + 1) / 2) if n_d > 0 else 0
    except Exception:
        stat, p, r_rb = np.nan, np.nan, np.nan

    # Direction interpretation
    mineral_name = col.replace("SI_", "").replace("si_", "")
    if median_delta > 0:
        direction = "more saturated (mineral more stable)"
    elif median_delta < 0:
        direction = "less saturated (mineral less stable)"
    else:
        direction = "no change"

    paired_stats.append({
        "Mineral": mineral_name,
        "n_pairs": int(mask.sum()),
        "SI_old_median": round(median_old, 3),
        "SI_new_median": round(median_new, 3),
        "Delta_SI_median": round(median_delta, 3),
        "Wilcoxon_stat": round(stat, 1) if pd.notna(stat) else np.nan,
        "p_value": p,
        "rank_biserial_r": round(r_rb, 3) if pd.notna(r_rb) else np.nan,
        "Direction": direction,
        "Significant": p < ALPHA if pd.notna(p) else False,
    })

stats_df = pd.DataFrame(paired_stats)

# FDR correction
if len(stats_df) > 0:
    from statsmodels.stats.multitest import multipletests
    pvals = stats_df["p_value"].fillna(1.0).values
    reject, pvals_fdr, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    stats_df["p_FDR"] = pvals_fdr
    stats_df["Significant_FDR"] = reject

stats_df.to_csv(TABLE_DIR / "T10b_si_paired_stats.csv", index=False)

print("\nPAIRED SI CHANGES (Wilcoxon signed-rank, FDR-corrected):")
print("-" * 80)
for _, row in stats_df.iterrows():
    sig = "***" if row.get("p_FDR", 1) < 0.001 else "**" if row.get("p_FDR", 1) < 0.01 else "*" if row.get("p_FDR", 1) < 0.05 else "ns"
    print(f"  {row['Mineral']:20s}  ΔSI={row['Delta_SI_median']:+.3f}  "
          f"p_FDR={row.get('p_FDR', row['p_value']):.4f} {sig}  "
          f"r={row['rank_biserial_r']:+.3f}  n={row['n_pairs']}")

# ─────────────────────────────────────────────────────────
# 5. INVERSE MODELING (batch — representative wells)
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PHREEQC INVERSE MODELING")
print("="*60)

def build_inverse_model(row_old, row_new, sol_nums=(1, 2)):
    """Build PHREEQC input for inverse model between two solutions."""
    lines = []

    # Solution 1 (old)
    lines.append(build_solution_block(row_old, sol_num=sol_nums[0]))

    # Solution 2 (new)
    lines.append(build_solution_block(row_new, sol_num=sol_nums[1]))

    # Inverse modeling block
    lines.append(f"INVERSE_MODELING 1")
    lines.append(f"    -solutions {sol_nums[0]} {sol_nums[1]}")
    lines.append("    -uncertainty 0.10")  # 10% analytical uncertainty
    lines.append("    -range")
    lines.append("    -tolerance 1e-10")
    lines.append("    -mineral_water false")

    # Phases that could dissolve or precipitate
    phases = [
        "Calcite",          # CaCO3 dissolution/precipitation
        "Siderite",         # FeCO3
        "Fe(OH)3(a)",       # Ferrihydrite dissolution
        "Goethite",         # More crystalline Fe-oxide
        "Rhodochrosite",    # MnCO3
        "Vivianite",        # Fe3(PO4)2 — PO4 mineral
        "Gypsum",           # CaSO4
        "Halite",           # NaCl — ion exchange proxy
        "CO2(g)",           # CO2 degassing/dissolution
        "Pyrolusite",       # MnO2 reduction
        "Hydroxyapatite",   # Ca5(PO4)3OH — PO4 source
    ]

    for phase in phases:
        lines.append(f"    -phases {phase}")

    lines.append("    -balances")
    lines.append("        Fe    0.10")
    lines.append("        Mn    0.10")
    lines.append("        As    0.20")  # wider tolerance for trace element
    lines.append("        P     0.15")

    lines.append("END")
    return "\n".join(lines)


def parse_inverse_output(output_str):
    """Parse PHREEQC inverse modeling output for phase transfers."""
    results = []
    in_model = False
    model_num = 0

    for line in output_str.split("\n"):
        if "Beginning of inverse" in line:
            in_model = True
            model_num += 1
            continue
        if in_model and "Phase mance" in line:
            continue
        # Look for phase transfer lines (phase name followed by numbers)
        if in_model:
            parts = line.strip().split()
            if len(parts) >= 2:
                phase = parts[0]
                try:
                    transfer = float(parts[1])
                    results.append({
                        "Model": model_num,
                        "Phase": phase,
                        "Transfer_mol": transfer,
                        "Direction": "dissolve" if transfer > 0 else "precipitate"
                    })
                except (ValueError, IndexError):
                    pass

    return results


# Run inverse models on representative paired wells
# Select wells with complete chemistry in both periods
inv_cols = ["pH", "Eh", "Ca", "Mg", "Na", "K", "Cl", "HCO3", "SO4", "Fe", "Mn", "PO4", "As"]
old_complete = old.dropna(subset=inv_cols)
new_complete = new.dropna(subset=inv_cols)

# Match on pair_key
inv_keys = set(old_complete["pair_key"]) & set(new_complete["pair_key"])
print(f"Wells with complete chemistry for inverse modeling: {len(inv_keys)}")

# Sample up to 50 representative wells (stratified by depth)
inv_old = old_complete[old_complete["pair_key"].isin(inv_keys)].set_index("pair_key")
inv_new = new_complete[new_complete["pair_key"].isin(inv_keys)].set_index("pair_key")

# Stratify: take ~equal from each depth bin
inv_sample_keys = []
for dbin in DEPTH_LABELS:
    keys_in_bin = [k for k in inv_keys if k in inv_old.index and inv_old.loc[k, "Depth_bin"] == dbin]
    n_sample = min(len(keys_in_bin), 20)
    if n_sample > 0:
        sampled = np.random.choice(keys_in_bin, n_sample, replace=False)
        inv_sample_keys.extend(sampled)

print(f"Sampled {len(inv_sample_keys)} wells for inverse modeling")

inverse_results = []
n_success = 0
n_fail = 0

for pk in inv_sample_keys:
    try:
        row_old = inv_old.loc[pk]
        row_new = inv_new.loc[pk]

        # Handle case where pk matches multiple rows
        if isinstance(row_old, pd.DataFrame):
            row_old = row_old.iloc[0]
        if isinstance(row_new, pd.DataFrame):
            row_new = row_new.iloc[0]

        phreeqc = IPhreeqc()
        phreeqc.load_database(DB_PATH)

        input_str = build_inverse_model(row_old, row_new)

        # Add selected output
        sel = """SELECTED_OUTPUT 1
    -reset false
    -inverse_modeling true
END"""
        full_input = input_str + "\n" + sel
        phreeqc.run_string(full_input)

        # Get output string for parsing
        # The inverse model results are in the accumulated output
        # Try to get them from the output array
        output = phreeqc.get_selected_output_array()

        if len(output) >= 2:
            headers = output[0]
            for row_data in output[1:]:
                row_dict = dict(zip(headers, row_data))
                row_dict["pair_key"] = pk
                row_dict["Depth_bin"] = row_old.get("Depth_bin", "")
                inverse_results.append(row_dict)
            n_success += 1
        else:
            n_fail += 1

    except Exception as e:
        n_fail += 1
        if n_fail <= 5:
            print(f"  Inverse model FAIL {pk}: {str(e)[:80]}")

print(f"\nInverse modeling: {n_success} success, {n_fail} failed")

if inverse_results:
    inv_df = pd.DataFrame(inverse_results)
    inv_df.to_csv(TABLE_DIR / "T10c_inverse_models.csv", index=False)
    print(f"Saved T10c_inverse_models.csv ({len(inv_df)} rows)")
else:
    print("No inverse modeling results obtained — saving empty file")
    inv_df = pd.DataFrame()
    inv_df.to_csv(TABLE_DIR / "T10c_inverse_models.csv", index=False)

# ─────────────────────────────────────────────────────────
# 6. FIGURES — 4-PANEL COMPOSITE
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

# ── Panel A: Key mineral SI distributions (paired, both periods) ──
ax1 = fig.add_subplot(gs[0, 0])
key_minerals = ["Fe(OH)3(a)", "Goethite", "Siderite", "Vivianite", "Calcite"]
key_si_cols = [f"si_{m}" for m in key_minerals]
# Check which columns actually exist
existing_cols = []
existing_labels = []
for m, c in zip(key_minerals, key_si_cols):
    alt_c = f"SI_{m}"
    if c in si_old.columns:
        existing_cols.append(c)
        existing_labels.append(m)
    elif alt_c in si_old.columns:
        existing_cols.append(alt_c)
        existing_labels.append(m)

if existing_cols:
    positions = np.arange(len(existing_cols))
    width = 0.35

    old_medians = [si_old[c].median() for c in existing_cols]
    new_medians = [si_new[c].median() for c in existing_cols]
    old_q25 = [si_old[c].quantile(0.25) for c in existing_cols]
    old_q75 = [si_old[c].quantile(0.75) for c in existing_cols]
    new_q25 = [si_new[c].quantile(0.25) for c in existing_cols]
    new_q75 = [si_new[c].quantile(0.75) for c in existing_cols]

    old_err = [np.array([m - q25, q75 - m]) for m, q25, q75 in zip(old_medians, old_q25, old_q75)]
    new_err = [np.array([m - q25, q75 - m]) for m, q25, q75 in zip(new_medians, new_q25, new_q75)]

    ax1.barh(positions - width/2, old_medians, width, color="#2166ac", alpha=0.8,
             xerr=np.array(old_err).T, label="2012-13", capsize=3)
    ax1.barh(positions + width/2, new_medians, width, color="#b2182b", alpha=0.8,
             xerr=np.array(new_err).T, label="2020-21", capsize=3)

    ax1.axvline(0, color="black", lw=1.5, ls="--", alpha=0.7)
    ax1.set_yticks(positions)
    ax1.set_yticklabels(existing_labels, fontsize=8)
    ax1.set_xlabel("Saturation Index (SI)")
    ax1.set_title("(a) Mineral Saturation Indices", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.text(0.02, 0.98, "← Undersaturated | Supersaturated →",
             transform=ax1.transAxes, fontsize=7, va="top", style="italic")
else:
    ax1.text(0.5, 0.5, "No SI data", transform=ax1.transAxes, ha="center")

# ── Panel B: Paired ΔSI for key minerals ──
ax2 = fig.add_subplot(gs[0, 1])
if existing_cols and len(common_keys) > 0:
    delta_data = []
    delta_labels = []
    for c, label in zip(existing_cols, existing_labels):
        o = si_old_pk.loc[common_keys, c].astype(float)
        n_v = si_new_pk.loc[common_keys, c].astype(float)
        mask = np.isfinite(o) & np.isfinite(n_v)
        delta = (n_v[mask] - o[mask]).values
        if len(delta) > 10:
            delta_data.append(delta)
            delta_labels.append(label)

    if delta_data:
        bp = ax2.boxplot(delta_data, vert=True, patch_artist=True, widths=0.6,
                         showfliers=False, medianprops=dict(color="black", lw=2))
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(delta_data)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.7)
        ax2.set_xticklabels(delta_labels, fontsize=8, rotation=30, ha="right")
        ax2.set_ylabel("ΔSI (2020-21 minus 2012-13)")
        ax2.set_title("(b) Paired Change in Saturation Index", fontweight="bold")

        # Add significance stars
        for i, label in enumerate(delta_labels):
            match = stats_df[stats_df["Mineral"] == label]
            if len(match) > 0:
                p_fdr = match["p_FDR"].values[0] if "p_FDR" in match.columns else match["p_value"].values[0]
                if p_fdr < 0.001:
                    sig_text = "***"
                elif p_fdr < 0.01:
                    sig_text = "**"
                elif p_fdr < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"
                ymax = np.percentile(delta_data[i], 95)
                ax2.text(i + 1, ymax + 0.1, sig_text, ha="center", fontsize=9, fontweight="bold")
else:
    ax2.text(0.5, 0.5, "No paired data", transform=ax2.transAxes, ha="center")

# ── Panel C: Fe(OH)3(a) SI vs As concentration ──
ax3 = fig.add_subplot(gs[1, 0])
fh_col = None
for c in si_all.columns:
    if "Fe(OH)3(a)" in c:
        fh_col = c
        break

if fh_col:
    for period, color, marker in [("2012-2013", "#2166ac", "o"), ("2020-2021", "#b2182b", "^")]:
        sub = si_all[si_all["Period"] == period]
        mask = sub[fh_col].notna() & sub["As"].notna() & (sub["As"] > 0)
        x = sub.loc[mask, fh_col].values
        y = np.log10(sub.loc[mask, "As"].values)
        ax3.scatter(x, y, c=color, marker=marker, alpha=0.3, s=15, label=period)

    ax3.axvline(0, color="gray", lw=1.5, ls="--", alpha=0.7)
    ax3.axhline(np.log10(10), color="red", lw=1, ls=":", alpha=0.7)
    ax3.text(ax3.get_xlim()[1] * 0.95, np.log10(10) + 0.1, "WHO 10 µg/L",
             ha="right", fontsize=7, color="red")

    ax3.set_xlabel("Ferrihydrite SI")
    ax3.set_ylabel("log₁₀(As, µg/L)")
    ax3.set_title("(c) Ferrihydrite SI vs Arsenic", fontweight="bold")
    ax3.legend(fontsize=8)

    # Add annotation
    ax3.text(0.02, 0.02, "SI<0: dissolving\nSI>0: stable",
             transform=ax3.transAxes, fontsize=7, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
else:
    ax3.text(0.5, 0.5, "No ferrihydrite SI", transform=ax3.transAxes, ha="center")

# ── Panel D: Summary volcano plot — ΔSI vs significance ──
ax4 = fig.add_subplot(gs[1, 1])
if len(stats_df) > 0:
    x = stats_df["Delta_SI_median"].values
    y = -np.log10(stats_df["p_FDR"].clip(lower=1e-20).values) if "p_FDR" in stats_df.columns else -np.log10(stats_df["p_value"].clip(lower=1e-20).values)
    sig_mask = stats_df["Significant_FDR"].values if "Significant_FDR" in stats_df.columns else stats_df["Significant"].values

    ax4.scatter(x[~sig_mask], y[~sig_mask], c="gray", s=60, alpha=0.6, edgecolors="black", lw=0.5, label="ns")
    ax4.scatter(x[sig_mask], y[sig_mask], c="#b2182b", s=80, alpha=0.8, edgecolors="black", lw=0.5, label="p<0.05 FDR")

    ax4.axhline(-np.log10(0.05), color="gray", ls="--", lw=1, alpha=0.5)
    ax4.axvline(0, color="gray", ls="--", lw=1, alpha=0.5)

    for i, mineral in enumerate(stats_df["Mineral"]):
        ax4.annotate(mineral, (x[i], y[i]), fontsize=6.5, ha="center",
                     va="bottom", xytext=(0, 5), textcoords="offset points")

    ax4.set_xlabel("Median ΔSI (2020-21 minus 2012-13)")
    ax4.set_ylabel("-log₁₀(p_FDR)")
    ax4.set_title("(d) Mineral Stability Change Significance", fontweight="bold")
    ax4.legend(fontsize=8)
else:
    ax4.text(0.5, 0.5, "No stats", transform=ax4.transAxes, ha="center")

plt.savefig(FIGURE_DIR / "F11_phreeqc_speciation.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved F11_phreeqc_speciation.png")

# ─────────────────────────────────────────────────────────
# 7. FINDINGS SUMMARY
# ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("KEY FINDINGS SUMMARY")
print("="*60)

findings_path = Path(__file__).resolve().parents[1] / "output" / "findings"
findings_path.mkdir(parents=True, exist_ok=True)

findings = []
findings.append("A9 — PHREEQC GEOCHEMICAL SPECIATION & INVERSE MODELING")
findings.append("=" * 58)
findings.append("")

findings.append("═" * 50)
findings.append("PART 1: MINERAL SATURATION INDICES (PHREEQC wateq4f.dat)")
findings.append("═" * 50)
findings.append("")
findings.append("Method: Full PHREEQC speciation using IPhreeqc DLL + wateq4f.dat")
findings.append("  Input: pH, pe (from Eh), T, major ions, Fe, Mn, PO4, As")
findings.append(f"  Wells speciated: {len(si_all)} total")
findings.append("")

findings.append("PAIRED SI STATISTICS (Wilcoxon, FDR-corrected):")
findings.append("-" * 70)
for _, row in stats_df.iterrows():
    sig = "***" if row.get("p_FDR", 1) < 0.001 else "**" if row.get("p_FDR", 1) < 0.01 else "*" if row.get("p_FDR", 1) < 0.05 else "ns"
    findings.append(f"  {row['Mineral']:20s}  ΔSI={row['Delta_SI_median']:+.3f}  "
                    f"p_FDR={row.get('p_FDR', row['p_value']):.4f} {sig}  "
                    f"r={row['rank_biserial_r']:+.3f}  n={row['n_pairs']}")

findings.append("")

# Key mineral interpretations
sig_minerals = stats_df[stats_df.get("Significant_FDR", stats_df["Significant"])].sort_values("p_FDR" if "p_FDR" in stats_df.columns else "p_value")
if len(sig_minerals) > 0:
    findings.append(f"SIGNIFICANTLY CHANGED MINERALS ({len(sig_minerals)}/{len(stats_df)} after FDR):")
    for _, row in sig_minerals.iterrows():
        findings.append(f"  • {row['Mineral']}: ΔSI={row['Delta_SI_median']:+.3f} ({row['Direction']})")
else:
    findings.append("NO minerals showed significant SI change after FDR correction.")

findings.append("")
findings.append("═" * 50)
findings.append("PART 2: INVERSE MODELING")
findings.append("═" * 50)
findings.append(f"Wells attempted: {len(inv_sample_keys)}")
findings.append(f"Successful models: {n_success}")
findings.append(f"Failed models: {n_fail}")

if len(inv_df) > 0:
    findings.append(f"Results rows: {len(inv_df)}")
    findings.append("See T10c_inverse_models.csv for phase transfer details")
else:
    findings.append("No successful inverse models — likely needs wider uncertainty bounds")
    findings.append("or the composition change is too complex for simple mineral dissolution/precipitation")

findings.append("")
findings.append("═" * 50)
findings.append("ES&T RELEVANCE")
findings.append("═" * 50)
findings.append("")
findings.append("The PHREEQC speciation provides THERMODYNAMIC validation of the")
findings.append("empirical SI calculations in A8. Key value:")
findings.append("  1. Full aqueous speciation (not just Fe3+ approximation)")
findings.append("  2. Comprehensive mineral SI survey (13 minerals)")
findings.append("  3. Statistical comparison with FDR correction")
findings.append("  4. Inverse modeling attempts to quantify mass transfers")
findings.append("")
findings.append("OUTPUT FILES")
findings.append("------------")
findings.append("T10_phreeqc_speciation.csv")
findings.append("T10b_si_paired_stats.csv")
findings.append("T10c_inverse_models.csv")
findings.append("F11_phreeqc_speciation.png (4-panel composite)")

findings_text = "\n".join(findings)
with open(findings_path / "A9_phreeqc_speciation.txt", "w") as f:
    f.write(findings_text)

print(findings_text)
print(f"\nSaved findings to {findings_path / 'A9_phreeqc_speciation.txt'}")
print("\n✓ A9 COMPLETE")
