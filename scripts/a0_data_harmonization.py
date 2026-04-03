"""
A0 — Data Harmonization & Well Matching
========================================
Loads both datasets (2012-13 and 2020-21), harmonizes column names and units,
matches wells by Sample ID, and produces:
  1. matched_wells.csv     — 115 paired wells (long format: one row per well × period)
  2. old_harmonized.csv    — full old dataset with standardized columns
  3. new_harmonized.csv    — full new dataset with standardized columns
  4. district_summary.csv  — district-level aggregated statistics for both periods
  5. QC report printed to stdout
  6. Charge balance error (CBE) calculation and filtering
"""

import pandas as pd
import numpy as np
import re
from config import (
    OLD_DATA, PAPER1_DATA, TABLE_DIR, OLD_COL_MAP, NEW_COL_MAP,
    TEMPORAL_PARAMS, DEPTH_BINS, DEPTH_LABELS
)

# ─────────────────────────────────────────────────────────
# CHARGE BALANCE ERROR (CBE) CALCULATION
# ─────────────────────────────────────────────────────────
# CBE (%) = [(Σ cations - Σ anions) / (Σ cations + Σ anions)] × 100
# All concentrations converted from mg/L to meq/L

# Molecular weights (g/mol) and valence for meq/L conversion
# meq/L = (mg/L × valence) / molecular_weight
CATION_CONV = {
    "Ca":  (2, 40.078),
    "Mg":  (2, 24.305),
    "Na":  (1, 22.990),
    "K":   (1, 39.098),
    "Fe":  (2, 55.845),
    "Mn":  (2, 54.938),
}
ANION_CONV = {
    "HCO3": (1, 61.017),
    "CO3":  (2, 60.009),
    "Cl":   (1, 35.453),
    "SO4":  (2, 96.06),
    "NO3":  (1, 62.004),
    "PO4":  (3, 94.971),
}
CBE_THRESHOLD = 10.0  # ±10% threshold for acceptable CBE


def calc_cbe(df):
    """Calculate charge balance error (%) for each sample.

    Returns a Series of CBE values. NaN where insufficient ion data.
    Requires at least 3 cations and 3 anions to be present.
    """
    sum_cat = pd.Series(0.0, index=df.index)
    n_cat = pd.Series(0, index=df.index)
    for ion, (valence, mw) in CATION_CONV.items():
        if ion in df.columns:
            vals = pd.to_numeric(df[ion], errors="coerce").fillna(0).clip(lower=0)
            meq = vals * valence / mw
            sum_cat += meq
            n_cat += (vals > 0).astype(int)

    sum_an = pd.Series(0.0, index=df.index)
    n_an = pd.Series(0, index=df.index)
    for ion, (valence, mw) in ANION_CONV.items():
        if ion in df.columns:
            vals = pd.to_numeric(df[ion], errors="coerce").fillna(0).clip(lower=0)
            meq = vals * valence / mw
            sum_an += meq
            n_an += (vals > 0).astype(int)

    total = sum_cat + sum_an
    cbe = np.where(
        (total > 0) & (n_cat >= 3) & (n_an >= 3),
        (sum_cat - sum_an) / total * 100.0,
        np.nan,
    )
    return pd.Series(cbe, index=df.index)

# ─────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────

print("=" * 70)
print("A0 — DATA HARMONIZATION & WELL MATCHING")
print("=" * 70)

old_raw = pd.read_csv(OLD_DATA)
new_raw = pd.read_csv(PAPER1_DATA)

print(f"\nOld dataset (2012-13): {old_raw.shape}")
print(f"New dataset (2020-21): {new_raw.shape}")

# ─────────────────────────────────────────────────────────
# 2. FILTER: GROUNDWATER ONLY (old dataset)
# ─────────────────────────────────────────────────────────

old_gw = old_raw[old_raw["Well_Type"] != "Surface Water"].copy()
print(f"\nOld GW only: {len(old_gw)} samples (removed {len(old_raw) - len(old_gw)} surface water)")

# ─────────────────────────────────────────────────────────
# 3. HARMONIZE COLUMN NAMES
# ─────────────────────────────────────────────────────────

def harmonize(df, col_map, period_label):
    """Rename columns using mapping, add period label."""
    # Only keep columns that exist in the mapping
    available = {k: v for k, v in col_map.items() if k in df.columns}
    renamed = df.rename(columns=available)

    # Keep only standardized columns
    std_cols = list(available.values())
    extra_cols = [c for c in renamed.columns if c not in std_cols]
    renamed = renamed[std_cols].copy()

    renamed["Period"] = period_label
    return renamed

old_h = harmonize(old_gw, OLD_COL_MAP, "2012-2013")
new_h = harmonize(new_raw, NEW_COL_MAP, "2020-2021")

print(f"\nHarmonized old: {old_h.shape} ({len(old_h.columns)} cols)")
print(f"Harmonized new: {new_h.shape} ({len(new_h.columns)} cols)")

# Check common columns
common_cols = set(old_h.columns) & set(new_h.columns)
old_only = set(old_h.columns) - set(new_h.columns)
new_only = set(new_h.columns) - set(old_h.columns)
print(f"\nCommon columns: {len(common_cols)}")
print(f"Old-only columns: {old_only}")
print(f"New-only columns: {new_only}")

# ─────────────────────────────────────────────────────────
# 4. UNIT HARMONIZATION
# ─────────────────────────────────────────────────────────

# As: old is already in µg/L, new is in µg/L — confirmed from data inspection
# Mn, Fe, PO4: both in mg/L — confirmed
# Eh/ORP: both in mV — note ORP ≈ Eh for practical purposes
# EC: both in µS/cm
# All other parameters: same units in both datasets

# Standardize district names across campaigns
DISTRICT_NAME_MAP = {
    "Barisal": "Barishal",
    "Jessore": "Jashore",
    "Cox's Bazar": "Cox'S Bazar",
    "Cox's bazar": "Cox'S Bazar",
    "Shariyatpur": "Shariatpur",
    "Shathkhira": "Satkhira",
    "Chittagong": "Chattogram",
    "Sariatpur": "Shariatpur",
    "Meherpur.": "Meherpur",
}
for df in [old_h, new_h]:
    df["District"] = df["District"].str.strip().replace(DISTRICT_NAME_MAP)

# Convert numeric columns
numeric_cols = ["As", "Mn", "Fe", "PO4", "pH", "Eh", "EC", "TDS",
                "Temperature", "Depth", "Latitude", "Longitude",
                "Ca", "Mg", "Na", "K", "Cl", "HCO3", "SO4", "NO3",
                "CO3", "Salinity"]

for col in numeric_cols:
    if col in old_h.columns:
        old_h[col] = pd.to_numeric(old_h[col], errors="coerce")
    if col in new_h.columns:
        new_h[col] = pd.to_numeric(new_h[col], errors="coerce")

# ─────────────────────────────────────────────────────────
# 4b. CHARGE BALANCE ERROR (CBE) CALCULATION & FILTERING
# ─────────────────────────────────────────────────────────

old_h["CBE"] = calc_cbe(old_h)
new_h["CBE"] = calc_cbe(new_h)

print(f"\n{'=' * 50}")
print(f"CHARGE BALANCE ERROR (CBE)")
print(f"{'=' * 50}")
for label, df in [("Old (2012-13)", old_h), ("New (2020-21)", new_h)]:
    cbe_valid = df["CBE"].dropna()
    n_valid = len(cbe_valid)
    n_pass = (cbe_valid.abs() <= CBE_THRESHOLD).sum()
    n_fail = (cbe_valid.abs() > CBE_THRESHOLD).sum()
    n_missing = df["CBE"].isna().sum()
    print(f"\n  {label}: {n_valid} samples with CBE computed, {n_missing} insufficient ions")
    if n_valid > 0:
        print(f"    CBE range: [{cbe_valid.min():.1f}%, {cbe_valid.max():.1f}%]")
        print(f"    CBE mean ± std: {cbe_valid.mean():.1f}% ± {cbe_valid.std():.1f}%")
        print(f"    Within ±{CBE_THRESHOLD}%: {n_pass} ({n_pass/n_valid*100:.1f}%)")
        print(f"    Outside ±{CBE_THRESHOLD}%: {n_fail} ({n_fail/n_valid*100:.1f}%) — FLAGGED")

# Flag samples (keep all but add CBE_flag column for downstream filtering)
old_h["CBE_flag"] = np.where(old_h["CBE"].abs() > CBE_THRESHOLD, "FAIL", "PASS")
old_h.loc[old_h["CBE"].isna(), "CBE_flag"] = "NO_DATA"
new_h["CBE_flag"] = np.where(new_h["CBE"].abs() > CBE_THRESHOLD, "FAIL", "PASS")
new_h.loc[new_h["CBE"].isna(), "CBE_flag"] = "NO_DATA"

# ─────────────────────────────────────────────────────────
# 5. SAMPLE ID NORMALIZATION & WELL MATCHING
# ─────────────────────────────────────────────────────────

def normalize_id(sid):
    """Normalize Sample ID for matching across datasets.

    Handles variations like:
      BABAPZ 1 vs BABAPZ-1/1
      BNAMPZ_1 vs BNAMPZ-1
    """
    s = str(sid).strip().upper()
    # Remove trailing /1, /2 etc. (sub-sample indicators in new dataset)
    s = re.sub(r"/\d+$", "", s)
    # Normalize separators: spaces, underscores → hyphens
    s = re.sub(r"[\s_]+", "-", s)
    # Remove trailing whitespace
    s = s.strip()
    return s

old_h["ID_norm"] = old_h["Sample_ID"].apply(normalize_id)
new_h["ID_norm"] = new_h["Sample_ID"].apply(normalize_id)

# Find overlapping normalized IDs
old_ids = set(old_h["ID_norm"].unique())
new_ids = set(new_h["ID_norm"].unique())
matched_ids = old_ids & new_ids

print(f"\n{'=' * 50}")
print(f"WELL MATCHING RESULTS")
print(f"{'=' * 50}")
print(f"Old unique IDs (normalized): {len(old_ids)}")
print(f"New unique IDs (normalized): {len(new_ids)}")
print(f"Matched IDs: {len(matched_ids)}")
print(f"Examples: {sorted(list(matched_ids))[:10]}")

# ─────────────────────────────────────────────────────────
# 6. BUILD PAIRED DATASET (depth-aware matching)
# ─────────────────────────────────────────────────────────

# Strategy: match on (ID_norm, Depth_bin) to avoid averaging across
# different depth zones in nested piezometers. Within a matching
# (ID_norm, Depth_bin) group, seasonal replicates are averaged (correct).

pair_cols = sorted(list(common_cols - {"Period", "Date", "Season", "Well_Type"}))

def aggregate_matched(df, matched_ids_set):
    """For matched wells, aggregate by (ID_norm, Depth_bin).

    This preserves depth information: nested piezometers at different
    depths are treated as separate wells. Only seasonal replicates
    (same well, same depth, different season) are averaged.
    """
    matched = df[df["ID_norm"].isin(matched_ids_set)].copy()

    num_cols = [c for c in pair_cols if c in df.columns and df[c].dtype in ["float64", "int64"]]
    cat_cols = ["District", "Upazila", "Latitude", "Longitude"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    agg_dict = {c: "mean" for c in num_cols if c not in cat_cols}
    for c in cat_cols:
        if c in ["Latitude", "Longitude"]:
            agg_dict[c] = "mean"
        else:
            agg_dict[c] = "first"

    # Group by both ID and depth bin to preserve depth structure
    grouped = matched.groupby(["ID_norm", "Depth_bin"]).agg(agg_dict).reset_index()
    grouped["Period"] = df["Period"].iloc[0]

    return grouped

# Depth-bin the harmonized data BEFORE matching (needed for depth-aware pairing)
for df in [old_h, new_h]:
    if "Depth" in df.columns:
        df["Depth_bin"] = pd.cut(
            df["Depth"], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=True
        )

old_matched = aggregate_matched(old_h, matched_ids)
new_matched = aggregate_matched(new_h, matched_ids)

print(f"\nOld matched (aggregated by ID+depth): {len(old_matched)} well-depth units")
print(f"New matched (aggregated by ID+depth): {len(new_matched)} well-depth units")

# Match on (ID_norm, Depth_bin) — explicit merge instead of positional alignment
old_matched["pair_key"] = old_matched["ID_norm"] + "_" + old_matched["Depth_bin"].astype(str)
new_matched["pair_key"] = new_matched["ID_norm"] + "_" + new_matched["Depth_bin"].astype(str)

common_keys = set(old_matched["pair_key"]) & set(new_matched["pair_key"])
old_matched = old_matched[old_matched["pair_key"].isin(common_keys)].sort_values("pair_key").reset_index(drop=True)
new_matched = new_matched[new_matched["pair_key"].isin(common_keys)].sort_values("pair_key").reset_index(drop=True)

print(f"Final paired well-depth units: {len(old_matched)}")

# Verify alignment
assert (old_matched["pair_key"].values == new_matched["pair_key"].values).all(), \
    "Pair keys must match after alignment"

# Combine into long format
paired = pd.concat([old_matched, new_matched], ignore_index=True)
paired = paired.sort_values(["pair_key", "Period"]).reset_index(drop=True)

# Depth-bin the paired data (already present from aggregation)
if "Depth" in paired.columns and "Depth_bin" not in paired.columns:
    paired["Depth_bin"] = pd.cut(
        paired["Depth"], bins=DEPTH_BINS, labels=DEPTH_LABELS, right=True
    )

# Recompute CBE on aggregated paired data (means of ions are valid for CBE)
paired["CBE"] = calc_cbe(paired)
paired["CBE_flag"] = np.where(paired["CBE"].abs() > CBE_THRESHOLD, "FAIL", "PASS")
paired.loc[paired["CBE"].isna(), "CBE_flag"] = "NO_DATA"

# ─────────────────────────────────────────────────────────
# 8. DISTRICT-LEVEL SUMMARY
# ─────────────────────────────────────────────────────────

# Normalize district names
old_h["District_norm"] = old_h["District"].str.strip().str.title()
new_h["District_norm"] = new_h["District"].str.strip().str.title()

old_districts = set(old_h["District_norm"].dropna().unique())
new_districts = set(new_h["District_norm"].dropna().unique())
common_districts = old_districts & new_districts

print(f"\n{'=' * 50}")
print(f"DISTRICT MATCHING")
print(f"{'=' * 50}")
print(f"Old districts: {len(old_districts)}")
print(f"New districts: {len(new_districts)}")
print(f"Common districts: {len(common_districts)}")
print(f"Names: {sorted(common_districts)}")

# Build district summary for common districts
summary_rows = []
for district in sorted(common_districts):
    for period, df in [("2012-2013", old_h), ("2020-2021", new_h)]:
        ddf = df[df["District_norm"] == district]
        row = {
            "District": district,
            "Period": period,
            "n_samples": len(ddf),
        }
        for param in TEMPORAL_PARAMS:
            if param in ddf.columns:
                vals = ddf[param].dropna()
                row[f"{param}_mean"] = vals.mean() if len(vals) > 0 else np.nan
                row[f"{param}_median"] = vals.median() if len(vals) > 0 else np.nan
                row[f"{param}_std"] = vals.std() if len(vals) > 0 else np.nan
                row[f"{param}_n"] = len(vals)
        summary_rows.append(row)

district_summary = pd.DataFrame(summary_rows)

# ─────────────────────────────────────────────────────────
# 9. QC REPORT
# ─────────────────────────────────────────────────────────

print(f"\n{'=' * 50}")
print(f"QC REPORT")
print(f"{'=' * 50}")

print("\n--- Paired wells: parameter availability ---")
for param in TEMPORAL_PARAMS:
    if param in old_matched.columns and param in new_matched.columns:
        old_n = old_matched[param].notna().sum()
        new_n = new_matched[param].notna().sum()
        both_n = (old_matched[param].notna() & new_matched[param].notna()).sum()
        print(f"  {param:>5}: old={old_n}, new={new_n}, both_available={both_n}")

print("\n--- As measurement caveat ---")
print("  Old: field kit (ranges → midpoints, ±5 µg/L precision)")
print("  New: ICP-MS (precise, ±0.1 µg/L)")
print("  NOTE: Direct As comparison has systematic uncertainty from method difference.")
print("  Mn, Fe, PO4 measured by same lab methods in both campaigns — more reliable.")

print("\n--- Charge balance error (paired wells) ---")
if "CBE" in paired.columns:
    cbe_paired = paired["CBE"].dropna()
    n_cbe = len(cbe_paired)
    n_pass = (cbe_paired.abs() <= CBE_THRESHOLD).sum()
    n_fail = (cbe_paired.abs() > CBE_THRESHOLD).sum()
    print(f"  Paired wells with CBE: {n_cbe}")
    print(f"  Within ±{CBE_THRESHOLD}%: {n_pass} ({n_pass/n_cbe*100:.1f}%)")
    print(f"  Outside ±{CBE_THRESHOLD}%: {n_fail} ({n_fail/n_cbe*100:.1f}%) — flagged")

print("\n--- Outlier check (paired wells) ---")
for param in ["As", "Mn", "Fe", "PO4"]:
    if param in paired.columns:
        vals = paired[param].dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)).sum()
        print(f"  {param}: {n_outliers} extreme outliers (>3×IQR)")

# ─────────────────────────────────────────────────────────
# 10. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────

old_h.to_csv(TABLE_DIR / "old_harmonized.csv", index=False)
new_h.to_csv(TABLE_DIR / "new_harmonized.csv", index=False)
paired.to_csv(TABLE_DIR / "matched_wells.csv", index=False)
district_summary.to_csv(TABLE_DIR / "district_summary.csv", index=False)

print(f"\n{'=' * 50}")
print(f"OUTPUTS SAVED")
print(f"{'=' * 50}")
print(f"  {TABLE_DIR / 'old_harmonized.csv'} ({len(old_h)} rows)")
print(f"  {TABLE_DIR / 'new_harmonized.csv'} ({len(new_h)} rows)")
print(f"  {TABLE_DIR / 'matched_wells.csv'} ({len(paired)} rows, {len(common_keys)} pairs × 2 periods)")
print(f"  {TABLE_DIR / 'district_summary.csv'} ({len(district_summary)} rows)")

# Final summary table
print(f"\n{'=' * 50}")
print(f"PRELIMINARY PAIRED COMPARISON (n={len(old_matched)} wells)")
print(f"{'=' * 50}")
print(f"{'Param':>8} {'Old mean':>12} {'New mean':>12} {'Change':>10} {'% Change':>10}")
print(f"{'-' * 54}")
for param in TEMPORAL_PARAMS:
    if param in old_matched.columns and param in new_matched.columns:
        old_mean = old_matched[param].mean()
        new_mean = new_matched[param].mean()
        if pd.notna(old_mean) and pd.notna(new_mean):
            change = new_mean - old_mean
            pct = (change / abs(old_mean) * 100) if old_mean != 0 else np.nan
            print(f"{param:>8} {old_mean:>12.3f} {new_mean:>12.3f} {change:>+10.3f} {pct:>+9.1f}%")

print("\nDone.")
