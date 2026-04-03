"""
Paper3 — Shared configuration for all analysis scripts.
Paths, constants, and style settings.
"""
import os
from pathlib import Path

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Paper3/
DATA_DIR = PROJECT_ROOT / "data"
PAPER1_DATA = Path("/Users/rakibhhridoy/AsGW/GroundWater/Paper1/hypo/hypo1/output/data_phase1_integrated.csv")
OLD_DATA = DATA_DIR / "water_quality_2012_2013.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

for d in [TABLE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- DEPTH BINS ---
DEPTH_BINS = [0, 50, 150, 500]
DEPTH_LABELS = ["Shallow (<50m)", "Medium (50–150m)", "Deep (>150m)"]

# --- WHO / BANGLADESH STANDARDS ---
WHO_LIMITS = {
    "As": 10.0,      # µg/L
    "Mn": 0.4,       # mg/L (WHO health-based value)
    "Fe": 0.3,       # mg/L (aesthetic, not health)
}

BD_LIMITS = {
    "As": 50.0,      # µg/L (Bangladesh standard)
    "Mn": 0.1,       # mg/L (Bangladesh standard)
    "Fe": 1.0,       # mg/L
}

# --- COLUMN NAME MAPPING (harmonize old → new) ---
# Old dataset column → standardized name
OLD_COL_MAP = {
    "As (µg/l)": "As",          # µg/L
    "Mn (mg/l)": "Mn",          # mg/L
    "Fe (mg/l)": "Fe",          # mg/L
    "PO₄²‾ (mg/l)": "PO4",     # mg/L
    "pH": "pH",
    "Eh (mV)": "Eh",
    "EC (µS/cm)": "EC",
    "TDS (mg/l)": "TDS",
    "Temperature (°C)": "Temperature",
    "Salinity (ppt)": "Salinity",
    "Depth (m)": "Depth",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "District": "District",
    "Upazila": "Upazila",
    "Sample ID": "Sample_ID",
    "Date of Sampling": "Date",
    "Season": "Season",
    "Well_Type": "Well_Type",
    "Ca2+ (mg/l)": "Ca",
    "Mg2+ (mg/l)": "Mg",
    "Na+ (mg/l)": "Na",
    "K+ (mg/l)": "K",
    "Cl- (mg/l)": "Cl",
    "HCO3-(mg/l)": "HCO3",
    "SO₄2-(mg/l)": "SO4",
    "NO₃- (mg/l)": "NO3",
    "CO₃²⁻ (mg/l)": "CO3",
    "F- (mg/l)": "F",
    "B (mg/l)": "B",
    "I (mg/l)": "I",
    "Br (mg/l)": "Br",
    "SiO₂ (mg/l)": "SiO2",
    "CO₂ (mg/l)": "CO2",
}

# New dataset column → standardized name
NEW_COL_MAP = {
    "As": "As",                  # µg/L
    "Mn2+": "Mn",               # mg/L
    "Fe2+": "Fe",               # mg/L
    "PO43-": "PO4",             # mg/L
    "pH": "pH",
    "ORP": "Eh",                # mV (ORP ≈ Eh)
    "EC": "EC",
    "TDS": "TDS",
    "Temperature": "Temperature",
    "Salinity": "Salinity",
    "Depth": "Depth",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "District": "District",
    "Upazila": "Upazila",
    "Sample ID": "Sample_ID",
    "Date": "Date",
    "Season": "Season",
    "Ca2+": "Ca",
    "Mg2+": "Mg",
    "Na+": "Na",
    "K+": "K",
    "Cl-": "Cl",
    "HCO3-": "HCO3",
    "SO42-": "SO4",
    "NO3-": "NO3",
    "CO3-": "CO3",
    "Br-": "Br",
    "I-": "I",
    "B+": "B",
    "DO": "DO",
    "GW": "GW_level",
}

# Parameters for temporal comparison
TEMPORAL_PARAMS = ["As", "Mn", "Fe", "PO4", "pH", "Eh", "EC", "TDS"]
KEY_CONTAMINANTS = ["As", "Mn", "Fe", "PO4"]

# --- STATISTICAL SETTINGS ---
ALPHA = 0.05
N_BOOTSTRAP = 5000
RANDOM_SEED = 42

# --- FIGURE STYLE ---
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

def set_est_style():
    """Set matplotlib style for ES&T publication figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": FIGURE_DPI,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
