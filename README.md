# Arsenic Mobilization Mechanism Shift in Bangladesh Groundwater

Analysis scripts for the paper:

**Decadal Geochemical Evolution Reveals Imminent Arsenic Mobilization Threshold in Bangladesh Groundwater**

Md Rakib Hasan, Mst Anika Khatun Rupa, Anwar Zahid

*Submitted to Environmental Science & Technology, 2026*

## Dataset

The 2020-2021 groundwater quality dataset is available at Zenodo:
[DOI: 10.5281/zenodo.19148957](https://doi.org/10.5281/zenodo.19148957)

## Scripts

| Script | Description |
|--------|-------------|
| `config.py` | Shared configuration and paths |
| `a0_data_harmonization.py` | Data cleaning, pairing, and harmonization |
| `a1_paired_temporal_stats.py` | Paired temporal statistics (Wilcoxon, effect sizes) |
| `a2_spatial_hotspot_shift.py` | Spatial hotspot shift analysis (Moran's I, LISA) |
| `a3_redox_evolution.py` | Redox evolution analysis (Eh, Fe/Mn ratios) |
| `a4_depth_stratified.py` | Depth-stratified geochemical analysis |
| `a5_advanced_methods.py` | SEM, Bayesian estimation, mediation analysis |
| `a6_sem_enhancements.py` | SEM robustness and enhancement tests |
| `a7_satellite_landuse_linkage.py` | Satellite/land-use linkage analysis |
| `a8_tipping_point.py` | Tipping point and threshold analysis |
| `a9_phreeqc_speciation.py` | PHREEQC mineral speciation modeling |
| `a10_sensitivity.py` | Sensitivity analysis under filtering scenarios |
| `make_study_area_map.py` | Study area map generation |
| `make_toc_graphic.py` | TOC/graphical abstract generation |
| `regenerate_main_figures.py` | Regenerate all manuscript figures |

## Requirements

- Python 3.11+
- numpy, pandas, scipy, matplotlib, seaborn
- scikit-learn, statsmodels, semopy
- geopandas, pysal, phreeqpy

## License

Apache 2.0
