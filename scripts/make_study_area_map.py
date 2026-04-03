#!/usr/bin/env python3
"""Generate professional study area map (Figure 1) for ES&T submission.
Features: ADM1+ADM2 boundaries, As choropleth, wells, rivers, inset map.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
})

# --- Paths ---
BASE = '/Users/rakibhhridoy/AsGW/GroundWater'
P1_SHP = f'{BASE}/Paper1/data/raw/shapefiles'
P3 = f'{BASE}/Paper3'

# --- Load shapefiles ---
bd_national = gpd.read_file(f'{P1_SHP}/bangladesh.geojson')
bd_divisions = gpd.read_file(f'{P1_SHP}/bgd_divisions.geojson')
bd_districts = gpd.read_file(f'{P3}/data/bgd_districts.geojson')
bd_rivers = gpd.read_file(f'{P1_SHP}/bgd_hydrorivers.geojson')

# --- Identify study districts ---
study_names_gadm = [
    'Bagerhat', 'Barguna', 'Barisal', 'Bhola', 'Chandpur', 'Chittagong',
    'Feni', 'Gopalganj', 'Jessore', 'Jhalokati', 'Khulna', 'Lakshmipur',
    'Narail', 'Noakhali', 'Patuakhali', 'Pirojpur', 'Satkhira', 'Shariatpur'
]
name_map_to_data = {
    'Jhalokati': 'Jhalokathi', 'Lakshmipur': 'Laksmipur',
    'Barisal': 'Barishal', 'Jessore': 'Jashore', 'Chittagong': 'Chattogram',
}

study_districts = bd_districts[bd_districts['NAME_2'].isin(study_names_gadm)].copy()
non_study_districts = bd_districts[~bd_districts['NAME_2'].isin(study_names_gadm)].copy()

# --- Compute As exceedance per district for choropleth ---
df_exc = pd.read_csv(f'{P3}/analysis/output/tables/T02b_district_exceedance.csv')
as_new = df_exc[df_exc['Parameter'] == 'As'][['District', 'New_rate_pct']].copy()
as_new.columns = ['District', 'As_exceed_pct']

study_districts['District'] = study_districts['NAME_2'].map(
    lambda x: name_map_to_data.get(x, x))
study_districts = study_districts.merge(as_new, on='District', how='left')
study_districts['As_exceed_pct'] = study_districts['As_exceed_pct'].fillna(0)

# --- Well locations ---
df_wells = pd.read_csv(f'{P3}/data/water_quality_2012_2013.csv')
our_districts = list(name_map_to_data.values()) + [
    n for n in study_names_gadm if n not in name_map_to_data] + [
    'Shariyatpur', 'Barisal', 'Jessore', 'Chittagong']
df_wells_study = df_wells[df_wells['District'].isin(our_districts)].copy()
df_wells_study['Latitude'] = pd.to_numeric(df_wells_study['Latitude'], errors='coerce')
df_wells_study['Longitude'] = pd.to_numeric(df_wells_study['Longitude'], errors='coerce')
df_wells_study = df_wells_study.dropna(subset=['Latitude', 'Longitude'])

# --- View bounds (study area with padding) ---
sb = study_districts.total_bounds  # [minx, miny, maxx, maxy]
pad = 0.25
xmin, ymin, xmax, ymax = sb[0]-pad, sb[1]-pad, sb[2]+pad, sb[3]+pad

# Clip rivers to view
view_box = gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=bd_rivers.crs)
if 'ORD_STRA' in bd_rivers.columns:
    major_rivers = bd_rivers[bd_rivers['ORD_STRA'] <= 5]
elif 'RIV_ORD' in bd_rivers.columns:
    major_rivers = bd_rivers[bd_rivers['RIV_ORD'] <= 5]
else:
    major_rivers = bd_rivers
rivers_clipped = gpd.clip(major_rivers, view_box)

# --- Create figure ---
# Full-width layout: map takes full width, colorbar horizontal at bottom
fig = plt.figure(figsize=(7, 6.2), dpi=300)

# Main map: full width
ax = fig.add_axes([0.02, 0.12, 0.96, 0.86])

# 1. Non-study districts (light gray fill)
non_study_districts.plot(ax=ax, facecolor='#EDEDED', edgecolor='#BBBBBB',
                          linewidth=0.3, zorder=1)

# 2. Choropleth: study districts colored by As exceedance
bounds = [0, 15, 30, 40, 50, 60, 75]
cmap = plt.cm.YlOrRd
norm = BoundaryNorm(bounds, cmap.N)

study_districts.plot(ax=ax, column='As_exceed_pct', cmap=cmap, norm=norm,
                      edgecolor='#222222', linewidth=0.9, zorder=3, legend=False)

# 3. Rivers
rivers_clipped.plot(ax=ax, color='#6BAED6', linewidth=0.5, alpha=0.7, zorder=2)

# 4. Division boundaries (thick overlay)
bd_divisions.boundary.plot(ax=ax, color='#444444', linewidth=1.4, zorder=4)

# 5. Well locations on top
ax.scatter(df_wells_study['Longitude'], df_wells_study['Latitude'],
           s=35, c='black', alpha=0.9, edgecolors='#00BCD4', linewidths=1.2,
           zorder=6, label='Sampling wells')

# 6. District labels
for idx, row in study_districts.iterrows():
    cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
    name = row['NAME_2']
    ax.text(cx, cy, name, fontsize=5, ha='center', va='center',
            fontweight='bold', color='#1a1a1a', zorder=8,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      alpha=0.75, edgecolor='none', linewidth=0))

# 7. Set extent
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Remove axis labels, keep tick values only
ax.set_xlabel('')
ax.set_ylabel('')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
ax.set_xticks(np.arange(np.ceil(xmin * 2) / 2, np.floor(xmax * 2) / 2 + 0.5, 0.5))
ax.set_yticks(np.arange(np.ceil(ymin * 2) / 2, np.floor(ymax * 2) / 2 + 0.5, 0.5))
ax.tick_params(labelsize=7, length=3)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_facecolor('white')
# Bay of Bengal text removed

# North arrow (bottom left)
ax.annotate('N', xy=(xmin + 0.12, ymin + 0.65), fontsize=10, fontweight='bold',
            ha='center', va='bottom', zorder=10)
ax.annotate('', xy=(xmin + 0.12, ymin + 0.63),
            xytext=(xmin + 0.12, ymin + 0.23),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.8), zorder=10)

# Scale bar (bottom center)
sb_x = (xmin + xmax) / 2 - 0.45 / 2
sb_y = ymin + 0.20
sb_len_km = 50
sb_len_deg = sb_len_km / 111.0
ax.plot([sb_x, sb_x + sb_len_deg], [sb_y, sb_y], 'k-', lw=2.5, zorder=10)
ax.plot([sb_x, sb_x], [sb_y - 0.03, sb_y + 0.03], 'k-', lw=1.5, zorder=10)
ax.plot([sb_x + sb_len_deg, sb_x + sb_len_deg], [sb_y - 0.03, sb_y + 0.03],
        'k-', lw=1.5, zorder=10)
ax.text(sb_x + sb_len_deg / 2, sb_y + 0.06, '50 km', fontsize=6,
        ha='center', va='bottom', zorder=10)

# --- Horizontal colorbar at bottom center (50% width) ---
cax = fig.add_axes([0.25, 0.02, 0.50, 0.018])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, ticks=bounds, orientation='horizontal')
cbar.set_label('As exceedance rate (% wells > 10 \u00b5g/L, 2020\u201321)',
               fontsize=7, labelpad=4)
cbar.ax.tick_params(labelsize=6)

# --- Inset map (top left) ---
ax_inset = fig.add_axes([0.80, 0.69, 0.22, 0.22])
bd_national.plot(ax=ax_inset, facecolor='#F0F0F0', edgecolor='#333', linewidth=0.8)

# Highlight study area
rect = Rectangle((sb[0] - 0.1, sb[1] - 0.1), sb[2] - sb[0] + 0.2, sb[3] - sb[1] + 0.2,
                  linewidth=1.8, edgecolor='red', facecolor='red', alpha=0.15, zorder=5)
ax_inset.add_patch(rect)

ax_inset.set_xlim(87.8, 92.8)
ax_inset.set_ylim(20.5, 26.8)
ax_inset.set_title('Bangladesh', fontsize=7, fontweight='bold', pad=3)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
# Bay of Bengal text removed from inset
ax_inset.set_facecolor('white')
for spine in ax_inset.spines.values():
    spine.set_visible(False)

# --- Legend (below inset, upper left area) ---
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
           markeredgecolor='#00838F', markeredgewidth=0.6, markersize=5,
           label=f'Sampling wells (n={len(df_wells_study)})'),
    Line2D([0], [0], color='#6BAED6', linewidth=1.2, label='Rivers'),
    Line2D([0], [0], color='#444444', linewidth=1.4, label='Division (ADM1)'),
    Line2D([0], [0], color='#222222', linewidth=0.9, label='District (ADM2)'),
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=6,
          framealpha=0.92, edgecolor='#BBB', fancybox=False, borderpad=0.6,
          bbox_to_anchor=(0.5, -0.16), ncol=4)

# No title (caption handles it)

# Save
outpath = f'{P3}/analysis/output/figures/Fig1_study_area.png'
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {outpath}")
plt.close()
