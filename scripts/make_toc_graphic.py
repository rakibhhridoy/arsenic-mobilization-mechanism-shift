#!/usr/bin/env python3
"""Generate TOC/Abstract graphic for ES&T submission.
ES&T requires 3.25 x 1.75 inches at 300 DPI.
Visual summary: mechanism shift from reductive dissolution to competitive desorption.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Setup
fig, ax = plt.subplots(1, 1, figsize=(3.25, 1.75), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.axis('off')

# Colors
OLD_COLOR = '#D35F5F'
NEW_COLOR = '#4A90D9'
GOLD = '#E8A838'
GRAY = '#888888'

# Title
ax.text(5, 5.2, 'Mechanism Shift in Bangladesh Groundwater', fontsize=6,
        fontweight='bold', ha='center', va='top', fontfamily='Arial')

# --- Left panel: 2012-2013 ---
ax.text(2.5, 4.5, '2012\u20132013', fontsize=5.5, fontweight='bold', ha='center',
        color=OLD_COLOR, fontfamily='Arial')

# Fe-oxide box
fe_box_l = mpatches.FancyBboxPatch((0.8, 2.8), 1.6, 0.9, boxstyle="round,pad=0.1",
                                     facecolor='#D4A574', edgecolor='#8B6914', linewidth=0.8)
ax.add_patch(fe_box_l)
ax.text(1.6, 3.25, 'Fe-oxide', fontsize=4.5, ha='center', va='center',
        fontweight='bold', fontfamily='Arial')

# As sorbed on Fe-oxide
ax.text(1.6, 2.55, 'As sorbed', fontsize=3.5, ha='center', va='center',
        color=GOLD, fontweight='bold', fontfamily='Arial')

# Arrow: Eh drives reductive dissolution
ax.annotate('', xy=(1.6, 2.2), xytext=(1.6, 1.2),
            arrowprops=dict(arrowstyle='->', color=OLD_COLOR, lw=1.5))
ax.text(0.4, 1.7, 'Reductive\ndissolution', fontsize=3.2, ha='center', va='center',
        color=OLD_COLOR, fontfamily='Arial', fontstyle='italic')

# Released As (small)
ax.text(1.6, 0.8, 'As release (low)', fontsize=3.5, ha='center', va='center',
        color=GOLD, fontfamily='Arial')

# PO4 (small, faded)
ax.text(3.5, 3.25, 'PO$_4$', fontsize=4, ha='center', va='center',
        color='#CCCCCC', fontfamily='Arial', fontweight='bold')

# --- Right panel: 2020-2021 ---
ax.text(7.5, 4.5, '2020\u20132021', fontsize=5.5, fontweight='bold', ha='center',
        color=NEW_COLOR, fontfamily='Arial')

# Depleted Fe-oxide box (thinner, lighter)
fe_box_r = mpatches.FancyBboxPatch((5.8, 3.0), 1.6, 0.6, boxstyle="round,pad=0.1",
                                     facecolor='#E8D4B8', edgecolor='#8B6914', linewidth=0.5,
                                     linestyle='--')
ax.add_patch(fe_box_r)
ax.text(6.6, 3.3, 'Fe-oxide', fontsize=4, ha='center', va='center',
        color='#999', fontfamily='Arial')
ax.text(6.6, 3.0, '(\u221224%)', fontsize=3, ha='center', va='top',
        color=OLD_COLOR, fontfamily='Arial')

# PO4 (large, prominent)
po4_circle = plt.Circle((8.8, 3.3), 0.45, facecolor='#90C695', edgecolor='#2E7D32',
                          linewidth=0.8)
ax.add_patch(po4_circle)
ax.text(8.8, 3.3, 'PO$_4$\n+82%', fontsize=3.5, ha='center', va='center',
        fontweight='bold', color='#1B5E20', fontfamily='Arial')

# Arrow: PO4 displaces As
ax.annotate('', xy=(7.5, 2.2), xytext=(8.5, 2.8),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.8))
ax.text(9.2, 2.5, 'Competitive\ndesorption', fontsize=3.2, ha='center', va='center',
        color='#2E7D32', fontfamily='Arial', fontstyle='italic')

# Arrow: Fe-oxide dissolution (faded)
ax.annotate('', xy=(6.6, 2.2), xytext=(6.6, 2.8),
            arrowprops=dict(arrowstyle='->', color='#CCCCCC', lw=0.8))

# Released As (large)
as_box = mpatches.FancyBboxPatch((6.2, 1.2), 2.5, 0.7, boxstyle="round,pad=0.1",
                                   facecolor='#FFF3E0', edgecolor=GOLD, linewidth=1.0)
ax.add_patch(as_box)
ax.text(7.45, 1.55, 'As surge imminent', fontsize=4, ha='center', va='center',
        color='#E65100', fontweight='bold', fontfamily='Arial')

# Central arrow (shift)
ax.annotate('', xy=(5.3, 2.5), xytext=(4.2, 2.5),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.2,
                            connectionstyle='arc3,rad=0'))
ax.text(4.75, 2.9, '8 years', fontsize=3.5, ha='center', va='center',
        color='#333', fontfamily='Arial')

# Bottom bar: key stats
ax.text(5, 0.3, '705 wells  |  15 districts  |  PO$_4$$\\rightarrow$As 17\u00d7 activation  |  74% past threshold',
        fontsize=3.2, ha='center', va='center', color=GRAY, fontfamily='Arial')

plt.tight_layout(pad=0.1)
fig.savefig('/Users/rakibhhridoy/AsGW/GroundWater/Paper3/article/toc_graphic.png',
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('/Users/rakibhhridoy/AsGW/GroundWater/Paper3/article/toc_graphic.tiff',
            dpi=300, bbox_inches='tight', facecolor='white')
print("TOC graphic saved")
plt.close()
