import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Arc
from pathlib import Path

# Set seaborn theme for a clean appearance
sns.set_theme(style="whitegrid")

# Define angle in degrees and convert to radians
angle_deg = 60
angle_rad = np.radians(angle_deg)

# Vector coordinates (unit vector)
v_len = 1.0
vx = v_len * np.cos(angle_rad)
vy = v_len * np.sin(angle_rad)

# Create figure and axis using subplots
fig, ax = plt.subplots(figsize=(6, 4))

# Set limits and maintain an equal aspect ratio to prevent distortion
ax.set_xlim(-0.5, 1.2)
ax.set_ylim(-0.5, 1.2)
ax.set_aspect('equal')

# Draw major axes lines at x=0 and y=0
ax.axhline(0, color='black', linewidth=1.2, zorder=1)
ax.axvline(0, color='black', linewidth=1.2, zorder=1)

# Draw dashed projection lines from the vector tip to the axes
ax.plot([vx, vx], [0, vy], color='gray', linestyle='--', linewidth=1.5, zorder=2)
ax.plot([0, vx], [vy, vy], color='gray', linestyle='--', linewidth=1.5, zorder=2)

# Highlight components on the axes (shadow projections)
# Using seaborn palette colors for distinction
colors = sns.color_palette()
ax.annotate('', xy=(vx, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colors[3], lw=3, mutation_scale=15),
            zorder=3)
ax.annotate('', xy=(0, vy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=colors[0], lw=3, mutation_scale=15),
            zorder=3)

# Draw the main aspect vector
ax.annotate('', xy=(vx, vy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color='black', lw=4, mutation_scale=20),
            zorder=4)

# Draw the angle arc to specify the orientation
arc = Arc((0, 0), 0.4, 0.4, angle=0, theta1=0, theta2=angle_deg, color='black', lw=1.5, zorder=5)
ax.add_patch(arc)

# Labels, mathematical text, and components descriptions
ax.text(vx + 0.05, vy + 0.05, r'Aspect Vector $\vec{v}$', fontsize=12, fontweight='bold')
ax.text(vx / 2, -0.08, 'East-West\nComponent ($x$)', color=colors[3], fontsize=11, ha='center', fontweight='bold')
ax.text(-0.08, vy / 2, 'North-South\nComponent ($y$)', color=colors[0], fontsize=11, va='center', ha='right', fontweight='bold')
ax.text(0.25 * np.cos(np.radians(angle_deg/2)), 0.25 * np.sin(np.radians(angle_deg/2)), r'$\phi$', fontsize=14, ha='center', va='center')

# Labeling the axes as geographic directions
ax.set_xlabel('East (+) / West (-)', fontsize=12)
ax.set_ylabel('North (+) / South (-)', fontsize=12)
# Ensure nothing is truncated or overlapping
plt.tight_layout()

# Save the figure
out_path = Path('exports', 'figs_misc', 'aspect_decomposition.png')
plt.savefig(out_path, dpi=300)