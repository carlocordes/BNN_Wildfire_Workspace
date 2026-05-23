import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['savefig.dpi'] = 300

def get_2d_julian_embedding(days):
    """
    Maps days (0-365) to a continuous 2D cyclical embedding space.
    """
    # Convert days to radians (0 to 2*pi)
    angles = (2 * np.pi * days) / 365.25
    
    # Generate the sine and cosine dimensions
    sin_dim = np.sin(angles)
    cos_dim = np.cos(angles)
    
    # Stack into a 2D array of shape (N, 2)
    return np.stack([sin_dim, cos_dim], axis=-1)

# 1. Generate all days of the year
days_of_year = np.arange(366)
embeddings = get_2d_julian_embedding(days_of_year)

# 2. Extract components for plotting
sin_coords = embeddings[:, 0]
cos_coords = embeddings[:, 1]

# 3. Create the 2D plot
plt.figure(figsize=(9, 4))
plt.plot(sin_coords, cos_coords, label="Yearly Cycle", color="teal", linewidth=2)

# Highlight specific milestone days to see where they land
milestones = {0: "Jan 1 (Day 0)", 90: "Apr 1 (Day 90)", 181: "Jul 1 (Day 181)", 273: "Oct 1 (Day 273)"}
for day, label in milestones.items():
    plt.scatter(sin_coords[day], cos_coords[day], color="darkorange", s=80, zorder=5)
    plt.annotate(label, (sin_coords[day], cos_coords[day]), textcoords="offset points", 
                 xytext=(10,10), ha='center', fontsize=9, fontweight='bold')

#plt.title("2D Cyclical Julian Day Embedding Space")
plt.xlabel("Dimension 1: Sin(Day)")
plt.ylabel("Dimension 2: Cos(Day)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.axhline(0, color='black',linewidth=0.5, alpha=0.3)
plt.axvline(0, color='black',linewidth=0.5, alpha=0.3)
plt.gca().set_aspect('auto') 

# Optional: Add tight_layout to pull the edges out even further
plt.tight_layout()
plt.legend()


out_path = Path('exports', 'julian_day_enc', 'julian_day.png')

plt.savefig(out_path)