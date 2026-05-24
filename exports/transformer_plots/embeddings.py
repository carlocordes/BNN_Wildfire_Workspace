import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['savefig.dpi'] = 300

out_path = Path('exports', 'transformer_plots')
out_path.mkdir(parents=True, exist_ok=True)
epsilon = 0.1

terms = {
    'brother' : np.array([-1, 0]),
    'sister' : np.array([-1, -1]),
    'uncle' : np.array([1 + epsilon, 1 + epsilon]),
    'aunt' : np.array([1, 0]),
}

# --- Fixed Logic: Both represent Male -> Female (Female minus Male) ---
e_gender_sib = terms['sister'] - terms['brother'] # [0, -1]
e_gender_av = terms['aunt'] - terms['uncle']     # [-0.3, -1.3]

# Create a figure with specific dimensions
plt.figure(figsize=(14, 5))

# --- First Subplot (2D Vector Plot) ---
ax1 = plt.subplot(1, 2, 1)

# 1. Plot the base embedding vectors from the origin
for term, vector in terms.items():
    x, y = vector
    ax1.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.4)
    ax1.scatter(x, y, color='red', s=40)
    ax1.text(x + 0.05, y + 0.05, f'{term}', size=11)

# 2. Sibling Difference Vector: Starts at brother, points to sister
ax1.quiver(terms['brother'][0], terms['brother'][1], 
           e_gender_sib[0], e_gender_sib[1], 
           angles='xy', scale_units='xy', scale=1, color='brown', width=0.007, 
           label='Gender Vector')

# 3. Uncle/Aunt Difference Vector: Starts at uncle, points to aunt
ax1.quiver(terms['uncle'][0], terms['uncle'][1], 
           e_gender_av[0], e_gender_av[1], 
           angles='xy', scale_units='xy', scale=1, color='green', width=0.007)

# Set axis labels and styling
ax1.set_xlabel('$X$')
ax1.set_ylabel('$Y$')
ax1.set_xlim([-1.8, 1.8])
ax1.set_ylim([-1.8, 1.8])
ax1.set_title('2D Word Embedding Vectors')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')

# --- Second Subplot (Relative Gender Encodings) ---
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Relative Gender Encodings (Centered)')

# Plotting the vectors centered at the origin (0,0)
# Adding a label and altering line style/width slightly to make overlapping distinct
ax2.quiver(0, 0, e_gender_sib[0], e_gender_sib[1], 
           angles='xy', scale_units='xy', scale=1, color='brown', width=0.01, 
           label='brother $\\rightarrow$ sister')

ax2.quiver(0, 0, e_gender_av[0], e_gender_av[1], 
           angles='xy', scale_units='xy', scale=1, color='darkgreen', width=0.01, 
           label='uncle $\\rightarrow$ aunt')

# Formatting for the relative plot
ax2.set_xlabel('$X$')
ax2.set_ylabel('$Y$')
ax2.set_xlim([-1.8, 1.8])
ax2.set_ylim([-1.8, 1])
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

# Save the figure
plt.savefig(out_path / '2d_vectors_and_relative.png', bbox_inches='tight', dpi=150)