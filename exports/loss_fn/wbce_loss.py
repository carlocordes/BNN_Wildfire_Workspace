# Internal


# External
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pathlib import Path




# ---- CONFIG ---- #
pos_weight = 8
eps = 1e-7 # Small constant to avoid log(0)
out_path = Path("exports", "loss_fn") / "wbce_3d.svg"


def wbce_vectorized(pred, gt, p_weight):
    # Clip predictions to [eps, 1-eps] to prevent infinity in logs
    pred = np.clip(pred, eps, 1 - eps)
    loss = -1 * (p_weight * gt * np.log(pred) + (1 - gt) * np.log(1 - pred))
    return loss

predictions = np.linspace(start = 0.1, stop = 0.99, num = 101)
ground_truths = np.linspace(start = 0.1, stop = 0.99, num = 101)

P, G = np.meshgrid(predictions, ground_truths)
Z = wbce_vectorized(P, G, pos_weight)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(P, G, Z, cmap=cm.magma, 
                       linewidth=0, antialiased=True, alpha=0.9)

# Add labels and styling
ax.set_xlabel('Prediction (p)', fontsize=12)
ax.set_ylabel('Ground Truth (y)', fontsize=12)
ax.set_zlabel('WBCE Loss', fontsize=12)
#ax.set_title(f'WBCE Loss Surface (pos_weight={pos_weight})', fontsize=15)

# Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=10)

# Adjust viewing angle to see the 'asymmetry' better
ax.view_init(elev=30, azim=220)
plt.show()
#plt.savefig(out_path)