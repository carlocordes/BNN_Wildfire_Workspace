from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from ViT import STViT

plt.rcParams['savefig.dpi'] = 300

def visualize_pos_embed(grid_h, grid_w, save_path):
    pos_embed = model_vit.get_2d_pos_embed(grid_h, grid_w, 'cpu')

    pos_grid = pos_embed.view(grid_h, grid_w, model_vit.embedding_dim)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i, ax in enumerate(axes):
        # Pick a different channel to show different wave frequencies
        channel_idx = i * (model_vit.embedding_dim // 4)
        im = ax.imshow(pos_grid[:, :, channel_idx].detach().numpy(), cmap='RdBu')
        ax.set_title(f"Encoding Dim {channel_idx}")
        plt.colorbar(im, ax=ax)
    plt.savefig(save_path)


def plot_patch_similarity(model, grid_h, grid_w, save_path : Path, anchor_pos=(100, 200)):
    # 1. Generate the embeddings for the grid
    # Shape: (1, H*W, E)
    pos_embed = model.get_2d_pos_embed(grid_h, grid_w, "cpu")
    
    # 2. Reshape to (H*W, E) and normalize for Cosine Similarity
    flat_embed = pos_embed.squeeze(0) # (num_patches, E)
    norm_embed = F.normalize(flat_embed, p=2, dim=1)
    
    # 3. Pick the Anchor Patch
    # Convert (y, x) to flat index
    anchor_idx = anchor_pos[0] * grid_w + anchor_pos[1]
    anchor_vec = norm_embed[anchor_idx].unsqueeze(0) # (1, E)
    
    # 4. Calculate similarity between anchor and ALL patches
    # (1, E) @ (E, num_patches) -> (1, num_patches)
    similarities = torch.mm(anchor_vec, norm_embed.t())
    
    # 5. Reshape back to grid for plotting
    sim_grid = similarities.view(grid_h, grid_w).detach().numpy()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(sim_grid, cmap='magma')
    plt.colorbar(label='Cosine Similarity')
    plt.scatter(anchor_pos[1], anchor_pos[0], marker='x', color='red', s=100, label='Anchor Patch')
    plt.title(f"Positional Similarity relative to Patch {anchor_pos}")
    plt.legend()
    plt.savefig(save_path)

def figure_patch_similarity(embedding_dims : list,
                            grid_h, grid_w, save_path : Path,
                            anchor_pos=(100, 200)):

    num_splots = len(embedding_dims)
    fig, axs = plt.subplots(1, num_splots, figsize=(12, 6))

    for i, e_dim in zip(range(num_splots), embedding_dims):

        model_vit = STViT(
            num_modules=2,
            patch_size=16,
            embedding_dim=e_dim
        )

        pos_embed = model_vit.get_2d_pos_embed(grid_h, grid_w, "cpu")
        flat_embed = pos_embed.squeeze(0)
        norm_embed = F.normalize(flat_embed, p=2, dim=1)

        anchor_idx = anchor_pos[0] * grid_w + anchor_pos[1]
        anchor_vec = norm_embed[anchor_idx].unsqueeze(0)

        similarities = torch.mm(anchor_vec, norm_embed.t())
        sim_grid = similarities.view(grid_h, grid_w).detach().numpy()

        im = axs[i].imshow(sim_grid, cmap='magma')
        axs[i].scatter(
            anchor_pos[1],
            anchor_pos[0],
            marker='x',
            color='red',
            s=100,
            label='Anchor Patch'
        )

        axs[i].set_title(f'Embedding dim {e_dim}')

    # Colorbar
    cbar = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.04)
    cbar.set_label("Cosine Similarity")

    #plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path)


model_vit = STViT(num_modules = 2,
              patch_size = 16,
              embedding_dim = 128)


visualize_pos_embed(80, 40, Path('exports', 'graphs', 'encoding_dims_visual.png'))

figure_patch_similarity(embedding_dims = [32, 128, 2048], grid_h = 70, grid_w = 40,
                        anchor_pos = (35, 20), save_path = Path('exports', 'graphs', 'encoding_simil_visual.png'))