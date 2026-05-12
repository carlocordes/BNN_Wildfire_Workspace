# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.train import WildfireDataset, build_dataloader

# External
import torch
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---- CONFIG ---- #
NUM_TIMESTEPS = 50


def main(config_path : Path, model_path : Path, dataset_path : Path):

    # Check that paths exist
    assert(config_path.exists() & model_path.exists() & dataset_path.exists())


    # Get cfg
    cfg = load_config(config_path)
    cfg_model = cfg['model']
    cfg_training = cfg['training']
    cfg_training['batch_size'] = 1 # Force param



    # Load model from parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STViT(
        num_dynamic_channels=4,
        num_static_channels=3,
        num_timestamps_per_sample=10,
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    # Load Dataset
    dataset_dict = torch.load(dataset_path, weights_only=False)
    dataloader = build_dataloader(dataset_dict, cfg_training)


    # Iterate over batches
    all_tokens = []
    all_patch_idx = []
    all_time_idx = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            if batch_idx == NUM_TIMESTEPS: # Remove, test purposes only
                break

            print(f'Processing batch idx: {batch_idx}')
            
            static_x = batch['static'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)

            logits, tokens = model(static_x, dynamic_x, return_tokens=True)

            # Store prediction (one per batch)
            #preds = torch.sigmoid(logits).mean().item()

            tokens = tokens.squeeze(0)   # (780, 512)
            all_tokens.append(tokens)  # store ALL tokens

            # Metadata
            num_patches = tokens.shape[0]

            patch_idx = torch.arange(num_patches, device=tokens.device)
            all_patch_idx.append(patch_idx)

            time_idx = torch.full((num_patches,), batch_idx, device=tokens.device)
            all_time_idx.append(time_idx)


    # Stack everything
    all_tokens = torch.cat(all_tokens, dim=0)   # (N*780, 512)
    all_patch_idx = torch.cat(all_patch_idx, dim=0)
    all_time_idx = torch.cat(all_time_idx, dim=0)

    # TSNE
    tsne = TSNE(n_components = 2, perplexity = 30, random_state = 42)
    embedded = tsne.fit_transform(all_tokens.cpu().numpy())

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: colored by patch_idx ---
    sc1 = axs[0].scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=all_patch_idx.cpu().numpy(),
        cmap='magma',
        s=5
    )
    axs[0].set_title("Patch Index")
    axs[0].set_xlabel("Dim 1")
    axs[0].set_ylabel("Dim 2")

    cbar1 = fig.colorbar(sc1, ax=axs[0])
    cbar1.set_label("Patch idx")


    # --- Right: colored by time_idx ---
    sc2 = axs[1].scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=all_time_idx.cpu().numpy(),
        cmap='magma',
        s=5
    )
    axs[1].set_title("Time Index")
    axs[1].set_xlabel("Dim 1")
    axs[1].set_ylabel("Dim 2")

    cbar2 = fig.colorbar(sc2, ax=axs[1])
    cbar2.set_label("Time idx")


    plt.tight_layout()
    plt.savefig(Path('exports', 'tsne', 'tsne_split.png'))
    plt.close()


if __name__ == '__main__':
    config_path = Path('files', 'configs', '3year_0lead.yaml')
    dataset_path = Path('files', 'datasets', '1year_0lead_v.pt')
    model_path = Path('files', 'experiments', 't001', 't001_small', 't001_small_model_best.pt')

    main(config_path=config_path,
         model_path=model_path,
         dataset_path=dataset_path)