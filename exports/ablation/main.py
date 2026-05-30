"""
Takes an instance of STViT and measures predictional variability
of a baseline model against ablated versions of the same model
"""

# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn


# External
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime


# --- Helper functions

class ablation():
    def __init__(self, static_ablations, dynamic_ablations):
        self.static_ablations = static_ablations
        self.dynamic_ablations = dynamic_ablations

def load_trained_model(cfg_model : nn.Module, cfg_data, model_path, device):
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def evaluate_ablated_model(model, dataloader : DataLoader, cfg_path, device, ablation_cfg : ablation):



    model.eval()

    all_probs = []

    # Primary sweep with full model
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f'Processing test batch_idx {batch_idx + 1} / {len(dataloader)}')
            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            #targets = batch['target'].to(device)
            #first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')

            ## Predict
            logits = model(x_static=static_x, x_dynamic=dynamic_x, x_single_dynamic=single_dynamic_x,
                           ablate_static_idxs = ablation_cfg.static_ablations,
                           ablate_dynamic_idxs = ablation_cfg.dynamic_ablations # Ablated versions
                           )

            probs = torch.sigmoid(logits).squeeze()

            all_probs.append(probs)
            

    # Stack all results together
    all_preds_tensor = torch.stack(all_probs, dim = 0)
    
    return all_preds_tensor.detach().cpu().numpy()






if __name__ == '__main__':
    
    # Load cfg
    exp_name = 't009_4' # 't009_1'

    cfg_file = 'config_' + exp_name + '.yaml'
    cfg_path = Path('files', 'configs', cfg_file)
    cfg = load_config(cfg_path)
    cfg_model = cfg['model']
    cfg_data = cfg['data']

    # Load model path
    model_path = Path('files', 'experiments', exp_name, f'{exp_name}_model_best.pt')

    # Device
    device = 'mps'

    # Select baseline model
    model = load_trained_model(
        cfg_model=cfg_model,
        cfg_data=cfg_data,
        model_path=model_path,
        device = device
    )

    # Data
    dataset = SpatialTemporalDataset(cfg_path, split_type="test", benchmark_mode=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False, collate_fn=skip_missing_collate_fn
    )


    # Build ablation configurations
    ablations = [
        ablation(static_ablations = None, dynamic_ablations = None), # Baseline case
        ablation(static_ablations = [0, 1, 2], dynamic_ablations = None), # All terrain sets
        #ablation(static_ablations = [3], dynamic_ablations = None), # Roads
        #ablation(static_ablations = [4], dynamic_ablations = None), # burn hiustory
        #ablation(static_ablations=[5], dynamic_ablations=None), # precipitation
        #ablation(static_ablations=None, dynamic_ablations=[0]), # NDVI
        #ablation(static_ablations=None, dynamic_ablations=[1]), # NDWI
        #ablation(static_ablations=None, dynamic_ablations=[2, 3, 4]), # wind
        #ablation(static_ablations=None, dynamic_ablations=[5]), # Land surface temperature
    ]
    

    all_predictions = []
    for abltn in ablations:
        
        # Precict with ablationo configuration
        ablation_predictions = evaluate_ablated_model(
            model=model,
            dataloader=dataloader,
            cfg_path=cfg_path,
            device=device,
            ablation_cfg=abltn)
        
        # Store
        all_predictions.append(ablation_predictions)

    # Separate
    baseline_preds = all_predictions[0]
    ablated_preds = all_predictions[1:] 

    print(baseline_preds.shape)