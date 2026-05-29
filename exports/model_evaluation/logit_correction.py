
# Internal
from src.models.vit.vit import STViT

# External
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path


# --------------- CONFIG --------------- #
cfg_base = Path('files', 'configs')
exp_base = Path('files', 'experiments')

out_path = Path('exports', 'model_evaluation', 'benchmarks.parquet')

# Name configurations
run_names = {
    'SampleExt3' : 't009_1',
    'SampleExt5' : 't009_2',
    'SampleExt7' : 't009_3',
    'SampleExt1' : 't009_4',

    'Lead-5' : 't010_1',
    'Lead0' : 't009_2', 
    'Lead5' : 't010_2',
    'Lead10' : 't010_3',
    'Lead20' : 't010_4',
}
# -------------------------------------- #


def load_model(cfg_model, cfg_data, model_path, device):
    model = STViT(
        num_dynamic_channels=cfg_model['num_dynamic_channels'],
        num_static_channels=cfg_model['num_static_channels'] + cfg_model['num_single_dynamic_channels'],
        num_timestamps_per_sample=cfg_data['temporal_extent']['sample_extent'],
        patch_size=cfg_model['patch_size'],
        embedding_dim=cfg_model['embedding_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)






if __name__ == '__main__':
    df_runs = pd.DataFrame([
        { 
            'name' : name,
            'cfg_path' : cfg_base / f'config_{exp}.yaml',
            'model_path' : exp_base / exp / f'{exp}_model_best.pt',
            'log_path' : exp_base / exp / 'training.log'
        } for name, exp in run_names.items()
    ])
