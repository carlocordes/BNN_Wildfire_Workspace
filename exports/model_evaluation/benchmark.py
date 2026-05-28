"""
Evaluate performance metrics of multiple models accross disciplines
"""

# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn
from scripts.train import build_model

# External
import re
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# --------------- CONFIG --------------- #
cfg_base = Path('files', 'configs')
exp_base = Path('files', 'experiments')

out_path = Path('exports', 'model_evaluation', 'benchmarks.parquet')

# Name configurations
run_names = {
    'SampleExt3' : 't009_1',
    #'SampleExt5' : 't009_2',
    #'SampleExt7' : 't009_3',
    #'SampleExt1' : 't009_4',

    #'Lead-5' : 't010_1',
    #'Lead0' : 't009_2', #Grabbed from old run
    #'Lead5' : 't010_2',
    #'Lead10' : 't010_3',
    #'Lead20' : 't010_4',
}

THR_CLASS = 0.1
# -------------------------------------- #

# Helper functions
def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """
    Parses a training log file to extract Train/Val losses per epoch 
    and the final Test loss.
    """
    # Regex patterns matching your log format
    epoch_pattern = re.compile(
        r"Epoch \[\d+/100\] Train Loss: (?P<train_loss>[\d\.]+) \| Val Loss: (?P<val_loss>[\d\.]+)"
    )
    test_pattern = re.compile(r"Final Test Loss: (?P<test_loss>[\d\.]+)")

    results = {
        "train_losses": [],
        "val_losses": [],
        "test_loss": None
    }

    # Read and parse line by line
    with open(log_path, 'r') as f:
        for line in f:
            # Check for epoch losses
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                results["train_losses"].append(float(epoch_match.group("train_loss")))
                results["val_losses"].append(float(epoch_match.group("val_loss")))
                continue
                
            # Check for final test loss
            test_match = test_pattern.search(line)
            if test_match:
                results["test_loss"] = float(test_match.group("test_loss"))

    return results


def evaluate(model, loss_fn, cfg_training, cfg_path, device):

    # Initialize data structures
    dates_list = []
    loss_list = []
    recall_list = []
    precision_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_samples = 0


    # Create dataset and dataloader
    dataset = SpatialTemporalDataset(cfg_path, split_type = "test")

    set_dataloader = DataLoader(
        dataset, 
        batch_size=1, #cfg_training['batch_size'], 
        shuffle=False,         
        num_workers=1,         
        pin_memory=False,       
        collate_fn=skip_missing_collate_fn
    )

    model.eval()


    with torch.no_grad():
        for batch_idx, batch in enumerate(set_dataloader):

            print(f'Processing batch_idx {batch_idx + 1} / {len(set_dataloader)}')

            if batch is None:
                continue

            # Get data
            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)
            first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')

            ## Predict
            preds = model(
                x_static=static_x,
                x_dynamic=dynamic_x,
                x_single_dynamic=single_dynamic_x
            )

            ## Convert outputs to Binary
            targets_int = targets.int()
            probs = torch.sigmoid(preds)
            print('Max value in prediction:', probs.max().item())
            binary_preds = (probs > THR_CLASS).int()

            # Calculate raw pixel counts for this 300x600 grid sample
            tp = torch.sum((binary_preds == 1) & (targets_int == 1)).item()
            fp = torch.sum((binary_preds == 1) & (targets_int == 0)).item()
            fn = torch.sum((binary_preds == 0) & (targets_int == 1)).item()

            # --- Precision Lane ---
            if (tp + fp) > 0:
                batch_precision = tp / (tp + fp)
            else:
                # The model predicted zero fires. 
                # If there actually WERE fires (fn > 0), precision is 0.0 (it completely missed).
                # If there were NO fires (fn == 0), it was a perfect clear-sky prediction. We set to NaN to ignore it.
                batch_precision = 0.0 if fn > 0 else float('nan')

            # --- Recall Lane ---
            if (tp + fn) > 0:
                batch_recall = tp / (tp + fn)
            else:
                # There were zero actual fires on the ground.
                # Recall is mathematically undefined here, so we ignore this day's recall.
                batch_recall = float('nan')

            # Append values to histories
            precision_list.append(batch_precision)
            recall_list.append(batch_recall)

            ## Loss (Still tracked globally to assess background convergence)
            loss = loss_fn(preds, targets)
            loss_list.append(loss.item())

            ## Dates
            dates_list.append(first_date)

        # Global averages computed strictly from days where fire pixels existed/were predicted
        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        'batch_history' : {
            'dates' : dates_list,
            'losses' : loss_list,
            'precision' : precision_list,
            'recall' : recall_list,
        },
        'global_averages' : {
            'precision' : global_precision,
            'recall' : global_recall,
        }
    }


if __name__ == '__main__':
    # Initiate df structure
    df_runs = pd.DataFrame([
        { 
            'name' : name,
            'cfg_path' : cfg_base / f'config_{exp}.yaml',
            'model_path' : exp_base / exp / f'{exp}_model_best.pt',
            'log_path' : exp_base / exp / 'training.log'
        } for name, exp in run_names.items()
    ])

    # Lists to temporarily hold data to avoid Pandas iterable assignment bugs
    train_histories = []
    val_histories = []
    test_losses = []
    best_val_losses = []

    test_run_dates = []
    test_run_losses = []
    test_run_precision_history = []
    test_run_recall_history = []

    test_run_precision = []
    test_run_recall = []

    for idx, row in df_runs.iterrows():

        ## 1. Parse logfile
        # Read logfile:
        log_path = row['log_path']
        log_data = parse_training_log(log_path)

        # Append to temporary lists
        train_histories.append(log_data['train_losses'])
        val_histories.append(log_data['val_losses'])
        test_losses.append(log_data['test_loss'])
        
        if log_data['val_losses']:
            best_val_losses.append(min(log_data['val_losses']))
        else:
            best_val_losses.append(None)


        ## 2. Evaluate model accuracy
        # Get config from path
        cfg = load_config(row['cfg_path'])
        cfg_training = cfg['training']
        cfg_model = cfg['model']
        cfg_data = cfg['data']

        # Other dependencies
        device = 'mps'
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg_training["pos_weight"]])).to(device)


        # Initiate model
        model = build_model(cfg_model = cfg_model, cfg_data = cfg_data, device = device)

        # Evaluate and gather metrics
        metrics = evaluate(model = model,
                        loss_fn = loss_fn,
                        cfg_training = cfg_training,
                        cfg_path = row['cfg_path'],
                        device = device)
        
        # Extract and append
        metrics_history = metrics['batch_history']
        metrics_global = metrics['global_averages']

        test_run_dates.append(metrics_history['dates'])
        test_run_losses.append(metrics_history['losses'])
        test_run_recall_history.append(metrics_history['recall'])
        test_run_precision_history.append(metrics_history['precision'])

        test_run_recall.append(metrics_global['recall'])
        test_run_precision.append(metrics_global['precision'])
        



    # Safely assign entire columns at once
    df_runs['train_loss_history'] = train_histories
    df_runs['val_loss_history'] = val_histories
    df_runs['final_test_loss'] = test_losses
    df_runs['best_val_loss'] = best_val_losses
    df_runs['dates'] = test_run_dates
    df_runs['loss_history'] = test_run_losses
    df_runs['precision_history'] = test_run_precision_history
    df_runs['recall_history'] = test_run_recall_history
    df_runs['precision'] = test_run_precision
    df_runs['recall'] = test_run_recall

    # Drop path columns as parquet cannot write these
    df_runs = df_runs.drop(columns = ['cfg_path', 'model_path', 'log_path'])

    print(df_runs)

    # Store to parquet
    #df_runs.to_parquet(path = out_path, index = False)
