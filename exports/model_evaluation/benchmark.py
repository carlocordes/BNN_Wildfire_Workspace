"""
Evaluate performance metrics of multiple models accross disciplines with Temperature Scaling calibration
"""

# Internal
from src.core.utils import load_config
from src.models.vit.vit import STViT
from scripts.dataset import SpatialTemporalDataset, skip_missing_collate_fn

# External
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from datetime import datetime
import scipy.optimize as opt
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryCalibrationError
from sklearn.calibration import calibration_curve


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

# Helper functions
def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Parses training log file to extract histories."""
    epoch_pattern = re.compile(
        r"Epoch \[\d+/100\] Train Loss: (?P<train_loss>[\d\.]+) \| Val Loss: (?P<val_loss>[\d\.]+)"
    )
    test_pattern = re.compile(r"Final Test Loss: (?P<test_loss>[\d\.]+)")

    results = {"train_losses": [], "val_losses": [], "test_loss": None}

    with open(log_path, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                results["train_losses"].append(float(epoch_match.group("train_loss")))
                results["val_losses"].append(float(epoch_match.group("val_loss")))
                continue
            test_match = test_pattern.search(line)
            if test_match:
                results["test_loss"] = float(test_match.group("test_loss"))
    return results



def evaluate(model, loss_fn, cfg_training, cfg_path, device):
    dates_list, loss_list, recall_list, precision_list = [], [], [], []
    high_tier_coverage, med_tier_coverage, low_tier_coverage = [], [], []
    all_flat_probs, all_flat_targets = [], []

    total_tp, total_fp, total_fn = 0, 0, 0

    dataset = SpatialTemporalDataset(cfg_path, split_type="test", benchmark_mode=True)
    set_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False, collate_fn=skip_missing_collate_fn
    )

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(set_dataloader):
            print(f'Processing test batch_idx {batch_idx + 1} / {len(set_dataloader)}')
            if batch is None:
                continue

            static_x = batch['static'].to(device)
            single_dynamic_x = batch['single_dynamic'].to(device)
            dynamic_x = batch['dynamic'].to(device)
            targets = batch['target'].to(device)
            first_date = datetime.strptime(batch['meta_first_date'][0], '%Y-%m-%d')

            ## Predict (Raw Logits)
            preds = model(x_static=static_x, x_dynamic=dynamic_x, x_single_dynamic=single_dynamic_x)

            ## Convert outputs via Temperature Correction
            targets_int = targets.int()
            probs = torch.sigmoid(preds)
            binary_preds = (probs > THR_CLASS).int()

            tp = torch.sum((binary_preds == 1) & (targets_int == 1)).item()
            fp = torch.sum((binary_preds == 1) & (targets_int == 0)).item()
            fn = torch.sum((binary_preds == 0) & (targets_int == 1)).item()

            # --- Categorical Risk Tier Formulation ---
            total_actual_fires = torch.sum(targets_int).item()
            
            if total_actual_fires > 0:
                high_risk_mask = (probs >= 0.40).int()
                med_risk_mask  = ((probs >= 0.15) & (probs < 0.40)).int()
                low_risk_mask  = (probs < 0.15).int()

                h_cov = torch.sum((high_risk_mask == 1) & (targets_int == 1)).item() / total_actual_fires
                m_cov = torch.sum((med_risk_mask == 1) & (targets_int == 1)).item() / total_actual_fires
                l_cov = torch.sum((low_risk_mask == 1) & (targets_int == 1)).item() / total_actual_fires
            else:
                h_cov, m_cov, l_cov = float('nan'), float('nan'), float('nan')

            high_tier_coverage.append(h_cov)
            med_tier_coverage.append(m_cov)
            low_tier_coverage.append(l_cov)

            # --- Precision & Recall ---
            if (tp + fp) > 0:
                batch_precision = tp / (tp + fp)
            else:
                batch_precision = 0.0 if fn > 0 else float('nan')

            if (tp + fn) > 0:
                batch_recall = tp / (tp + fn)
            else:
                batch_recall = float('nan')

            if (tp + fp + fn) > 0:
                total_tp += tp; total_fp += fp; total_fn += fn

            precision_list.append(batch_precision)
            recall_list.append(batch_recall)
            loss_list.append(loss_fn(preds, targets).item())
            dates_list.append(first_date)

            all_flat_probs.append(probs.cpu().flatten())
            all_flat_targets.append(targets_int.cpu().flatten())

        global_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        global_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        probs_tensor = torch.cat(all_flat_probs)
        targets_tensor = torch.cat(all_flat_targets)

        # --- Global Expected Calibration Error (ECE) Math ---
        ece_metric = BinaryCalibrationError(n_bins=10, norm='l1')
        global_ece = ece_metric(probs_tensor, targets_tensor).item()

        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true = targets_tensor.numpy(),
            y_prob = probs_tensor.numpy(),
            n_bins = 10,
        )

    return {
        'batch_history' : {
            'dates' : dates_list,
            'losses' : loss_list,
            'precision' : precision_list,
            'recall' : recall_list,
            'high_risk_coverage': high_tier_coverage,
            'med_risk_coverage': med_tier_coverage,
            'low_risk_coverage': low_tier_coverage,
        },
        'global_averages' : {
            'precision' : global_precision,
            'recall' : global_recall,
            'ece': global_ece,
        },
        'calibration' : {
            'prob_true' : prob_true,
            'prob_pred' : prob_pred,
        }
    }


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

    train_histories, val_histories, test_losses, best_val_losses = [], [], [], []
    test_run_dates, test_run_losses, test_run_precision_history, test_run_recall_history = [], [], [], []
    test_run_precision, test_run_recall, test_run_ece = [], [], []
    test_run_low_risk, test_run_med_risk, test_run_high_risk = [], [], []
    prob_pred, prob_true = [], []

    for idx, row in df_runs.iterrows():
        print(f"\n Evaluating model run: {row['name']}")
        
        ## 1. Parse logfile
        log_data = parse_training_log(row['log_path'])
        train_histories.append(log_data['train_losses'])
        val_histories.append(log_data['val_losses'])
        test_losses.append(log_data['test_loss'])
        best_val_losses.append(min(log_data['val_losses']) if log_data['val_losses'] else None)

        ## 2. Setup Configs and Models
        cfg = load_config(row['cfg_path'])
        device = 'mps'
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg['training']["pos_weight"]])).to(device)
        model = load_model(cfg_model=cfg['model'], cfg_data=cfg['data'], model_path=row['model_path'], device=device)


        ## PASS 2: Evaluate with the newly found T scaling factor
        metrics = evaluate(model=model, loss_fn=loss_fn, cfg_training=cfg['training'], 
                           cfg_path=row['cfg_path'], device=device)
        
        metrics_history = metrics['batch_history']
        metrics_global = metrics['global_averages']
        metrics_cal = metrics['calibration']

        test_run_dates.append(metrics_history['dates'])
        test_run_losses.append(metrics_history['losses'])
        test_run_recall_history.append(metrics_history['recall'])
        test_run_precision_history.append(metrics_history['precision'])
        test_run_recall.append(metrics_global['recall'])
        test_run_precision.append(metrics_global['precision'])
        test_run_low_risk.append(metrics_history['low_risk_coverage'])
        test_run_med_risk.append(metrics_history['med_risk_coverage'])
        test_run_high_risk.append(metrics_history['high_risk_coverage'])
        test_run_ece.append(metrics_global['ece'])

        prob_pred.append(metrics_cal['prob_pred'])
        prob_true.append(metrics_cal['prob_true'])


    # Assign Columns
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
    df_runs['ece'] = test_run_ece
    df_runs['low_risk_cov'] = test_run_low_risk
    df_runs['med_risk_cov'] = test_run_med_risk
    df_runs['high_risk_cov'] = test_run_high_risk 
    df_runs['prob_pred'] = prob_pred
    df_runs['prob_true'] = prob_true   


    df_runs = df_runs.drop(columns=['cfg_path', 'model_path', 'log_path'])
    print("\n Final Benchmarking Evaluation Summary:")
    print(df_runs[['name', 'optimal_temperature', 'ece', 'precision', 'recall']])
    print(df_runs[['name', 'low_risk_cov', 'med_risk_cov', 'high_risk_cov']])
    print(df_runs[['prob_pred', 'prob_true']])

    print('prob_pred', prob_pred)
    print('prob_true', prob_true)


    # df_runs.to_parquet(path=out_path, index=False)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    plt.plot(prob_pred, prob_true, "s-", color="crimson")

    plt.xlabel("Confidence (Mean Predicted Probability)")
    plt.ylabel("Accuracy (Actual Burn Fraction)")
    plt.title("Wildfire Risk Reliability Diagram")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()