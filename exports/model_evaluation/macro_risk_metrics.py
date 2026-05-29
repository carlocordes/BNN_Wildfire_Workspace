
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns


def plot_macro_risk_metrics(df_runs: pd.DataFrame):
    """
    Computes and plots Fuzzy IoU, Actuarial Bias, and Brier Scores 
    for the yearly aggregated models.
    """
    sns.set_theme(style="whitegrid")
    metrics_list = []
    
    for idx, row in df_runs.iterrows():

        print(np.max(row['yearly_exp_risk']))
        print(np.mean(row['yearly_exp_risk']))

        model_name = row['name']
        P = np.asarray(row['yearly_exp_risk']).squeeze().flatten()
        Y = np.asarray(row['yearly_agg_burn']).squeeze().flatten()
        
        if P.ndim != 1 or len(P) == 0:
            continue
            
        # 1. Fuzzy IoU
        fuzzy_iou = np.sum(np.minimum(P, Y)) / np.sum(np.maximum(P, Y))
        
        # 2. Actuarial Bias (Predicted Macro-Fraction vs Actual Macro-Fraction)
        pred_fraction = np.mean(P)
        actual_fraction = np.mean(Y)
        actuarial_bias = pred_fraction - actual_fraction
        
        # 3. Brier Score
        brier_score = np.mean((P - Y) ** 2)
        
        metrics_list.append({
            'Model': model_name,
            'Fuzzy IoU': fuzzy_iou,
            'Actuarial Bias': actuarial_bias,
            'Brier Score': brier_score
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    
    # Melt dataframe for easy multi-panel plotting with Seaborn Catplot
    df_melted = df_metrics.melt(id_vars='Model', var_name='Metric', value_name='Value')
    
    g = sns.catplot(
        data=df_melted,
        x='Model',
        y='Value',
        col='Metric',
        kind='bar',
        sharey=False,
        height=5,
        aspect=0.8,
        palette='muted'
    )
    g.set_xticklabels(rotation=45)
    g.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Load parquet
    path_to_pq = Path('exports', 'model_evaluation', 'benchmarks.parquet')
    df = pd.read_parquet(path=path_to_pq)


    # Select relevant cols
    df = df[['name', 'yearly_exp_risk', 'yearly_agg_burn']]



    seq_models = ['SampleExt1', 'SampleExt3', 'SampleExt5', 'SampleExt7']
    seq_data = df[df['name'].isin(seq_models)]
    seq_data['name'] = pd.Categorical(seq_data['name'], categories = seq_models, ordered = True)


    lead_models = ['Lead-5', 'Lead0', 'Lead5', 'Lead10', 'Lead20']
    lead_data = df[df['name'].isin(lead_models)]
    lead_data['name'] = pd.Categorical(lead_data['name'], categories = lead_models, ordered = True)

    plot_macro_risk_metrics(seq_data)