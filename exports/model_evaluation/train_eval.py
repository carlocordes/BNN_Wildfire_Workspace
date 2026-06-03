"""
Produces figures of training runs of comparable models
"""

# Internal

# External
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUT_PATH = Path('exports', 'model_evaluation', 'model_sample_compare.png')

def plot_train_eval(data : pd.DataFrame):

    # Set Seaborn theme styling and middle-ground text sizes
    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    df_sorted = data.reset_index(drop = True)
    
    # Set up a 2x2 grid with custom width ratios (History: 4, Bar Chart: 1)
    fig = plt.figure(figsize=(14, 8.5)) 
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 1])
    
    ax0 = fig.add_subplot(gs[0, 0])  # Top-Left: Train History
    ax3 = fig.add_subplot(gs[0, 1])  # Top-Right: Final Test Loss
    ax1 = fig.add_subplot(gs[1, 0])  # Bottom-Left: Val History
    ax2 = fig.add_subplot(gs[1, 1])  # Bottom-Right: Best Val Loss

    if len(df_sorted) == 5:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#A015E0"]
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    palette = sns.color_palette(colors)

    # --- ROW 1, PLOT 1: Train Loss History (Time Series) ---
    for idx, row in df_sorted.iterrows():
        sns.lineplot(
            x=range(len(row['train_history'])), 
            y=row['train_history'], 
            label=row['name'], 
            marker='o',
            color=colors[idx], 
            linewidth=2.2, 
            ax=ax0
        )
    ax0.set_title('Train Loss History')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')
    ax0.legend()

    # --- ROW 1, PLOT 2: Final Test Loss (Static Bar Chart) ---
    sns.barplot(
        x='name', 
        y='test_loss', 
        data=df_sorted, 
        palette=palette, 
        edgecolor='black', 
        alpha=0.85, 
        ax=ax3
    )
    ax3.set_title('Final Test Loss')
    ax3.set_xlabel(None)
    ax3.set_ylabel('Loss')
    ax3.set_xticklabels(df_sorted['name'], rotation=45, ha='right')
    ax3.set_ylim(0, 1)

    # Native, safe method to pull text values directly from the plotted container
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.4f', padding=3, fontsize=14)

    # --- ROW 2, PLOT 1: Validation Loss History (Time Series) ---
    for idx, row in df_sorted.iterrows():
        sns.lineplot(
            x=range(len(row['val_history'])), 
            y=row['val_history'], 
            label=row['name'], 
            marker='o', 
            color=colors[idx], 
            linewidth=2.2, 
            ax=ax1
        )
    ax1.set_title('Validation Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # --- ROW 2, PLOT 2: Best Validation Loss (Static Bar Chart) ---
    sns.barplot(
        x='name',
        y='best_val_loss',
        data=df_sorted,
        palette=palette,
        edgecolor='black',
        alpha=0.85,
        ax=ax2
    )
    ax2.set_title('Best Validation Loss')
    ax2.set_xlabel(None)
    ax2.set_ylabel('Loss')
    ax2.set_xticklabels(df_sorted['name'], rotation=45, ha='right')
    ax2.set_ylim(0, 1)

    # Native, safe method to pull text values directly from the plotted container
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.4f', padding=3, fontsize=14)

    # Optimize layout to ensure labels do not overlap or get truncated
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi = 300)

if __name__ == '__main__':

    # Load parquet
    path_to_pq = Path('exports', 'model_evaluation', 'benchmarks.parquet')
    data = pd.read_parquet(path=path_to_pq)

    data = data[['name', 'train_history', 'val_history', 'test_loss', 'best_val_loss']]

    # Define explicit name mappings
    name_mapping = {
        'SampleExt1': 'Sample_extent_1',
        'SampleExt3': 'Sample_extent_3',
        'SampleExt5': 'Sample_extent_5',
        'SampleExt7': 'Sample_extent_7',
        'Lead-5': 'lead_time_-5',
        'Lead0': 'lead_time_0',
        'Lead5': 'lead_time_5',
        'Lead10': 'lead_time_10',
        'Lead20': 'lead_time_20'
    }

    # 1. Process Sequence Models
    seq_models_old = ['SampleExt1', 'SampleExt3', 'SampleExt5', 'SampleExt7']
    seq_models_new = [name_mapping[m] for m in seq_models_old]
    
    seq_data = data[data['name'].isin(seq_models_old)].copy()
    seq_data['name'] = seq_data['name'].map(name_mapping)
    seq_data['name'] = pd.Categorical(seq_data['name'], categories=seq_models_new, ordered=True)
    seq_data = seq_data.sort_values('name')

    # 2. Process Lead Models
    lead_models_old = ['Lead-5', 'Lead0', 'Lead5', 'Lead10', 'Lead20']
    lead_models_new = [name_mapping[m] for m in lead_models_old]
    
    lead_data = data[data['name'].isin(lead_models_old)].copy()
    lead_data['name'] = lead_data['name'].map(name_mapping)
    lead_data['name'] = pd.Categorical(lead_data['name'], categories=lead_models_new, ordered=True)
    lead_data = lead_data.sort_values('name')

    # Plot whichever dataset you currently need (e.g., lead_data)
    plot_train_eval(seq_data)