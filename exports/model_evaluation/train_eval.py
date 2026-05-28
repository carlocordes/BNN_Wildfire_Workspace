"""
Produces figures of training runs of comparable models
"""

# Internal

# External
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUT_PATH = Path('exports', 'model_evaluation', 'model_train_compare.png')

def plot_train_eval(data : pd.DataFrame):

    # Set Seaborn theme styling
    sns.set_theme(style="whitegrid")

    df_sorted = data.reset_index(drop = True)
    fig, axes = plt.subplots(1, 4, figsize=(13, 5), gridspec_kw={'width_ratios': [2.4, 2.4, 1.2, 1.2]})


    if len(df_sorted) == 5:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "#A015E0"]
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    palette = sns.color_palette(colors)

    # --- Plot 1: Train Loss History (Time Series) ---
    ax0 = axes[0]
    for idx, row in df_sorted.iterrows():
        sns.lineplot(
            x=range(len(row['train_loss_history'])), 
            y=row['train_loss_history'], 
            label=row['name'], 
            marker='o', 
            color=colors[idx], 
            linewidth=2, 
            ax=ax0
        )
    ax0.set_title('Train Loss History')
    ax0.set_xlabel('Epoch')
    ax0.set_ylabel('Loss')
    ax0.legend()

    # --- Plot 2: Validation Loss History (Time Series) ---
    ax1 = axes[1]
    for idx, row in df_sorted.iterrows():
        sns.lineplot(
            x=range(len(row['val_loss_history'])), 
            y=row['val_loss_history'], 
            label=row['name'], 
            marker='o', 
            color=colors[idx], 
            linewidth=2, 
            ax=ax1
        )
    ax1.set_title('Validation Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # --- Plot 3: Final Test Loss (Static Bar Chart) ---
    ax3 = axes[3]
    sns.barplot(
        x='name', 
        y='final_test_loss', 
        data=df_sorted, 
        palette=palette, 
        hue='name',
        legend=False,
        edgecolor='black', 
        alpha=0.85, 
        ax=ax3
    )
    ax3.set_title('Final Test Loss')
    ax3.set_xlabel(None)  # Remove redundant 'name' label on x-axis
    ax3.set_ylabel('Loss')
    
    # Set tick positions explicitly to prevent warnings
    ax3.set_xticks(range(len(df_sorted)))
    ax3.set_xticklabels(df_sorted['name'], rotation=45, ha='right')

    ax3.set_ylim(0, 1)

    # Add exact value labels on top of each bar
    for bar in ax3.patches:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)


    # --- Plot 4: Best Validation Loss
    ax2 = axes[2]
    sns.barplot(
        x = 'name',
        y = 'best_val_loss',
        data = df_sorted,
        palette = palette,
        hue = 'name',
        legend = False,
        edgecolor = 'black',
        alpha = 0.85,
        ax = ax2
    )
    ax2.set_title('Best Validation Loss')
    ax2.set_xlabel(None),
    ax2.set_ylabel('Loss')

    ax2.set_xticks(range(len(df_sorted)))
    ax2.set_xticklabels(df_sorted['name'], rotation=45, ha='right')

    ax2.set_ylim(0, 1)

    # Add values on top of bar
    for bar in ax2.patches:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)



    # Optimize layout to ensure labels do not overlap or get truncated
    plt.tight_layout()

    # Save figure
    plt.savefig(OUT_PATH, dpi=300)

if __name__ == '__main__':

    # Load parquet
    path_to_pq = Path('exports', 'model_evaluation', 'benchmarks.parquet')
    data = pd.read_parquet(path=path_to_pq)


    data = data[['name', 'train_loss_history', 'val_loss_history', 'final_test_loss', 'best_val_loss']]



    seq_models = ['SampleExt1', 'SampleExt3', 'SampleExt5', 'SampleExt7']
    seq_data = data[data['name'].isin(seq_models)]
    seq_data['name'] = pd.Categorical(seq_data['name'], categories = seq_models, ordered = True)


    lead_models = ['Lead-5', 'Lead0', 'Lead5', 'Lead10', 'Lead20']
    lead_data = data[data['name'].isin(lead_models)]
    lead_data['name'] = pd.Categorical(lead_data['name'], categories = lead_models, ordered = True)


    plot_train_eval(lead_data)