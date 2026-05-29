"""
Evaluate performance metrics of multiple models accross disciplines with Temperature Scaling calibration
"""

# Internal


# External
import numpy as np
from pathlib import Path
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns

out_path = Path('exports', 'model_evaluation', 'seq_lift.png')

def plot_aggregate_captured_fire_ratio(df_runs: pd.DataFrame, out_path : Path):
    """
    Plots the Spatially Aggregated Captured-Fire Ratio using Seaborn.
    Collapses the temporal dimension first by calculating the accumulated expected
    risk map vs. the cumulative annual burn map.
    """
    # Set standard Seaborn styling context
    sns.set_theme(style="whitegrid")
    
    # We will accumulate tidy data for Seaborn lineplot
    plot_data_list = []
    
    for idx, row in df_runs.iterrows():
        model_name = row['name']

        # --- THE FIX: Force extraction into flat 1D NumPy arrays ---
        spatial_expected_risk = np.asarray(row['yearly_exp_risk']).squeeze().flatten()
        spatial_actual_burns = np.asarray(row['yearly_agg_burn']).squeeze().flatten()

        # Debugging step: Let's make sure we actually have flat pixel arrays
        if spatial_expected_risk.ndim != 1 or len(spatial_expected_risk) == 0:
            print(f"Warning: Unexpected data shape for {model_name}. "
                  f"Risk shape: {spatial_expected_risk.shape}, Burn shape: {spatial_actual_burns.shape}")
            continue

        # 3. Sort pixels by cumulative annual risk in descending order
        sort_idx = np.argsort(spatial_expected_risk)[::-1]
        sorted_actual_burns = spatial_actual_burns[sort_idx]
        
        # 4. Compute cumulative spatial fires caught
        cum_fires = np.cumsum(sorted_actual_burns)
        total_fires = cum_fires[-1] if len(cum_fires) > 0 else 0
        
        if total_fires == 0 or np.isnan(total_fires):
            print(f"Warning: No actual fires found in the aggregated map for {model_name} (Total: {total_fires}). Skipping.")
            continue
            
        # 5. Scale to percentages
        y_captured_fraction = (cum_fires / total_fires) * 100
        x_area_fraction = np.linspace(0, 100, len(cum_fires))
        
        # Downsample data points for cleaner plotting memory footprint (e.g., every 1000th pixel or 500 points total)
        # Highly recommended if your satellite images have millions of pixels
        downsample_factor = max(1, len(x_area_fraction) // 1000)
        
        # Build temp dataframe for this model execution
        model_df = pd.DataFrame({
            'Area Monitored (%)': x_area_fraction[::downsample_factor],
            'Fires Captured (%)': y_captured_fraction[::downsample_factor],
            'Model': model_name
        })
        plot_data_list.append(model_df)
        
        # Print operational spatial metrics to the console
        print(f"\n--- Spatially Aggregated Metrics [{model_name}] ---")
        for pct in [1, 5, 10, 30]: 
            idx_pct = int((pct / 100) * (len(x_area_fraction) - 1))
            print(f"Top {pct}% highest-risk land area accounts for {y_captured_fraction[idx_pct]:.2f}% of the year's total burned pixels.")

    if not plot_data_list:
        print("Error: No valid model metrics could be calculated. Plotting aborted.")
        return

    # Combine all individual model runs into one tidy dataframe
    tidy_plot_df = pd.concat(plot_data_list, ignore_index=True)

    # Initialize the matplotlib figure figure container
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 6. Plot using Seaborn lineplot
    # Pass 'name' categorical order directly to hue_order so it maintains your structure
    hue_order = df_runs['name'].cat.categories if isinstance(df_runs['name'].dtype, pd.CategoricalDtype) else None
    
    sns.lineplot(
        data=tidy_plot_df,
        x='Area Monitored (%)',
        y='Fires Captured (%)',
        hue='Model',
        hue_order=hue_order,
        linewidth=2.5,
        ax=ax
    )
    
    # Reference baseline line
    ax.plot([0, 100], [0, 100], color='grey', linestyle='--', label='Random Spatial Baseline', alpha=0.8)
    
    # Adjust formatting via matplotlib axis layer
    ax.set_title('Spatially Aggregated Captured-Fire Ratio (Annual Risk)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('% of Geographic Land Area Monitored\n(Sorted by Annual Accumulated Risk)', fontsize=12, labelpad=10)
    ax.set_ylabel('% of Total Annual Burned Pixels Captured', fontsize=12, labelpad=10)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Refresh/combine custom legends cleanly
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='lower right', fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi = 300)

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

    plot_aggregate_captured_fire_ratio(seq_data, out_path)