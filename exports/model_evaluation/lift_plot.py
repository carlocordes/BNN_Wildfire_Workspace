"""
Evaluate performance metrics of multiple models across disciplines with Temperature Scaling calibration
"""

# External
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

out_path = Path('exports', 'model_evaluation', 'seq_lift.png')

def plot_aggregate_captured_fire_ratio(df_runs: pd.DataFrame, out_path: Path):
    """
    Plots the Spatially Aggregated Captured-Fire Ratio using Seaborn.
    Calculates and appends Gini Coefficients directly to the model metrics and plot legend.
    """
    # Set standard Seaborn styling context
    sns.set_theme(style="whitegrid")
    
    # We will accumulate tidy data for Seaborn lineplot
    plot_data_list = []
    
    # Dictionary to map model names to their updated names with Gini info for the legend
    legend_labels_map = {}
    
    for idx, row in df_runs.iterrows():
        model_name = row['name']

        # Force extraction into flat 1D NumPy arrays
        spatial_expected_risk = np.asarray(row['yearly_exp_risk']).squeeze().flatten()
        spatial_actual_burns = np.asarray(row['yearly_agg_burn']).squeeze().flatten()

        # Debugging step: Make sure we actually have flat pixel arrays
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
            
        # 5. Scale to fractions and percentages
        y_captured_fraction = cum_fires / total_fires
        x_area_fraction = np.linspace(0, 1, len(cum_fires))
        
        # --- GINI COEFFICIENT CALCULATION ---
        # Gini is defined as: (Area under the model curve - Area under random baseline) / (Area under perfect model - Area under random baseline)
        # Area under random baseline = 0.5
        # Max theoretical area under a perfect model with severe class imbalance approaches 1.0
        area_under_model = np.trapezoid(y_captured_fraction, x_area_fraction)
        gini_coefficient = (area_under_model - 0.5) / 0.5
        
        # Create an updated label string to house our Gini metric inside the visual legend wrapper
        legend_model_label = f"{model_name} (Gini: {gini_coefficient:.3f})"
        legend_labels_map[model_name] = legend_model_label

        # Downsample data points for cleaner plotting memory footprint
        downsample_factor = max(1, len(x_area_fraction) // 1000)
        
        # Build temp dataframe for this model execution
        model_df = pd.DataFrame({
            'Area Monitored (%)': x_area_fraction[::downsample_factor] * 100,
            'Fires Captured (%)': y_captured_fraction[::downsample_factor] * 100,
            'Model': legend_model_label  # Assigning the labeled string directly for Seaborn grouping
        })
        plot_data_list.append(model_df)
        
        # Print operational spatial metrics alongside Gini to the console
        print(f"\n--- Spatially Aggregated Metrics [{model_name}] ---")
        print(f"Calculated Gini Coefficient: {gini_coefficient:.4f}")
        for pct in [1, 5, 10, 30]: 
            idx_pct = int((pct / 100) * (len(x_area_fraction) - 1))
            print(f"Top {pct}% highest-risk land area accounts for {y_captured_fraction[idx_pct]*100:.2f}% of the year's total burned pixels.")

    if not plot_data_list:
        print("Error: No valid model metrics could be calculated. Plotting aborted.")
        return

    # Combine all individual model runs into one tidy dataframe
    tidy_plot_df = pd.concat(plot_data_list, ignore_index=True)

    # Initialize the matplotlib figure container
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 6. Plot using Seaborn lineplot
    # Reconstruct the categorical sorting order using our new modified strings
    hue_order = [legend_labels_map[cat] for cat in df_runs['name'].cat.categories if cat in legend_labels_map]
    
    sns.lineplot(
        data=tidy_plot_df,
        x='Area Monitored (%)',
        y='Fires Captured (%)',
        hue='Model',
        hue_order=hue_order if hue_order else None,
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
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved Gini-integrated lift plot to: {out_path}")


if __name__ == '__main__':

    # Load parquet
    path_to_pq = Path('exports', 'model_evaluation', 'benchmarks.parquet')
    df = pd.read_parquet(path=path_to_pq)

    # Select relevant cols
    df = df[['name', 'yearly_exp_risk', 'yearly_agg_burn']].copy()

    seq_models = ['SampleExt1', 'SampleExt3', 'SampleExt5', 'SampleExt7']
    seq_data = df[df['name'].isin(seq_models)].copy()
    seq_data['name'] = pd.Categorical(seq_data['name'], categories=seq_models, ordered=True)
    seq_data = seq_data.sort_values('name')

    lead_models = ['Lead-5', 'Lead0', 'Lead5', 'Lead10', 'Lead20']
    lead_data = df[df['name'].isin(lead_models)].copy()
    lead_data['name'] = pd.Categorical(lead_data['name'], categories=lead_models, ordered=True)
    lead_data = lead_data.sort_values('name')

    # Execute plotting sequence
    plot_aggregate_captured_fire_ratio(lead_data, out_path)