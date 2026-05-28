# Internal

# External
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def produce_plots(df):
    # 1. Flatten the list-structured columns into a long-form DataFrame
    rows = []
    for _, row in df.iterrows():
        name = row['name']
        dates = pd.to_datetime(row['dates'])
        recalls = row['recall_history']  # Matches your custom column slice
        precisions = row['precision_history']
        
        for d, r, p in zip(dates, recalls, precisions):
            rows.append({'name': name, 'date': d, 'recall': r, 'precision': p})
            
    long_df = pd.DataFrame(rows)
    
    # Extract unique ordered models from the categorical configuration to maintain legend sequence
    models = df['name'].cat.categories if isinstance(df['name'].dtype, pd.CategoricalDtype) else long_df['name'].unique()
    
    # 2. Exclude NaNs and compute a true 10-day calendar rolling average per model
    processed_models = []
    for model in models:
        model_data = long_df[long_df['name'] == model].sort_values('date').copy()
        
        # Recall rolling average
        model_recall = model_data.dropna(subset=['recall']).set_index('date')
        model_recall['recall_rolling'] = model_recall['recall'].rolling('10D').mean()
        
        # Precision rolling average
        model_precision = model_data.dropna(subset=['precision']).set_index('date')
        model_precision['precision_rolling'] = model_precision['precision'].rolling('10D').mean()
        
        # Combine the results cleanly for the current model
        model_combined = pd.DataFrame(index=model_data['date'].unique())
        model_combined = model_combined.join(model_recall['recall_rolling'], how='left')
        model_combined = model_combined.join(model_precision['precision_rolling'], how='left')
        model_combined['name'] = model
        model_combined = model_combined.reset_index().rename(columns={'index': 'date'})
        
        processed_models.append(model_combined)
        
    # Reassemble back to a single DataFrame and re-apply categorical ordering for Seaborn plotting
    plot_df = pd.concat(processed_models, ignore_index=True)
    plot_df['name'] = pd.Categorical(plot_df['name'], categories=models, ordered=True)
    
    # 3. Create the stacked layout with Seaborn
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    
    # Top Subplot: Recall
    sns.lineplot(
        data=plot_df, 
        x='date', 
        y='recall_rolling', 
        hue='name', 
        ax=axs[0], 
        linewidth=2.5
    )
    axs[0].set_title('10-Day Rolling Window Average of Recall by Model', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Recall (10-Day Rolling Average)', fontsize=12)
    axs[0].set_xlabel('')  # Clear top X-label since timelines are synchronized
    axs[0].get_legend().set_title('Model')
    
    # Bottom Subplot: Precision
    sns.lineplot(
        data=plot_df, 
        x='date', 
        y='precision_rolling', 
        hue='name', 
        ax=axs[1], 
        linewidth=2.5
    )
    axs[1].set_title('10-Day Rolling Window Average of Precision by Model', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Precision (10-Day Rolling Average)', fontsize=12)
    axs[1].set_xlabel('Date', fontsize=12)
    axs[1].get_legend().set_title('Model')
    
    # Format global components and save figure
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = Path('exports', 'model_evaluation', 'rolling_evaluation_metrics.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == '__main__':

    # Load parquet
    path_to_pq = Path('exports', 'model_evaluation', 'benchmarks.parquet')
    data = pd.read_parquet(path=path_to_pq)

    samp_models = ['SampleExt3'] #['SampleExt1', 'SampleExt3', 'SampleExt5', 'SampleExt7']
    samp_data = data[data['name'].isin(samp_models)]
    samp_data['name'] = pd.Categorical(samp_data['name'], categories = samp_models, ordered = True)


    lead_models = ['Lead-5', 'Lead0', 'Lead5', 'Lead10', 'Lead20']
    lead_data = data[data['name'].isin(lead_models)]
    lead_data['name'] = pd.Categorical(lead_data['name'], categories = lead_models, ordered = True)


    columns = ['name', 'dates', 'recall_history', 'precision_history']
    
    data = samp_data
    data = data[columns]

    produce_plots(data)