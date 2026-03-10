import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import sys
import numpy as np
import ast
from collections.abc import Iterable
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import os
import re
import pickle
from tqdm import tqdm
from matplotlib.patches import Patch

############################################
#converts conversion dataframe to a drug usage matrix
############################################

def build_conversion_drug_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a dataframe with columns
        starting_cl | target_cl | path
    into a binary drug-usage matrix plus a `conversion` label.
    """
    df = df.copy()
    # This function assumes the 'conversion' column already exists in the input df
    def parse_path(value) -> set[str]:
        """Return the set of drug names found in `value`."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return set()
        if isinstance(value, str):
            try: value = ast.literal_eval(value)
            except (ValueError, SyntaxError): value = [value]
        if not isinstance(value, Iterable): value = [value]
        stack = list(value)
        drugs = set()
        while stack:
            item = stack.pop()
            if isinstance(item, str):
                try: item = ast.literal_eval(item)
                except Exception:
                    drugs.add(item.strip())
                    continue
            if isinstance(item, tuple):
                if item: drugs.add(str(item[0]).strip())
            elif isinstance(item, Iterable): stack.extend(item)
            else:
                try: drugs.add(str(item[0]).strip())
                except Exception: pass
        return drugs
    # ----------------------------

    drug_sets = df["drug_sequence"].apply(parse_path)
    all_drugs = sorted({drug for s in drug_sets for drug in s})
    drug_matrix = pd.DataFrame(0, index=df.index, columns=all_drugs, dtype=int)
    for idx, drugs in drug_sets.items():
        if drugs: drug_matrix.loc[idx, list(drugs)] = 1
    return pd.concat([df[["conversion"]], drug_matrix], axis=1)

############################################
#Heirarchical clustering of drug usage matrix
############################################

def _cluster_single_conversion(
    sub_df: pd.DataFrame, drug_cols: list[str], metric: str = "jaccard",
    distance_threshold: float = 0.5
) -> pd.Series:
    """Helper to cluster paths for a single conversion."""
    if len(sub_df) < 2:
        return pd.Series([0], index=sub_df.index, name="cluster")
    X = sub_df[drug_cols].astype(bool).to_numpy()
    dist = pairwise_distances(X, metric=metric)
    model = AgglomerativeClustering(
        metric="precomputed", linkage="average", n_clusters=None,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(dist)
    return pd.Series(labels, index=sub_df.index, name="cluster")

def cluster_all_conversions(
    df_bin: pd.DataFrame, conversion_col: str = "conversion",
    metric: str = "jaccard", distance_threshold: float = 0.5
):
    """
    Calculates per-conversion summary stats including the
    Coefficient of Variation (CV) of cluster sizes for uniformity.
    """
    drug_cols = [c for c in df_bin.columns if c != conversion_col]
    summaries = []
    cluster_labels = pd.Series(dtype=int, name="cluster")

    for conv, sub_df in df_bin.groupby(conversion_col):
        labels = _cluster_single_conversion(
            sub_df, drug_cols, metric=metric, distance_threshold=distance_threshold
        )
        cluster_labels = pd.concat([cluster_labels, labels])
        
        n_c = labels.nunique()
        cluster_sizes = labels.value_counts()
        
        if len(cluster_sizes) > 1:
            size_std = cluster_sizes.std()
            size_mean = cluster_sizes.mean()
            size_cv = (size_std / size_mean) if size_mean > 0 else 0.0
        else:
            size_cv = 0.0

        summaries.append({
            "conversion": conv,
            "n_clusters": int(n_c),
            "cv_cluster_size": size_cv,
        })

    summary_df = pd.DataFrame(summaries)
    df_with_clusters = df_bin.join(cluster_labels)
    return summary_df, df_with_clusters


############################################
# Plotting per-conversion trends (5 metrics)
############################################

def plot_per_conversion_trends(
    file_paths: list[str],
    param_name: str,
    param_regex: str,
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6), 
    show: bool = True,
):
    """
    Analyzes and plots 5 per-conversion metrics as individual figures 
    across a parameter sweep.
    """
    all_results = []

    for path in tqdm(sorted(file_paths), desc="Analyzing files"):
        match = re.search(param_regex, os.path.basename(path))
        if not match: continue
        param_value = float(match.group(1))

        try:
            df = pd.read_pickle(path)
        except Exception as e:
            print(f"Error reading {path}: {e}. Skipping.")
            continue
        
        df = df[df['covered_threshold'] == True].copy()
        if df.empty: continue

        df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)
            
        df['path_length'] = df['drug_sequence'].apply(len)
        avg_path_length_df = df.groupby('conversion')['path_length'].mean().reset_index()
        avg_path_length_df.rename(columns={'path_length': 'avg_path_length'}, inplace=True)

        df['progress_perc'] = (df['starting_distance'] - df['final_distance']) / (df['starting_distance'] + 1e-9)
        best_progress_df = df.groupby('conversion')['progress_perc'].max().reset_index()
        best_progress_df.rename(columns={'progress_perc': 'best_path_progress'}, inplace=True)

        paths_per_conv = df['conversion'].value_counts().reset_index()
        paths_per_conv.columns = ['conversion', 'num_paths']
        
        df_bin = build_conversion_drug_matrix(df)
        summary_df, _ = cluster_all_conversions(df_bin)
        
        combined_metrics = pd.merge(paths_per_conv, summary_df, on='conversion')
        combined_metrics = pd.merge(combined_metrics, avg_path_length_df, on='conversion')
        combined_metrics = pd.merge(combined_metrics, best_progress_df, on='conversion')
        combined_metrics[param_name] = param_value
        all_results.append(combined_metrics)

    if not all_results:
        print("No data was processed. Exiting.")
        return

    full_results_df = pd.concat(all_results, ignore_index=True)
    top_conversions = full_results_df.groupby('conversion')['num_paths'].sum().nlargest(top_n).index
    plot_df = full_results_df[full_results_df['conversion'].isin(top_conversions)]

    if plot_df.empty:
        print("No data available to plot after filtering for top N conversions.")
        return

    # --- Plotting Individual Graphs ---
    
    # List of (Column, Title, Y-Label)
    metrics_to_plot = [
        ('num_paths', 'Total Successful Paths', 'Path Count'),
        ('n_clusters', 'Solution Uniqueness', 'Cluster Count'),
        ('cv_cluster_size', 'Solution Uniformity', 'Coefficient of Variation (CV)'),
        ('avg_path_length', 'Average Path Length', 'Average Path Length'),
        ('best_path_progress', 'Best Path Progress', 'Best Progress (%)')
    ]

    for col_y, title, label_y in metrics_to_plot:
        plt.figure(figsize=figsize)
        
        sns.lineplot(
            data=plot_df, 
            x=param_name, 
            y=col_y, 
            hue='conversion', 
            marker='o'
        )
        
        plt.title(f'{title} vs. {param_name} (Top {top_n} Conversions)', fontsize=14)
        plt.ylabel(label_y)
        plt.xlabel(param_name)
        
        # Move the legend outside to the right
        plt.legend(title='Conversion', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Optional: Save each plot automatically
        # plt.savefig(f"{col_y}_vs_{param_name}.pdf", bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close() # Close to save memory if not showing

    return full_results_df

############################################
# Pairwise Parameter Comparison (6 metrics)
############################################

def plot_pairwise_parameter_comparison(
    file_paths: list[str],
    param_name: str,
    param_regex: str,
    figsize: tuple[int, int] = (8, 6), # Standard size for individual heatmaps
    show: bool = True,
):
    """
    Analyzes and plots 6 pairwise comparison matrices as individual figures.
    """
    all_results = []
    conversions_per_param = {}

    for path in tqdm(sorted(file_paths), desc="Analyzing files"):
        match = re.search(param_regex, os.path.basename(path))
        if not match: continue
        param_value = float(match.group(1))

        try:
            df = pd.read_pickle(path)
        except Exception as e:
            print(f"Error reading {path}: {e}. Skipping.")
            continue
        
        df = df[df['covered_threshold'] == True].copy()
        if df.empty:
            conversions_per_param[param_value] = 0
            continue
            
        df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)
        conversions_per_param[param_value] = df['conversion'].nunique()

        df['path_length'] = df['drug_sequence'].apply(len)
        avg_path_length_df = df.groupby('conversion')['path_length'].mean().reset_index()
        avg_path_length_df.rename(columns={'path_length': 'avg_path_length'}, inplace=True)
            
        df['progress_perc'] = (df['starting_distance'] - df['final_distance']) / (df['starting_distance'] + 1e-9)
        best_progress_df = df.groupby('conversion')['progress_perc'].max().reset_index()
        best_progress_df.rename(columns={'progress_perc': 'best_path_progress'}, inplace=True)

        paths_per_conv = df['conversion'].value_counts().reset_index()
        paths_per_conv.columns = ['conversion', 'num_paths']
        
        df_bin = build_conversion_drug_matrix(df)
        summary_df, _ = cluster_all_conversions(df_bin)
        
        combined_metrics = pd.merge(paths_per_conv, summary_df, on='conversion')
        combined_metrics = pd.merge(combined_metrics, avg_path_length_df, on='conversion')
        combined_metrics = pd.merge(combined_metrics, best_progress_df, on='conversion')
        combined_metrics[param_name] = param_value
        all_results.append(combined_metrics)

    if not all_results:
        print("No data processed.")
        return

    full_results_df = pd.concat(all_results, ignore_index=True)
    param_values = sorted(full_results_df[param_name].unique())
    n_params = len(param_values)

    # Initialize matrices
    n_conv_matrix = np.zeros((n_params, n_params))
    paths_matrix = np.zeros((n_params, n_params))
    clusters_matrix = np.zeros((n_params, n_params))
    cv_matrix = np.zeros((n_params, n_params))
    length_matrix = np.zeros((n_params, n_params))
    best_progress_matrix = np.zeros((n_params, n_params))

    for i, p1 in enumerate(param_values):
        for j, p2 in enumerate(param_values):
            if i == j: continue
            n_conv_matrix[i, j] = conversions_per_param[p1] - conversions_per_param[p2]
            df1 = full_results_df[full_results_df[param_name] == p1]
            df2 = full_results_df[full_results_df[param_name] == p2]
            
            outer_merged_df = pd.merge(df1, df2, on='conversion', how='outer', suffixes=('_p1', '_p2')).fillna(0)
            if not outer_merged_df.empty:
                total_conversions = len(outer_merged_df)
                paths_matrix[i, j] = (outer_merged_df['num_paths_p1'] > outer_merged_df['num_paths_p2']).sum() / total_conversions * 100
                best_progress_matrix[i, j] = (outer_merged_df['best_path_progress_p1'] > outer_merged_df['best_path_progress_p2']).sum() / total_conversions * 100

            inner_merged_df = pd.merge(df1, df2, on='conversion', how='inner', suffixes=('_p1', '_p2'))
            if not inner_merged_df.empty:
                total_common = len(inner_merged_df)
                clusters_matrix[i, j] = (inner_merged_df['n_clusters_p1'] > inner_merged_df['n_clusters_p2']).sum() / total_common * 100
                cv_matrix[i, j] = (inner_merged_df['cv_cluster_size_p1'] < inner_merged_df['cv_cluster_size_p2']).sum() / total_common * 100
                length_matrix[i, j] = (inner_merged_df['avg_path_length_p1'] < inner_merged_df['avg_path_length_p2']).sum() / total_common * 100

    # Configuration for individual plots
    param_labels = [f"{p:.0f}" for p in param_values]
    
    plot_configs = [
        (paths_matrix, "% where Row > Col\n(Total Successful Paths)", "YlOrRd", ".1f"),
        (n_conv_matrix, "Row - Col\n(Unique Task Coverage)", "vlag", ".0f"),
        (clusters_matrix, "% where Row > Col\n(Solution Uniqueness - Common)", "YlOrRd", ".1f"),
        (cv_matrix, "% where Row < Col\n(Solution Uniformity CV - Common)", "YlOrRd", ".1f"),
        (length_matrix, "% where Row < Col\n(Average Path Length - Common)", "YlOrRd", ".1f"),
        (best_progress_matrix, "% where Row > Col\n(Best Path Progress)", "YlOrRd", ".1f")
    ]

    for matrix, title, cmap, fmt in plot_configs:
        plt.figure(figsize=figsize)
        sns.heatmap(
            matrix, 
            annot=True, 
            fmt=fmt, 
            cmap=cmap, 
            xticklabels=param_labels, 
            yticklabels=param_labels,
            center=0 if cmap=="vlag" else None
        )
        plt.title(title, fontsize=14)
        plt.xlabel(param_name)
        plt.ylabel(param_name)
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()

    return full_results_df


############################################
# Algorithm Efficiency Comparison (7 metrics)
############################################

def plot_algorithm_efficiency(
    file_paths: list[str],
    times_csv_path: str,
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
):
    """
    Generates individual scatter plots comparing algorithm efficiency across 
    7 metrics relative to computation time.
    """
    all_metrics = []
    
    try:
        times_df = pd.read_csv(times_csv_path)
        times_df.rename(columns={'time': 'time_seconds'}, inplace=True)
    except FileNotFoundError:
        print(f"Error: Timing file not found at '{times_csv_path}'")
        return

    for path in tqdm(file_paths, desc="Calculating metrics"):
        filename = os.path.basename(path)
        k_val, n_paths_val, strategy = None, None, None
        
        if 'tree' in filename:
            strategy = 'k_path_tree'
            match = re.search(r'_k(\d+)_', filename)
            if match: k_val = float(match.group(1))
        elif 'beam' in filename:
            strategy = 'k_path_beam'
            match = re.search(r'_k(\d+)_', filename)
            if match: k_val = float(match.group(1))
        elif 'probabilistic' in filename:
            strategy = 'prob'
            match = re.search(r'paths(\d+)_', filename)
            if match: n_paths_val = float(match.group(1))
        
        if not strategy: continue

        try:
            df = pd.read_pickle(path)
        except Exception as e:
            print(f"Error reading {filename}: {e}. Skipping.")
            continue
            
        df = df[df['covered_threshold'] == True].copy()
        if df.empty: continue

        df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)

        num_conversions_found = df['conversion'].nunique()
        total_paths = len(df)
        
        df['path_length'] = df['drug_sequence'].apply(len)
        avg_path_length_df = df.groupby('conversion')['path_length'].mean().reset_index()
        median_path_length = avg_path_length_df['path_length'].median()

        df['progress_perc'] = (df['starting_distance'] - df['final_distance']) / (df['starting_distance'] + 1e-9)
        best_progress_df = df.groupby('conversion')['progress_perc'].max().reset_index()
        median_best_progress = best_progress_df['progress_perc'].median()

        overall_best_progress = df['progress_perc'].max() if not df.empty else 0
        
        df_bin = build_conversion_drug_matrix(df)
        summary_df, _ = cluster_all_conversions(df_bin)
        
        median_uniqueness = summary_df['n_clusters'].median()
        median_uniformity_cv = summary_df['cv_cluster_size'].median()

        all_metrics.append({
            'strategy': strategy,
            'k': k_val,
            'n_paths': n_paths_val,
            'total_paths': total_paths,
            'num_conversions_found': num_conversions_found,
            'median_uniqueness': median_uniqueness,
            'median_uniformity_cv': median_uniformity_cv,
            'median_avg_path_length': median_path_length,
            'median_best_progress': median_best_progress,
            'overall_best_progress': overall_best_progress,
        })

    if not all_metrics:
        print("No valid data processed.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    comparison_df = pd.merge(metrics_df, times_df, on=['strategy', 'k', 'n_paths'], how='inner')
    
    algo_map = {'k_path_tree': 'Tree Search', 'k_path_beam': 'Beam Search', 'prob': 'Probabilistic'}
    comparison_df['algorithm'] = comparison_df['strategy'].map(algo_map)

    # Define the 7 metrics for individual plots
    plot_configs = [
        ('total_paths', 'Total Successful Paths', 'Path Count'),
        ('num_conversions_found', 'Unique Task Coverage', 'Unique Conversions Count'),
        ('median_uniqueness', 'Median Solution Uniqueness', 'Median Cluster Count'),
        ('median_uniformity_cv', 'Median Solution Uniformity', 'Median CV'),
        ('median_avg_path_length', 'Median Average Path Length', 'Median Number of Drugs'),
        ('median_best_progress', 'Median Best Path Progress', 'Median Progress (%)'),
        ('overall_best_progress', 'Overall Top 1 Path Progress', 'Max Progress (%)')
    ]

    for col_y, title, label_y in plot_configs:
        plt.figure(figsize=figsize)
        
        sns.scatterplot(
            data=comparison_df, 
            x='time_seconds', 
            y=col_y, 
            hue='algorithm', 
            style='algorithm', 
            s=120
        )
        
        plt.title(f'{title} vs. Computation Time', fontsize=14)
        plt.ylabel(label_y)
        plt.xlabel('Computation Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.close()

    return comparison_df


def _calculate_and_summarize_metrics(df: pd.DataFrame) -> pd.Series:
    """Helper function to calculate the 7 key comparison metrics for a dataframe."""
    df_filtered = df[(df['starting_distance']-df['final_distance'])/df['starting_distance'] > 0.5].copy()
    
    if df_filtered.empty:
        return pd.Series({
            "Total Successful Paths": 0,
            "Number of Unique Conversions": 0,
            "Median Solution Uniqueness": 0,
            "Median Solution Uniformity (CV)": 0,
            "Median Path Length": 0,
            "Median Best Path Progress (%)": 0,
            "Overall Top 1 Path Progress (%)": 0
        })

    df_filtered["conversion"] = df_filtered["starting_cl"].astype(str) + ":" + df_filtered["target_cl"].astype(str)
    
    # Calculate path length and progress percentage
    df_filtered['path_length'] = df_filtered['drug_sequence'].apply(len)
    df_filtered['progress_perc'] = (df_filtered['starting_distance'] - df_filtered['final_distance']) / (df_filtered['starting_distance'] + 1e-9) * 100

    # Metric 1: Total successful paths
    total_paths = len(df_filtered)
    
    # Metric 2: Number of unique conversions
    num_conversions = df_filtered['conversion'].nunique()
    
    # For clustering metrics
    df_bin = build_conversion_drug_matrix(df_filtered)
    summary_df, _ = cluster_all_conversions(df_bin)
    
    # Metric 3: Median solution uniqueness
    median_uniqueness = summary_df['n_clusters'].median()
    
    # Metric 4: Median solution uniformity
    median_uniformity_cv = summary_df['cv_cluster_size'].median()
    
    # Metric 5: Median path length
    median_path_length = df_filtered.groupby('conversion')['path_length'].mean().median()
    
    # Metric 6: Median best path progress
    median_best_progress = df_filtered.groupby('conversion')['progress_perc'].max().median()
    
    # Metric 7: Overall top 1 path progress
    overall_top_progress = df_filtered['progress_perc'].max()

    return pd.Series({
        "Total Successful Paths": total_paths,
        "Number of Unique Conversions": num_conversions,
        "Median Solution Uniqueness": median_uniqueness,
        "Median Solution Uniformity (CV)": median_uniformity_cv,
        "Median Path Length": median_path_length,
        "Median Best Path Progress (%)": median_best_progress,
        "Overall Top 1 Path Progress (%)": overall_top_progress
    })


def plot_approach_comparison(
    new_approach_file: str,
    baseline_file: str,
    target_cell: str,
    new_approach_name: str = "New Approach",
    baseline_name: str = "Baseline",
    figsize: tuple[int, int] = (15, 18) # Adjusted figsize for subplot layout
):
    """
    Compares two search approaches on 7 key metrics for a specific target cell.
    
    Generates a summary table and a multi-plot bar chart for direct comparison.
    """
    try:
        df_new = pd.read_pickle(new_approach_file)
        df_baseline = pd.read_pickle(baseline_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Please check paths.\n{e}")
        return

    # Filter baseline to only the target cell for a fair comparison
    df_baseline_filtered = df_baseline[df_baseline['target_cl'] == target_cell].copy()
    
    # Also filter the new approach data to be certain
    df_new_filtered = df_new[df_new['target_cl'] == target_cell].copy()

    # Calculate metrics for both dataframes
    metrics_new = _calculate_and_summarize_metrics(df_new_filtered)
    metrics_baseline = _calculate_and_summarize_metrics(df_baseline_filtered)
    
    # Create comparison dataframe and print it
    comparison_df = pd.DataFrame({
        new_approach_name: metrics_new,
        baseline_name: metrics_baseline
    })
    
    print(f"--- Performance Comparison for Target Cell: {target_cell} ---")
    print(comparison_df.round(3))
    print("---------------------------------------------------------")

    # MODIFIED: Plotting each metric on its own subplot
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
    
    fig.suptitle(f'Comparison: "{new_approach_name}" vs. "{baseline_name}" for Target {target_cell}', fontsize=20)

    # Define distinct colors for the bars
    colors = ['#1f77b4', '#ff7f0e'] # Matplotlib's default blue and orange

    for i, metric in enumerate(comparison_df.index):
        ax = axes[i]
        # Plot data for the current metric with specified colors
        comparison_df.loc[metric].plot(kind='bar', ax=ax, width=0.7, rot=0, color=colors)
        
        ax.set_title(metric, fontsize=14)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add annotations
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=10)
        
        # Make y-axis limits a bit larger to accommodate labels
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15 if ax.get_ylim()[1] > 0 else 1)


    # Turn off any unused subplots
    for i in range(len(comparison_df.index), len(axes)):
        axes[i].axis('off')

    # Add a single, clear legend for the entire figure
    legend_elements = [
        Patch(facecolor=colors[0], label=comparison_df.columns[0]),
        Patch(facecolor=colors[1], label=comparison_df.columns[1])
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12, title="Approach")


    plt.tight_layout(rect=[0, 0, 0.9, 0.96]) # Adjust layout to make room for suptitle and legend
    plt.show()

    return comparison_df


############################################
# NEW: Summary Table Generation
############################################

# --- Customizable Thresholds ---
# Define the percentage thresholds for what constitutes a notable improvement or decline.
# Format: [Level 1, Level 2, Level 3] -> corresponds to [✓/✗, ✓✓/✗✗, ✓✓✓/✗✗✗]
PERFORMANCE_THRESHOLDS = [0.05, 0.10, 0.15] # Represents 5%, 10%, 15%

def _get_rating(value, baseline, higher_is_better=True):
    """Assigns a tick/cross rating based on percentage difference from baseline."""
    # Handle cases where the baseline is zero or near-zero
    if abs(baseline) < 1e-9:
        if value > 1e-9: return '✅✅✅'
        if value < -1e-9: return '❌❌❌'
        return '-'
    
    diff = (value - baseline) / abs(baseline)

    # Return '-' only if the difference is effectively zero
    if abs(diff) < 1e-9:
        return '-'
    
    if not higher_is_better:
        diff *= -1 # Invert difference for metrics where lower is better

    # Positive ratings (better performance)
    if diff > 0:
        if diff > PERFORMANCE_THRESHOLDS[1]: # e.g., > 10%
            return '✅✅✅'
        if diff > PERFORMANCE_THRESHOLDS[0]: # e.g., (5%, 10%]
            return '✅✅'
        return '✅' # e.g., (0%, 5%]
    
    # Negative ratings (worse performance)
    else: # diff < 0
        abs_diff = abs(diff)
        if abs_diff > PERFORMANCE_THRESHOLDS[1]: # e.g., > 10% worse
            return '❌❌❌'
        if abs_diff > PERFORMANCE_THRESHOLDS[0]: # e.g., (5%, 10%] worse
            return '❌❌'
        return '❌' # e.g., (0%, 5%] worse

def generate_summary_tables(
    algo_eff_files: list[str],
    times_csv_path: str,
    new_approach_file: str,
    baseline_file: str,
    target_cell: str,
    new_approach_name: str,
    baseline_name: str,
):
    """
    Generates and prints two summary tables with tick/cross performance ratings.
    """
    # --- Table 1: Algorithm Efficiency Comparison ---
    print("--- Generating Table 1: Algorithm Efficiency Summary (Time-Capped) ---")
    try:
        # 1a. Calculate metrics for all runs
        all_metrics = []
        for path in tqdm(algo_eff_files, desc="Calculating algo metrics"):
            df = pd.read_pickle(path)
            metrics = _calculate_and_summarize_metrics(df)
            
            filename = os.path.basename(path)
            strategy, k_val, n_paths_val = None, None, None
            if 'tree' in filename:
                strategy = 'k_path_tree'; k_val = float(re.search(r'_k(\d+)_', filename).group(1))
            elif 'beam' in filename:
                strategy = 'k_path_beam'; k_val = float(re.search(r'_k(\d+)_', filename).group(1))
            elif 'probabilistic' in filename:
                strategy = 'prob'; n_paths_val = float(re.search(r'paths(\d+)_', filename).group(1))
            
            if strategy:
                metrics['strategy'] = strategy
                metrics['k'] = k_val
                metrics['n_paths'] = n_paths_val
                all_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(all_metrics)
        times_df = pd.read_csv(times_csv_path)
        times_df.rename(columns={'time': 'time_seconds'}, inplace=True)
        
        # 1b. Merge with timing data
        comparison_df = pd.merge(metrics_df, times_df, on=['strategy', 'k', 'n_paths'], how='inner')
        algo_map = {'k_path_tree': 'Tree Search', 'k_path_beam': 'Beam Search', 'prob': 'Probabilistic'}
        comparison_df['algorithm'] = comparison_df['strategy'].map(algo_map)

        # 2. Determine Time Cap
        max_times = comparison_df.groupby('algorithm')['time_seconds'].max()
        time_cap = max_times.min()
        print(f"Time cap set to: {time_cap:.2f} seconds (min of max times)")

        # 3. Filter and 4. Average Metrics
        filtered_df = comparison_df[comparison_df['time_seconds'] <= time_cap]
        avg_metrics = filtered_df.groupby('algorithm').mean(numeric_only=True)
        
        # 5. Calculate Ratings
        rating_table = pd.DataFrame(index=avg_metrics.columns)
        overall_avg = avg_metrics.mean()

        higher_is_better_map = {
            "Total Successful Paths": True, "Number of Unique Conversions": True,
            "Median Solution Uniqueness": True, "Median Solution Uniformity (CV)": False,
            "Median Path Length": False, "Median Best Path Progress (%)": True,
            "Overall Top 1 Path Progress (%)": True
        }

        for algo in avg_metrics.index:
            ratings = {}
            for metric, is_higher_better in higher_is_better_map.items():
                ratings[metric] = _get_rating(avg_metrics.loc[algo, metric], overall_avg[metric], is_higher_better)
            rating_table[algo] = pd.Series(ratings)
        
        print("\nTable 1: Algorithm Performance Rating (vs. Average)")
        # Drop all non-metric columns that might exist before printing
        cols_to_drop = ['k', 'n_paths', 'time_seconds', 'n_steps', 'Unnamed: 0']
        existing_cols_to_drop = [col for col in cols_to_drop if col in rating_table.index]
        print(rating_table.drop(existing_cols_to_drop))

    except Exception as e:
        print(f"Could not generate Table 1. Error: {e}")

    # --- Table 2: Head-to-Head Comparison ---
    print("\n--- Generating Table 2: Head-to-Head Summary ---")
    try:
        df_new = pd.read_pickle(new_approach_file)
        df_baseline = pd.read_pickle(baseline_file)
        
        df_baseline_filtered = df_baseline[df_baseline['target_cl'] == target_cell]
        df_new_filtered = df_new[df_new['target_cl'] == target_cell]
        
        metrics_new = _calculate_and_summarize_metrics(df_new_filtered)
        metrics_baseline = _calculate_and_summarize_metrics(df_baseline_filtered)

        rating_table_2 = pd.DataFrame(index=metrics_new.index)
        ratings = {}
        for metric, is_higher_better in higher_is_better_map.items():
            ratings[metric] = _get_rating(metrics_new[metric], metrics_baseline[metric], is_higher_better)
        
        rating_table_2[new_approach_name] = pd.Series(ratings)
        rating_table_2[baseline_name] = '-' # Baseline is always the reference
        
        print(f"\nTable 2: {new_approach_name} Rating (vs. {baseline_name})")
        print(rating_table_2)

    except Exception as e:
        print(f"Could not generate Table 2. Error: {e}")