# Functions to analyze the results of the Successful paths across all conversions 

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
# Plotting per-conversion trends (4 metrics)
############################################

def plot_per_conversion_trends(
    file_paths: list[str],
    param_name: str,
    param_regex: str,
    top_n: int = 10,
    figsize: tuple[int, int] = (14, 18),
    show: bool = True,
):
    """
    Analyzes and plots 4 per-conversion metrics, including
    average path length, across a parameter sweep.
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

        # --- FIX: Create the 'conversion' column before it is used ---
        df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)
            
        df['path_length'] = df['drug_sequence'].apply(len)
        avg_path_length_df = df.groupby('conversion')['path_length'].mean().reset_index()
        avg_path_length_df.rename(columns={'path_length': 'avg_path_length'}, inplace=True)

        paths_per_conv = df['conversion'].value_counts().reset_index()
        paths_per_conv.columns = ['conversion', 'num_paths']
        
        df_bin = build_conversion_drug_matrix(df)
        summary_df, _ = cluster_all_conversions(df_bin)
        
        combined_metrics = pd.merge(paths_per_conv, summary_df, on='conversion')
        combined_metrics = pd.merge(combined_metrics, avg_path_length_df, on='conversion')
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

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(f'Per-Conversion Performance vs. Parameter "{param_name}" (Top {top_n} Conversions)', fontsize=16)

    sns.lineplot(data=plot_df, x=param_name, y='num_paths', hue='conversion', marker='o', ax=axes[0], legend='auto')
    axes[0].set_title("Number of Successful Paths per Conversion")
    axes[0].set_ylabel("Path Count")
    axes[0].legend(title='Conversion', bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.lineplot(data=plot_df, x=param_name, y='n_clusters', hue='conversion', marker='o', ax=axes[1], legend=False)
    axes[1].set_title("Number of Unique Solutions (Uniqueness)")
    axes[1].set_ylabel("Cluster Count")

    sns.lineplot(data=plot_df, x=param_name, y='cv_cluster_size', hue='conversion', marker='o', ax=axes[2], legend=False)
    axes[2].set_title("Uniformity of Solutions (Lower CV is Better)")
    axes[2].set_ylabel("Coefficient of Variation (CV)")
    
    sns.lineplot(data=plot_df, x=param_name, y='avg_path_length', hue='conversion', marker='o', ax=axes[3], legend=False)
    axes[3].set_title("Path Efficiency (Lower is Better)")
    axes[3].set_ylabel("Average Path Length")
    axes[3].set_xlabel(f"Parameter: {param_name}")

    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    if show:
        plt.show()

    return full_results_df, fig, axes


############################################
# Pairwise Parameter Comparison (4 metrics)
############################################

def plot_pairwise_parameter_comparison(
    file_paths: list[str],
    param_name: str,
    param_regex: str,
    figsize: tuple[int, int] = (24, 5),
    show: bool = True,
):
    """
    Creates heatmaps to show pairwise "win percentages" for 4 metrics.
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
            
        # --- FIX: Create the 'conversion' column before it is used ---
        df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)
            
        df['path_length'] = df['drug_sequence'].apply(len)
        avg_path_length_df = df.groupby('conversion')['path_length'].mean().reset_index()
        avg_path_length_df.rename(columns={'path_length': 'avg_path_length'}, inplace=True)
            
        paths_per_conv = df['conversion'].value_counts().reset_index()
        paths_per_conv.columns = ['conversion', 'num_paths']
        
        df_bin = build_conversion_drug_matrix(df)
        summary_df, _ = cluster_all_conversions(df_bin)
        
        combined_metrics = pd.merge(paths_per_conv, summary_df, on='conversion')
        combined_metrics = pd.merge(combined_metrics, avg_path_length_df, on='conversion')
        combined_metrics[param_name] = param_value
        all_results.append(combined_metrics)

    if not all_results:
        print("No data processed.")
        return

    full_results_df = pd.concat(all_results, ignore_index=True)
    param_values = sorted(full_results_df[param_name].unique())
    n_params = len(param_values)

    paths_matrix = np.zeros((n_params, n_params))
    clusters_matrix = np.zeros((n_params, n_params))
    cv_matrix = np.zeros((n_params, n_params))
    length_matrix = np.zeros((n_params, n_params))

    for i, p1 in enumerate(param_values):
        for j, p2 in enumerate(param_values):
            if i == j: continue

            df1 = full_results_df[full_results_df[param_name] == p1]
            df2 = full_results_df[full_results_df[param_name] == p2]
            
            outer_merged_df = pd.merge(df1, df2, on='conversion', how='outer', suffixes=('_p1', '_p2')).fillna(0)
            if not outer_merged_df.empty:
                total_conversions = len(outer_merged_df)
                paths_matrix[i, j] = (outer_merged_df['num_paths_p1'] > outer_merged_df['num_paths_p2']).sum() / total_conversions * 100

            inner_merged_df = pd.merge(df1, df2, on='conversion', how='inner', suffixes=('_p1', '_p2'))
            if not inner_merged_df.empty:
                total_common = len(inner_merged_df)
                clusters_matrix[i, j] = (inner_merged_df['n_clusters_p1'] > inner_merged_df['n_clusters_p2']).sum() / total_common * 100
                cv_matrix[i, j] = (inner_merged_df['cv_cluster_size_p1'] < inner_merged_df['cv_cluster_size_p2']).sum() / total_common * 100
                length_matrix[i, j] = (inner_merged_df['avg_path_length_p1'] < inner_merged_df['avg_path_length_p2']).sum() / total_common * 100

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(f'Pairwise Performance Comparison for Parameter "{param_name}"', fontsize=16)
    
    param_labels = [f"{p:.0f}" for p in param_values]

    sns.heatmap(paths_matrix, annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=param_labels, yticklabels=param_labels, ax=axes[0])
    axes[0].set_title("% where Row > Col\n(# of Paths)")
    axes[0].set_xlabel(f"{param_name}"); axes[0].set_ylabel(f"{param_name}")

    sns.heatmap(clusters_matrix, annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=param_labels, yticklabels=param_labels, ax=axes[1])
    axes[1].set_title("% where Row > Col\n(# of Clusters - Common Only)")
    axes[1].set_xlabel(f"{param_name}"); axes[1].set_ylabel(f"{param_name}")

    sns.heatmap(cv_matrix, annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=param_labels, yticklabels=param_labels, ax=axes[2])
    axes[2].set_title("% where Row < Col\n(Uniformity CV - Common Only)")
    axes[2].set_xlabel(f"{param_name}"); axes[2].set_ylabel(f"{param_name}")
    
    sns.heatmap(length_matrix, annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=param_labels, yticklabels=param_labels, ax=axes[3])
    axes[3].set_title("% where Row < Col\n(Avg Path Length - Common Only)")
    axes[3].set_xlabel(f"{param_name}"); axes[3].set_ylabel(f"{param_name}")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    if show:
        plt.show()

    return fig, axes


############################################
# Algorithm Efficiency Comparison (4 metrics)
############################################

def plot_algorithm_efficiency(
    file_paths: list[str],
    times_csv_path: str,
    figsize: tuple[int, int] = (14, 18),
    show: bool = True,
):
    """
    Generates scatter plots comparing algorithm efficiency for 4 metrics,
    including median average path length.
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

        # --- FIX: Create the 'conversion' column before it is used ---
        df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)

        total_paths = len(df)
        df['path_length'] = df['drug_sequence'].apply(len)
        avg_path_length_df = df.groupby('conversion')['path_length'].mean().reset_index()
        median_path_length = avg_path_length_df['path_length'].median()
        
        df_bin = build_conversion_drug_matrix(df)
        summary_df, _ = cluster_all_conversions(df_bin)
        
        median_uniqueness = summary_df['n_clusters'].median()
        median_uniformity_cv = summary_df['cv_cluster_size'].median()

        all_metrics.append({
            'strategy': strategy,
            'k': k_val,
            'n_paths': n_paths_val,
            'total_paths': total_paths,
            'median_uniqueness': median_uniqueness,
            'median_uniformity_cv': median_uniformity_cv,
            'median_avg_path_length': median_path_length,
        })

    if not all_metrics:
        print("No valid data processed from result files.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    comparison_df = pd.merge(metrics_df, times_df, on=['strategy', 'k', 'n_paths'], how='inner')
    
    algo_map = {'k_path_tree': 'Tree Search', 'k_path_beam': 'Beam Search', 'prob': 'Probabilistic'}
    comparison_df['algorithm'] = comparison_df['strategy'].map(algo_map)

    if comparison_df.empty:
        print("Error: No matching runs found between result files and the times CSV.")
        return

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle('Algorithm Efficiency Comparison (Performance vs. Time)', fontsize=16)

    sns.scatterplot(data=comparison_df, x='time_seconds', y='total_paths', hue='algorithm', style='algorithm', s=100, ax=axes[0])
    axes[0].set_title("Quantity: Total Successful Paths Found")
    axes[0].set_ylabel("Total Paths")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    sns.scatterplot(data=comparison_df, x='time_seconds', y='median_uniqueness', hue='algorithm', style='algorithm', s=100, ax=axes[1])
    axes[1].set_title("Quality: Median Solution Uniqueness (Higher is Better)")
    axes[1].set_ylabel("Median # of Unique Drug Sets")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    sns.scatterplot(data=comparison_df, x='time_seconds', y='median_uniformity_cv', hue='algorithm', style='algorithm', s=100, ax=axes[2])
    axes[2].set_title("Quality: Median Solution Uniformity (Lower is Better)")
    axes[2].set_ylabel("Median Coefficient of Variation")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    sns.scatterplot(data=comparison_df, x='time_seconds', y='median_avg_path_length', hue='algorithm', style='algorithm', s=100, ax=axes[3])
    axes[3].set_title("Efficiency: Median Path Length (Lower is Better)")
    axes[3].set_ylabel("Median Avg. Path Length")
    axes[3].set_xlabel("Computation Time (seconds)")
    axes[3].grid(True, linestyle='--', alpha=0.6)

    for ax in axes:
        ax.legend(title='Algorithm')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()

    return comparison_df, fig, axes
