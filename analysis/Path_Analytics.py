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


############################################
#converts conversion dataframe to a drug usage matrix
############################################

def build_conversion_drug_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turns a dataframe with columns
        starting_cl | target_cl | path
    into a binary drug-usage matrix plus a `conversion` label.

    `path` may contain:
      • real lists/tuples of drug-tuples
      • strings that literal-eval to the above
      • hybrids / oddities — handled best-effort
    """
    df = df.copy()
    df["conversion"] = df["starting_cl"].astype(str) + ":" + df["target_cl"].astype(str)

    # ---------- helper ----------
    def parse_path(value) -> set[str]:
        """Return the set of drug names found in `value`."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return set()

        # Ensure we start with an iterable container
        if isinstance(value, str):
            # The *whole* cell might be a string representation of a list/tuple
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = [value]             # treat as raw name

        if not isinstance(value, Iterable):
            value = [value]

        # Depth-first walk through nested containers
        stack = list(value)
        drugs = set()

        while stack:
            item = stack.pop()

            # Try to normalise strings that still look like tuples
            if isinstance(item, str):
                try:
                    item = ast.literal_eval(item)
                except Exception:
                    drugs.add(item.strip())
                    continue

            if isinstance(item, tuple):
                if item:
                    drugs.add(str(item[0]).strip())
            elif isinstance(item, Iterable):
                stack.extend(item)
            else:                           # defensive fallback
                try:
                    drugs.add(str(item[0]).strip())
                except Exception:
                    pass

        return drugs
    # ----------------------------

    drug_sets = df["path"].apply(parse_path)

    # Unique drug list in deterministic order
    all_drugs = sorted({drug for s in drug_sets for drug in s})

    # One-hot encode
    drug_matrix = pd.DataFrame(0, index=df.index, columns=all_drugs, dtype=int)
    for idx, drugs in drug_sets.items():
        if drugs:
            drug_matrix.loc[idx, list(drugs)] = 1

    return pd.concat([df[["conversion"]], drug_matrix], axis=1)

############################################
#Plots QC on the binary drug matrix
############################################

def plot_path_and_conversion_histograms(
    df_bin: pd.DataFrame,
    conversion_col: str = "conversion",
    bins_drugs: int = 10,
    bins_conv: int = 25,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
):
    """
    Draw two side-by-side histograms:

        • Total number of drugs per path (row)
        • Total number of paths per conversion

    Each panel displays mean and median in a text box.
    """
    # ------------------------------------------------------------------
    # 1) Drugs per path
    # ------------------------------------------------------------------
    drug_cols = [c for c in df_bin.columns if c != conversion_col]
    total_drugs = df_bin[drug_cols].sum(axis=1)

    # ------------------------------------------------------------------
    # 2) Paths per conversion
    # ------------------------------------------------------------------
    conversion_counts = (
        df_bin[conversion_col].value_counts()
        .rename_axis(conversion_col)
        .reset_index(name="count")
    )["count"]

    # ------------------------------------------------------------------
    # 3) Plotting
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ---------------- Histogram 1 ----------------
    sns.histplot(total_drugs, bins=bins_drugs, ax=axes[0], color="steelblue")
    axes[0].set_title("Total # Drugs per Path")
    axes[0].set_xlabel("Number of Drugs")
    axes[0].set_ylabel("Frequency")

    # stats box
    mean_drugs = total_drugs.mean()
    median_drugs = total_drugs.median()
    txt1 = f"mean = {mean_drugs:.2f}\nmedian = {median_drugs:.2f}"
    axes[0].text(
        0.97, 0.97, txt1,
        transform=axes[0].transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    # ---------------- Histogram 2 ----------------
    sns.histplot(conversion_counts, bins=bins_conv, ax=axes[1], color="seagreen")
    axes[1].set_title("Successful Paths per Conversion")
    axes[1].set_xlabel("Number of Paths")
    axes[1].set_ylabel("Frequency")

    # stats box
    mean_conv = conversion_counts.mean()
    median_conv = conversion_counts.median()
    txt2 = f"mean = {mean_conv:.2f}\nmedian = {median_conv:.2f}"
    axes[1].text(
        0.97, 0.97, txt2,
        transform=axes[1].transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes

def plot_path_and_conversion_stats(
    drug_matrix: pd.DataFrame,
    conversion_col: str = "conversion",
    bins_drugs: int = 10,
    bins_conv: int = 25,
    top_n: int = 75,
    figsize: tuple[int, int] = (14, 12),
    show: bool = True,
):
    """
    Draw a 3-row figure:

        Row 1 ── Two histograms
            • Total # drugs per path (row)
            • Total # paths per conversion

        Row 2 ── Bar plot
            • Frequency of each drug across *all* successful conversions (top N)

        Row 3 ── Bar plot
            • Frequency of each drug across *unique* conversions (top N)

    All panels show mean and median inside a textbox.

    Parameters
    ----------
    drug_matrix : pd.DataFrame
        Binary matrix (0/1) whose rows are successful paths and whose
        columns are individual drugs, plus a `conversion` column.
    conversion_col : str
        Name of the column that holds the "start→target" conversion IDs.
    bins_drugs, bins_conv : int
        Histogram bin counts.
    top_n : int
        How many top-ranking drugs to show in the bar plots.
    figsize : (w, h)
        Overall figure size.
    show : bool
        Call plt.show() if True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict[str, matplotlib.axes.Axes]
        Mapping of panel names to Axes objects.
    """
    drug_matrix_cleaned = drug_matrix.copy()
    #clean column names in drug_matrix
    #column_names = drug_matrix_cleaned.columns.tolist()
    #column_names = [c.split(' (')[0] for c in column_names]
    #drug_matrix_cleaned.columns = column_names

    # ------------------------------------------------------------------
    # 0) Identify drug columns
    # ------------------------------------------------------------------
    drug_cols = [c for c in drug_matrix_cleaned.columns if c != conversion_col]

    # ------------------------------------------------------------------
    # 1) Histograms
    # ------------------------------------------------------------------
    total_drugs = drug_matrix_cleaned[drug_cols].sum(axis=1)

    conversion_counts = (
        drug_matrix_cleaned[conversion_col]
        .value_counts()
        .rename_axis(conversion_col)
        .reset_index(name="count")["count"]
    )

    # ------------------------------------------------------------------
    # 2) Bar-plot data
    # ------------------------------------------------------------------
    # 2a. Overall frequency across *all* paths
    all_freq = (
        drug_matrix_cleaned.set_index(conversion_col)[drug_cols]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("sum")
        .to_frame()
    )

    # 2b. Presence/absence per conversion (unique conversions)
    unique_conv = (
        drug_matrix_cleaned.groupby(conversion_col)[drug_cols]
        .sum()
        .map(lambda x: 1 if x > 0 else 0)
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("sum")
        .to_frame()
    )

    # ------------------------------------------------------------------
    # 3) Figure & layout (GridSpec)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    # ── 1)  OUTER GridSpec ────────────────────────────────────────────────
    # Two big stripes: 0 = rows 1-2 together, 1 = row-3 alone.
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[2, 1],   # give the top stripe more height
        hspace=0.5             # ← space *only* between stripe-0 and stripe-1
    )

    # ── 2)  Nested GridSpec inside the top stripe ─────────────────────────
    # Two rows, two columns; *no* vertical gap here.
    top = outer[0].subgridspec(
        2, 2,
        height_ratios=[1, 1],   # equal height for row-1 and row-2
        hspace=0.5,               # ← zero gap between row-1 & row-2
        wspace=0.25
    )

    # Axes mapping
    ax_hist_drugs   = fig.add_subplot(top[0, 0])   # row-1, col-1
    ax_hist_paths   = fig.add_subplot(top[0, 1])   # row-1, col-2
    ax_bar_all      = fig.add_subplot(top[1, :])   # row-2 spans both cols
    ax_bar_unique   = fig.add_subplot(outer[1, 0]) # row-3 (full width)

    axes = {
        "hist_drugs_per_path": ax_hist_drugs,
        "hist_paths_per_conv": ax_hist_paths,
        "bar_all":            ax_bar_all,
        "bar_unique":         ax_bar_unique,
    }

    # ---------------- Histogram 1 ----------------
    ax = axes["hist_drugs_per_path"]
    sns.histplot(total_drugs, bins=bins_drugs, ax=ax, color="steelblue")
    ax.set_title("Total # Drugs per Path")
    ax.set_xlabel("Number of Drugs")
    ax.set_ylabel("Frequency")
    _annotate_stats(ax, total_drugs.mean(), total_drugs.median())

    # ---------------- Histogram 2 ----------------
    ax = axes["hist_paths_per_conv"]
    sns.histplot(conversion_counts, bins=bins_conv, ax=ax, color="seagreen")
    ax.set_title("Successful Paths per Conversion")
    ax.set_xlabel("Number of Paths")
    ax.set_ylabel("Frequency")
    _annotate_stats(ax, conversion_counts.mean(), conversion_counts.median())

    # ---------------- Bar plot 1 (all paths) ----------------
    ax = axes["bar_all"]
    sns.barplot(
        x=all_freq.index,
        y="sum",
        data=all_freq.reset_index(drop=False),
        ax=ax,
        color="mediumvioletred",
    )
    ax.set_title(
        f"Drug Frequency in All Successful Conversions (top {len(all_freq)})"
    )
    ax.set_ylabel("Total Occurrences")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    _annotate_stats(ax, all_freq["sum"].mean(), all_freq["sum"].median())

    # ---------------- Bar plot 2 (unique conversions) ----------------
    ax = axes["bar_unique"]
    sns.barplot(
        x=unique_conv.index,
        y="sum",
        data=unique_conv.reset_index(drop=False),
        ax=ax,
        color="peru",
    )
    ax.set_title(
        f"Drug Frequency in Unique Cell(A) - Cell(B) Conversions (top {len(unique_conv)})"
    )
    ax.set_ylabel("Total Occurrences")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    _annotate_stats(ax, unique_conv["sum"].mean(), unique_conv["sum"].median())

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


# ----------------------------------------------------------------------
# Helper: overlay mean & median textbox on an Axes
# ----------------------------------------------------------------------
def _annotate_stats(ax, mean_val: float, median_val: float) -> None:
    txt = f"mean = {mean_val:.2f}\nmedian = {median_val:.2f}"
    ax.text(
        0.97,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


############################################
#Heirarchical clustering of drug usage matrix
############################################

# ---------------------------------------------------------------------
# Single-conversion clustering  
# ---------------------------------------------------------------------
def _cluster_single_conversion(
    sub_df: pd.DataFrame,
    drug_cols: list[str],
    metric: str = "jaccard",
    distance_threshold: float = 0.5,
) -> pd.Series:
    """
    Return cluster labels for `sub_df`; index aligns with `sub_df`.
    Converts the binary matrix to a NumPy array to satisfy scikit-learn.
    """
    if len(sub_df) < 2:                       # nothing to cluster
        return pd.Series([0], index=sub_df.index, name="cluster")

    # --- Convert to Boolean NumPy array so sklearn sees `.dtype` ---
    X = sub_df[drug_cols].astype(bool).to_numpy()

    # Jaccard distances on binary vectors
    dist = pairwise_distances(X, metric=metric)

    model = AgglomerativeClustering(
        metric="precomputed",                # newer sklearn uses 'metric'
        linkage="average",
        n_clusters=None,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(dist)

    return pd.Series(labels, index=sub_df.index, name="cluster")


# ---------------------------------------------------------------------
# All-conversion wrapper 
# ---------------------------------------------------------------------
def cluster_all_conversions(
    df_bin: pd.DataFrame,
    conversion_col: str = "conversion",
    metric: str = "jaccard",
    distance_threshold: float = 0.5,
):
    drug_cols = [c for c in df_bin.columns if c != conversion_col]
    summaries = []
    cluster_labels = pd.Series(dtype=int, name="cluster")   # named at start

    for conv, sub_df in df_bin.groupby(conversion_col):
        labels = _cluster_single_conversion(
            sub_df,
            drug_cols,
            metric=metric,
            distance_threshold=distance_threshold,
        )
        cluster_labels = pd.concat([cluster_labels, labels])

        n_c = labels.nunique()
        summaries.append(
            {
                "conversion": conv,
                "n_clusters": int(n_c),
                "mean_cluster_size": len(labels) / n_c,
                "median_cluster_size": labels.value_counts().median(),
            }
        )

    summary_df = pd.DataFrame(summaries)
    df_with_clusters = df_bin.join(cluster_labels)

    return summary_df, df_with_clusters

############################################
#Plotting results of clustering
############################################

def plot_cluster_summary(
    summary: pd.DataFrame,
    x_label: str = "k-path",
    figsize: tuple[int, int] = (12, 4),
    annotate: bool = True,
    show: bool = True,
):
    """
    Draw three box-plots (n_clusters, mean_cluster_size, median_cluster_size)
    with optional per-panel annotation of descriptive stats.

    Parameters
    ----------
    summary : pd.DataFrame
        Must include columns 'n_clusters', 'mean_cluster_size',
        'median_cluster_size'.
    x_label : str
        Label printed under every box-plot (same for all three).
    figsize : (w, h)
        Matplotlib figure size in inches.
    annotate : bool
        Whether to add a text box with stats to each panel.
    show : bool
        Call plt.show() automatically.
    """
    required = {"n_clusters", "mean_cluster_size", "median_cluster_size"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Summary DataFrame missing columns: {missing}")

    metrics = ["n_clusters", "mean_cluster_size", "median_cluster_size"]
    titles  = ["Uniqueness \n(# Unique Drug Sets per Conversion)", "Redundancy (mean)\ntotal paths per cluster", "Redundancy (median)\ntotal paths per cluster"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    for ax, metric, title in zip(axes, metrics, titles):
        data = summary[metric].dropna()
        ax.boxplot(data, showfliers=True)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric)

        # Tighten y-axis
        pad = 0.05 * (data.max() - data.min() or 1)
        ax.set_ylim(data.min() - pad, data.max() + pad)

        # ---------------- annotation ----------------
        if annotate:
            q1, med, q3 = np.percentile(data, [25, 50, 75])
            txt = (
                f"n = {len(data)}\n"
                f"min = {data.min():.2f}\n"
                f"Q1  = {q1:.2f}\n"
                f"med = {med:.2f}\n"
                f"Q3  = {q3:.2f}\n"
                f"max = {data.max():.2f}"
            )
            ax.text(
                0.95,
                0.95,
                txt,
                va="top",
                ha="right",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
    # ----------------------------------------------

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes