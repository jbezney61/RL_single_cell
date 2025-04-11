import pickle
import pandas as pd
import numpy as np 
import ast
import requests
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns



def exact_matching(drugs, datasets):
    """
    Perform exact matching of drug names with multiple datasets.
    """
    # Normalize all drug names to lowercase and strip spaces (preserve NaNs)
    for name, df in datasets.items():
        column = f'compound_{name.split("_")[0]}'

    # Create base drug DataFrame
    mixed_df = pd.DataFrame({'compound': drugs})

    # Merge sequentially with each dataset
    for name, df in datasets.items():
        column = f'compound_{name.split("_")[0]}'
        mixed_df = mixed_df.merge(df, left_on='compound', right_on=column, how='left')

    return mixed_df

def get_series_of_genes(df, column_name):
    """
    Given a DataFrame and a column of ChEMBL target IDs (strings or lists of strings),
    returns a list of gene name lists retrieved via the ChEMBL API.
    """

    def get_gene_name_from_chembl(chembl_id):
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            gene_names = []
            for comp in data.get("target_components", []):
                for synonym in comp.get("target_component_synonyms", []):
                    if synonym.get("syn_type") == "GENE_SYMBOL":
                        gene_names.append(synonym.get("component_synonym"))
            return gene_names if gene_names else np.nan
        else:
            return np.nan

    out = []    
    for chembl_id in df[column_name]:
        if isinstance(chembl_id, list):
            gene_names = []
            for id in chembl_id:
                if isinstance(id, str): # it is never a list of lists
                    genes = get_gene_name_from_chembl(id)
                    if isinstance(genes, list):
                        if len(genes) > 1:
                            print(f'{id} has more than one gene name associated to it')
                        gene_names.append(genes if len(genes) > 1 else genes[0])
                    else:
                        gene_names.append(genes)  # np.nan
                else:
                    gene_names.append(np.nan)
            out.append(gene_names)

        elif isinstance(chembl_id, str):
            genes = get_gene_name_from_chembl(chembl_id)
            if isinstance(genes, list):
                if len(genes) > 1:
                    print(f'{chembl_id} has more than one gene associated to it')
                    out.append(genes)
                else:
                    out.append(genes[0])
            else:
                out.append(genes)  # np.nan

        else:
            out.append(np.nan)
            
    return out

def normalize_genes(x):
    """
    Normalize gene names to lowercase strings or lists of strings (used fo chembl_gene_names)
    """
    if isinstance(x, str):
        return x.lower()
    elif isinstance(x, list):
        return [normalize_genes(i) for i in x]
    elif pd.isna(x):
        return x
    else:
        return x
    
def compute_gene_count_percentiles(df, columns, percentiles=[0, 20, 40, 60, 80]):
    """
    Computes percentiles of the number of unique genes per drug
    for each column in `columns`.
    """
    gene_counts = {}

    for col in columns:
        def count_unique_genes(x):
            if isinstance(x, list):
                one_list = [item for sublist in x for item in (sublist if isinstance(sublist, list) else [sublist])]
                cleaned = [v for v in one_list if isinstance(v, str)] # all NaN (inside or outside a list) are np.nan which is a float
                cleaned_set = set(cleaned)
                return len(cleaned_set) if cleaned_set else np.nan
            elif isinstance(x, str):
                return 1
            else:
                return np.nan

        gene_counts[col] = df[col].apply(count_unique_genes)

    gene_counts_df = pd.DataFrame(gene_counts)
    percentile_summary = gene_counts_df.quantile([p / 100 for p in percentiles])

    return percentile_summary
    
def clean_columns_with_lists(df, columns):
    """
    Cleans columns with lists of strings or floats by flattening them and removing duplicates.
    """

    def flatten_and_unique(x):
        if isinstance(x, str) or isinstance(x, float):
            return x
        elif isinstance(x, list):
            # Flatten if it's a list of lists
            flat = []
            for item in x:
                if isinstance(item, list):
                    item = [i for i in item if isinstance(i, str)]
                    flat.extend(item)
                else:
                    flat.append(item) if isinstance(item, str) else None
            result = list(set(flat))
            return result
        else:
            return x  # unexpected types are left as-is

    df = df.copy()
    for col in columns:
        df[col] = df[col].apply(flatten_and_unique)

    return df

def dataset_per_compound(simplified_df):
    """
    Given a DataFrame with a 'compound' column and other columns representing datasets,
    returns a DataFrame with 'compound' and 'datasets' columns.
    The 'datasets' column contains lists of dataset names for each compound.
    """
    df = pd.DataFrame({'compound': simplified_df['compound']})
    df['datasets'] = simplified_df.apply(
        lambda row: [
            col.split('_', 2)[-1] for col in simplified_df.columns
            if col != 'compound' and not (
                pd.isna(row[col]).all() if hasattr(row[col], '__iter__') and not isinstance(row[col], str)
                else pd.isna(row[col])
            )
        ],
        axis=1
    )
    return df

def normalize_gene_entry(x):
    """
    Converts a gene field (which could be a string, list, or nested list) to a cleaned set of unique gene names.
    Filters out 'none' (str) and np.nan.
    """
    if isinstance(x, str):
        return {x}
    elif isinstance(x, list):
        return {v for v in x if isinstance(v, str)} 
    else:
        return set()

def jaccard_similarity(set1, set2):
    """
    Computes the Jaccard similarity between two sets.
    """
    if not set1 or not set2:
        return np.nan
    return len(set1 & set2) / len(set1 | set2)

def compute_jaccard_matrix(df, columns):
    """
    Computes a symmetric matrix of average Jaccard similarities between all pairs of specified columns.
    """
    similarity_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    for col1, col2 in combinations(columns, 2):
        sims = df.apply(
            lambda row: jaccard_similarity(
                normalize_gene_entry(row[col1]),
                normalize_gene_entry(row[col2])
            ),
            axis=1
        )
        mean_sim = sims.mean()
        similarity_matrix.loc[col1, col2] = mean_sim
        similarity_matrix.loc[col2, col1] = mean_sim

    # Set diagonal to 1.0
    for col in columns:
        similarity_matrix.loc[col, col] = 1.0

    return similarity_matrix

def unite_datasets(df_mixed, together, not_together, type='union', threshold=None):
    """
    Unites datasets by combining gene names from specified columns.
    The resulting DataFrame contains the original columns and a new column with the combined gene names.
    """
    def normalize_to_set(val):
        if isinstance(val, str):
            return {val}
        elif isinstance(val, list):
            return set(val)
        elif pd.isna(val):
            return None
        else:
            return None

    def unite(row, together, type='union', threshold=None):
        sets = []
        for col in together:
            col_name = f'gene_names_{col}'
            val = normalize_to_set(row[col_name])
            if val is not None:
                sets.append(val)
        
        if not sets:
            return np.nan

        if type == 'union':
            result = set().union(*sets)

        elif type == 'intersection':
            if len(sets) < len(together):  # i.e., one or more are NaN
                    return np.nan
            result = set.intersection(*sets)

        elif type == 'lax_intersection':
            if threshold is None:
                raise ValueError("You must specify a threshold for lax_intersection.")
            if threshold > len(together):
                raise ValueError(f"Threshold {threshold} cannot be greater than the number of datasets: {len(together)}.")
            count = Counter()
            for s in sets:
                count.update(s)
            result = {k for k, v in count.items() if v >= threshold}

        else:
            raise ValueError("type must be 'union', 'intersection', or 'lax_intersection'")

        return list(result) if result else np.nan

    united_dataset = pd.DataFrame()
    united_dataset['compound'] = df_mixed['compound']

    for column in not_together:
        col_name = f'gene_names_{column}'
        united_dataset[column] = df_mixed[col_name]
        united_dataset[column] = united_dataset[column].apply(
            lambda x: x if isinstance(x, list) else [x] if isinstance(x, str) else np.nan
        )

    name = ' + '.join(together)
    united_dataset[name] = df_mixed.apply(lambda row: unite(row, together, type, threshold), axis=1)

    return united_dataset

def compute_gene_overlap(df, united_col, comparison_col):
    """
    Computes the overlap ratio between two columns of gene names.
    The overlap ratio is defined as the number of genes in the intersection
    divided by the minimum number of genes in either column.
    """
    def overlap_ratio(row):
        united_genes = row[united_col]
        comparison_genes = row[comparison_col]
        
        # Ensure both are lists
        if not isinstance(united_genes, list) or not isinstance(comparison_genes, list):
            return np.nan
        
        # Compute percentage of united genes found in comparison
        intersection = set(united_genes) & set(comparison_genes)
        return len(intersection) / min(len(united_genes), len(comparison_genes))
    
    return df.apply(overlap_ratio, axis=1)

def list_length_percentiles(series, percentiles=np.arange(0, 101, 5)):
    """
    Compute percentiles of list lengths in a pandas Series.
    """
    # Drop NaN values
    non_nan = series.dropna()
    
    # Compute lengths of each list
    lengths = non_nan.apply(len)
    print(f'{lengths.mean()} is the mean length of the lists')
    
    # Compute percentiles
    values = np.percentile(lengths, percentiles)
    
    return pd.DataFrame({
        'Percentile': [f"{p}%" for p in percentiles],
        'List Length': values
    })





