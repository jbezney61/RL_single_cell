import pickle
import pandas as pd
import numpy as np 
import ast
from collections import Counter

def normalize_columns(df, columns):
    """
    Normalizes the specified columns in a DataFrame by stripping whitespace, and converting to lowercase.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ' '.join(x.strip().lower().split()) if isinstance(x, str) else x)
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    return df

def prefix_columns_inplace(datasets):
    """
    Modifies the input datasets in-place: for each DataFrame, adds a prefix to column names
    based on the part before '_' in the dataset name, except for 'drug'.
    """
    
    for name, df in datasets.items():
        prefix = name.split('_')[0]
        df.columns = [f"{col}_{prefix}" for col in df.columns]

def adjust_to_list(df, columns):
    """
    Adjusts the specified columns in a DataFrame to ensure that each entry is a list (applied to gpt_df)
    """
    for col in columns:
        def clean_and_split(value):
            if pd.isna(value):
                return value  # keep NaN as-is
            value = value.strip()
            # split on commas, strip whitespace
            parts = [v.strip() for v in value.split(',')]
            return parts[0] if len(parts) == 1 else parts
        df[col] = df[col].apply(clean_and_split)
    return df

def group_and_collapse_all(datasets):
    """
    Groups the datasets by the specified columns and collapses them into a single row per group.
    """
    def split_if_needed(x):
        if pd.isna(x):
            return x
        x = x.strip()
        parts = [part.strip() for part in x.split(',')]
        return parts if len(parts) > 1 else parts[0]

    grouped_datasets = {}

    for name, df in datasets.items():
        prefix = name.split('_')[0]
        gb_col = 'compound'

        if name == 'gpt_unprocessed':
            df = df.copy()
            df = adjust_to_list(df, ['gene_names_gpt'])
            grouped_datasets[f'{prefix}_processed'] = df
            continue

        if name == 'chembl_unprocessed':
            gb_col = 'Parent Molecule ChEMBL ID'

        if name == 'dtc_unprocessed':
            gb_col = 'compound_id'

        groupby_col = f'{gb_col}_{prefix}'
        df = df.copy()

        grouped = df.groupby(groupby_col)
        result_rows = []
        constant_cols = []
        varying_cols = []

        for col in df.columns:
            if col == groupby_col:
                continue
            nunique_per_group = grouped[col].nunique()
            is_constant_or_nan = (nunique_per_group <= 1).all()
            if is_constant_or_nan:
                constant_cols.append(col)
            else:
                varying_cols.append(col)

        print(f"\nDataset: {name}")
        print("Constant columns:", constant_cols)
        print("Varying columns:", varying_cols)

        for group_name, group in grouped:
            row = {groupby_col: group_name}
            for col in constant_cols + varying_cols:
                val = group[col].iloc[0] if len(group[col].unique()) == 1 else group[col]
                if isinstance(val, pd.Series):
                    val = val.tolist()
                    if col == f'gene_names_{prefix}':
                        val = [split_if_needed(x) for x in val]
                    row[col] = val
                else:
                    if col == f'gene_names_{prefix}':
                        val = split_if_needed(val)
                    row[col] = val

            result_rows.append(row)

        grouped_datasets[f'{prefix}_processed'] = pd.DataFrame(result_rows)

    return grouped_datasets
