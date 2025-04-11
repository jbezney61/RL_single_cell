
# imports

from preprocess_utils import *
from matching_utils import *
import pickle
import pandas as pd
import numpy as np 
import ast
from collections import Counter
import requests
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# read unprocessed data

gpt_df = pd.read_csv('gpt.csv')
dtc_df = pd.read_csv('dtc.csv')
dgidb_df = pd.read_csv('dgidb.tsv', sep='\t')
getdb_df = pd.read_csv('getdb.txt', delimiter='\t')
chembl_df = pd.read_csv('chEMBL.tsv', sep='\t')
with open('drugs.pkl', 'rb') as f:
    drugs= pickle.load(f)
with open('drugs_changed.pkl', 'rb') as f:
    drugs_changed = pickle.load(f)
with open('drugs_gpt2.pkl', 'rb') as f:
    drugs_gpt2 = pickle.load(f)
with open('gpt_df2.pkl', 'rb') as f:
    gpt_df2 = pickle.load(f)

# normalize name 'none' to np.nan

gpt_df['targets'] = gpt_df['targets'].replace('none', np.nan)
gpt_df2['targets'] = gpt_df2['targets'].replace('none', np.nan)

# variables we are interested in

variables_dtc = ['compound_id', 'standard_inchi_key', 'compound_name', 'gene_names', 'wildtype_or_mutant', 'ep_action_mode', 'title', 'journal']
variables_dgidb = ['gene_claim_name', 'gene_concept_id', 'gene_name', 'interaction_source_db_name', 'interaction_type', 'interaction_score', 'drug_claim_name', 'drug_concept_id']
variables_getdb = ['drug_name', 'target_name', 'gene_name', 'target_action', 'interaction_score', 'source']
variables_chEMBL = ['Parent Molecule ChEMBL ID', 'Parent Molecule Name', 'Parent Molecule Type','Mechanism of Action', 'Target ChEMBL ID', 'Target Name', 'Action Type','Target Type','Binding Site Name','References','Synonyms']

dtc_df = dtc_df[variables_dtc]
dgidb_df = dgidb_df[variables_dgidb]
getdb_df = getdb_df[variables_getdb]
chembl_df = chembl_df[variables_chEMBL]

# normalize compound column name

dtc_df.rename(columns={'compound_name': 'compound'}, inplace=True) # no NaN after we add the extras (see below) --> recheck for drugs_changed
dgidb_df.rename(columns={'drug_claim_name': 'compound'}, inplace=True) # no NaN 
dgidb_df.rename(columns={'gene_name': 'gene_names'}, inplace=True) # no NaN 
getdb_df.rename(columns={'drug_name': 'compound'}, inplace=True) # no NaN
getdb_df.rename(columns={'gene_name': 'gene_names'}, inplace=True) # no NaN 
chembl_df.rename(columns={'Parent Molecule Name': 'compound'}, inplace=True) # no NaN, no need to change the name of the gene column (will do later)
gpt_df.rename(columns={'targets': 'gene_names'}, inplace=True) # no NaN 
gpt_df2.rename(columns={'targets': 'gene_names'}, inplace=True) # no NaN

# add some extra ids for those compounds that are in the list (either normal or changed) and you have NaN ids for them 

extra = {'LAROTRECTINIB': 'inexistent_1', 'NVP-BHG712': 'inexistent_2', 'RADOTINIB': 'inexistent_3', 'SAPANISERTIB': 'inexistent_4',
       'TUCATINIB': 'inexistent_5', 'PEXIDARTINIB': 'inexistent_6'}

for i in range(len(dtc_df)):
    if dtc_df['compound'].iloc[i] in extra:
        dtc_df['compound_id'].iloc[i] = extra[dtc_df['compound'].iloc[i]]
        
dtc_df.dropna(subset=['compound_id'], inplace=True)

# normalize all columns of all datasets

gpt_df = normalize_columns(gpt_df, gpt_df.columns.tolist())
gpt_df2 = normalize_columns(gpt_df2, gpt_df2.columns.tolist())
dtc_df = normalize_columns(dtc_df, dtc_df.columns.tolist())
dtc_df = dtc_df.reset_index(drop=True) # as when normalizing columns the index gets messed up
dgidb_df = normalize_columns(dgidb_df, dgidb_df.columns.tolist())
getdb_df = normalize_columns(getdb_df, getdb_df.columns.tolist())
chembl_df = normalize_columns(chembl_df, chembl_df.columns.tolist())

# need to do this replaces (otherwise I get more than one observation after the groupby downstream)

dtc_df['compound_id'] = dtc_df['compound_id'].replace('chembl1201236', 'chembl1200748')
dtc_df['compound_id'] = dtc_df['compound_id'].replace('chembl1201258', 'chembl225072')

# unprocessed datasets in dictionary

datasets_unprocessed = {
    'gpt_unprocessed': gpt_df,
    'gpt2_unprocessed': gpt_df2,
    'dtc_unprocessed': dtc_df,
    'dgidb_unprocessed': dgidb_df,
    'getdb_unprocessed': getdb_df,
    'chembl_unprocessed': chembl_df
}

# adding name of dataset to column names

prefix_columns_inplace(datasets_unprocessed)

# groupby datasets and collapse them 

# before you run this make sure that there is no NaN in the column that you want to group by and that, if you do not group by the compound column, this last one is in the constant columns (note still not enough).
# 'standard_inchi_key_dtc' is not constant for what we did when we changed the chembl id.

datasets_processed = group_and_collapse_all(datasets_unprocessed)

# save processed datasets

for name, df in datasets_processed.items():
    df.to_pickle(f'{name}.pkl')

# read the processed datasets

gpt_processed = pd.read_pickle('gpt_processed.pkl')
gpt2_processed = pd.read_pickle('gpt2_processed.pkl')
dtc_processed = pd.read_pickle('dtc_processed.pkl')
dgidb_processed = pd.read_pickle('dgidb_processed.pkl')
getdb_processed = pd.read_pickle('getdb_processed.pkl')
chembl_processed = pd.read_pickle('chembl_processed.pkl')

with open('drugs.pkl', 'rb') as f:
    drugs = pickle.load(f)
with open('drugs_changed.pkl', 'rb') as f:
    drugs_changed = pickle.load(f)
with open('drugs_gpt2.pkl', 'rb') as f:
    drugs_gpt2 = pickle.load(f)

# datasets into a dictionary

datasets = {
    'gpt_processed': gpt_processed, 
    'gpt2_processed': gpt2_processed, # has chemebl_id
    'dtc_processed': dtc_processed, # has chemebl_id, standard_ichi_key
    'dgidb_processed': dgidb_processed, # has different ids which can be found in ids_dgidb
    'getdb_processed': getdb_processed, # no id, also checking the preprocessed
    'chembl_processed': chembl_processed # has chemebl_id
}

# do exact matching 

mixed_df2 = exact_matching(drugs_changed, datasets)

# get gene names from ChEMBL using ChEMBL API

chembl_genes2 = get_series_of_genes(mixed_df2, 'Target ChEMBL ID_chembl')
chembl_genes2 = [normalize_genes(entry) for entry in chembl_genes2]
col_to_insert_after = 'Target ChEMBL ID_chembl'
insert_index = mixed_df2.columns.get_loc(col_to_insert_after) + 1
mixed_df2.insert(insert_index, 'gene_names_chembl', chembl_genes2)

# saving mixed_df2

with open('mixed_df2.pkl', 'wb') as f:
    pickle.dump(mixed_df2, f)

# creating the simplified df (only compunds and genes for each dataset), from here on we will only have lists of unique genes

simplified_df2 = mixed_df2[['compound', 'gene_names_gpt2', 'gene_names_dtc', 'gene_names_dgidb', 'gene_names_getdb', 'gene_names_chembl']].copy()
simplified_df2 = clean_columns_with_lists(simplified_df2, simplified_df2.columns.tolist())

# saving simplified_df2

with open('simplified_df2.pkl', 'wb') as f:
    pickle.dump(simplified_df2, f)

# creating the datasets where for each drug I have the datasets for which I have gene names

dataset_x_drug2 = dataset_per_compound(simplified_df2)

# saving dataset_x_drug2

with open('dataset_x_drug2.pkl', 'wb') as f:
    pickle.dump(dataset_x_drug2, f)

# united dataset

united_dtc_out_laxi2_2 = unite_datasets(simplified_df2, ['gpt2','dgidb', 'getdb', 'chembl'], ['dtc'], type='lax_intersection', threshold=2)

# compute gene overlap 

gene_overlap_dtc_out_laxi2_2 = compute_gene_overlap(united_dtc_out_laxi2_2, 'gpt2 + dgidb + getdb + chembl', 'dtc')

# creating final dataset (we used a lax_intersection with threshold 2 and datasets gpt2, dgidb, getdb and chembl)

mask = ((gene_overlap_dtc_out_laxi2_2.notna()) & (gene_overlap_dtc_out_laxi2_2 >= 0.2))
final_dataset = united_dtc_out_laxi2_2[mask]
final_dataset = final_dataset[['compound', 'gpt2 + dgidb + getdb + chembl']]
final_dataset = final_dataset.reset_index(drop=True)

# saving the final dataset

with open('final_dataset.pkl', 'wb') as f:
    pickle.dump(final_dataset, f)