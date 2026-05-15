#!/usr/bin/env python

"""
prepare_state_se_input.py

Prepare an AnnData .h5ad file for Arc STATE SE-600M embedding.

Expected behavior:
  - input .X is raw counts
  - optionally subset to cells with obs[pass_filter_col] == pass
  - optionally subset to a gene list matching STATE/Tahoe gene universe
  - save raw counts in ad.layers["counts"]
  - normalize each cell to 10,000 total counts
  - log1p transform
  - write output .h5ad with log-normalized expression in .X

Example:
  python prepare_state_se_input.py \
    --input SW480_cell_line_examples_merged_raw.h5ad \
    --output SW480_cell_line_examples_merged_SE_input_lognorm.h5ad \
    --pass-filter-col pass_filter \
    --cell-line-col cell_line \
    --pert-col drugname_drugconc
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input raw-count h5ad")
    p.add_argument("--output", required=True, help="Output SE-ready log-normalized h5ad")

    p.add_argument(
        "--pass-filter-col",
        default="pass_filter",
        help="obs column indicating QC-passing cells. If absent, no filtering is applied.",
    )
    p.add_argument(
        "--no-pass-filter",
        action="store_true",
        help="Do not filter cells by pass_filter column.",
    )

    p.add_argument(
        "--gene-list",
        default=None,
        help=(
            "Optional file with one gene per line. If provided, subset/reorder genes "
            "to this list where present."
        ),
    )
    p.add_argument(
        "--gene-symbol-col",
        default=None,
        help=(
            "Optional var column containing gene symbols/IDs to match against --gene-list. "
            "If omitted, matches using ad.var_names."
        ),
    )

    p.add_argument(
        "--cell-line-col",
        default="cell_line",
        help="obs column to copy into obs['cell_type'] for STATE transition model.",
    )
    p.add_argument(
        "--pert-col",
        default="drugname_drugconc",
        help="Perturbation column to check/report.",
    )

    p.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Library-size normalization target. Tahoe/STATE uses 10000.",
    )
    p.add_argument(
        "--counts-layer",
        default="counts",
        help="Layer name where raw counts will be saved.",
    )
    return p.parse_args()


def boolean_pass_mask(series: pd.Series) -> np.ndarray:
    """
    Robustly convert common pass_filter encodings to boolean.
    Handles True/False, 1/0, pass/fail, yes/no.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.values

    vals = series.astype(str).str.lower().str.strip()
    pass_values = {"true", "1", "yes", "y", "pass", "passed"}
    return vals.isin(pass_values).values


def summarize_matrix(ad, label):
    x = ad.X.data if sparse.issparse(ad.X) else np.asarray(ad.X).ravel()
    if x.size == 0:
        print(f"{label}: empty matrix")
        return

    sample = x[: min(x.size, 500_000)]
    non_integer_frac = np.mean(np.abs(sample - np.round(sample)) > 1e-6)

    print(f"\n{label}")
    print(f"  shape: {ad.n_obs} cells × {ad.n_vars} genes")
    print(f"  min sampled value: {sample.min():.4f}")
    print(f"  max sampled value: {sample.max():.4f}")
    print(f"  fraction non-integer-like sampled values: {non_integer_frac:.6f}")

    cell_sums = np.asarray(ad.X.sum(axis=1)).ravel()
    print(f"  median per-cell sum: {np.median(cell_sums):.4f}")
    print(f"  min/max per-cell sum: {cell_sums.min():.4f} / {cell_sums.max():.4f}")


def subset_to_gene_list(ad, gene_list_file, gene_symbol_col=None):
    wanted = pd.read_csv(gene_list_file, header=None)[0].astype(str).tolist()

    if gene_symbol_col is None:
        gene_ids = pd.Series(ad.var_names.astype(str), index=ad.var_names)
    else:
        if gene_symbol_col not in ad.var.columns:
            raise ValueError(f"--gene-symbol-col {gene_symbol_col!r} not found in ad.var")
        gene_ids = ad.var[gene_symbol_col].astype(str)

    gene_to_varname = {}
    for varname, gene_id in zip(ad.var_names, gene_ids):
        if gene_id not in gene_to_varname:
            gene_to_varname[gene_id] = varname

    keep_varnames = [gene_to_varname[g] for g in wanted if g in gene_to_varname]
    missing = [g for g in wanted if g not in gene_to_varname]

    print(f"\nGene-list filtering:")
    print(f"  requested genes: {len(wanted)}")
    print(f"  matched genes:   {len(keep_varnames)}")
    print(f"  missing genes:   {len(missing)}")

    if len(keep_varnames) == 0:
        raise ValueError("No genes from --gene-list matched the AnnData object.")

    return ad[:, keep_varnames].copy()


def main():
    args = parse_args()

    print(f"Reading: {args.input}")
    ad = sc.read_h5ad(args.input)
    print(ad)

    summarize_matrix(ad, "Input matrix before preprocessing")

    # 1. Filter cells by pass_filter if requested and available
    if not args.no_pass_filter and args.pass_filter_col in ad.obs.columns:
        mask = boolean_pass_mask(ad.obs[args.pass_filter_col])
        print(f"\nFiltering by obs[{args.pass_filter_col!r}]")
        print(f"  cells before: {ad.n_obs}")
        print(f"  cells passing: {mask.sum()}")
        ad = ad[mask].copy()
    elif not args.no_pass_filter:
        print(f"\nNo obs[{args.pass_filter_col!r}] column found; skipping pass_filter step.")
    else:
        print("\nSkipping pass_filter step because --no-pass-filter was set.")

    # 2. Optional gene subset/reorder
    if args.gene_list is not None:
        ad = subset_to_gene_list(ad, args.gene_list, args.gene_symbol_col)

    # 3. Add STATE-friendly context alias
    if args.cell_line_col in ad.obs.columns:
        ad.obs["cell_type"] = ad.obs[args.cell_line_col].astype(str)
    else:
        print(f"\nWarning: obs[{args.cell_line_col!r}] not found; not creating obs['cell_type'].")

    # 4. Report perturbation labels
    if args.pert_col in ad.obs.columns:
        print(f"\nPerturbation counts from obs[{args.pert_col!r}]:")
        print(ad.obs[args.pert_col].astype(str).value_counts().head(20))
    else:
        print(f"\nWarning: obs[{args.pert_col!r}] not found.")

    # 5. Preserve raw counts
    ad.layers[args.counts_layer] = ad.X.copy()

    # 6. Normalize and log1p
    print(f"\nNormalizing to target_sum={args.target_sum:g} and applying scanpy.pp.log1p")
    sc.pp.normalize_total(ad, target_sum=args.target_sum)
    sc.pp.log1p(ad)

    # 7. Make indices unique and save
    ad.obs_names_make_unique()
    ad.var_names_make_unique()

    summarize_matrix(ad, "Output matrix after normalize_total + log1p")

    print(f"\nWriting: {args.output}")
    ad.write_h5ad(args.output)
    print("Done.")


if __name__ == "__main__":
    main()