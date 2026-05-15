#!/usr/bin/env python
"""
test_data_loader.py

Small test script for data_loader.py.

Example:
    python test_data_loader.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --cell-col cell_name \
      --embed-key X_state \
      --start-sample 256 \
      --target-sample 256 \
      --save-npz J82_to_A172_states.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loader import load_start_target_embeddings, list_cell_counts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adata", required=True, help="Input h5ad containing SE embeddings in obsm")
    p.add_argument("--start-cell", required=True, help="Starting cell label, e.g. J82")
    p.add_argument("--target-cell", required=True, help="Target cell label, e.g. A-172")
    p.add_argument("--cell-col", default="cell_name", help="obs column containing cell labels")
    p.add_argument("--embed-key", default="X_state", help="obsm key containing SE embeddings")
    p.add_argument("--start-sample", default="256", help="Number of start cells to sample, or 'all'")
    p.add_argument("--target-sample", default="256", help="Number of target cells to sample, or 'all'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-replace-if-needed", action="store_true")
    p.add_argument("--save-npz", default=None, help="Optional output .npz cache")
    p.add_argument("--show-counts", action="store_true", help="Print counts per cell label before loading")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show_counts:
        counts = list_cell_counts(args.adata, cell_col=args.cell_col)
        print("\nTop cell counts:")
        print(counts.head(20))
        print("\nCount summary:")
        print(counts.describe())

    pair = load_start_target_embeddings(
        h5ad_path=args.adata,
        start_cell=args.start_cell,
        target_cell=args.target_cell,
        cell_col=args.cell_col,
        embed_key=args.embed_key,
        start_sample=args.start_sample,
        target_sample=args.target_sample,
        seed=args.seed,
        replace_if_needed=not args.no_replace_if_needed,
    )

    print("\nLoaded start/target states")
    print(json.dumps(pair.metadata(), indent=2))

    print("\nEmbedding shapes:")
    print("  start:", pair.start_embeddings.shape, pair.start_embeddings.dtype)
    print("  target:", pair.target_embeddings.shape, pair.target_embeddings.dtype)

    print("\nBasic embedding stats:")
    print("  start mean/std:", float(np.mean(pair.start_embeddings)), float(np.std(pair.start_embeddings)))
    print("  target mean/std:", float(np.mean(pair.target_embeddings)), float(np.std(pair.target_embeddings)))

    start_centroid = pair.start_embeddings.mean(axis=0)
    target_centroid = pair.target_embeddings.mean(axis=0)
    l2 = float(np.linalg.norm(start_centroid - target_centroid))
    cosine = float(
        1.0
        - np.dot(start_centroid, target_centroid)
        / ((np.linalg.norm(start_centroid) * np.linalg.norm(target_centroid)) + 1e-8)
    )
    print("\nCentroid distances:")
    print("  L2:", l2)
    print("  cosine distance:", cosine)

    if args.save_npz:
        pair.save_npz(args.save_npz)
        print(f"\nSaved cache: {args.save_npz}")

        # Quick reload sanity check
        from data_loader import LoadedCellStates

        reloaded = LoadedCellStates.load_npz(args.save_npz)
        print("Reloaded cache:")
        print("  start:", reloaded.start_embeddings.shape)
        print("  target:", reloaded.target_embeddings.shape)


if __name__ == "__main__":
    main()
