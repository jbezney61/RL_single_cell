#!/usr/bin/env python
"""
test_converter.py

Small smoke test for StateSEConverter using an h5ad file containing one cell type
and SE embeddings in adata.obsm['X_state'].

Examples
--------
# Test one exact drug label
python test_converter.py \
  --adata SW480_256.SE600M.h5ad \
  --model-dir /path/to/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state \
  --checkpoint /path/to/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state/checkpoints/final.ckpt \
  --perturbation "[('Trametinib', 0.05, 'uM')]" \
  --output-npy SW480_Trametinib_pred_X_state.npy

# Search for labels containing Trametinib and run the first match
python test_converter.py \
  --adata SW480_256.SE600M.h5ad \
  --model-dir $ST_RUN \
  --find Trametinib

# Run first 10 non-control perturbations as a chunked test
python test_converter.py \
  --adata SW480_256.SE600M.h5ad \
  --model-dir $ST_RUN \
  --run-first-n 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from converter import StateSEConverter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adata", required=True, help="Input h5ad containing obsm[embed_key]")
    p.add_argument("--model-dir", required=True, help="ST-SE model/run directory")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to model-dir/checkpoints/final.ckpt")
    p.add_argument("--embed-key", default="X_state", help="Embedding key in adata.obsm")
    p.add_argument("--n-cells", type=int, default=256, help="Number of cells to use from h5ad")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu; defaults to cuda if available")

    p.add_argument("--perturbation", default=None, help="Exact perturbation label to run")
    p.add_argument("--find", default=None, help="Substring to search perturbation labels; first match is used if --perturbation omitted")
    p.add_argument("--run-first-n", type=int, default=0, help="Run first N non-control perturbations as many-pert test")
    p.add_argument("--chunk-size", type=int, default=8)

    p.add_argument("--output-npy", default=None, help="Optional .npy path for one-perturbation output")
    p.add_argument("--output-many-npz", default=None, help="Optional .npz path for many-perturbation output")
    return p.parse_args()


def load_embeddings(adata_path: str, embed_key: str, n_cells: int, seed: int) -> np.ndarray:
    ad = sc.read_h5ad(adata_path)
    if embed_key not in ad.obsm:
        raise KeyError(f"{embed_key!r} not found in adata.obsm. Available keys: {list(ad.obsm.keys())}")

    x = np.asarray(ad.obsm[embed_key], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {x.shape}")

    if x.shape[0] < n_cells:
        raise ValueError(f"Requested {n_cells} cells, but h5ad only has {x.shape[0]}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=n_cells, replace=False)
    return x[idx]


def main() -> None:
    args = parse_args()

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    x = load_embeddings(args.adata, args.embed_key, args.n_cells, args.seed)
    print(f"Loaded embeddings: {x.shape} from {args.adata}::{args.embed_key}")

    converter = StateSEConverter(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        max_set_len=args.n_cells,
        verbose=True,
    )

    if args.find:
        matches = converter.find_perturbations(args.find)
        print(f"Found {len(matches)} perturbation labels matching {args.find!r}")
        for m in matches[:20]:
            print(" ", repr(m))
        if args.perturbation is None:
            if not matches:
                raise SystemExit(f"No perturbation matched {args.find!r}")
            args.perturbation = matches[0]
            print(f"Using first match: {args.perturbation!r}")

    if args.perturbation is None and args.run_first_n <= 0:
        print("No --perturbation, --find, or --run-first-n provided. Listing first 20 non-control perturbations:")
        for p in converter.list_perturbations(include_control=False)[:20]:
            print(" ", repr(p))
        return

    if args.perturbation is not None:
        y = converter.convert_one(x, args.perturbation, return_cpu=True)
        print(f"One-perturbation output: {tuple(y.shape)}")
        delta = y.numpy() - x
        print(f"Delta mean L2 per cell: {np.linalg.norm(delta, axis=1).mean():.6f}")
        print(f"Input mean/std:  {x.mean():.6f} / {x.std():.6f}")
        print(f"Output mean/std: {y.numpy().mean():.6f} / {y.numpy().std():.6f}")

        if args.output_npy:
            np.save(args.output_npy, y.numpy().astype(np.float32))
            print(f"Saved one-perturbation prediction: {args.output_npy}")

    if args.run_first_n > 0:
        perts = converter.list_perturbations(include_control=False)[: args.run_first_n]
        all_labels = []
        all_preds = []
        for labels, preds in converter.convert_many_iter(
            x,
            perturbations=perts,
            chunk_size=args.chunk_size,
            return_cpu=True,
        ):
            print(f"Chunk: {len(labels)} perturbations -> {tuple(preds.shape)}")
            all_labels.extend(labels)
            all_preds.append(preds.numpy().astype(np.float32))

        pred_arr = np.concatenate(all_preds, axis=0)
        print(f"Many-perturbation output: {pred_arr.shape}")

        if args.output_many_npz:
            np.savez_compressed(
                args.output_many_npz,
                labels=np.array(all_labels, dtype=object),
                predictions=pred_arr,
            )
            print(f"Saved many-perturbation predictions: {args.output_many_npz}")


if __name__ == "__main__":
    main()
