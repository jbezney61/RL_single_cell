#!/usr/bin/env python
"""
test_search.py

Small smoke test for search.py.

This script intentionally uses a very small search by default:
    max_depth=2
    beam_size=2
    max_drugs_to_consider=6
    prefilter_multiplier=3

Example:
    export CUDA_VISIBLE_DEVICES=0

    python test_search.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --cell-col cell_name \
      --embed-key X_state \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --config search_config.yaml \
      --output-dir test_search_J82_to_A172 \
      --algorithm deterministic_beam

Try diverse beam:
    python test_search.py ... --algorithm diverse_beam
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from converter import StateSEConverter
from data_loader import load_start_target_embeddings
from scoring import DistributionScorer
from search import load_search_config, run_search


def parse_args():
    p = argparse.ArgumentParser()

    # Input data
    p.add_argument("--adata", required=True)
    p.add_argument("--start-cell", required=True)
    p.add_argument("--target-cell", required=True)
    p.add_argument("--cell-col", default="cell_name")
    p.add_argument("--embed-key", default="X_state")
    p.add_argument("--start-sample", default="256")
    p.add_argument("--target-sample", default="256")

    # Model
    p.add_argument("--model-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)

    # Config / output
    p.add_argument("--config", default="search_config.yaml")
    p.add_argument("--output-dir", default="test_search_outputs")
    p.add_argument("--algorithm", default=None, choices=["deterministic_beam", "diverse_beam"])

    # Smoke-test overrides
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--beam-size", type=int, default=2)
    p.add_argument("--max-drugs-to-consider", type=int, default=30)
    p.add_argument("--prefilter-multiplier", type=int, default=3)
    p.add_argument("--converter-chunk-size", type=int, default=3)

    # Scoring
    p.add_argument("--n-projections", type=int, default=32)
    p.add_argument("--sinkhorn-iters", type=int, default=50)
    p.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = load_search_config(args.config)

    # Override config for a short smoke test.
    cfg.setdefault("search", {})
    cfg["search"]["max_depth"] = args.max_depth
    cfg["search"]["beam_size"] = args.beam_size
    cfg["search"]["max_drugs_to_consider"] = args.max_drugs_to_consider
    cfg["search"]["prefilter_multiplier"] = args.prefilter_multiplier
    cfg["search"]["converter_chunk_size"] = args.converter_chunk_size

    if args.algorithm is not None:
        cfg["search"]["algorithm"] = args.algorithm

    cfg.setdefault("output", {})
    cfg["output"]["output_dir"] = args.output_dir

    print("\nLoading start and target embeddings")
    pair = load_start_target_embeddings(
        h5ad_path=args.adata,
        start_cell=args.start_cell,
        target_cell=args.target_cell,
        cell_col=args.cell_col,
        embed_key=args.embed_key,
        start_sample=args.start_sample,
        target_sample=args.target_sample,
        seed=args.seed,
    )
    print(f"start:  {pair.start_cell} {pair.start_embeddings.shape}")
    print(f"target: {pair.target_cell} {pair.target_embeddings.shape}")

    print("\nLoading converter")
    converter = StateSEConverter(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=device,
        max_set_len=256,
    )

    print("\nInitializing scorer")
    scorer = DistributionScorer(
        target_state=pair.target_embeddings,
        device=device,
        normalize=True,
        n_projections=args.n_projections,
        projection_seed=args.seed,
        sinkhorn_metric="cosine",
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
    )

    print("\nRunning search")
    out = run_search(
        converter=converter,
        scorer=scorer,
        start_embeddings=pair.start_embeddings,
        cfg=cfg,
        perturbations=None,
        output_dir=args.output_dir,
        start_cell=args.start_cell,
        target_cell=args.target_cell,
    )

    print("\nDone")
    print(f"results:    {out['results_tsv']}")
    print(f"checkpoint: {out['checkpoint']}")


if __name__ == "__main__":
    main()
