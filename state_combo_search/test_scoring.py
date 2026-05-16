#!/usr/bin/env python
"""
test_scoring.py

Test and compare Sliced Wasserstein and Sinkhorn OT scoring on SE embedding states.

Mode A: compare start vs target from an h5ad using data_loader.py
    python test_scoring.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --cell-col cell_name \
      --embed-key X_state

Mode B: compare converter predictions for one or more drugs against target
    python test_scoring.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --find Trametinib \
      --run-first-n 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from data_loader import load_start_target_embeddings
from scoring import DistributionScorer


def parse_args():
    p = argparse.ArgumentParser()

    # Data args
    p.add_argument("--adata", required=True)
    p.add_argument("--start-cell", required=True)
    p.add_argument("--target-cell", required=True)
    p.add_argument("--cell-col", default="cell_name")
    p.add_argument("--embed-key", default="X_state")
    p.add_argument("--start-sample", default="256")
    p.add_argument("--target-sample", default="256")
    p.add_argument("--seed", type=int, default=42)

    # Optional converter args
    p.add_argument("--model-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--find", default=None, help="Substring to select one perturbation from the model map")
    p.add_argument("--run-first-n", type=int, default=0, help="Run first N non-control perturbations through converter")
    p.add_argument("--chunk-size", type=int, default=4)

    # Scoring args
    p.add_argument("--device", default=None)
    p.add_argument("--n-projections", type=int, default=64)
    p.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    p.add_argument("--sinkhorn-iters", type=int, default=100)
    p.add_argument("--top-m", type=int, default=10)

    return p.parse_args()


def synchronize_if_cuda(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_score_table(labels, sw_scores, sink_scores=None):
    print("\nScores; lower is better")
    print("-" * 100)
    if sink_scores is None:
        print(f"{'rank':>4}  {'label':<60}  {'sliced_wasserstein':>20}")
        order = np.argsort(sw_scores)
        for rank, idx in enumerate(order, start=1):
            print(f"{rank:>4}  {labels[idx]:<60.60}  {sw_scores[idx]:>20.6g}")
    else:
        order = np.argsort(sink_scores)
        print(f"{'rank':>4}  {'label':<60}  {'sliced_wasserstein':>20}  {'sinkhorn_ot':>14}")
        for rank, idx in enumerate(order, start=1):
            print(f"{rank:>4}  {labels[idx]:<60.60}  {sw_scores[idx]:>20.6g}  {sink_scores[idx]:>14.6g}")
    print("-" * 100)


def main():
    args = parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    print("\nLoaded embeddings")
    print(f"  start cell:  {pair.start_cell}")
    print(f"  target cell: {pair.target_cell}")
    print(f"  start shape: {pair.start_embeddings.shape}")
    print(f"  target shape:{pair.target_embeddings.shape}")

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

    # Baseline: score start state against target state.
    print("\nBaseline: start state versus target state")

    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    start_sw = scorer.sliced_wasserstein(pair.start_embeddings)
    synchronize_if_cuda(device)
    t_sw = time.perf_counter() - t0

    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    start_sink = scorer.sinkhorn(pair.start_embeddings)
    synchronize_if_cuda(device)
    t_sink = time.perf_counter() - t0

    print(f"  Sliced Wasserstein: {float(start_sw.item()):.6g}  time={t_sw:.4f}s")
    print(f"  Sinkhorn OT:        {float(start_sink.item()):.6g}  time={t_sink:.4f}s")

    print("\nHow to interpret these two scores")
    print("  Sliced Wasserstein:")
    print("    - Projects the cell cloud onto random 1D axes.")
    print("    - Sorts projected cells and compares quantiles.")
    print("    - Very fast and batch-friendly; use this to screen many drugs during beam expansion.")
    print("    - Approximate: can miss geometry not captured by the sampled projections.")
    print("  Sinkhorn OT:")
    print("    - Computes pairwise cell-cell costs and solves a soft optimal transport matching.")
    print("    - More faithful to single-cell heterogeneity and distributional overlap.")
    print("    - Slower; use for reranking top candidates or final analysis.")
    print("  Recommended search pattern:")
    print("    - Score all candidates with Sliced Wasserstein.")
    print("    - Keep top M, e.g. 5x beam size.")
    print("    - Rerank top M with Sinkhorn OT.")

    # Optional converter predictions.
    if args.model_dir is None:
        print("\nNo --model-dir supplied; skipping converter prediction scoring.")
        return

    from converter import StateSEConverter

    converter = StateSEConverter(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=device,
        max_set_len=256,
    )

    candidate_labels = []
    candidate_states = []

    if args.find:
        matches = [p for p in converter.list_perturbations(include_control=False) if args.find.lower() in p.lower()]
        if not matches:
            raise ValueError(f"No perturbations matched --find {args.find!r}")
        p = matches[0]
        print(f"\nRunning converter for matched perturbation: {p}")
        pred = converter.convert_one(pair.start_embeddings, p, return_cpu=False)
        candidate_labels.append(p)
        candidate_states.append(pred)

    if args.run_first_n > 0:
        perts = converter.list_perturbations(include_control=False)[: args.run_first_n]
        print(f"\nRunning converter for first {len(perts)} perturbations")
        for labels, preds in converter.convert_many_iter(
            pair.start_embeddings,
            perturbations=perts,
            chunk_size=args.chunk_size,
            return_cpu=False,
        ):
            for j, lab in enumerate(labels):
                candidate_labels.append(lab)
                candidate_states.append(preds[j])

    if not candidate_states:
        print("\nNo converter candidates requested; use --find or --run-first-n.")
        return

    candidate_batch = torch.stack(candidate_states, dim=0)

    print(f"\nCandidate batch shape: {tuple(candidate_batch.shape)}")

    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    sw_scores = scorer.sliced_wasserstein(candidate_batch)
    synchronize_if_cuda(device)
    t_sw = time.perf_counter() - t0

    synchronize_if_cuda(device)
    t0 = time.perf_counter()
    sink_scores = scorer.sinkhorn(candidate_batch)
    synchronize_if_cuda(device)
    t_sink = time.perf_counter() - t0

    sw_np = sw_scores.detach().cpu().numpy()
    sink_np = sink_scores.detach().cpu().numpy()

    print(f"\nScoring time for {len(candidate_labels)} candidates")
    print(f"  Sliced Wasserstein: {t_sw:.4f}s total, {t_sw / len(candidate_labels):.4f}s/candidate")
    print(f"  Sinkhorn OT:        {t_sink:.4f}s total, {t_sink / len(candidate_labels):.4f}s/candidate")

    print_score_table(candidate_labels, sw_np, sink_np)

    if len(candidate_labels) > args.top_m:
        print(f"\nTwo-stage ranking example with top_m={args.top_m}")
        out = scorer.two_stage_rank(candidate_batch, top_m=args.top_m)
        final_indices = out["final_indices"].detach().cpu().numpy()
        for rank, idx in enumerate(final_indices, start=1):
            print(
                f"{rank:>3}  {candidate_labels[idx]:<60.60}  "
                f"SW={float(out['final_sliced_wasserstein_scores'][rank-1]):.6g}  "
                f"Sinkhorn={float(out['final_sinkhorn_scores'][rank-1]):.6g}"
            )


if __name__ == "__main__":
    main()
