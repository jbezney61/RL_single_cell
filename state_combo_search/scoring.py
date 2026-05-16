#!/usr/bin/env python
"""
scoring.py

Distribution-aware scoring functions for STATE/ST-SE sequential drug search.

Inputs are cell-state embeddings:
    predicted_states: [B, N, D] or [N, D]
    target_state:     [M, D]

where:
    B = number of candidate perturbations / beam nodes
    N = number of predicted cells, usually 256
    M = number of target cells, often 256 now but can be larger later
    D = SE embedding dimension, e.g. 2058

Implemented scores
------------------
1. Sliced Wasserstein distance
   - Very fast, GPU-friendly, batch-friendly.
   - Projects both cell populations onto random 1D axes, sorts, compares quantiles.
   - Good for screening many candidate perturbations inside beam search.

2. Sinkhorn optimal transport distance
   - More biologically faithful distributional matching.
   - Builds a full pairwise cell-cell cost matrix and solves soft optimal transport.
   - Good for reranking top candidates or final evaluation.

Recommended use in beam search
------------------------------
Use Sliced Wasserstein to score all candidates quickly, keep top M, then rerank
those top M using Sinkhorn OT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch


TensorLike = Union[np.ndarray, torch.Tensor]


def as_3d_tensor(x: TensorLike, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert [N, D] or [B, N, D] input to [B, N, D]."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype, non_blocking=True)
    else:
        t = torch.as_tensor(x, device=device, dtype=dtype)

    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected [N, D] or [B, N, D], got shape {tuple(t.shape)}")
    return t


def l2_normalize_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize embeddings along the embedding dimension."""
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)


def make_random_projections(
    embedding_dim: int,
    n_projections: int = 64,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create fixed random unit vectors for sliced Wasserstein.

    Reusing the same projection matrix across the whole search makes scores
    comparable across candidates and beam depths.
    """
    device = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    proj = torch.randn((embedding_dim, n_projections), generator=generator, dtype=dtype)
    proj = proj / (torch.linalg.norm(proj, dim=0, keepdim=True) + 1e-8)
    return proj.to(device)


def _quantile_resample_sorted(sorted_values: torch.Tensor, n_out: int) -> torch.Tensor:
    """
    Resample sorted 1D empirical distributions to a common number of quantiles.

    sorted_values:
        [B, N, K] or [1, M, K]

    returns:
        [B, n_out, K] or [1, n_out, K]

    This lets Sliced Wasserstein compare predicted 256 cells to target 10,000 cells.
    """
    b, n, k = sorted_values.shape
    if n == n_out:
        return sorted_values

    # torch interpolate expects [B, C, L], so treat projections as channels.
    x = sorted_values.permute(0, 2, 1)  # [B, K, N]
    x = torch.nn.functional.interpolate(
        x,
        size=n_out,
        mode="linear",
        align_corners=True,
    )
    return x.permute(0, 2, 1)  # [B, n_out, K]


def sliced_wasserstein_distance(
    predicted_states: TensorLike,
    target_state: TensorLike,
    projections: Optional[torch.Tensor] = None,
    n_projections: int = 64,
    seed: int = 42,
    normalize: bool = True,
    p: Literal[1, 2] = 2,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """
    Batch Sliced Wasserstein distance.

    Lower is better.

    This is the fast screening score:
      1. L2-normalize embeddings, optionally.
      2. Project both cell clouds onto K random 1D axes.
      3. Sort projected values along the cell dimension.
      4. Compare sorted quantiles.

    Returns:
        scores: torch.Tensor [B]
    """
    if device is None:
        if torch.is_tensor(predicted_states):
            device = predicted_states.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    X = as_3d_tensor(predicted_states, device=device)
    Y = as_3d_tensor(target_state, device=device)

    if Y.shape[0] != 1 and Y.shape[0] != X.shape[0]:
        raise ValueError("target_state must be [M, D], [1, M, D], or [B, M, D] matching predicted batch size")

    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"Embedding dimensions differ: predicted D={X.shape[-1]}, target D={Y.shape[-1]}")

    if normalize:
        X = l2_normalize_embeddings(X)
        Y = l2_normalize_embeddings(Y)

    if projections is None:
        projections = make_random_projections(
            embedding_dim=X.shape[-1],
            n_projections=n_projections,
            device=device,
            dtype=X.dtype,
            seed=seed,
        )
    else:
        projections = projections.to(device=device, dtype=X.dtype)

    # Project cells onto random lines.
    X_proj = X @ projections  # [B, N, K]
    Y_proj = Y @ projections  # [1 or B, M, K]

    X_sorted = torch.sort(X_proj, dim=1).values
    Y_sorted = torch.sort(Y_proj, dim=1).values

    # Compare same number of quantiles even if N != M.
    Y_sorted = _quantile_resample_sorted(Y_sorted, n_out=X_sorted.shape[1])

    diff = X_sorted - Y_sorted

    if p == 1:
        return diff.abs().mean(dim=(1, 2))
    if p == 2:
        return (diff ** 2).mean(dim=(1, 2))

    raise ValueError("p must be 1 or 2")


def pairwise_cost_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric: Literal["cosine", "sqeuclidean", "euclidean"] = "cosine",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise cell-cell cost matrix C.

    X: [B, N, D]
    Y: [1, M, D] or [B, M, D]

    returns:
        C: [B, N, M]
    """
    if normalize:
        X = l2_normalize_embeddings(X)
        Y = l2_normalize_embeddings(Y)

    if metric == "cosine":
        sim = torch.bmm(X, Y.transpose(1, 2))
        return 1.0 - sim.clamp(-1.0, 1.0)

    if metric == "sqeuclidean":
        return torch.cdist(X, Y, p=2) ** 2

    if metric == "euclidean":
        return torch.cdist(X, Y, p=2)

    raise ValueError(f"Unsupported metric: {metric}")


def sinkhorn_ot_distance(
    predicted_states: TensorLike,
    target_state: TensorLike,
    metric: Literal["cosine", "sqeuclidean", "euclidean"] = "cosine",
    normalize: bool = True,
    epsilon: float = 0.05,
    n_iters: int = 100,
    device: Optional[str | torch.device] = None,
    return_transport: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch entropic Sinkhorn optimal transport distance.

    Lower is better.

    This is the faithful distribution-matching score:
      1. Compute all pairwise predicted-cell to target-cell costs.
      2. Solve a soft optimal transport problem with uniform cell weights.
      3. Return expected transport cost.

    For cosine cost on L2-normalized SE embeddings, reasonable epsilon values are
    often 0.03 to 0.10. Smaller epsilon is sharper but less stable/slower.
    """
    if device is None:
        if torch.is_tensor(predicted_states):
            device = predicted_states.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    X = as_3d_tensor(predicted_states, device=device)
    Y = as_3d_tensor(target_state, device=device)

    if Y.shape[0] == 1 and X.shape[0] > 1:
        Y = Y.expand(X.shape[0], -1, -1)
    elif Y.shape[0] != X.shape[0]:
        raise ValueError("target_state must be [M, D], [1, M, D], or [B, M, D] matching predicted batch size")

    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"Embedding dimensions differ: predicted D={X.shape[-1]}, target D={Y.shape[-1]}")

    B, N, _ = X.shape
    M = Y.shape[1]

    C = pairwise_cost_matrix(X, Y, metric=metric, normalize=normalize)  # [B, N, M]

    # Uniform weights in log-domain.
    log_a = torch.full((B, N), -np.log(N), dtype=X.dtype, device=device)
    log_b = torch.full((B, M), -np.log(M), dtype=X.dtype, device=device)

    log_K = -C / epsilon
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(log_K + v[:, None, :], dim=2)
        v = log_b - torch.logsumexp(log_K + u[:, :, None], dim=1)

    log_P = log_K + u[:, :, None] + v[:, None, :]
    P = torch.exp(log_P)
    score = torch.sum(P * C, dim=(1, 2))

    if return_transport:
        return score, P
    return score


@dataclass
class DistributionScorer:
    """
    Convenience scorer for beam search.

    Stores:
      - target state on GPU
      - fixed random projections for Sliced Wasserstein
      - Sinkhorn hyperparameters

    Use:
        scorer = DistributionScorer(target_embeddings, device="cuda:0")
        fast_scores = scorer.sliced_wasserstein(candidate_batch)
        final_scores = scorer.sinkhorn(top_candidate_batch)
    """

    target_state: TensorLike
    device: Optional[str | torch.device] = None
    normalize: bool = True
    n_projections: int = 64
    projection_seed: int = 42
    sw_p: Literal[1, 2] = 2
    sinkhorn_metric: Literal["cosine", "sqeuclidean", "euclidean"] = "cosine"
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 100

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        self.target = as_3d_tensor(self.target_state, device=self.device)
        if self.target.shape[0] != 1:
            raise ValueError("DistributionScorer expects one target distribution: [M, D] or [1, M, D]")

        self.projections = make_random_projections(
            embedding_dim=self.target.shape[-1],
            n_projections=self.n_projections,
            device=self.device,
            dtype=torch.float32,
            seed=self.projection_seed,
        )

    def sliced_wasserstein(self, predicted_states: TensorLike) -> torch.Tensor:
        return sliced_wasserstein_distance(
            predicted_states=predicted_states,
            target_state=self.target,
            projections=self.projections,
            normalize=self.normalize,
            p=self.sw_p,
            device=self.device,
        )

    def sinkhorn(self, predicted_states: TensorLike, return_transport: bool = False):
        return sinkhorn_ot_distance(
            predicted_states=predicted_states,
            target_state=self.target,
            metric=self.sinkhorn_metric,
            normalize=self.normalize,
            epsilon=self.sinkhorn_epsilon,
            n_iters=self.sinkhorn_iters,
            device=self.device,
            return_transport=return_transport,
        )

    def two_stage_rank(self, predicted_states: TensorLike, top_m: int = 50) -> dict:
        """
        Fast two-stage ranking:
          1. Score all candidates with Sliced Wasserstein.
          2. Rerank the best top_m candidates with Sinkhorn OT.
        """
        X = as_3d_tensor(predicted_states, device=self.device)
        B = X.shape[0]
        top_m = min(top_m, B)

        sw_scores = self.sliced_wasserstein(X)
        prelim_idx = torch.topk(-sw_scores, k=top_m).indices  # lower is better

        sink_scores = self.sinkhorn(X[prelim_idx])
        final_order = torch.argsort(sink_scores)

        final_idx = prelim_idx[final_order]
        final_sinkhorn = sink_scores[final_order]
        final_sw = sw_scores[final_idx]

        return {
            "final_indices": final_idx,
            "final_sinkhorn_scores": final_sinkhorn,
            "final_sliced_wasserstein_scores": final_sw,
            "all_sliced_wasserstein_scores": sw_scores,
            "prelim_indices": prelim_idx,
        }
