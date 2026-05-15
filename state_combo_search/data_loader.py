#!/usr/bin/env python
"""
data_loader.py

Utilities for loading start and target cell-state embeddings from an AnnData .h5ad
that already contains SE embeddings in adata.obsm[embed_key], usually "X_state".

This module is intentionally independent of the ST-SE converter and scoring modules.

Typical use:
    from data_loader import load_start_target_embeddings

    pair = load_start_target_embeddings(
        h5ad_path="WT_256_per_cell_name.SE600M.h5ad",
        start_cell="J82",
        target_cell="A-172",
        cell_col="cell_name",
        embed_key="X_state",
        start_sample=256,
        target_sample=256,
        seed=42,
    )

    start_embeddings = pair.start_embeddings  # np.ndarray [256, 2058]
    target_embeddings = pair.target_embeddings  # np.ndarray [256, 2058]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional, Sequence, Union, Dict, Any

import json
import numpy as np
import scanpy as sc


SampleSpec = Union[int, Literal["all"]]


@dataclass
class LoadedCellStates:
    """Container returned by load_start_target_embeddings()."""

    start_embeddings: np.ndarray
    target_embeddings: np.ndarray
    start_cell: str
    target_cell: str
    cell_col: str
    embed_key: str
    start_obs_names: list[str]
    target_obs_names: list[str]
    start_n_available: int
    target_n_available: int
    start_n_sampled: int
    target_n_sampled: int
    seed: int
    replace_start: bool
    replace_target: bool

    def metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata, excluding the embedding arrays."""
        d = asdict(self)
        d.pop("start_embeddings", None)
        d.pop("target_embeddings", None)
        return d

    def save_npz(self, output_npz: str | Path) -> None:
        """
        Save start/target embeddings and metadata in a compact NumPy archive.

        This is useful for:
          - caching the extracted start/target state
          - feeding the same target state into multiple scoring/search runs
          - making the search workflow reproducible

        Note:
          For the beam search inner loop, keep embeddings in memory. Use this file
          for checkpointing / reproducibility, not per-candidate I/O.
        """
        output_npz = Path(output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_npz,
            start_embeddings=self.start_embeddings.astype(np.float32, copy=False),
            target_embeddings=self.target_embeddings.astype(np.float32, copy=False),
            metadata=json.dumps(self.metadata()),
            start_obs_names=np.asarray(self.start_obs_names, dtype=object),
            target_obs_names=np.asarray(self.target_obs_names, dtype=object),
        )

    @staticmethod
    def load_npz(input_npz: str | Path) -> "LoadedCellStates":
        """Load a cache created by save_npz()."""
        z = np.load(input_npz, allow_pickle=True)
        metadata = json.loads(str(z["metadata"].item()))

        return LoadedCellStates(
            start_embeddings=z["start_embeddings"].astype(np.float32, copy=False),
            target_embeddings=z["target_embeddings"].astype(np.float32, copy=False),
            start_obs_names=list(z["start_obs_names"].astype(str)),
            target_obs_names=list(z["target_obs_names"].astype(str)),
            **{
                k: v
                for k, v in metadata.items()
                if k not in {"start_obs_names", "target_obs_names"}
            },
        )


def _parse_sample_spec(value: SampleSpec) -> SampleSpec:
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "all":
            return "all"
        try:
            ivalue = int(value)
        except ValueError as exc:
            raise ValueError(f"Sample spec must be an integer or 'all', got {value!r}") from exc
        return ivalue
    return int(value)


def _sample_indices(
    available_indices: np.ndarray,
    sample: SampleSpec,
    rng: np.random.Generator,
    replace_if_needed: bool,
    label: str,
) -> tuple[np.ndarray, bool]:
    """
    Sample indices reproducibly.

    If sample == "all", all available indices are returned.
    If sample is an integer larger than available cells:
      - if replace_if_needed=True, sample with replacement
      - otherwise raise ValueError
    """
    sample = _parse_sample_spec(sample)

    if len(available_indices) == 0:
        raise ValueError(f"No cells available for {label}")

    if sample == "all":
        return available_indices.copy(), False

    if sample <= 0:
        raise ValueError(f"Sample size for {label} must be positive or 'all', got {sample}")

    if len(available_indices) >= sample:
        chosen = rng.choice(available_indices, size=sample, replace=False)
        return chosen, False

    if not replace_if_needed:
        raise ValueError(
            f"Requested {sample} cells for {label}, but only {len(available_indices)} are available. "
            "Set replace_if_needed=True to sample with replacement."
        )

    chosen = rng.choice(available_indices, size=sample, replace=True)
    return chosen, True


def load_start_target_embeddings(
    h5ad_path: str | Path,
    start_cell: str,
    target_cell: str,
    cell_col: str = "cell_name",
    embed_key: str = "X_state",
    start_sample: SampleSpec = 256,
    target_sample: SampleSpec = 256,
    seed: int = 42,
    replace_if_needed: bool = True,
    dtype: np.dtype = np.float32,
) -> LoadedCellStates:
    """
    Load start and target cell-state embeddings from an h5ad file.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData object with SE embeddings in adata.obsm[embed_key].
    start_cell:
        Starting cell type / cell line label, for example "J82".
    target_cell:
        Target cell type / cell line label, for example "A-172".
    cell_col:
        adata.obs column containing cell-type / cell-line labels.
    embed_key:
        adata.obsm key containing SE embeddings, usually "X_state".
    start_sample:
        Number of starting cells to sample, or "all".
    target_sample:
        Number of target cells to sample, or "all".
        For scoring, target_sample can be much larger than start_sample if the
        h5ad contains more target cells.
    seed:
        Random seed for reproducible sampling.
    replace_if_needed:
        If True, sample with replacement when requested sample size exceeds the
        number of available cells.
    dtype:
        Output dtype for embeddings.

    Returns
    -------
    LoadedCellStates
        start_embeddings: np.ndarray [start_sample, emb_dim]
        target_embeddings: np.ndarray [target_sample, emb_dim]
    """
    h5ad_path = Path(h5ad_path)
    rng = np.random.default_rng(seed)

    ad = sc.read_h5ad(h5ad_path)

    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available columns: {list(ad.obs.columns)}")

    if embed_key not in ad.obsm:
        raise KeyError(f"embed_key={embed_key!r} not found in adata.obsm. Available keys: {list(ad.obsm.keys())}")

    labels = ad.obs[cell_col].astype(str).values
    start_mask = labels == str(start_cell)
    target_mask = labels == str(target_cell)

    start_available = np.where(start_mask)[0]
    target_available = np.where(target_mask)[0]

    if len(start_available) == 0:
        examples = sorted(set(labels))[:20]
        raise ValueError(f"No cells found for start_cell={start_cell!r} in {cell_col!r}. Example labels: {examples}")

    if len(target_available) == 0:
        examples = sorted(set(labels))[:20]
        raise ValueError(f"No cells found for target_cell={target_cell!r} in {cell_col!r}. Example labels: {examples}")

    start_idx, replace_start = _sample_indices(
        start_available,
        sample=start_sample,
        rng=rng,
        replace_if_needed=replace_if_needed,
        label=f"start_cell={start_cell}",
    )

    target_idx, replace_target = _sample_indices(
        target_available,
        sample=target_sample,
        rng=rng,
        replace_if_needed=replace_if_needed,
        label=f"target_cell={target_cell}",
    )

    X_state = np.asarray(ad.obsm[embed_key])

    start_embeddings = X_state[start_idx].astype(dtype, copy=True)
    target_embeddings = X_state[target_idx].astype(dtype, copy=True)

    return LoadedCellStates(
        start_embeddings=start_embeddings,
        target_embeddings=target_embeddings,
        start_cell=str(start_cell),
        target_cell=str(target_cell),
        cell_col=str(cell_col),
        embed_key=str(embed_key),
        start_obs_names=list(ad.obs_names[start_idx].astype(str)),
        target_obs_names=list(ad.obs_names[target_idx].astype(str)),
        start_n_available=int(len(start_available)),
        target_n_available=int(len(target_available)),
        start_n_sampled=int(start_embeddings.shape[0]),
        target_n_sampled=int(target_embeddings.shape[0]),
        seed=int(seed),
        replace_start=bool(replace_start),
        replace_target=bool(replace_target),
    )


def list_cell_counts(
    h5ad_path: str | Path,
    cell_col: str = "cell_name",
) -> "np.ndarray":
    """
    Convenience utility to print and return counts per cell label.
    """
    ad = sc.read_h5ad(h5ad_path)
    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available columns: {list(ad.obs.columns)}")
    counts = ad.obs[cell_col].astype(str).value_counts()
    return counts
