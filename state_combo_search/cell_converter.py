#!/usr/bin/env python
"""
cell_converter.py

End-to-end CLI for one ST-SE cell-state conversion search.

This wrapper runs the full workflow for a single conversion:

    1. Load starting and target cell-state embeddings from an SE-embedded h5ad.
    2. Load the ST-SE converter model onto GPU/CPU.
    3. Initialize the distributional scorer.
    4. Run deterministic or diverse beam search.
    5. Generate a summary report with plots and tables.

It expects these companion modules to be importable from the same directory or PYTHONPATH:

    data_loader.py
    converter.py
    scoring.py
    search.py
    make_search_report.py

Minimal example
---------------
    export CUDA_VISIBLE_DEVICES=0

    python cell_converter.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --cell-col cell_name \
      --embed-key X_state \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --output-dir runs/J82_to_A172 \
      --algorithm deterministic_beam \
      --max-depth 5 \
      --beam-size 32

Small smoke test
----------------
    python cell_converter.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --output-dir runs/test_J82_to_A172 \
      --max-depth 2 \
      --beam-size 2 \
      --max-drugs-to-consider 6 \
      --prefilter-multiplier 3 \
      --converter-chunk-size 3 \
      --n-projections 32 \
      --sinkhorn-iters 50

Outputs
-------
output_dir/
    search/
        results.tsv
        checkpoint.pt
        search_config.used.yaml
    cache/
        start_target_states.npz
    report/
        summary.md
        tables/
        figures/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def parse_csv_or_none(x: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into a list, preserving None."""
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"none", "null"}:
        return None
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an end-to-end ST-SE sequential drug search to convert one SE-embedded "
            "cell type/state into another."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = p.add_argument_group("Required inputs")
    required.add_argument("--adata", required=True, help="Input h5ad containing SE embeddings in adata.obsm[embed_key].")
    required.add_argument("--start-cell", required=True, help="Starting cell type/cell line label, e.g. J82.")
    required.add_argument("--target-cell", required=True, help="Target cell type/cell line label, e.g. A-172.")
    required.add_argument("--model-dir", required=True, help="ST-SE training run directory.")
    required.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")

    data = p.add_argument_group("Data loading")
    data.add_argument("--checkpoint", default=None, help="Path to ST-SE checkpoint. Defaults to model_dir/checkpoints/final.ckpt.")
    data.add_argument("--cell-col", default="cell_name", help="adata.obs column containing start/target labels.")
    data.add_argument("--embed-key", default="X_state", help="adata.obsm key containing SE embeddings.")
    data.add_argument("--start-sample", default="256", help="Number of start cells to sample, or 'all'.")
    data.add_argument("--target-sample", default="256", help="Number of target cells to sample, or 'all'.")
    data.add_argument("--seed", type=int, default=42, help="Random seed for sampling and projections.")
    data.add_argument("--no-replace-if-needed", action="store_true", help="Error if requested sample size exceeds available cells.")
    data.add_argument("--save-state-cache", action=argparse.BooleanOptionalAction, default=True, help="Save start/target embeddings as npz.")

    compute = p.add_argument_group("Compute")
    compute.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu. Defaults to cuda:0 if available.")
    compute.add_argument("--max-set-len", type=int, default=256, help="Maximum cells per ST-SE forward pass.")
    compute.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True, help="Use autocast mixed precision on CUDA.")
    compute.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16", help="Autocast dtype.")

    search = p.add_argument_group("Search")
    search.add_argument("--config", default=None, help="Optional YAML search config; CLI values override it.")
    search.add_argument("--algorithm", choices=["deterministic_beam", "diverse_beam"], default="deterministic_beam")
    search.add_argument("--max-depth", type=int, default=5, help="Maximum number of sequential drugs.")
    search.add_argument("--beam-size", type=int, default=32, help="Number of paths retained after each depth.")
    search.add_argument("--prefilter-multiplier", type=int, default=10, help="Sinkhorn rerank pool = beam_size * this value.")
    search.add_argument("--converter-chunk-size", type=int, default=16, help="Number of perturbations processed per converter chunk.")
    search.add_argument("--max-drugs-to-consider", type=int, default=None, help="Optional limiter for smoke tests only.")

    filt = p.add_argument_group("Perturbation filters and constraints")
    filt.add_argument("--allow-repeated-drug-names", action=argparse.BooleanOptionalAction, default=False,
                      help="If false, a path cannot use the same base drug more than once, even at another dose.")
    filt.add_argument("--allow-repeated-perturbation-labels", action=argparse.BooleanOptionalAction, default=False,
                      help="If false, exact perturbation labels cannot repeat.")
    filt.add_argument("--allow-control-like-drugs", action=argparse.BooleanOptionalAction, default=False,
                      help="If false, DMSO/control-like perturbations are excluded.")
    filt.add_argument("--banned-drug-names", default=None, help="Comma-separated base drug names to exclude.")
    filt.add_argument("--banned-perturbation-labels", default=None, help="Comma-separated exact perturbation labels to exclude.")
    filt.add_argument("--allowed-drug-names", default=None, help="Comma-separated base drug names to allow; excludes all others.")
    filt.add_argument("--allowed-drug-name-contains", default=None, help="Comma-separated substrings allowed in base drug names.")

    diversity = p.add_argument_group("Diverse beam settings")
    diversity.add_argument("--path-overlap-penalty", type=float, default=0.05, help="Diverse beam path-overlap penalty.")
    diversity.add_argument("--use-state-similarity-penalty", action=argparse.BooleanOptionalAction, default=False)
    diversity.add_argument("--state-similarity-penalty", type=float, default=0.02)

    scoring = p.add_argument_group("Scoring")
    scoring.add_argument("--n-projections", type=int, default=64, help="Random projections for Sliced Wasserstein.")
    scoring.add_argument("--sinkhorn-epsilon", type=float, default=0.05, help="Entropic regularization for Sinkhorn OT.")
    scoring.add_argument("--sinkhorn-iters", type=int, default=100, help="Sinkhorn iterations.")
    scoring.add_argument("--sinkhorn-metric", choices=["cosine", "sqeuclidean", "euclidean"], default="cosine")
    scoring.add_argument("--no-normalize-embeddings", action="store_true", help="Disable L2 normalization before scoring.")

    out = p.add_argument_group("Output and report")
    out.add_argument("--state-checkpoint-dtype", choices=["float16", "float32"], default="float16")
    out.add_argument("--skip-report", action="store_true", help="Run search only.")
    out.add_argument("--conversion-threshold", type=float, default=None, help="Report-only Sinkhorn threshold for counting conversions.")
    out.add_argument("--report-top-n-paths", type=int, default=25)
    out.add_argument("--report-top-n-drugs", type=int, default=20)
    out.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite non-empty output directory.")

    return p.parse_args()


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update base with update."""
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_search_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a search.py-compatible config from YAML + CLI args."""
    cfg = load_yaml_config(args.config)

    cli_cfg = {
        "search": {
            "algorithm": args.algorithm,
            "max_depth": args.max_depth,
            "beam_size": args.beam_size,
            "prefilter_multiplier": args.prefilter_multiplier,
            "converter_chunk_size": args.converter_chunk_size,
            "max_drugs_to_consider": args.max_drugs_to_consider,
        },
        "constraints": {
            "allow_repeated_drug_names": args.allow_repeated_drug_names,
            "allow_repeated_perturbation_labels": args.allow_repeated_perturbation_labels,
            "allow_control_like_drugs": args.allow_control_like_drugs,
            "banned_drug_names": parse_csv_or_none(args.banned_drug_names) or [],
            "banned_perturbation_labels": parse_csv_or_none(args.banned_perturbation_labels) or [],
            "allowed_drug_names": parse_csv_or_none(args.allowed_drug_names),
            "allowed_drug_name_contains": parse_csv_or_none(args.allowed_drug_name_contains),
        },
        "diversity": {
            "path_overlap_penalty": args.path_overlap_penalty,
            "use_state_similarity_penalty": args.use_state_similarity_penalty,
            "state_similarity_penalty": args.state_similarity_penalty,
        },
        "output": {
            "output_dir": "",
            "state_checkpoint_dtype": args.state_checkpoint_dtype,
        },
    }
    return deep_update(cfg, cli_cfg)


def prepare_output_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    search_dir = output_dir / "search"
    cache_dir = output_dir / "cache"
    report_dir = output_dir / "report"

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output_dir}\n"
                "Use --overwrite or choose a new --output-dir."
            )

    search_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "search_dir": search_dir,
        "cache_dir": cache_dir,
        "report_dir": report_dir,
    }


def save_run_manifest(args: argparse.Namespace, cfg: Dict[str, Any], dirs: Dict[str, Path]) -> Path:
    manifest = {
        "args": vars(args),
        "search_config": cfg,
        "paths": {k: str(v) for k, v in dirs.items()},
    }
    path = dirs["output_dir"] / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def main() -> None:
    args = parse_args()
    dirs = prepare_output_dirs(args)

    from data_loader import load_start_target_embeddings
    from converter import StateSEConverter
    from scoring import DistributionScorer
    from search import run_search, save_search_config

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n=== Cell converter workflow ===")
    print(f"start cell:  {args.start_cell}")
    print(f"target cell: {args.target_cell}")
    print(f"device:      {device}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    cfg = build_search_config(args)
    cfg["output"]["output_dir"] = str(dirs["search_dir"])

    save_search_config(cfg, dirs["output_dir"] / "cell_converter.search_config.yaml")
    manifest_path = save_run_manifest(args, cfg, dirs)
    print(f"manifest:    {manifest_path}")

    print("\n[1/5] Loading start and target SE embeddings")
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

    print(f"start embeddings:  {pair.start_embeddings.shape}")
    print(f"target embeddings: {pair.target_embeddings.shape}")
    print(f"start available/sample:  {pair.start_n_available}/{pair.start_n_sampled}")
    print(f"target available/sample: {pair.target_n_available}/{pair.target_n_sampled}")

    state_cache_path = dirs["cache_dir"] / "start_target_states.npz"
    if args.save_state_cache:
        pair.save_npz(state_cache_path)
        print(f"cached start/target embeddings: {state_cache_path}")

    print("\n[2/5] Loading ST-SE converter")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    converter = StateSEConverter(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=device,
        max_set_len=args.max_set_len,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
    )

    print("\n[3/5] Initializing distribution scorer")
    scorer = DistributionScorer(
        target_state=pair.target_embeddings,
        device=device,
        normalize=not args.no_normalize_embeddings,
        n_projections=args.n_projections,
        projection_seed=args.seed,
        sinkhorn_metric=args.sinkhorn_metric,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
    )

    print("\n[4/5] Running sequential perturbation search")
    search_out = run_search(
        converter=converter,
        scorer=scorer,
        start_embeddings=pair.start_embeddings,
        cfg=cfg,
        perturbations=None,
        output_dir=dirs["search_dir"],
        start_cell=args.start_cell,
        target_cell=args.target_cell,
    )
    print(f"search results:    {search_out['results_tsv']}")
    print(f"search checkpoint: {search_out['checkpoint']}")

    if args.skip_report:
        print("\n[5/5] Skipping report because --skip-report was set")
    else:
        print("\n[5/5] Generating summary report")
        from make_search_report import main as report_main

        old_argv = sys.argv[:]
        try:
            report_argv = [
                "make_search_report.py",
                "--search-dir", str(dirs["search_dir"]),
                "--output-dir", str(dirs["report_dir"]),
                "--target-npz", str(state_cache_path),
                "--top-n-paths", str(args.report_top_n_paths),
                "--top-n-drugs", str(args.report_top_n_drugs),
                "--seed", str(args.seed),
            ]
            if args.conversion_threshold is not None:
                report_argv.extend(["--conversion-threshold", str(args.conversion_threshold)])
            if args.device is not None:
                report_argv.extend(["--device", str(args.device)])

            sys.argv = report_argv
            report_main()
        finally:
            sys.argv = old_argv

    print("\n=== Workflow complete ===")
    print(f"output:  {dirs['output_dir']}")
    print(f"search:  {dirs['search_dir']}")
    print(f"cache:   {dirs['cache_dir']}")
    if not args.skip_report:
        print(f"report:  {dirs['report_dir'] / 'summary.md'}")


if __name__ == "__main__":
    main()
