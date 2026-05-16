#!/usr/bin/env python
"""
test_all.py

Integrated smoke test for the ST-SE cell-state conversion codebase.

This script tests the major modules without requiring a large production run:

    1. Import modules.
    2. Load start/target embeddings.
    3. Cache start/target states.
    4. Load ST-SE converter.
    5. Run one-drug conversion.
    6. Run Sliced Wasserstein and Sinkhorn scoring.
    7. Run a tiny deterministic beam search.
    8. Generate a report.
    9. Confirm expected output files exist.

Example:
    export CUDA_VISIBLE_DEVICES=0

    python test_all.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --output-dir test_all_output \
      --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--adata", required=True)
    p.add_argument("--start-cell", required=True)
    p.add_argument("--target-cell", required=True)
    p.add_argument("--cell-col", default="cell_name")
    p.add_argument("--embed-key", default="X_state")

    p.add_argument("--model-dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)

    p.add_argument("--output-dir", default="test_all_output")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--start-sample", default="256")
    p.add_argument("--target-sample", default="256")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--find-drug", default=None, help="Optional substring to choose one drug. Defaults to first non-control drug.")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--beam-size", type=int, default=2)
    p.add_argument("--max-drugs-to-consider", type=int, default=6)
    p.add_argument("--prefilter-multiplier", type=int, default=3)
    p.add_argument("--converter-chunk-size", type=int, default=3)
    p.add_argument("--n-projections", type=int, default=32)
    p.add_argument("--sinkhorn-iters", type=int, default=50)
    p.add_argument("--sinkhorn-epsilon", type=float, default=0.05)

    return p.parse_args()


class StepRunner:
    def __init__(self):
        self.results = []

    def run(self, name, func):
        print(f"\n=== TEST STEP: {name} ===")
        try:
            out = func()
            self.results.append((name, True, None))
            print(f"PASS: {name}")
            return out
        except Exception as e:
            self.results.append((name, False, repr(e)))
            print(f"FAIL: {name}")
            traceback.print_exc()
            raise

    def summary(self):
        print("\n=== TEST SUMMARY ===")
        for name, ok, err in self.results:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
            if err:
                print(f"      {err}")


def assert_exists(path: Path):
    if not path.exists():
        raise AssertionError(f"Expected file/directory does not exist: {path}")


def main():
    args = parse_args()
    runner = StepRunner()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"{out_dir} exists and is not empty. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = out_dir / "cache"
    search_dir = out_dir / "search"
    report_dir = out_dir / "report"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    modules = runner.run(
        "import modules",
        lambda: __import_modules(),
    )

    data_loader, converter_mod, scoring_mod, search_mod, report_mod = modules

    pair = runner.run(
        "load start and target embeddings",
        lambda: data_loader.load_start_target_embeddings(
            h5ad_path=args.adata,
            start_cell=args.start_cell,
            target_cell=args.target_cell,
            cell_col=args.cell_col,
            embed_key=args.embed_key,
            start_sample=args.start_sample,
            target_sample=args.target_sample,
            seed=args.seed,
        ),
    )

    runner.run(
        "validate embedding shapes",
        lambda: validate_pair(pair),
    )

    state_cache = cache_dir / "start_target_states.npz"
    runner.run(
        "save start/target cache",
        lambda: pair.save_npz(state_cache),
    )

    conv = runner.run(
        "load ST-SE converter",
        lambda: converter_mod.StateSEConverter(
            model_dir=args.model_dir,
            checkpoint=args.checkpoint,
            device=device,
            max_set_len=256,
        ),
    )

    perturbation = runner.run(
        "select test perturbation",
        lambda: select_perturbation(conv, args.find_drug),
    )
    print(f"Selected perturbation: {perturbation}")

    pred = runner.run(
        "run one-drug conversion",
        lambda: conv.convert_one(pair.start_embeddings, perturbation, return_cpu=False),
    )

    runner.run(
        "validate converted embedding shape",
        lambda: validate_prediction(pred, pair.start_embeddings),
    )

    scorer = runner.run(
        "initialize scorer",
        lambda: scoring_mod.DistributionScorer(
            target_state=pair.target_embeddings,
            device=device,
            normalize=True,
            n_projections=args.n_projections,
            projection_seed=args.seed,
            sinkhorn_metric="cosine",
            sinkhorn_epsilon=args.sinkhorn_epsilon,
            sinkhorn_iters=args.sinkhorn_iters,
        ),
    )

    runner.run(
        "score baseline and converted state",
        lambda: score_baseline_and_prediction(scorer, pair.start_embeddings, pred),
    )

    cfg = {
        "search": {
            "algorithm": "deterministic_beam",
            "max_depth": args.max_depth,
            "beam_size": args.beam_size,
            "prefilter_multiplier": args.prefilter_multiplier,
            "converter_chunk_size": args.converter_chunk_size,
            "max_drugs_to_consider": args.max_drugs_to_consider,
        },
        "constraints": {
            "allow_repeated_drug_names": False,
            "allow_repeated_perturbation_labels": False,
            "allow_control_like_drugs": False,
            "banned_drug_names": [],
            "banned_perturbation_labels": [],
            "allowed_drug_names": None,
            "allowed_drug_name_contains": None,
        },
        "diversity": {
            "path_overlap_penalty": 0.05,
            "use_state_similarity_penalty": False,
            "state_similarity_penalty": 0.02,
        },
        "output": {
            "output_dir": str(search_dir),
            "state_checkpoint_dtype": "float16",
        },
    }

    runner.run(
        "run tiny deterministic beam search",
        lambda: search_mod.run_search(
            converter=conv,
            scorer=scorer,
            start_embeddings=pair.start_embeddings,
            cfg=cfg,
            perturbations=None,
            output_dir=search_dir,
            start_cell=args.start_cell,
            target_cell=args.target_cell,
        ),
    )

    runner.run(
        "validate search outputs",
        lambda: validate_search_outputs(search_dir),
    )

    runner.run(
        "generate report",
        lambda: run_report(report_mod, search_dir, report_dir, state_cache, args.seed, device),
    )

    runner.run(
        "validate report outputs",
        lambda: validate_report_outputs(report_dir),
    )

    runner.summary()
    print("\nALL TESTS PASSED")
    print(f"Output directory: {out_dir}")


def __import_modules():
    import data_loader
    import converter
    import scoring
    import search
    import make_search_report

    return data_loader, converter, scoring, search, make_search_report


def validate_pair(pair):
    if pair.start_embeddings.ndim != 2:
        raise AssertionError(f"start_embeddings must be 2D, got {pair.start_embeddings.shape}")
    if pair.target_embeddings.ndim != 2:
        raise AssertionError(f"target_embeddings must be 2D, got {pair.target_embeddings.shape}")
    if pair.start_embeddings.shape[1] != pair.target_embeddings.shape[1]:
        raise AssertionError("Start/target embedding dimensions differ")
    print(f"start shape:  {pair.start_embeddings.shape}")
    print(f"target shape: {pair.target_embeddings.shape}")


def select_perturbation(conv, find_drug):
    perts = conv.list_perturbations(include_control=False)
    if not perts:
        raise AssertionError("No non-control perturbations available")

    if find_drug:
        matches = [p for p in perts if find_drug.lower() in p.lower()]
        if not matches:
            raise AssertionError(f"No perturbation matched substring: {find_drug}")
        return matches[0]

    return perts[0]


def validate_prediction(pred, start_embeddings):
    if not torch.is_tensor(pred):
        raise AssertionError("Prediction must be a torch.Tensor")
    expected = tuple(start_embeddings.shape)
    got = tuple(pred.shape)
    if got != expected:
        raise AssertionError(f"Prediction shape mismatch: got {got}, expected {expected}")
    if not torch.isfinite(pred).all():
        raise AssertionError("Prediction contains non-finite values")
    print(f"prediction shape: {got}")


def score_baseline_and_prediction(scorer, start_embeddings, pred):
    sw0 = scorer.sliced_wasserstein(start_embeddings)
    ot0 = scorer.sinkhorn(start_embeddings)
    sw1 = scorer.sliced_wasserstein(pred)
    ot1 = scorer.sinkhorn(pred)

    vals = {
        "baseline_sw": float(sw0.item()),
        "baseline_sinkhorn": float(ot0.item()),
        "converted_sw": float(sw1.item()),
        "converted_sinkhorn": float(ot1.item()),
    }
    print(vals)

    for k, v in vals.items():
        if not np.isfinite(v):
            raise AssertionError(f"Non-finite score: {k}={v}")

    return vals


def validate_search_outputs(search_dir: Path):
    assert_exists(search_dir / "results.tsv")
    assert_exists(search_dir / "checkpoint.pt")
    assert_exists(search_dir / "search_config.used.yaml")
    print(f"search outputs exist in {search_dir}")


def run_report(report_mod, search_dir: Path, report_dir: Path, state_cache: Path, seed: int, device: str):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "make_search_report.py",
            "--search-dir", str(search_dir),
            "--output-dir", str(report_dir),
            "--target-npz", str(state_cache),
            "--seed", str(seed),
            "--device", str(device),
        ]
        report_mod.main()
    finally:
        sys.argv = old_argv


def validate_report_outputs(report_dir: Path):
    assert_exists(report_dir / "summary.md")
    assert_exists(report_dir / "figures")
    assert_exists(report_dir / "tables")
    assert_exists(report_dir / "figures" / "01_best_score_by_depth.png")
    assert_exists(report_dir / "figures" / "06_path_similarity_heatmap.png")
    print(f"report outputs exist in {report_dir}")


if __name__ == "__main__":
    main()
