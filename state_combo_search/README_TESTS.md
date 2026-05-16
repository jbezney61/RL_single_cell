# Test Scripts README

This document describes how to test each module in the ST-SE cell-state conversion codebase.

The test scripts are designed for developers to run after updating modules.

---

## Test script overview

```text
test_data_loader.py
  Tests loading start and target X_state embeddings from an h5ad.

test_converter.py
  Tests loading ST-SE and applying one or many perturbations.

test_scoring.py
  Tests Sliced Wasserstein and Sinkhorn scoring.
  Optionally tests converter predictions scored against a target.

test_search.py
  Tests a small beam search using all modules except the end-to-end CLI wrapper.

test_all.py
  Runs an end-to-end smoke test across all major modules and reports pass/fail status.
```

---

## Shared assumptions

The tests assume you have:

```text
1. An h5ad file with SE embeddings in adata.obsm["X_state"]
2. A valid ST-SE model directory
3. A checkpoint, usually checkpoints/final.ckpt
4. Companion modules in the same directory:
   data_loader.py, converter.py, scoring.py, search.py, make_search_report.py
```

Recommended environment variables:

```bash
export ST_RUN=/oak/stanford/groups/larsms/Users/jbezney/tahoe100m/state_transition/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state
export ST_CKPT=$ST_RUN/checkpoints/final.ckpt
export CUDA_VISIBLE_DEVICES=0
```

Example h5ad:

```bash
export TEST_ADATA=WT_256_per_cell_name.SE600M.h5ad
```

Example cells:

```bash
export START_CELL=J82
export TARGET_CELL=A-172
```

---

## 1. Test the data loader

Purpose:

```text
Confirm that start and target cell labels exist.
Confirm that X_state exists.
Confirm expected embedding shapes.
Optionally save start/target embeddings to .npz.
```

Command:

```bash
python test_data_loader.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --cell-col cell_name \
  --embed-key X_state \
  --start-sample 256 \
  --target-sample 256 \
  --save-npz test_states.npz \
  --show-counts
```

Expected:

```text
start:  (256, 2058)
target: (256, 2058)
Saved cache: test_states.npz
```

Common failures:

```text
KeyError: embed_key='X_state' not found
  → Run SE embedding first or check adata.obsm keys.

No cells found for start/target
  → Check spelling in adata.obs[cell_name].
```

---

## 2. Test the converter

Purpose:

```text
Confirm ST-SE checkpoint loads.
Confirm CUDA/GPU is used if available.
Confirm convert_one returns [256, 2058].
Confirm convert_many_iter returns [n_drugs, 256, 2058].
```

One-drug test:

```bash
python test_converter.py \
  --adata "$TEST_ADATA" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --find Trametinib \
  --output-npy test_trametinib_pred.npy
```

Many-drug test:

```bash
python test_converter.py \
  --adata "$TEST_ADATA" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --run-first-n 10 \
  --chunk-size 4 \
  --output-many-npz test_first10_preds.npz
```

Expected:

```text
one drug output:       (256, 2058)
first 10 drug output:  (10, 256, 2058)
```

Common failures:

```text
Perturbation not found
  → Use exact labels from pert_onehot_map.pt or a different --find substring.

Expected embedding output dim 2058, got ...
  → Model may be returning gene-space output; check ST-SE checkpoint/config.
```

---

## 3. Test scoring

Purpose:

```text
Confirm Sliced Wasserstein and Sinkhorn run.
Compare runtime.
Confirm lower-is-better scores are produced.
Optionally score converter predictions.
```

Baseline start-versus-target:

```bash
python test_scoring.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --cell-col cell_name \
  --embed-key X_state \
  --start-sample 256 \
  --target-sample 256
```

With converter predictions:

```bash
python test_scoring.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --cell-col cell_name \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --find Trametinib \
  --run-first-n 10 \
  --chunk-size 4
```

Expected:

```text
Sliced Wasserstein score
Sinkhorn OT score
Candidate ranking table
```

---

## 4. Test search

Purpose:

```text
Run a tiny beam search.
Confirm results.tsv and checkpoint.pt are created.
Confirm no repeated base drugs are selected within each path.
```

Command:

```bash
python test_search.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --cell-col cell_name \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --config search_config.yaml \
  --output-dir test_search_output \
  --algorithm deterministic_beam \
  --max-depth 2 \
  --beam-size 2 \
  --max-drugs-to-consider 6 \
  --prefilter-multiplier 3 \
  --converter-chunk-size 3 \
  --n-projections 32 \
  --sinkhorn-iters 50
```

Expected outputs:

```text
test_search_output/
  results.tsv
  checkpoint.pt
  search_config.used.yaml
```

Check:

```bash
head test_search_output/results.tsv
```

If the search stops before `max_depth`, the most common reason in smoke tests is that `--max-drugs-to-consider 6` includes only a few unique base drugs once concentrations are collapsed.

---

## 5. Test full end-to-end CLI

Purpose:

```text
Confirm cell_converter.py runs the entire workflow:
data loading → converter → scoring → search → report
```

Command:

```bash
python cell_converter.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --cell-col cell_name \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --output-dir test_cell_converter_output \
  --algorithm deterministic_beam \
  --max-depth 2 \
  --beam-size 2 \
  --max-drugs-to-consider 6 \
  --prefilter-multiplier 3 \
  --converter-chunk-size 3 \
  --n-projections 32 \
  --sinkhorn-iters 50 \
  --overwrite
```

Expected outputs:

```text
test_cell_converter_output/
  run_manifest.json
  cell_converter.search_config.yaml
  cache/start_target_states.npz
  search/results.tsv
  search/checkpoint.pt
  report/summary.md
  report/figures/
  report/tables/
```

---

## 6. Run all tests with one command

Use `test_all.py` for a single integrated smoke test.

Command:

```bash
python test_all.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --output-dir test_all_output \
  --overwrite
```

What it checks:

```text
1. Imports all required modules.
2. Loads start and target X_state embeddings.
3. Loads ST-SE converter.
4. Runs one-drug conversion.
5. Runs scoring on baseline and converted state.
6. Runs tiny deterministic beam search.
7. Generates report.
8. Confirms required output files exist.
```

Expected final message:

```text
ALL TESTS PASSED
```

If something fails, `test_all.py` prints the failed step and traceback.

---

## Recommended development workflow

After editing a single module:

```bash
# if editing data_loader.py
python test_data_loader.py ...

# if editing converter.py
python test_converter.py ...

# if editing scoring.py
python test_scoring.py ...

# if editing search.py
python test_search.py ...

# if editing report or CLI behavior
python test_all.py ...
```

Before launching a large run:

```bash
python test_all.py \
  --adata "$TEST_ADATA" \
  --start-cell "$START_CELL" \
  --target-cell "$TARGET_CELL" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --output-dir test_all_output \
  --overwrite
```

Then run the production search.

