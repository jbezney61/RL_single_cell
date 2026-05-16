# ST-SE Cell-State Conversion Search

This codebase runs sequential drug-combination searches in ST-SE latent space. It starts from one SE-embedded cell state, repeatedly applies candidate drug perturbations with the ST-SE model, scores each predicted state against a target cell state, and saves the best ordered drug paths.

The core biological question is:

> Given 256 starting cells in `X_state`, which ordered sequence of up to N drugs most closely converts them toward the target cell-state distribution?

The system is modular so each component can be developed independently.

---

## Directory layout

Recommended working directory:

```text
tahoe100m/
  data_loader.py
  converter.py
  scoring.py
  search.py
  make_search_report.py
  cell_converter.py

  test_data_loader.py
  test_converter.py
  test_scoring.py
  test_search.py
  test_all.py

  search_config.yaml
  README.md
  README_TESTS.md
```

---

## Module overview

### `data_loader.py`

Loads start and target cell embeddings from an `.h5ad` file that already contains SE embeddings in:

```python
adata.obsm["X_state"]
```

Primary function:

```python
load_start_target_embeddings(...)
```

Inputs:

```text
--adata        h5ad containing all embedded cell types
--start-cell   starting cell label, e.g. J82
--target-cell  target cell label, e.g. A-172
--cell-col     obs column containing cell labels, e.g. cell_name
--embed-key    obsm key containing SE embeddings, usually X_state
```

Outputs:

```python
pair.start_embeddings   # [start_sample, embedding_dim]
pair.target_embeddings  # [target_sample, embedding_dim]
```

Optionality:

```text
start_sample  can be 256 or "all"
target_sample can be 256, 10000, or "all"
replace_if_needed allows sampling with replacement when too few cells are present
save_npz() caches the extracted start/target states
```

Use larger `target_sample` values when you have many target cells and want a more robust scoring distribution.

---

### `converter.py`

Loads the ST-SE model once and keeps it in memory. It applies one or many drug perturbations to an input cell-state embedding matrix.

Primary class:

```python
StateSEConverter
```

Primary methods:

```python
convert_one(embeddings, perturbation)
convert_many_iter(embeddings, perturbations, chunk_size)
list_perturbations()
```

Input:

```python
embeddings  # [256, 2058], usually torch.Tensor or np.ndarray
```

Output:

```python
predicted_embeddings  # [256, 2058]
```

Important behavior:

```text
The converter feeds the current node state directly as ctrl_cell_emb.
This is what enables sequential composition:

start → drug A → state_A → drug B → state_AB → ...
```

It does not write AnnData files during the search. This avoids expensive I/O.

---

### `scoring.py`

Scores a predicted cell-state distribution against the target cell-state distribution.

Implemented scoring functions:

```text
1. Sliced Wasserstein distance
2. Sinkhorn optimal transport distance
```

Recommended use:

```text
Sliced Wasserstein:
  Fast approximate score.
  Use to screen all candidate drugs during beam expansion.

Sinkhorn OT:
  More faithful distributional score.
  Use to rerank top candidates and for final reporting.
```

Primary class:

```python
DistributionScorer
```

Example:

```python
scorer = DistributionScorer(target_embeddings, device="cuda:0")
fast_scores = scorer.sliced_wasserstein(candidate_batch)
final_scores = scorer.sinkhorn(top_candidate_batch)
```

Expected shapes:

```python
candidate_batch  # [n_candidates, 256, 2058]
target_state     # [target_sample, 2058]
```

---

### `search.py`

Runs sequential drug-path search.

Implemented algorithms:

```text
1. deterministic_beam
2. diverse_beam
```

#### `deterministic_beam`

At each depth:

```text
1. Expand every retained path over all legal drugs.
2. Predict candidate next states with converter.py.
3. Score all candidates with Sliced Wasserstein.
4. Keep top beam_size × prefilter_multiplier candidates.
5. Rerank those candidates with Sinkhorn OT.
6. Keep top beam_size paths.
7. Save results.tsv and checkpoint.pt.
```

#### `diverse_beam`

Same as deterministic beam, but final beam selection penalizes redundant paths.

Diversity penalties:

```text
path_overlap_penalty:
  Penalizes paths using overlapping base drug names.

state_similarity_penalty:
  Optional penalty for predicted states with similar centroids.
```

#### Same-drug / different-dose constraint

Tahoe perturbation labels often include multiple concentrations:

```text
[('Trametinib', 0.05, 'uM')]
[('Trametinib', 0.5, 'uM')]
[('Trametinib', 5.0, 'uM')]
```

By default:

```yaml
constraints:
  allow_repeated_drug_names: false
```

This means once a path uses one concentration of a drug, it cannot use another concentration of the same base drug later in the same path.

This is important for biologically sensible combinations.

---

### `make_search_report.py`

Generates a summary report from a search output directory.

Input:

```text
search/
  results.tsv
  checkpoint.pt
  search_config.used.yaml
```

Output:

```text
report/
  summary.md
  tables/
  figures/
```

Figures:

```text
01_best_score_by_depth.png
02_top_path_trajectories.png
03_delta_score_by_step.png
04_drug_frequency_top_paths.png
05_drug_position_heatmap.png
06_path_similarity_heatmap.png
07_variance_ratio_by_depth.png
08_target_neighbor_coverage.png
```

Plots 7 and 8 require target embeddings, usually provided with:

```text
--target-npz cache/start_target_states.npz
```

---

### `cell_converter.py`

End-to-end CLI that runs the full workflow:

```text
data_loader.py
→ converter.py
→ scoring.py
→ search.py
→ make_search_report.py
```

This is the primary user-facing command for a single conversion.

---

## Search configuration

You can run the search either with CLI arguments or a YAML config.

Example `search_config.yaml`:

```yaml
search:
  algorithm: deterministic_beam
  max_depth: 5
  beam_size: 32
  prefilter_multiplier: 10
  converter_chunk_size: 16
  max_drugs_to_consider: null

constraints:
  allow_repeated_drug_names: false
  allow_repeated_perturbation_labels: false
  allow_control_like_drugs: false
  banned_drug_names: []
  banned_perturbation_labels: []
  allowed_drug_names: null
  allowed_drug_name_contains: null

diversity:
  path_overlap_penalty: 0.05
  use_state_similarity_penalty: false
  state_similarity_penalty: 0.02

output:
  output_dir: search_outputs
  state_checkpoint_dtype: float16
```

CLI options override the YAML.

---

## End-to-end CLI usage

### Small smoke test

Use this first to verify that the whole pipeline runs.

```bash
export CUDA_VISIBLE_DEVICES=0

python cell_converter.py \
  --adata WT_256_per_cell_name.SE600M.h5ad \
  --start-cell J82 \
  --target-cell A-172 \
  --cell-col cell_name \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --output-dir runs/test_J82_to_A172 \
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
runs/test_J82_to_A172/
  run_manifest.json
  cell_converter.search_config.yaml
  cache/
    start_target_states.npz
  search/
    results.tsv
    checkpoint.pt
    search_config.used.yaml
  report/
    summary.md
    tables/
    figures/
```

### Production-style deterministic beam

```bash
python cell_converter.py \
  --adata WT_256_per_cell_name.SE600M.h5ad \
  --start-cell J82 \
  --target-cell A-172 \
  --cell-col cell_name \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --output-dir runs/J82_to_A172_beam32_depth5 \
  --algorithm deterministic_beam \
  --max-depth 5 \
  --beam-size 32 \
  --prefilter-multiplier 10 \
  --converter-chunk-size 16 \
  --target-sample 256 \
  --n-projections 64 \
  --sinkhorn-epsilon 0.05 \
  --sinkhorn-iters 100 \
  --conversion-threshold 0.025
```

### Production-style diverse beam

```bash
python cell_converter.py \
  --adata WT_256_per_cell_name.SE600M.h5ad \
  --start-cell J82 \
  --target-cell A-172 \
  --cell-col cell_name \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --output-dir runs/J82_to_A172_diverse_depth5 \
  --algorithm diverse_beam \
  --max-depth 5 \
  --beam-size 32 \
  --prefilter-multiplier 10 \
  --converter-chunk-size 16 \
  --path-overlap-penalty 0.05 \
  --use-state-similarity-penalty \
  --state-similarity-penalty 0.02 \
  --target-sample 256 \
  --n-projections 64 \
  --sinkhorn-epsilon 0.05 \
  --sinkhorn-iters 100
```

---

## Important parameters

### Search depth

```bash
--max-depth 5
```

Maximum number of sequential drugs. Use 5 or 6 for full searches.

### Beam size

```bash
--beam-size 32
```

Number of paths retained after each depth. Larger values explore more paths but cost more.

### Prefilter multiplier

```bash
--prefilter-multiplier 10
```

The number of candidates reranked with Sinkhorn is:

```text
beam_size × prefilter_multiplier
```

For `beam_size=32`, `prefilter_multiplier=10`, Sinkhorn reranks 320 candidates per depth.

### Drug limiter

```bash
--max-drugs-to-consider 6
```

Use only for smoke tests. For production, omit this option.

Because there are multiple concentrations per drug, a small limiter like 6 may contain only two unique base drugs. In that case the search can stop early after depth 2 because no legal new base drugs remain.

### Repeated drug rule

Default:

```bash
--no-allow-repeated-drug-names
```

This prevents selecting the same base drug twice at different concentrations.

To allow repeated base drugs:

```bash
--allow-repeated-drug-names
```

Usually not recommended for biological combination searches.

### Target sample size

```bash
--target-sample 256
```

For larger target distributions:

```bash
--target-sample 10000
```

or:

```bash
--target-sample all
```

A larger target sample can improve scoring robustness if enough target cells are available.

---

## Interpreting outputs

### `search/results.tsv`

Main search results. Important columns:

```text
depth
rank
num_drugs
path_json
drug_names_json
path_string
drug_name_string
score_sinkhorn_ot
score_sliced_wasserstein
adjusted_score
delta_sinkhorn_from_parent
```

Lower `score_sinkhorn_ot` is better.

### `search/checkpoint.pt`

Contains saved beam states per depth:

```python
{
  "config": ...,
  "depths": {
    depth: {
      "paths": ...,
      "drug_names": ...,
      "scores_sinkhorn": ...,
      "scores_sliced_wasserstein": ...,
      "states": torch.Tensor[beam_size, n_cells, embedding_dim]
    }
  }
}
```

States are stored on CPU as `float16` by default to save disk.

### `report/summary.md`

Human-readable summary with:

```text
best path overall
best path by depth
top paths
drug frequency
conversion threshold summary
heterogeneity diagnostics
```

### Report figures

Key plots:

```text
01_best_score_by_depth:
  Whether additional drugs improve distance to target.

02_top_path_trajectories:
  How top paths move over sequential drug steps.

03_delta_score_by_step:
  Which steps improve or worsen the score.

04_drug_frequency_top_paths:
  Recurrently selected drugs.

05_drug_position_heatmap:
  Which drugs appear early vs late.

06_path_similarity_heatmap:
  Whether top paths are diverse or redundant.

07_variance_ratio_by_depth:
  Population-collapse diagnostic.

08_target_neighbor_coverage:
  How much of the target distribution is covered.
```

---

## Development notes

The inner search loop should avoid disk I/O. Keep candidate states in GPU or CPU memory, score them immediately, and only save retained beam states.

For debugging, use:

```bash
--max-depth 2
--beam-size 2
--max-drugs-to-consider 6
```

For production, remove `--max-drugs-to-consider`.

