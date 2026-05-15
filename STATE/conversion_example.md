
#need to alter two files to improve performance massively 
#right now it isn't leveraging all GPU support 

1) need to alter the inference.py to ensure using CUDA for performance, so in state --> src --> state --> emb --> inference

2) need to alter the infer.py to ensure using CUDA, so in state --> src --> state --> cli --> tx --> infer.py

#prepare the raw data 
#log1p and normalize to 10,000 reads
python prepare_state_se_input.py \
  --input SW480_cell_line_examples_merged_raw.h5ad \
  --output SW480_cell_line_examples_merged_SE_input_lognorm.h5ad \
  --pass-filter-col pass_filter \
  --cell-line-col cell_line \
  --pert-col drugname_drugconc

#location of the models and checkpoints
SE_DIR=/oak/stanford/groups/larsms/Users/jbezney/tahoe100m/state_embedding/SE-600M
SE_CKPT=$SE_DIR/se600m_epoch16.ckpt

ST_RUN=/oak/stanford/groups/larsms/Users/jbezney/tahoe100m/state_transition/ST-SE-Tahoe/zeroshot/state_generalization_zeroshot_X_state
ST_CKPT=$ST_RUN/checkpoints/final.ckpt

#run the embedding 
state emb transform \
  --model-folder "$SE_DIR" \
  --checkpoint "$SE_CKPT" \
  --input SW480_cell_line_examples_merged_SE_input_lognorm.h5ad \
  --output SW480_cell_line_examples_merged_SE_input_lognorm.SE600M.h5ad \
  --embed-key X_state \
  --batch-size 32

#make a drug file 
python - <<'PY'
from pathlib import Path

Path("drug1.tsv").write_text(
    "perturbation\tnum_cells\n"
    "[('Trametinib', 0.05, 'uM')]\t256\n"
)
PY

#run the conversion 
state tx infer \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --adata SW480_cell_line_examples_merged_SE_input_lognorm.SE600M.h5ad  \
  --embed-key X_state \
  --pert-col drugname_drugconc \
  --control-pert "[('DMSO_TF', 0.0, 'uM')]" \
  --celltype-col cell_name \
  --tsv drug1.tsv \
  --max-set-len 256 \
  --output SW480_cell_line_DMSO_to_Trametinib.STSE.h5ad

#make second drug file 
python - <<'PY'
from pathlib import Path

Path("drug2.tsv").write_text(
    "perturbation\tnum_cells\n"
    "[('Dabrafenib', 0.5, 'uM')]\t256\n"
)
PY

#run second conversion
state tx infer \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --adata SW480_cell_line_DMSO_to_Trametinib.STSE.h5ad \
  --embed-key X_state \
  --pert-col drugname_drugconc \
  --control-pert "[('Trametinib', 0.05, 'uM')]" \
  --celltype-col cell_name \
  --tsv drug2.tsv \
  --max-set-len 256 \
  --output SW480_Trametinib_to_Drug2.STSE.h5ad


















