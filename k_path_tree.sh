#!/bin/bash
#SBATCH --job-name="k_search_tree_k8_n5"
#SBATCH --account=3126294
#SBATCH --partition=compute
#SBATCH --nodelist=cnode07
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --time=10:00:00

# --- Set the working directory to where your k_search.py file is ---
#SBATCH --chdir=/home/3126294/RL/RL_single_cell/searches

# --- Use unique log files for each job run ---
#SBATCH --output=/home/3126294/RL/outputs/k_search_tree_k8_n5.out
#SBATCH --error=/home/3126294/RL/outputs/k_search_tree_k8_n5.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=carlo.ruggeri@studbocconi.it

# --- Run your python script using the absolute path to the conda environment's python ---
/home/3126294/miniconda3/envs/average_search_env/bin/python k_search.py --k 8 --n_steps 5 --strategy tree

echo "Job finished."
