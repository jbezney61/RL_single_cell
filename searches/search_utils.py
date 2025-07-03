import pandas as pd
import numpy as np
import random
import heapq
from scipy.spatial import cKDTree
import torch
import torch.nn as nn



class AverageCellPerturbationSearch:
    def __init__(self, df):
        df = df.copy()
        df['genes_targeted'] = df['genes_targeted'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        embedding_cols = [col for col in df.columns if col.startswith("PC")]
        drugs = df[df['drugname_drugconc'] != "[('DMSO_TF', 0.0, 'uM')]"]['drugname_drugconc'].unique().tolist()
        cell_lines = df['cell_line'].unique().tolist()
        drugs_to_genes = df[['drugname_drugconc', 'genes_targeted']].dropna().drop_duplicates().set_index('drugname_drugconc')['genes_targeted'].to_dict()
        wild_id = "[('DMSO_TF', 0.0, 'uM')]"
        wild_df = df[df['drugname_drugconc'] == wild_id]
        drugs_df = df[df['drugname_drugconc'] != wild_id]

        displacements = []
        for _, row in drugs_df.iterrows():
            cl = row['cell_line']
            drug = row['drugname_drugconc']
            wild_row = wild_df[wild_df['cell_line'] == cl]
            disp = row[embedding_cols].values - wild_row[embedding_cols].values[0]
            displacements.append({'drug': drug, 'cell_line': cl, 'vector': disp})
        disp_df = pd.DataFrame(displacements)

        rows = []
        for line1 in cell_lines:
            row = []
            centroid1 = wild_df[wild_df['cell_line'] == line1][embedding_cols].values[0]
            for line2 in cell_lines:
                centroid2 = wild_df[wild_df['cell_line'] == line2][embedding_cols].values[0]
                row.append(np.linalg.norm(centroid1 - centroid2))
            rows.append(row)
        wild_dist_df = pd.DataFrame(rows, index=cell_lines, columns=cell_lines)

        wild_df = wild_df.copy()
        wild_df['radius'] = 1.0
        radiuses = {cell: radius for cell, radius in zip(wild_df['cell_line'], wild_df['radius'])}

        self.df = df
        self.embedding_cols = embedding_cols
        self.drugs = drugs
        self.cell_lines = cell_lines
        self.drugs_to_genes = drugs_to_genes
        self.wild_df = wild_df
        self.drugs_df = drugs_df
        self.disp_df = disp_df
        self.wild_dist_df = wild_dist_df
        self.radiuses = radiuses
        self.centroids = {row['cell_line']: row[self.embedding_cols].values for _, row in self.wild_df.iterrows()} # better for searches
        self.cell_drug_to_disp = {(row['cell_line'], row['drug']): row['vector'] for _, row in self.disp_df.iterrows()} # better for searches
        self.centroid_keys = list(self.centroids.keys()) # better for searches
        centroid_mat = np.vstack([self.centroids[c] for c in self.centroid_keys])
        self.centroid_tree  = cKDTree(centroid_mat) # better for searches
    
    def blended_disp(self, centroid_weight_list, drug):
        vals = []
        valid_weights = []
        centroids = []
        all_none = True

        for c, w in centroid_weight_list:
            v = self.cell_drug_to_disp.get((c, drug))
            if v is not None:
                all_none = False
                vals.append(v)
                valid_weights.append(w)
                centroids.append(c)
            else:
                centroids.append(None)
        
        if all_none:
            return None, centroids
        valid_weights = np.array(valid_weights)
        valid_weights /= valid_weights.sum()

        blended = sum(w * v for w, v in zip(valid_weights, vals))
        return blended, centroids
    

    def path_key(self, drugs, step):
        return (tuple(drugs), step)


    def search_path_k_paths(self, search_id, starting_cl, ending_cl,
                    strategy: str = 'beam',
                    n_steps: int = 5,
                    k: int = 10,
                    threshold: float = 0.4,
                    blend: int = 2):
        """
        Performs a path search using one of several selectable strategies.

        Args:
            search_id (str): A unique identifier for this search.
            starting_cl (str): The starting cell line.
            ending_cl (str): The target cell line.
            strategy (str): The search strategy to use. One of:
                            'beam' -> Classic beam search (expand all, prune to k).
                            'tree' -> Full k-ary tree expansion (k, k*k, k*k*k, ...).
                            Defaults to 'beam'.
            n_steps (int): The maximum number of steps (drugs) in a path.
            k (int): The beam width or branching factor.
            threshold (float): The progress threshold for recording paths.
            blend (int): The number of nearest WT centroids to blend for displacement.
        """
        # --- 1. SETUP ---
        start = self.centroids[starting_cl]
        end = self.centroids[ending_cl]
        end_radius = self.radiuses[ending_cl]
        all_drugs = self.drugs
        init_dist = np.linalg.norm(start - end)
        progress_threshold = (1 - threshold) * init_dist

        paths = [(start, [], set(), [], init_dist, [])]
        seen_paths, best_at_step, added_keys = [], {}, set()

        # --- 2. MAIN SEARCH LOOP ---
        for step in range(n_steps + 1):
            new_paths = []
            
            for cur_pos, ord_drugs, drug_set, gene_list, dist, cl_path in paths:
                # --- Recording logic ---
                key = self.path_key(ord_drugs, step)
                if step != 0:
                    path_for_recording = (cur_pos, ord_drugs, drug_set, gene_list, dist, cl_path)
                    if step not in best_at_step or dist < best_at_step[step][4]:
                        best_at_step[step] = path_for_recording
                    if dist < end_radius and key not in added_keys:
                        seen_paths.append({'step': step, 'is_best_at_step': False, 'is_success': True, 'covered_threshold': True, 'path': path_for_recording})
                        added_keys.add(key)
                        continue
                    if dist <= progress_threshold and key not in added_keys:
                        seen_paths.append({'step': step, 'is_best_at_step': False, 'is_success': False, 'covered_threshold': True, 'path': path_for_recording})
                        added_keys.add(key)
                if step == n_steps:
                    continue

                # --- Expansion Logic ---
                drug_candidates = []
                dists, idxs = self.centroid_tree.query(cur_pos, k=blend)
                weights = 1.0 / (dists + 1e-6)
                weights /= weights.sum()
                centroid_weight_list = [(self.centroid_keys[idx], weight) for idx, weight in zip(idxs, weights)]

                for drug in all_drugs:
                    if drug in drug_set: continue
                    disp, wt_pair = self.blended_disp(centroid_weight_list, drug)
                    if disp is None: continue
                    
                    new_pos = cur_pos + disp
                    new_dist = np.linalg.norm(new_pos - end)
                    
                    if new_dist < dist:
                        drug_candidates.append((
                            new_pos, ord_drugs + [drug], drug_set | {drug},
                            gene_list + [self.drugs_to_genes.get(drug, ())],
                            new_dist, cl_path + [wt_pair]
                        ))

                # --- STRATEGY-DEPENDENT EXPANSION ---
                if strategy == 'beam':
                    new_paths.extend(drug_candidates)
                elif strategy == 'tree':
                    top_k_for_this_path = heapq.nsmallest(k, drug_candidates, key=lambda p: p[4])
                    new_paths.extend(top_k_for_this_path)
                else:
                    raise ValueError(f"Unknown strategy: '{strategy}'. Must be 'beam' or 'tree'.")

            if not new_paths:
                break

            # --- STRATEGY-DEPENDENT PRUNING ---
            if step == n_steps:
                break

            if strategy == 'beam':
                paths = heapq.nsmallest(k, new_paths, key=lambda p: p[4])
            elif strategy == 'tree':
                paths = new_paths
            
            if not paths:
                break
                
        # --- 3. POST-PROCESSING AND RETURN ---
        for step_num, best_path in best_at_step.items():
            key = self.path_key(best_path[1], step_num)
            if key in added_keys:
                for entry in seen_paths:
                    if self.path_key(entry['path'][1], entry['step']) == key:
                        entry['is_best_at_step'] = True
            else:
                seen_paths.append({
                    'step': step_num, 'is_best_at_step': True, 'is_success': False,
                    'covered_threshold': False, 'path': best_path
                })
                added_keys.add(key)

        df = pd.DataFrame(seen_paths)
        return {
            'search_id': search_id,
            'type_of_search': f'k_path_{strategy}',
            'n_steps': n_steps, 'k': k, 'threshold': threshold, 'blend': blend,
            'starting_cl': starting_cl, 'target_cl': ending_cl,
            'starting_position': start, 'target_position': end,
            'target_radius': end_radius, 'progress_table': df
        }



    def probabilistic_search(self,
        search_id,
        starting_cl,
        ending_cl,
        *,                 # keyword-only from here on
        n_paths: int = 100,
        n_steps: int = 5,
        beta: float = 1.0,
        blend: int = 2,               # how many WT centroids to blend
        threshold: float = 0.4        # same meaning as in k-path
    ):
        """
        Monte-Carlo search that:
        • uses weighted‐average displacements over the *blend* nearest WT centroids
        • samples moves with a Gibbs/Boltzmann distribution (temperature 1/β)
        • records the same progress table format produced by search_path_k_paths
        """

        # ------------------------------------------------------------------ setup
        start          = self.centroids[starting_cl]
        end            = self.centroids[ending_cl]
        end_radius     = self.radiuses[ending_cl]
        init_dist      = np.linalg.norm(start - end)
        progress_thr   = (1 - threshold) * init_dist

        all_drugs      = self.drugs
        centroid_keys  = self.centroid_keys                    # local alias
        seen_paths     = []                                    # rows for df
        added_keys     = set()                                 # uniqueness filter
        best_at_step   = {}                                    # best path / step


        # ===================================================== main Monte-Carlo loop
        for _ in range(n_paths):
            cur_pos          = start.copy()
            ordered_drugs    = []
            drug_set         = set()
            gene_list        = []
            cl_path          = []   # for symmetry with k-path
            step             = 0

            while step <= n_steps:
                dist = np.linalg.norm(cur_pos - end)
                key  = self.path_key(ordered_drugs, step)

                if step != 0:

                    # ---------- track “best so far” at this depth -------------------
                    if step not in best_at_step or dist < best_at_step[step][4]:
                        best_at_step[step] = (
                            cur_pos.copy(), ordered_drugs.copy(), drug_set.copy(),
                            gene_list.copy(), dist, cl_path.copy()
                        )

                    # ---------- record success / threshold coverage -----------------
                    covered_thr = dist <= progress_thr
                    if (dist < end_radius or covered_thr) and key not in added_keys:
                        seen_paths.append({
                            'step'             : step,
                            'is_best_at_step'  : False,   # temp –  patched later
                            'is_success'       : dist < end_radius,
                            'covered_threshold': covered_thr,
                            'path'             : (
                                cur_pos.copy(), ordered_drugs.copy(), drug_set.copy(),
                                gene_list.copy(), dist, cl_path.copy()
                            )
                        })
                        added_keys.add(key)

                if dist < end_radius:          # reached target – stop this walk
                    break
                if step == n_steps:            # depth limit
                    break

                # ---------- optimisation A: k nearest WT centroids --------------
                dists, idxs = self.centroid_tree.query(cur_pos, k=blend)
                weights     = 1.0 / (dists + 1e-6)
                weights    /= weights.sum()
                centroid_wt = [(centroid_keys[i], w) for i, w in zip(idxs, weights)]

                # ---------- build Gibbs distribution over unused drugs ----------
                deltas, cand_drugs, cand_disp, cand_wts = [], [], [], []
                for drug in all_drugs:
                    if drug in drug_set:
                        continue
                    disp_vec, wt_pair = self.blended_disp(centroid_wt, drug)
                    if disp_vec is None:
                        continue
                    new_dist = np.linalg.norm(cur_pos + disp_vec - end)
                    deltas.append(dist - new_dist)      # improvement (>0 better)
                    cand_drugs.append(drug)
                    cand_disp.append(disp_vec)
                    cand_wts.append(wt_pair) 

                if not cand_drugs:                       # dead end
                    break

                scores = np.exp(beta * np.array(deltas)) # Boltzmann
                probs  = scores / scores.sum()
                idx_chosen = random.choices(range(len(cand_drugs)), weights=probs)[0]

                # ---------- advance state ---------------------------------------
                chosen_drug  = cand_drugs[idx_chosen]
                disp_vec     = cand_disp[idx_chosen]
                chosen_wt_pair = cand_wts[idx_chosen]
                cur_pos     += disp_vec
                ordered_drugs.append(chosen_drug)
                drug_set.add(chosen_drug)
                gene_list.append(self.drugs_to_genes.get(chosen_drug, ()))
                cl_path.append(chosen_wt_pair)
                step += 1

        # ================================================= post-processing
        # Patch “is_best_at_step” flags and add missing best-paths
        for s, best_path in best_at_step.items():
            key = self.path_key(best_path[1], s)
            if key in added_keys:
                # find existing entry and flag it
                for entry in seen_paths:
                    if self.path_key(entry['path'][1], entry['step']) == key:
                        entry['is_best_at_step'] = True
                        break
            else:
                seen_paths.append({
                    'step'             : s,
                    'is_best_at_step'  : True,
                    'is_success'       : False,
                    'covered_threshold': False,
                    'path'             : best_path
                })
                added_keys.add(key)

        progress_df = pd.DataFrame(seen_paths)

        return {
            'search_id'        : search_id,
            'type_of_search'   : 'probabilistic',
            'n_steps'          : n_steps,
            'n_paths'          : n_paths,
            'beta'             : beta,
            'blend'            : blend,
            'starting_cl'      : starting_cl,
            'target_cl'        : ending_cl,
            'starting_position': start,
            'target_position'  : end,
            'target_radius'    : end_radius,
            'progress_table'   : progress_df
        }
    

    def search_path_dqn(self, search_id: str, starting_cl: str, ending_cl: str, 
                        q_network: nn.Module, device: torch.device, 
                        strategy: str = 'beam',
                        n_steps: int = 8, k: int = 10, threshold: float = 0.4, blend: int = 2):
        """
        Performs a k-path search guided by a trained DQN model, using a selectable strategy.

        Args:
            search_id (str): A unique identifier for this search.
            starting_cl (str): The starting cell line.
            ending_cl (str): The target cell line.
            q_network (nn.Module): The trained PyTorch Q-network model.
            device (torch.device): The device ('mps' or 'cpu') the model should run on.
            strategy (str): The search strategy to use. One of:
                            'beam' -> Expands all options, then prunes the entire pool to k.
                            'tree' -> Each path expands its own top k, without pruning the total.
                            Defaults to 'beam'.
            n_steps (int): The maximum number of steps (drugs) in a path.
            k (int): The beam width or branching factor.
            threshold (float): The progress threshold for recording paths.
            blend (int): The number of nearest WT centroids to blend for displacement.
        """
        # --- 1. SETUP ---
        q_network.eval() 

        # Warning for the computationally expensive tree search
        if strategy == 'tree' and k**n_steps > 100000:
            print(f"WARNING: Tree search with k={k} and n_steps={n_steps} will explore up to {k**n_steps} paths, which may be very slow or run out of memory.")

        start = self.centroids[starting_cl]
        end = self.centroids[ending_cl]
        end_radius = self.radiuses[ending_cl]
        all_drugs = self.drugs
        init_dist = np.linalg.norm(start - end)
        progress_threshold = (1 - threshold) * init_dist

        paths = [(start, [], set(), [], init_dist, [], 0.0)]
        seen_paths, best_at_step, added_keys = [], {}, set()

        # --- 2. MAIN SEARCH LOOP ---
        for step in range(n_steps + 1):
            new_paths = []

            for cur_pos, ord_drugs, drug_set, gene_list, dist, cl_path, _ in paths:
                # --- Recording Logic ---
                key = self.path_key(ord_drugs, step)
                if step != 0:
                    path_for_recording = (cur_pos, ord_drugs, drug_set, gene_list, dist, cl_path)
                    if step not in best_at_step or dist < best_at_step[step][4]:
                        best_at_step[step] = path_for_recording
                    if dist < end_radius and key not in added_keys:
                        seen_paths.append({'step': step, 'is_best_at_step': False, 'is_success': True, 'covered_threshold': True, 'path': path_for_recording})
                        added_keys.add(key)
                        continue
                    if dist <= progress_threshold and key not in added_keys:
                        seen_paths.append({'step': step, 'is_best_at_step': False, 'is_success': False, 'covered_threshold': True, 'path': path_for_recording})
                        added_keys.add(key)
                if step == n_steps:
                    continue
                
                # --- Guided Expansion using DQN ---
                steps_remaining = n_steps - step
                state_vec = np.hstack([cur_pos, np.array([steps_remaining])]).astype(np.float32)
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(device)
                with torch.no_grad():
                    all_q_values = q_network(state_tensor).squeeze(0)

                drug_candidates = []
                for action_idx, q_value in enumerate(all_q_values):
                    drug_name = all_drugs[action_idx]
                    if drug_name in drug_set: continue

                    dists, idxs = self.centroid_tree.query(cur_pos, k=blend)
                    weights = 1.0 / (dists + 1e-6)
                    weights /= weights.sum()
                    centroid_weight_list = [(self.centroid_keys[idx], weight) for idx, weight in zip(idxs, weights)]
                    
                    disp, wt_pair = self.blended_disp(centroid_weight_list, drug_name)
                    if disp is None: continue

                    new_pos = cur_pos + disp
                    new_dist = np.linalg.norm(new_pos - end)
                    
                    drug_candidates.append((
                        new_pos, ord_drugs + [drug_name], drug_set | {drug_name},
                        gene_list + [self.drugs_to_genes.get(drug_name, ())],
                        new_dist, cl_path + [wt_pair], q_value.item()
                    ))

                # ====================================================================
                # STRATEGY-DEPENDENT EXPANSION
                # ====================================================================
                if strategy == 'beam':
                    # Add all valid moves from this path to the global pool
                    new_paths.extend(drug_candidates)
                elif strategy == 'tree':
                    # Find the top-k moves for this specific path and add them
                    top_k_for_this_path = heapq.nlargest(k, drug_candidates, key=lambda p: p[6])
                    new_paths.extend(top_k_for_this_path)
                else:
                    raise ValueError(f"Unknown strategy: '{strategy}'. Must be 'beam' or 'tree'.")

            if not new_paths:
                break
                
            if step == n_steps:
                break
            
            # ====================================================================
            # STRATEGY-DEPENDENT PRUNING
            # ====================================================================
            if strategy == 'beam':
                # Prune the entire pool of candidates back down to the top k
                paths = heapq.nlargest(k, new_paths, key=lambda p: p[6])
            elif strategy == 'tree':
                # Do not prune. The number of paths grows exponentially.
                paths = new_paths

            if not paths:
                break

        # --- 3. POST-PROCESSING AND RETURN ---
        # (This logic is identical for both strategies)
        for step_num, best_path_tuple in best_at_step.items():
            key = self.path_key(best_path_tuple[1], step_num)
            if key in added_keys:
                for entry in seen_paths:
                    entry_key = self.path_key(entry['path'][1], entry['step'])
                    if entry_key == key:
                        entry['is_best_at_step'] = True
                        break
            else:
                seen_paths.append({
                    'step': step_num, 'is_best_at_step': True, 'is_success': False,
                    'covered_threshold': False, 'path': best_path_tuple
                })
                added_keys.add(key)

        df = pd.DataFrame(seen_paths)

        return {
            'search_id': search_id,
            'type_of_search': f'dqn_guided_{strategy}',
            'n_steps': n_steps, 'k': k, 'threshold': threshold, 'blend': blend,
            'starting_cl': starting_cl, 'target_cl': ending_cl,
            'starting_position': start, 'target_position': end,
            'target_radius': end_radius, 'progress_table': df
        }

            

def search_to_df(search_results):
    """
    Convert k-paths, probabilistic, or DQN-guided search output into a flat DataFrame.

    Handles:
      • k-paths           → has keys: k
      • probabilistic     → has keys: n_paths, beta, blend
      • dqn_guided_k_paths → has keys: k, threshold, blend
    """
    meta = {
        'search_id'        : search_results['search_id'],
        'type_of_search'   : search_results['type_of_search'],
        'n_steps'          : search_results['n_steps'],
        
        # --- MODIFICATION ---
        # Added all optional keys from all search types.
        # .get() will return None if a key doesn't exist for a given search type.
        'k'                : search_results.get('k'),
        'n_paths'          : search_results.get('n_paths'),
        'beta'             : search_results.get('beta'),
        'blend'            : search_results.get('blend'),
        'threshold'        : search_results.get('threshold'),

        # common fields
        'starting_cl'      : search_results['starting_cl'],
        'target_cl'        : search_results['target_cl'],
        'starting_position': search_results['starting_position'],
        'target_position'  : search_results['target_position'],
        'starting_distance': np.linalg.norm(
                                 search_results['starting_position']
                                 - search_results['target_position']),
        'target_radius'    : search_results['target_radius'],
    }

    # This part of the logic can remain the same as the 'progress_table'
    # and the 'path' tuple structure are consistent across all search types.
    records = []
    if 'progress_table' in search_results and not search_results['progress_table'].empty:
        for _, row in search_results['progress_table'].iterrows():
            pos, drug_seq, drug_set, gene_targets, final_dist, cl_path = row['path']
            records.append({
                **meta,
                'step'              : row['step'],
                'path_type'         : (
                    'success'   if row['is_success']
                    else 'threshold' if row['covered_threshold']
                    else 'best_at_step' if row['is_best_at_step']
                    else 'unclassified'
                ),
                'is_best_at_step'   : row['is_best_at_step'],
                'is_success'        : row['is_success'],
                'covered_threshold' : row['covered_threshold'],
                'end_position'      : pos,
                'drug_sequence'     : drug_seq,
                'genes_targeted'    : gene_targets,
                'cell_path'         : cl_path,
                'final_distance'    : final_dist,
            })

    # If no paths were recorded but we still want a record of the search attempt
    if not records:
        records.append(meta)

    return pd.DataFrame(records)