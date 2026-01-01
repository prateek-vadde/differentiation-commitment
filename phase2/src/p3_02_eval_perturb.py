"""
Phase 3: Evaluate Perturbations (Mode A - Exact Recomputation)

For each perturbed state x', compute C'(x') using Phase 2A encoder:
1. Embed x' → e'
2. Compute similarities to all targets
3. Apply TopM + softmax → new row distribution T'_{i·}
4. Compose forward for h=1,2,3
5. Compute C'(x') exactly as in Phase 2B
6. Compute ΔC = C' - C

Statistical analysis:
- Bin cells by baseline C into deciles
- For each decile, compare directional vs matched null ΔC
- Report: median(D), bootstrap CI, paired Wilcoxon p-value

Outputs:
- results/<species>/pair_<k>/p3_pair_effects.json
- results/<species>/p3_effects.json (aggregated)
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from p2a_02_model import SharedEncoder
from common_math import percentile_ranks, row_entropy_csr, bootstrap_ci_median, paired_wilcoxon
from common_io import load_pair_bundle


def load_encoder(species, models_root, config_phase2a):
    """Load Phase 2A encoder model."""
    model_path = models_root / species / 'best_model.pt'

    # Build encoder model
    encoder = SharedEncoder(config_phase2a)

    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Extract encoder weights from full model state dict
    encoder_state = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('encoder.'):
            encoder_state[k.replace('encoder.', '', 1)] = v

    encoder.load_state_dict(encoder_state)
    encoder.eval()

    return encoder


def compute_C_from_perturbed_batch(X_perturbed_batch, encoder, Y_targets_embedded_gpu, T_composed_by_horizon,
                                    topM, horizons, horizon_weights, config, device):
    """
    Compute C for a BATCH of perturbed states using exact recomputation.
    GPU-OPTIMIZED batched version with PRE-COMPUTED transition matrices.

    Args:
        X_perturbed_batch: Perturbed states (batch_size, d)
        encoder: Phase 2A encoder
        Y_targets_embedded_gpu: Pre-embedded target states ON GPU (torch tensor, N_tgt x embed_dim)
        T_composed_by_horizon: Dict mapping horizon -> pre-composed T_hat matrix (h>1)
        topM: TopM for this pair
        horizons: [1, 2, 3]
        horizon_weights: weights dict
        config: config dict
        device: cuda/cpu

    Returns:
        C_values: (batch_size,) Commitment scores
    """
    batch_size = len(X_perturbed_batch)

    # BATCHED: Embed all perturbed states at once
    with torch.no_grad():
        X_batch_tensor = torch.from_numpy(X_perturbed_batch).float().to(device)  # (batch_size, d)
        E_perturbed = encoder(X_batch_tensor)  # (batch_size, embed_dim)
        E_perturbed = F.normalize(E_perturbed, p=2, dim=1)  # (batch_size, embed_dim)

        # Y_targets already on GPU (passed as tensor)
        similarities = torch.mm(E_perturbed, Y_targets_embedded_gpu.T)  # (batch_size, N_tgt) on GPU

    # Apply TopM for all rows
    N_tgt = similarities.shape[1]
    if topM < N_tgt:
        # Vectorized TopM on GPU using topk
        _, top_indices = torch.topk(similarities, topM, dim=1)  # (batch_size, topM)

        # Create mask efficiently on GPU
        row_sparse_batch = torch.full_like(similarities, -torch.inf)
        # Use advanced indexing to set topM values
        row_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
        row_sparse_batch[row_idx, top_indices] = similarities[row_idx, top_indices]
    else:
        row_sparse_batch = similarities

    # DENSE GPU: Keep similarities on GPU, softmax on GPU, dense @ dense
    # similarities: (batch_size, N_tgt) dense GPU tensor
    row_logits = row_sparse_batch - row_sparse_batch.max(dim=1, keepdim=True)[0]
    row_dist_gpu = torch.exp(row_logits)
    row_dist_gpu = row_dist_gpu / row_dist_gpu.sum(dim=1, keepdim=True)

    # Compute max entropy for normalization
    N_targets = Y_targets_embedded_gpu.shape[0]
    max_entropy = np.log(N_targets) if config['entropy_log_base'] == 'e' else np.log2(N_targets)

    U_batch = np.zeros(batch_size)

    for h in horizons:
        if h == 1:
            # h=1: Use row distribution directly
            T_composed_gpu = row_dist_gpu
        else:
            # h>1: dense matrix multiplication
            if T_composed_by_horizon[h] is None:
                continue

            # T_composed_by_horizon[h] is dense GPU tensor (N_tgt x N_tgt)
            # Dense @ dense: (batch_size x N_tgt) @ (N_tgt x N_tgt) = (batch_size x N_tgt)
            T_composed_gpu = torch.mm(row_dist_gpu, T_composed_by_horizon[h])

        # Compute entropy on GPU from dense tensor
        # T_composed_gpu: (batch_size, N_tgt) dense
        T_clipped = torch.clamp(T_composed_gpu, min=config['numerical_tolerance']['min_prob_clip'])
        log_base = config['entropy_log_base']

        # Convert log_base to numeric divisor
        if log_base == 'e':
            log_divisor = 1.0  # torch.log is natural log
        elif log_base == '2':
            log_divisor = np.log(2)
        else:
            log_divisor = np.log(float(log_base))

        H_h_batch_gpu = -torch.sum(T_clipped * torch.log(T_clipped) / log_divisor, dim=1)
        H_h_batch = H_h_batch_gpu.cpu().numpy()

        normalized_H_batch = H_h_batch / max_entropy
        U_batch += horizon_weights[h] * normalized_H_batch

    C_values = 1.0 - U_batch

    return C_values


def eval_pair(pair_idx, species, data_root, results_root, models_root, config, config_phase2a, device):
    """
    Evaluate all perturbations for one pair.

    Returns:
        pair_idx for tracking
    """
    print(f"\n  Processing pair {pair_idx}...", flush=True)

    # Load encoder
    encoder = load_encoder(species, models_root, config_phase2a).to(device)

    # Load pair data
    pair = load_pair_bundle(species, pair_idx, data_root)

    # Load TopM
    topM_path = data_root / species / f'pair_{pair_idx}' / 'topM.json'
    with open(topM_path, 'r') as f:
        topM = json.load(f)

    # Pre-embed all targets for this pair
    with torch.no_grad():
        Y_targets_tensor = torch.from_numpy(pair.Y_pca).float().to(device)
        Y_targets_embedded_gpu = encoder(Y_targets_tensor)  # (N_tgt, embed_dim)
        Y_targets_embedded_gpu = F.normalize(Y_targets_embedded_gpu, p=2, dim=1)  # Normalize once

    # Load next pairs for composition (if available)
    horizons = config['horizons']
    max_horizon = max(horizons)
    T_hat_next_pairs = []

    for h in range(1, max_horizon):
        next_pair_idx = pair_idx + h
        try:
            next_pair = load_pair_bundle(species, next_pair_idx, data_root)
            T_hat_next_pairs.append(next_pair.T_hat)
        except:
            break

    # Pre-compute composed transition matrices
    print(f"    Pre-computing composed transition matrices + GPU transfer...", flush=True)
    T_composed_by_horizon = {}
    for h in horizons:
        if h == 1:
            T_composed_by_horizon[h] = None  # Will use row_dist directly
        else:
            if h - 1 <= len(T_hat_next_pairs):
                # Compose T_hat matrices for this horizon
                T_composed = T_hat_next_pairs[0]
                for j in range(1, h - 1):
                    if j < len(T_hat_next_pairs):
                        T_composed = T_composed @ T_hat_next_pairs[j]

                # Convert to dense GPU tensor
                T_composed_dense = torch.from_numpy(T_composed.toarray()).float().to(device)
                T_composed_by_horizon[h] = T_composed_dense
            else:
                T_composed_by_horizon[h] = None

    # Load perturbations
    perturb_path = results_root / species / f'pair_{pair_idx}' / 'perturbations.npz'

    if not perturb_path.exists():
        print(f"    No perturbations found, skipping", flush=True)
        return pair_idx

    perturb_data = np.load(perturb_path)

    cell_indices = perturb_data['cell_indices']
    baseline_Cs = perturb_data['baseline_Cs']
    families = perturb_data['families']
    is_nulls = perturb_data['is_nulls']
    null_ids = perturb_data['null_ids']
    X_perturbed = perturb_data['X_perturbed']

    n_perturb = len(cell_indices)

    # Compute C' for all perturbations
    print(f"    Computing C' for {n_perturb} perturbations (GPU-batched)...", flush=True)

    horizon_weights = {h: 1.0 / h for h in horizons}
    weight_sum = sum(horizon_weights.values())
    horizon_weights = {h: w / weight_sum for h, w in horizon_weights.items()}

    batch_size = 512  # Larger batch for GPU efficiency

    C_primes = []
    for start_idx in range(0, n_perturb, batch_size):
        end_idx = min(start_idx + batch_size, n_perturb)

        # BATCHED: Process entire batch at once
        X_batch = X_perturbed[start_idx:end_idx]
        batch_C_values = compute_C_from_perturbed_batch(
            X_batch, encoder, Y_targets_embedded_gpu, T_composed_by_horizon,
            topM, horizons, horizon_weights, config, device
        )

        C_primes.extend(batch_C_values)

        if (end_idx // batch_size) % 5 == 0:
            print(f"      Progress: {end_idx}/{n_perturb}", flush=True)

    C_primes = np.array(C_primes)

    # Compute ΔC = C' - C_baseline
    delta_Cs = C_primes - baseline_Cs

    print(f"    Computing statistics by decile...", flush=True)

    # Bin by baseline C into deciles
    n_deciles = config['perturb']['evaluate_deciles']
    decile_edges = np.percentile(baseline_Cs, np.linspace(0, 100, n_deciles + 1))

    # Results per family and decile
    results = {}

    for family_name in config['perturb']['families']:
        results[family_name] = {'deciles': []}

        for decile_idx in range(n_deciles):
            lower = decile_edges[decile_idx]
            upper = decile_edges[decile_idx + 1]

            # Cells in this decile
            if decile_idx == n_deciles - 1:
                in_decile = (baseline_Cs >= lower) & (baseline_Cs <= upper)
            else:
                in_decile = (baseline_Cs >= lower) & (baseline_Cs < upper)

            # Filter for this family
            is_family = families == family_name

            # Directional
            is_dir = is_family & ~is_nulls & in_decile
            delta_dir = delta_Cs[is_dir]

            # Matched nulls
            is_null_family = is_family & is_nulls & in_decile
            delta_null = delta_Cs[is_null_family]

            if len(delta_dir) == 0:
                continue

            # Paired difference D = delta_dir - delta_null
            # Match by cell_idx using broadcasting
            cells_dir = cell_indices[is_dir]
            deltas_dir = delta_Cs[is_dir]

            cells_null = cell_indices[is_null_family]
            deltas_null = delta_Cs[is_null_family]

            # Build pairing matrix: dir_cells vs null_cells
            # pairing[i, j] = True if cells_dir[i] == cells_null[j]
            pairing = cells_dir[:, None] == cells_null[None, :]  # (n_dir, n_null)

            # VECTORIZED: Compute ALL pairwise differences where pairing is True
            # Use broadcasting: deltas_dir[:, None] - deltas_null[None, :] gives (n_dir, n_null) matrix
            delta_matrix = deltas_dir[:, None] - deltas_null[None, :]  # (n_dir, n_null)

            # Extract only valid pairs (where pairing is True)
            D_values = delta_matrix[pairing]

            if len(D_values) == 0:
                continue

            # Statistics
            median_D = float(np.median(D_values))

            # Bootstrap CI on median
            if len(D_values) >= 10:
                _, ci_lower, ci_upper = bootstrap_ci_median(
                    D_values,
                    n_boot=config['num_bootstrap'],
                    seed=config['seed']
                )
            else:
                ci_lower = median_D
                ci_upper = median_D

            # Paired Wilcoxon (D vs 0)
            if len(D_values) >= 5:
                p_value = paired_wilcoxon(D_values, np.zeros(len(D_values)), alternative='less')
            else:
                p_value = 1.0

            results[family_name]['deciles'].append({
                'decile_idx': decile_idx,
                'baseline_C_range': [float(lower), float(upper)],
                'n_cells': int(len(cells_dir)),
                'n_comparisons': len(D_values),
                'median_D': median_D,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'p_value': float(p_value),
                'mean_delta_dir': float(delta_dir.mean()),
                'mean_delta_null': float(delta_null.mean()) if len(delta_null) > 0 else None
            })

    # Save results
    output_path = results_root / species / f'pair_{pair_idx}' / 'p3_pair_effects.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Pair {pair_idx} complete", flush=True)

    return pair_idx


def process_species(species, config, config_phase2a, data_root, results_root, models_root, device):
    """Process all pairs for one species."""
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluate Perturbations: {species}", flush=True)
    print(f"{'='*60}", flush=True)

    # Find all pairs with perturbations
    species_dir = results_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_') and
                       (d / 'perturbations.npz').exists()])

    num_pairs = len(pair_dirs)
    print(f"Found {num_pairs} pairs with perturbations", flush=True)

    # Process all pairs in parallel
    from functools import partial
    eval_fn = partial(eval_pair, species=species, data_root=data_root, results_root=results_root,
                     models_root=models_root, config=config, config_phase2a=config_phase2a, device=device)

    with ProcessPoolExecutor(max_workers=min(num_pairs, 4)) as executor:  # Reduced to 4 for GPU memory
        pair_indices = [int(d.name.split('_')[1]) for d in pair_dirs]
        futures = {executor.submit(eval_fn, pair_idx): pair_idx for pair_idx in pair_indices}

        for future in as_completed(futures):
            pair_idx = future.result()

    print(f"\n[{species}] ✓ DONE\n", flush=True)


def main():
    """Evaluate perturbations for all species."""
    # Set spawn method for CUDA multiprocessing compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Phase 3 perturbations")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for prepared data')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory for results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config_p2b_p3.json')
    parser.add_argument('--species', type=str, nargs='+', default=['mouse', 'zebrafish'],
                        help='List of species to process')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    results_root = Path(args.results_root)

    # Models are in project/phase2/models
    models_root = Path(args.config).parent / 'models'
    config_phase2a_path = Path(args.config).parent / 'config_phase2A.json'

    with open(args.config, 'r') as f:
        config = json.load(f)

    with open(config_phase2a_path, 'r') as f:
        config_phase2a = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 3: Evaluate Perturbations (Mode A)", flush=True)
    print(f"Device: {device}", flush=True)

    # Process each species
    for species in args.species:
        process_species(species, config, config_phase2a, data_root, results_root, models_root, device)

    print("✓ ALL SPECIES COMPLETE\n", flush=True)


if __name__ == '__main__':
    main()
