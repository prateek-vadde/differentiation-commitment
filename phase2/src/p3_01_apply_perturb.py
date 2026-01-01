"""
Phase 3: Apply Perturbations

Generate perturbed states for all families + matched nulls.

Families:
1. local_random: x' = x + epsilon_k * u/||u||
2. directional_downC: x' = x - epsilon_k * g/||g|| (+ matched nulls)
3. directional_upC: x' = x + epsilon_k * g/||g|| (+ matched nulls)

Matched nulls: orthogonal random directions, same magnitude

Outputs per pair:
- perturbations.npz (compressed arrays)
"""

import json
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def generate_matched_nulls(x, gradient, epsilon_k, num_nulls, rng):
    """
    Generate matched null perturbations: orthogonal to gradient, same magnitude.
    VECTORIZED implementation.

    Args:
        x: Original state (d,)
        gradient: Direction vector (d,)
        epsilon_k: Step size
        num_nulls: Number of nulls to generate
        rng: Random state

    Returns:
        x_nulls: (num_nulls, d) perturbed states
    """
    d = len(x)

    # VECTORIZED: Generate all random directions at once
    V = rng.randn(num_nulls, d)  # (num_nulls, d)

    # VECTORIZED: Orthogonalize all to gradient at once
    # v_perp = v - (v · g / ||g||^2) * g
    g_norm_sq = np.dot(gradient, gradient)
    projections = np.dot(V, gradient) / g_norm_sq  # (num_nulls,)
    V_perp = V - projections[:, None] * gradient[None, :]  # (num_nulls, d)

    # VECTORIZED: Normalize and scale
    norms = np.linalg.norm(V_perp, axis=1, keepdims=True) + 1e-12  # (num_nulls, 1)
    V_perp_norm = V_perp / norms  # (num_nulls, d)

    # VECTORIZED: Apply perturbation
    x_nulls = x[None, :] + epsilon_k * V_perp_norm  # (num_nulls, d)

    return x_nulls


def apply_perturbations_pair(pair_idx, species, data_root, results_root, config):
    """
    Apply all perturbations for one pair.

    Returns:
        pair_idx for tracking
    """
    # Load data
    X_pca = np.load(data_root / species / f'pair_{pair_idx}' / 'X_pca.npy')
    C = np.load(results_root / species / f'pair_{pair_idx}' / 'C.npy')

    # Load perturbation definitions
    pair_dir = results_root / species / f'pair_{pair_idx}'

    with open(pair_dir / 'epsilon_k.json', 'r') as f:
        epsilon_k = json.load(f)['epsilon_k']

    sampled_indices = np.load(pair_dir / 'sampled_cells.npy')
    gradients = np.load(pair_dir / 'gradients.npy')

    num_sample = len(sampled_indices)
    d = X_pca.shape[1]
    num_null_per_cell = config['perturb']['num_null_per_cell']

    rng = np.random.RandomState(config['seed'] + pair_idx)  # Pair-specific seed

    # Generate all perturbations
    X_sampled = X_pca[sampled_indices]  # (num_sample, d)
    C_sampled = C[sampled_indices]  # (num_sample,)

    # Normalize all gradients at once
    g_norms = np.linalg.norm(gradients, axis=1, keepdims=True) + 1e-12  # (num_sample, 1)
    gradients_norm = gradients / g_norms  # (num_sample, d)

    # Family 1: local_random - vectorized
    U = rng.randn(num_sample, d)  # (num_sample, d)
    U_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
    U_norm = U / U_norms
    X_local_random = X_sampled + epsilon_k * U_norm  # (num_sample, d)

    # Family 2 & 3: directional - vectorized
    X_down = X_sampled - epsilon_k * gradients_norm  # (num_sample, d)
    X_up = X_sampled + epsilon_k * gradients_norm  # (num_sample, d)

    # Generate ALL matched nulls at once for downC and upC
    # For each cell, generate num_null_per_cell orthogonal perturbations
    V_all = rng.randn(num_sample, num_null_per_cell, d)  # (num_sample, num_null, d)

    # Orthogonalize: v_perp = v - (v·g / ||g||^2) * g
    # gradients: (num_sample, d) -> (num_sample, 1, d)
    g_expanded = gradients[:, None, :]  # (num_sample, 1, d)
    g_norm_sq = np.sum(gradients * gradients, axis=1, keepdims=True)[:, :, None]  # (num_sample, 1, 1)

    # Dot product: (num_sample, num_null, d) · (num_sample, 1, d) -> (num_sample, num_null)
    projections = np.sum(V_all * g_expanded, axis=2, keepdims=True) / g_norm_sq  # (num_sample, num_null, 1)
    V_perp = V_all - projections * g_expanded  # (num_sample, num_null, d)

    # Normalize and scale
    V_norms = np.linalg.norm(V_perp, axis=2, keepdims=True) + 1e-12  # (num_sample, num_null, 1)
    V_perp_norm = V_perp / V_norms

    # Apply perturbations for downC and upC nulls
    X_nulls_down = X_sampled[:, None, :] + epsilon_k * V_perp_norm  # (num_sample, num_null, d)
    X_nulls_up = X_sampled[:, None, :] + epsilon_k * V_perp_norm  # (num_sample, num_null, d) - same nulls

    # Array construction using numpy stacking
    # Per cell: 1 local_random + 1 down + num_null downNulls + 1 up + num_null upNulls
    n_per_cell = 3 + 2 * num_null_per_cell

    # Stack all X arrays: (num_sample, n_per_cell, d)
    # local, down, downNulls, up, upNulls
    X_all = np.concatenate([
        X_local_random[:, None, :],  # (num_sample, 1, d)
        X_down[:, None, :],  # (num_sample, 1, d)
        X_nulls_down,  # (num_sample, num_null, d)
        X_up[:, None, :],  # (num_sample, 1, d)
        X_nulls_up  # (num_sample, num_null, d)
    ], axis=1)  # -> (num_sample, n_per_cell, d)

    # Reshape to (n_total, d)
    X_perturbed = X_all.reshape(-1, d)
    n_perturb = X_perturbed.shape[0]

    # Replicate cell_idx and baseline_C
    cell_indices = np.repeat(sampled_indices, n_per_cell)  # (n_total,)
    baseline_Cs = np.repeat(C_sampled, n_per_cell)  # (n_total,)

    # Build families array
    family_pattern = (['local_random', 'directional_downC'] +
                     ['directional_downC'] * num_null_per_cell +
                     ['directional_upC'] +
                     ['directional_upC'] * num_null_per_cell)
    families = np.tile(family_pattern, num_sample).astype('U32')

    # Build is_nulls array
    null_pattern = ([False, False] +
                   [True] * num_null_per_cell +
                   [False] +
                   [True] * num_null_per_cell)
    is_nulls = np.tile(null_pattern, num_sample)

    # Build null_ids array
    id_pattern = ([-1, -1] +
                 list(range(num_null_per_cell)) +
                 [-1] +
                 list(range(num_null_per_cell)))
    null_ids = np.tile(id_pattern, num_sample).astype(np.int32)

    # Save as compressed npz
    np.savez_compressed(
        pair_dir / 'perturbations.npz',
        cell_indices=cell_indices,
        baseline_Cs=baseline_Cs,
        families=families,
        is_nulls=is_nulls,
        null_ids=null_ids,
        X_perturbed=X_perturbed
    )

    print(f"  Pair {pair_idx}: {n_perturb} perturbations ({num_sample} cells)")

    return pair_idx


def process_species(species, config, data_root, results_root):
    """Process all pairs for one species (parallelized)."""
    print(f"\n{'='*60}")
    print(f"Apply Perturbations: {species}")
    print(f"{'='*60}")

    # Find all pairs
    species_dir = data_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])
    num_pairs = len(pair_dirs)

    print(f"Found {num_pairs} pairs")

    # Parallelize across pairs
    from functools import partial
    apply_fn = partial(
        apply_perturbations_pair,
        species=species,
        data_root=data_root,
        results_root=results_root,
        config=config
    )

    with ProcessPoolExecutor(max_workers=min(num_pairs, 8)) as executor:
        futures = {executor.submit(apply_fn, pair_idx): pair_idx
                   for pair_idx in range(num_pairs)}

        for future in as_completed(futures):
            pair_idx = future.result()

    print(f"\n[{species}] ✓ DONE\n")


def main():
    """Apply perturbations for all species."""
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')
    data_root = project_root / 'data_phase2A'
    results_root = project_root / 'results_p2b'

    config_path = project_root / 'config_p2b_p3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Phase 3: Apply Perturbations")
    print(f"Families: {config['perturb']['families']}")
    print(f"Matched nulls per cell: {config['perturb']['num_null_per_cell']}")

    # Process each species
    for species in ['mouse', 'zebrafish']:
        process_species(species, config, data_root, results_root)

    print("✓ ALL SPECIES COMPLETE\n")


if __name__ == '__main__':
    main()
