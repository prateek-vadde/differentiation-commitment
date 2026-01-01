"""
Phase 3: Perturbation Definitions

Compute epsilon_k (biologically scaled step size) and gradients for directional perturbations.

Per pair:
- epsilon_k = median distance to 10th nearest neighbor (interpretable scale)
- Sample up to 5000 cells uniformly
- For directional: compute local C gradient via kNN regression (k=50)

Outputs per pair:
- epsilon_k.json
- sampled_cells.npy
- gradients.npy (directional families only)
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def compute_epsilon_and_sample(pair_idx, species, data_root, results_root, config):
    """
    Compute epsilon_k and sample cells for one pair.

    Returns:
        pair_idx for tracking
    """
    # Load X_pca
    X_pca_path = data_root / species / f'pair_{pair_idx}' / 'X_pca.npy'
    X_pca = np.load(X_pca_path)

    # Load baseline C
    C_path = results_root / species / f'pair_{pair_idx}' / 'C.npy'
    C = np.load(C_path)

    n_cells = len(X_pca)

    # Sample cells (up to max, or all if fewer)
    num_sample = min(config['perturb']['num_cells_sample_per_pair'], n_cells)
    rng = np.random.RandomState(config['seed'])
    sampled_indices = rng.choice(n_cells, size=num_sample, replace=False)

    # Compute epsilon_k: median distance to 10th nearest neighbor
    m = config['eps_scale_knn']  # 10
    nbrs = NearestNeighbors(n_neighbors=m+1, algorithm='auto').fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)

    # distances[:, 0] is self (distance=0), distances[:, m] is m-th neighbor
    distances_to_mth = distances[:, m]
    epsilon_k = float(np.median(distances_to_mth))

    # Compute gradients for directional perturbations
    # BATCH all kNN queries at once, then vectorize linear regressions
    k_grad = config['knn_k_gradient']  # 50
    nbrs_grad = NearestNeighbors(n_neighbors=k_grad, algorithm='auto').fit(X_pca)

    # BATCH: Find k neighbors for ALL sampled cells at once
    X_sampled = X_pca[sampled_indices]  # (num_sample, d)
    _, neighbor_indices = nbrs_grad.kneighbors(X_sampled)  # (num_sample, k)

    # Gradient computation using batched linear algebra
    # Build (num_sample, k, d) array of all neighbor features
    k_grad = neighbor_indices.shape[1]
    d = X_pca.shape[1]

    X_neighbors_all = X_pca[neighbor_indices]  # (num_sample, k, d)
    C_neighbors_all = C[neighbor_indices]  # (num_sample, k)

    # Center data for numerical stability (vectorized across all cells)
    X_mean = X_neighbors_all.mean(axis=1, keepdims=True)  # (num_sample, 1, d)
    C_mean = C_neighbors_all.mean(axis=1, keepdims=True)  # (num_sample, 1)
    X_centered = X_neighbors_all - X_mean  # (num_sample, k, d)
    C_centered = C_neighbors_all - C_mean  # (num_sample, k)

    # Solve all least squares problems at once
    # For each sample: solve X_centered[i] @ w[i] = C_centered[i]
    # Solution: w = (X^T X)^{-1} X^T y
    # Compute X^T X for all samples: (num_sample, d, d)
    XTX = np.einsum('nkd,nke->nde', X_centered, X_centered)  # (num_sample, d, d)

    # Compute X^T y for all samples: (num_sample, d)
    XTy = np.einsum('nkd,nk->nd', X_centered, C_centered)  # (num_sample, d)

    # Solve all systems: (num_sample, d, d) @ (num_sample, d) = (num_sample, d)
    gradients = np.linalg.solve(XTX, XTy)  # (num_sample, d)

    # Save outputs
    out_dir = results_root / species / f'pair_{pair_idx}'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'epsilon_k.json', 'w') as f:
        json.dump({'epsilon_k': epsilon_k}, f)

    np.save(out_dir / 'sampled_cells.npy', sampled_indices)
    np.save(out_dir / 'gradients.npy', gradients)

    print(f"  Pair {pair_idx}: epsilon_k={epsilon_k:.6f}, n_sampled={num_sample}")

    return pair_idx


def process_species(species, config, data_root, results_root):
    """Process all pairs for one species (parallelized)."""
    print(f"\n{'='*60}")
    print(f"Perturbation Definitions: {species}")
    print(f"{'='*60}")

    # Find all pairs
    species_dir = data_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])
    num_pairs = len(pair_dirs)

    print(f"Found {num_pairs} pairs")

    # Parallelize across pairs
    from functools import partial
    compute_fn = partial(
        compute_epsilon_and_sample,
        species=species,
        data_root=data_root,
        results_root=results_root,
        config=config
    )

    with ProcessPoolExecutor(max_workers=min(num_pairs, 8)) as executor:
        futures = {executor.submit(compute_fn, pair_idx): pair_idx
                   for pair_idx in range(num_pairs)}

        for future in as_completed(futures):
            pair_idx = future.result()

    print(f"\n[{species}] ✓ DONE\n")


def main():
    """Compute perturbation definitions for all species."""
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')
    data_root = project_root / 'data_phase2A'
    results_root = project_root / 'results_p2b'

    config_path = project_root / 'config_p2b_p3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Phase 3: Perturbation Definitions")
    print(f"Sampling: up to {config['perturb']['num_cells_sample_per_pair']} cells/pair")
    print(f"Epsilon scale: {config['eps_scale_knn']}-th nearest neighbor")
    print(f"Gradient k: {config['knn_k_gradient']}")

    # Process each species
    for species in ['mouse', 'zebrafish']:
        process_species(species, config, data_root, results_root)

    print("✓ ALL SPECIES COMPLETE\n")


if __name__ == '__main__':
    main()
