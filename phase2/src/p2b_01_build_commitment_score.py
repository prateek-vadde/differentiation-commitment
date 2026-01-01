"""
Phase 2B: Build Commitment Score

Computes continuous commitment score C for each cell in each pair.

Algorithm:
1. For each pair k and horizon h ∈ {1,2,3}:
   - Compose T^(h)_k = T_k · T_{k+1} · ... · T_{k+h-1} (GPU-accelerated)
   - Compute row entropy H^(h)_{k,i} for all sources (GPU-accelerated)
   - Convert to within-pair percentile ranks R^(h)_{k,i}

2. Compute weighted average: U_k = Σ w_h R^(h)_{k} / Σ w_h where w_h = 1/h

3. Define commitment: C_k = 1 - U_k

Outputs per pair:
- C.npy: Commitment score (N_sources,)
- U.npy: Competence/uncertainty score (N_sources,)
- H_h{1,2,3}.npy: Entropies at each horizon
- R_h{1,2,3}.npy: Percentile ranks at each horizon

Uses GPU acceleration for massive speedup on GH200.
"""

import json
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from common_io import build_alignment, compose_T_hat
from common_math import row_entropy_csr, percentile_ranks


def compute_commitment_for_pair(
    pair_idx: int,
    species: str,
    data_root: Path,
    horizons: list,
    horizon_weights_type: str,
    config: dict,
    results_root: Path,
    cache_dir: Path
):
    """
    Compute commitment score C for one pair (parallelizable).

    Args:
        pair_idx: Pair index
        species: Species name
        data_root: Path to data
        horizons: List of horizon values [1, 2, 3]
        horizon_weights_type: 'inverse_h' for w_h = 1/h
        config: Configuration dict
        results_root: Path to results directory
        cache_dir: Path for caching composed matrices

    Returns:
        pair_idx: For tracking completion
    """
    # Load ONLY this pair and neighbors needed for composition
    from common_io import load_pair_bundle

    max_horizon = max(horizons)
    pairs_needed = []
    for k in range(pair_idx, min(pair_idx + max_horizon, 100)):
        try:
            pair_k = load_pair_bundle(species, k, data_root)
            pairs_needed.append(pair_k)
        except:
            break

    pair = pairs_needed[0]

    # Storage for entropies and ranks at each horizon
    entropies_all_h = {}
    ranks_all_h = {}

    # Horizon weights
    if horizon_weights_type == 'inverse_h':
        weights = {h: 1.0 / h for h in horizons}
    else:
        raise ValueError(f"Unknown horizon_weights_type: {horizon_weights_type}")

    # Normalize weights
    weight_sum = sum(weights.values())
    weights = {h: w / weight_sum for h, w in weights.items()}

    # Compute entropy at each horizon
    for h in horizons:
        # Check if we have enough pairs loaded for this horizon
        if h > len(pairs_needed):
            continue

        # Compose manually (cached)
        if h == 1:
            T_composed = pairs_needed[0].T_hat
        else:
            # Multiply h matrices
            T_composed = pairs_needed[0].T_hat
            for i in range(1, h):
                T_composed = T_composed @ pairs_needed[i].T_hat

        # Compute entropy
        base = config['entropy_log_base']
        clip = config['numerical_tolerance']['min_prob_clip']

        H_h = row_entropy_csr(
            T_composed,
            clip=clip,
            base=base,
            use_gpu=True
        )

        # Convert to percentile ranks
        R_h = percentile_ranks(H_h)

        # Store
        entropies_all_h[h] = H_h
        ranks_all_h[h] = R_h

    # Compute weighted average of ranks: U (vectorized)
    n_sources = len(pair.source_ids)
    ranks_matrix = np.zeros((len(horizons), n_sources))
    weights_vec = np.zeros(len(horizons))

    for idx, h in enumerate(horizons):
        if h in ranks_all_h:
            ranks_matrix[idx] = ranks_all_h[h]
            weights_vec[idx] = weights[h]

    U = np.sum(ranks_matrix * weights_vec[:, np.newaxis], axis=0)

    # Commitment: C = 1 - U
    C = 1.0 - U

    print(f"    Pair {pair_idx}: C range [{C.min():.4f}, {C.max():.4f}], mean {C.mean():.4f}")

    # Save outputs
    out_dir = results_root / species / f'pair_{pair_idx}'
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / 'C.npy', C)
    np.save(out_dir / 'U.npy', U)

    for h in horizons:
        if h in entropies_all_h:
            np.save(out_dir / f'H_h{h}.npy', entropies_all_h[h])
            np.save(out_dir / f'R_h{h}.npy', ranks_all_h[h])

    # Save metadata
    metadata = {
        'pair_idx': pair_idx,
        'species': species,
        'n_sources': len(pair.source_ids),
        'horizons_computed': list(entropies_all_h.keys()),
        'weights': weights,
        'C_stats': {
            'min': float(C.min()),
            'max': float(C.max()),
            'mean': float(C.mean()),
            'median': float(np.median(C)),
            'std': float(C.std())
        },
        'U_stats': {
            'min': float(U.min()),
            'max': float(U.max()),
            'mean': float(U.mean()),
            'median': float(np.median(U)),
            'std': float(U.std())
        }
    }

    with open(out_dir / 'commitment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Pair {pair_idx} complete: saved to {out_dir}")

    return pair_idx


def build_commitment_for_species(species: str, config: dict, data_root: Path, results_root: Path, cache_dir: Path):
    """
    Build commitment scores for all pairs in one species (PARALLELIZED).
    """
    print(f"\n[{species}] Starting...")

    # Count pairs from filesystem (DON'T load all data!)
    species_dir = data_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])
    num_pairs = len(pair_dirs)

    horizons = config['horizons']
    horizon_weights_type = config['horizon_weights']

    # PARALLELIZE across pairs
    from functools import partial
    compute_fn = partial(
        compute_commitment_for_pair,
        species=species,
        data_root=data_root,
        horizons=horizons,
        horizon_weights_type=horizon_weights_type,
        config=config,
        results_root=results_root,
        cache_dir=cache_dir
    )

    with ProcessPoolExecutor(max_workers=min(num_pairs, 8)) as executor:
        futures = {executor.submit(compute_fn, pair_idx): pair_idx
                   for pair_idx in range(num_pairs)}

        for future in as_completed(futures):
            pair_idx = future.result()

    print(f"[{species}] ✓ DONE\n")


def main():
    """Build commitment scores for all species (PARALLELIZED)."""
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')
    data_root = project_root / 'data_phase2A'
    results_root = project_root / 'results_p2b'
    cache_dir = project_root / 'cache_p2b'

    config_path = project_root / 'config_p2b_p3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Phase 2B: Build Commitment Scores (OPTIMIZED)")
    print(f"Horizons: {config['horizons']}, Weights: {config['horizon_weights']}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # PARALLELIZE across species
    from functools import partial
    compute_species_fn = partial(
        build_commitment_for_species,
        config=config,
        data_root=data_root,
        results_root=results_root,
        cache_dir=cache_dir
    )

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(compute_species_fn, species): species
                   for species in ['mouse', 'zebrafish']}

        for future in as_completed(futures):
            species = futures[future]
            future.result()

    print("✓ ALL COMPLETE\n")


if __name__ == '__main__':
    main()
