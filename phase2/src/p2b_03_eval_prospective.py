"""
Phase 2B: Prospective Evaluation

4 biologically interpretable benchmarks (NO arbitrary thresholds):

2B-A: Developmental monotonicity (C increases over time)
2B-B: Entropy drop rate (high C → faster collapse)
2B-C: Lock label consistency (diagnostic)
2B-D: Φ₃ concordance (universality)

All use permutation nulls, bootstrap CIs, and effect sizes.

Outputs:
- results/<species>/p2b_eval.json
"""

import json
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from common_math import (
    perm_test_spearman,
    bootstrap_ci_auroc,
    compute_auroc
)


def benchmark_2b_a_monotonicity(species, results_root, config):
    """
    Benchmark 2B-A: Developmental monotonicity.

    Tests if commitment score C increases over developmental time (pair index).

    Returns:
        dict with observed_rho, p_value, n_pairs
    """
    print(f"\n  [2B-A] Developmental monotonicity...")

    species_dir = results_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    pair_indices = []
    median_C_values = []

    for pair_dir in pair_dirs:
        pair_idx = int(pair_dir.name.split('_')[1])
        C = np.load(pair_dir / 'C.npy')

        pair_indices.append(pair_idx)
        median_C_values.append(np.median(C))

    pair_indices = np.array(pair_indices)
    median_C_values = np.array(median_C_values)

    # Spearman correlation + permutation test
    n_perm = config['num_permutations']
    seed = config['seed']

    obs_rho, p_value, null_dist = perm_test_spearman(
        pair_indices, median_C_values, n_perm, seed
    )

    print(f"    Observed rho: {obs_rho:.4f}, p-value: {p_value:.4e}")

    # Save full null distribution for supplementary materials
    null_dist_path = species_dir / 'null_dist_2B_A_monotonicity.npy'
    np.save(null_dist_path, null_dist)
    print(f"    Saved null distribution: {null_dist_path}")

    return {
        'observed_rho': float(obs_rho),
        'p_value': float(p_value),
        'n_pairs': len(pair_indices),
        'median_C_per_pair': {int(k): float(v) for k, v in zip(pair_indices, median_C_values)},
        'null_distribution_mean': float(null_dist.mean()),
        'null_distribution_std': float(null_dist.std())
    }


def benchmark_2b_b_entropy_drop(species, results_root, config):
    """
    Benchmark 2B-B: Entropy drop rate.

    Tests if committed cells (high C) have faster entropy collapse:
    ΔH = H^(2) - H^(1), expect Spearman(C, -ΔH) > 0

    Returns:
        dict with observed_rho, p_value, n_cells (or None if insufficient data)
    """
    print(f"\n  [2B-B] Entropy drop rate...")

    species_dir = results_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    C_all = []
    delta_H_all = []

    for pair_dir in pair_dirs:
        # Load C
        C_path = pair_dir / 'C.npy'
        if not C_path.exists():
            continue
        C = np.load(C_path)

        # Load H_h1 and H_h2
        H_h1_path = pair_dir / 'H_h1.npy'
        H_h2_path = pair_dir / 'H_h2.npy'

        if not H_h1_path.exists() or not H_h2_path.exists():
            # Skip pairs without h=2 (e.g., last pairs)
            continue

        H_h1 = np.load(H_h1_path)
        H_h2 = np.load(H_h2_path)

        # Compute ΔH = H^(2) - H^(1)
        delta_H = H_h2 - H_h1

        C_all.append(C)
        delta_H_all.append(delta_H)

    if len(C_all) == 0:
        print(f"    No pairs with H_h2 available, skipping")
        return None

    C_all = np.concatenate(C_all)
    delta_H_all = np.concatenate(delta_H_all)

    # Spearman: C vs -ΔH (negative because we expect drop)
    n_perm = config['num_permutations']
    seed = config['seed']

    obs_rho, p_value, null_dist = perm_test_spearman(
        C_all, -delta_H_all, n_perm, seed
    )

    print(f"    Observed rho: {obs_rho:.4f}, p-value: {p_value:.4e}, n_cells: {len(C_all)}")

    # Save full null distribution for supplementary materials
    null_dist_path = species_dir / 'null_dist_2B_B_entropy_drop.npy'
    np.save(null_dist_path, null_dist)
    print(f"    Saved null distribution: {null_dist_path}")

    return {
        'observed_rho': float(obs_rho),
        'p_value': float(p_value),
        'n_cells': len(C_all),
        'null_distribution_mean': float(null_dist.mean()),
        'null_distribution_std': float(null_dist.std())
    }


def benchmark_2b_c_lock_consistency(species, data_root, results_root, config):
    """
    Benchmark 2B-C: Lock label consistency (diagnostic).

    Tests if C discriminates Phase 1 "locked" cells via AUROC.

    Returns:
        dict with auroc, ci_lower, ci_upper, p_value (or None if no lock labels)
    """
    print(f"\n  [2B-C] Lock label consistency...")

    species_dir_data = data_root / species
    species_dir_results = results_root / species

    C_all = []
    lock_all = []

    pair_dirs = sorted([d for d in species_dir_results.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    for pair_dir in pair_dirs:
        pair_idx = int(pair_dir.name.split('_')[1])

        # Load C
        C_path = pair_dir / 'C.npy'
        if not C_path.exists():
            continue
        C = np.load(C_path)

        # Load lock_label from data_phase2A
        lock_path = data_root / species / f'pair_{pair_idx}' / 'lock_label.npy'
        if not lock_path.exists():
            continue

        lock = np.load(lock_path)

        # Validate dimensions
        if len(C) != len(lock):
            continue

        C_all.append(C)
        lock_all.append(lock)

    if len(C_all) == 0:
        print(f"    No lock labels available, skipping")
        return None

    C_all = np.concatenate(C_all)
    lock_all = np.concatenate(lock_all)

    # Check if we have both classes
    unique_labels = np.unique(lock_all)
    if len(unique_labels) < 2:
        print(f"    Lock labels are constant ({unique_labels}), skipping")
        return None

    # AUROC + bootstrap CI
    n_boot = config['num_bootstrap']
    seed = config['seed']

    auroc, ci_lower, ci_upper = bootstrap_ci_auroc(C_all, lock_all, n_boot, seed)

    # Null: shuffle labels
    rng = np.random.RandomState(seed)
    null_aurocs = []
    for _ in range(n_boot):
        lock_shuffled = rng.permutation(lock_all)
        null_auc = compute_auroc(C_all, lock_shuffled)
        null_aurocs.append(null_auc)

    null_aurocs = np.array(null_aurocs)
    p_value = np.mean(null_aurocs >= auroc)  # One-sided: observed better than null

    print(f"    AUROC: {auroc:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}], p-value: {p_value:.4e}")

    # Save full null distribution for supplementary materials
    null_dist_path = species_dir_results / 'null_dist_2B_C_lock_auroc.npy'
    np.save(null_dist_path, null_aurocs)
    print(f"    Saved null distribution: {null_dist_path}")

    return {
        'auroc': float(auroc),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p_value),
        'n_cells': len(C_all),
        'n_locked': int(lock_all.sum()),
        'frac_locked': float(lock_all.mean())
    }


def benchmark_2b_d_phi3_concordance(species, data_root, results_root, config):
    """
    Benchmark 2B-D: Φ₃ concordance (universality).

    Tests if Phase 1 Φ₃ (integrated info) aligns with 1-C.

    Returns:
        dict with observed_rho, p_value (or None if no Φ₃ data)
    """
    print(f"\n  [2B-D] Φ₃ concordance...")

    species_dir_data = data_root / species
    species_dir_results = results_root / species

    C_all = []
    phi3_all = []

    pair_dirs = sorted([d for d in species_dir_results.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    for pair_dir in pair_dirs:
        pair_idx = int(pair_dir.name.split('_')[1])

        # Load C
        C_path = pair_dir / 'C.npy'
        if not C_path.exists():
            continue
        C = np.load(C_path)

        # Load phi3_bits from data_phase2A
        phi3_path = data_root / species / f'pair_{pair_idx}' / 'phi3_bits.npy'
        if not phi3_path.exists():
            continue

        phi3 = np.load(phi3_path)

        # Validate dimensions
        if len(C) != len(phi3):
            continue

        C_all.append(C)
        phi3_all.append(phi3)

    if len(C_all) == 0:
        print(f"    No Φ₃ data available, skipping")
        return None

    C_all = np.concatenate(C_all)
    phi3_all = np.concatenate(phi3_all)

    # Spearman: rank(1-C) vs rank(Φ₃)
    n_perm = config['num_permutations']
    seed = config['seed']

    obs_rho, p_value, null_dist = perm_test_spearman(
        1 - C_all, phi3_all, n_perm, seed
    )

    print(f"    Observed rho: {obs_rho:.4f}, p-value: {p_value:.4e}, n_cells: {len(C_all)}")

    # Save full null distribution for supplementary materials
    null_dist_path = species_dir_results / 'null_dist_2B_D_phi3_concordance.npy'
    np.save(null_dist_path, null_dist)
    print(f"    Saved null distribution: {null_dist_path}")

    return {
        'observed_rho': float(obs_rho),
        'p_value': float(p_value),
        'n_cells': len(C_all),
        'null_distribution_mean': float(null_dist.mean()),
        'null_distribution_std': float(null_dist.std())
    }


def eval_species(species, config, data_root, results_root):
    """
    Run all prospective evaluations for one species.

    Args:
        species: 'mouse' or 'zebrafish'
        config: Configuration dict
        data_root: Path to data_phase2A
        results_root: Path to results_p2b
    """
    print(f"\n{'='*60}")
    print(f"Prospective Evaluation: {species}")
    print(f"{'='*60}")

    results = {}

    # Benchmark 2B-A: Monotonicity (always runs)
    results['2B-A_monotonicity'] = benchmark_2b_a_monotonicity(
        species, results_root, config
    )

    # Benchmark 2B-B: Entropy drop (always runs, skips pairs without h=2)
    results['2B-B_entropy_drop'] = benchmark_2b_b_entropy_drop(
        species, results_root, config
    )

    # Benchmark 2B-C: Lock consistency (optional, may skip)
    result_c = benchmark_2b_c_lock_consistency(
        species, data_root, results_root, config
    )
    if result_c is not None:
        results['2B-C_lock_consistency'] = result_c
    else:
        results['2B-C_lock_consistency'] = {'status': 'skipped', 'reason': 'no lock labels'}

    # Benchmark 2B-D: Φ₃ concordance (optional, may skip)
    result_d = benchmark_2b_d_phi3_concordance(
        species, data_root, results_root, config
    )
    if result_d is not None:
        results['2B-D_phi3_concordance'] = result_d
    else:
        results['2B-D_phi3_concordance'] = {'status': 'skipped', 'reason': 'no phi3 data'}

    # Save results
    output_path = results_root / species / 'p2b_eval.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {output_path}")
    print(f"\n[{species}] ✓ DONE\n")


def main():
    """Run prospective evaluations for all species."""
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')
    data_root = project_root / 'data_phase2A'
    results_root = project_root / 'results_p2b'

    config_path = project_root / 'config_p2b_p3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Phase 2B: Prospective Evaluation")
    print(f"Permutations: {config['num_permutations']}")
    print(f"Bootstraps: {config['num_bootstrap']}")

    # Evaluate each species
    for species in ['mouse', 'zebrafish']:
        eval_species(species, config, data_root, results_root)

    print("✓ ALL SPECIES COMPLETE\n")


if __name__ == '__main__':
    main()
