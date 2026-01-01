"""
Prepare data directory for Phase 2B/3 according to strict data contract.

Creates data_phase2A/<species>/pair_<k>/ with:
- T_hat_sparse.npz (from Phase 2A results)
- topM.json (pair-specific)
- source_ids.txt, target_ids.txt (ordered cell IDs)
- X_pca.npy, Y_pca.npy (features)
- T_sparse.npz, phi3_bits.npy, phi2.npy, lock_label.npy (Phase 1, optional)
"""

import json
import shutil
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz, save_npz
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

def prepare_pair(pair_dir: Path, species: str, project_root: Path, topM_species: dict) -> int:
    """Prepare data for one pair. Returns pair index."""

    pair_idx = int(pair_dir.name.split('_')[1])

    # Paths
    phase2a_results_dir = project_root / 'results' / species
    output_dir = project_root / 'data_phase2A' / species

    # Create output directory
    out_pair_dir = output_dir / f'pair_{pair_idx}'
    out_pair_dir.mkdir(parents=True, exist_ok=True)

    # 1. T_hat_sparse.npz (from Phase 2A results)
    T_hat_path = phase2a_results_dir / f'pair_{pair_idx}_That.npz'
    if not T_hat_path.exists():
        raise FileNotFoundError(f"Missing T_hat for {species} pair {pair_idx}: {T_hat_path}")

    T_hat = load_npz(T_hat_path)

    # Validate row-stochasticity
    row_sums = np.array(T_hat.sum(axis=1)).flatten()
    bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-5)[0]

    if len(bad_rows) > 0:
        worst_idx = bad_rows[np.argmax(np.abs(row_sums[bad_rows] - 1.0))]
        worst_sum = row_sums[worst_idx]
        raise ValueError(
            f"Pair {pair_idx}: T_hat violates row-stochasticity. "
            f"{len(bad_rows)}/{len(row_sums)} rows have |sum-1| > 1e-5. "
            f"Worst: row {worst_idx} sum={worst_sum:.10f}"
        )

    save_npz(out_pair_dir / 'T_hat_sparse.npz', T_hat, compressed=True)

    # 2. topM.json (pair-specific)
    topM = topM_species[str(pair_idx)]

    # Validate topM is a scalar integer
    if not isinstance(topM, (int, np.integer)):
        raise TypeError(
            f"Pair {pair_idx}: topM must be a scalar integer, got {type(topM).__name__}: {topM}"
        )

    topM = int(topM)  # Ensure Python int, not numpy int

    with open(out_pair_dir / 'topM.json', 'w') as f:
        json.dump(topM, f)

    # 3. X_pca.npy, Y_pca.npy (features)
    X_pca_src = pair_dir / 'X_pca.npy'
    Y_pca_src = pair_dir / 'Y_pca.npy'

    if not X_pca_src.exists():
        raise FileNotFoundError(f"Missing X_pca: {X_pca_src}")
    if not Y_pca_src.exists():
        raise FileNotFoundError(f"Missing Y_pca: {Y_pca_src}")

    shutil.copy(X_pca_src, out_pair_dir / 'X_pca.npy')
    shutil.copy(Y_pca_src, out_pair_dir / 'Y_pca.npy')

    X_pca = np.load(X_pca_src)
    Y_pca = np.load(Y_pca_src)

    # 4. source_ids.txt, target_ids.txt (use timepoint-based IDs for alignment)
    # Define canonical ordering
    n_source, n_target = T_hat.shape

    if X_pca.shape[0] != n_source:
        raise ValueError(f"X_pca rows ({X_pca.shape[0]}) != T_hat sources ({n_source})")
    if Y_pca.shape[0] != n_target:
        raise ValueError(f"Y_pca rows ({Y_pca.shape[0]}) != T_hat targets ({n_target})")

    # Load timepoint information
    time_id_path = pair_dir / 'time_id.json'
    if time_id_path.exists():
        with open(time_id_path, 'r') as f:
            time_info = json.load(f)
        source_time = time_info['source_time']
        target_time = time_info['target_time']
    else:
        # Fallback if time_id.json not available
        source_time = f"t{pair_idx}"
        target_time = f"t{pair_idx+1}"

    # Use timepoint-based IDs so targets of pair k match sources of pair k+1
    # Format: species_time_cell{idx}
    source_ids = [f"{species}_{source_time}_cell{i}" for i in range(n_source)]
    target_ids = [f"{species}_{target_time}_cell{j}" for j in range(n_target)]

    with open(out_pair_dir / 'source_ids.txt', 'w') as f:
        f.write('\n'.join(source_ids))

    with open(out_pair_dir / 'target_ids.txt', 'w') as f:
        f.write('\n'.join(target_ids))

    # 5. Phase 1 data (optional, for evaluation only)
    for phase1_file in ['T_sparse.npz', 'phi3_bits.npy', 'phi2.npy', 'lock_label.npy']:
        src = pair_dir / phase1_file
        if src.exists():
            shutil.copy(src, out_pair_dir / phase1_file)

    return pair_idx


def prepare_species_data(species: str, project_root: Path):
    """Prepare data for one species according to Phase 2B/3 contract (parallelized)."""

    print(f"\n{'='*60}")
    print(f"Preparing {species} data for Phase 2B/3 (parallelized)")
    print(f"{'='*60}")

    # Paths
    phase1_data_dir = project_root / 'data_phase1' / species
    output_dir = project_root / 'data_phase2A' / species

    # Load TopM values
    topM_path = project_root / 'topM_per_pair.json'
    with open(topM_path, 'r') as f:
        topM_all = json.load(f)
    topM_species = topM_all[species]

    # Find all pairs
    pair_dirs = sorted([d for d in phase1_data_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    print(f"Found {len(pair_dirs)} pairs - processing in parallel...")

    # Parallelize across pairs
    prepare_fn = partial(prepare_pair, species=species, project_root=project_root, topM_species=topM_species)

    with ProcessPoolExecutor(max_workers=min(len(pair_dirs), 16)) as executor:
        futures = {executor.submit(prepare_fn, pair_dir): pair_dir for pair_dir in pair_dirs}

        for future in as_completed(futures):
            pair_idx = future.result()
            print(f"  ✓ Pair {pair_idx} complete")

    print(f"\n✓ {species} data preparation complete: {output_dir}")


def main():
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')

    # Parallelize across species too
    species_list = ['mouse', 'zebrafish']

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(prepare_species_data, species, project_root): species
                   for species in species_list}

        for future in as_completed(futures):
            species = futures[future]
            future.result()  # Wait for completion, raise any exceptions

    print(f"\n{'='*60}")
    print("Data preparation complete for all species")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
