"""
Phase 2A - Input Validation
Fail-fast validation of all required Phase-1 outputs.
Parallelized across 64 cores for maximum throughput.
"""
import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.sparse import load_npz, issparse


def validate_pair(args):
    """Validate all required files for a single pair. Designed for multiprocessing."""
    pair_dir, pair_name, species = args
    errors = []

    # Required files
    required = {
        'X_pca.npy': None,
        'Y_pca.npy': None,
        'T_sparse.npz': None,
        'phi3_bits.npy': None,
        'phi2.npy': None,
        'lock_label.npy': None,
        'time_id.json': None
    }

    # Check existence
    for fname in required.keys():
        fpath = pair_dir / fname
        if not fpath.exists():
            errors.append(f"Missing {fname}")
        else:
            required[fname] = fpath

    if errors:
        return errors

    # Load and validate shapes/dtypes
    try:
        X = np.load(required['X_pca.npy'])
        Y = np.load(required['Y_pca.npy'])
        T = load_npz(required['T_sparse.npz'])
        phi3 = np.load(required['phi3_bits.npy'])
        phi2 = np.load(required['phi2.npy'])
        lock = np.load(required['lock_label.npy'])

        with open(required['time_id.json'], 'r') as f:
            time_ids = json.load(f)

        # Shape validation
        n_source = X.shape[0]
        n_target = Y.shape[0]
        pca_dim = X.shape[1]

        if X.ndim != 2:
            errors.append(f"X_pca must be 2D, got {X.ndim}D")
        if Y.ndim != 2:
            errors.append(f"Y_pca must be 2D, got {Y.ndim}D")
        if X.shape[1] != Y.shape[1]:
            errors.append(f"X and Y PCA dims mismatch: {X.shape[1]} vs {Y.shape[1]}")

        if not issparse(T):
            errors.append("T_sparse must be scipy sparse matrix")
        if T.shape != (n_source, n_target):
            errors.append(f"T shape {T.shape} != (n_source={n_source}, n_target={n_target})")

        if phi3.shape != (n_source,):
            errors.append(f"phi3_bits shape {phi3.shape} != ({n_source},)")
        if phi2.shape != (n_source,):
            errors.append(f"phi2 shape {phi2.shape} != ({n_source},)")
        if lock.shape != (n_source,):
            errors.append(f"lock_label shape {lock.shape} != ({n_source},)")

        # Dtype validation
        if not np.issubdtype(X.dtype, np.floating):
            errors.append(f"X_pca must be float, got {X.dtype}")
        if not np.issubdtype(Y.dtype, np.floating):
            errors.append(f"Y_pca must be float, got {Y.dtype}")
        if not np.issubdtype(phi3.dtype, np.floating):
            errors.append(f"phi3_bits must be float, got {phi3.dtype}")
        if not np.issubdtype(phi2.dtype, np.floating):
            errors.append(f"phi2 must be float, got {phi2.dtype}")
        if lock.dtype != bool:
            errors.append(f"lock_label must be bool, got {lock.dtype}")

        # Value validation
        if not np.all(np.isfinite(X)):
            errors.append("X_pca contains non-finite values")
        if not np.all(np.isfinite(Y)):
            errors.append("Y_pca contains non-finite values")
        if not np.all(np.isfinite(phi3)):
            errors.append("phi3_bits contains non-finite values")
        if not np.all(np.isfinite(phi2)):
            errors.append("phi2 contains non-finite values")

        # T normalization check (valid for UOT - allows mass variation)
        # UOT permits row sums != 1 (growth/death), so we only check for reasonable distribution
        row_sums = np.array(T.sum(axis=1)).flatten()
        # Check that median mass is reasonable (most cells should propagate)
        if np.median(row_sums) < 0.5:
            errors.append(f"T median row sum suspiciously low: {np.median(row_sums):.6f}")
        # Check that mean mass is reasonable
        if row_sums.mean() < 0.3:
            errors.append(f"T mean row sum suspiciously low: {row_sums.mean():.6f}")

        # T non-negativity
        if T.data.min() < -1e-10:
            errors.append(f"T contains negative values: min={T.data.min()}")

        # time_id validation
        if not isinstance(time_ids, dict):
            errors.append("time_id.json must contain a dict")
        if 'source_time' not in time_ids or 'target_time' not in time_ids:
            errors.append("time_id.json must contain 'source_time' and 'target_time'")

        # Summary info (not an error, just logging)
        if not errors:
            summary = (f"{species}/{pair_name}: {n_source} → {n_target}, PCA={pca_dim}, "
                      f"T_nnz={T.nnz}, locked={lock.sum()}/{n_source} ({100*lock.mean():.1f}%)")
        else:
            summary = None

    except Exception as e:
        errors.append(f"Exception during validation: {e}")
        summary = None

    return (species, pair_name, errors, summary)


def collect_validation_tasks(data_dir: Path):
    """Collect all validation tasks from both species."""
    tasks = []

    for species in ['mouse', 'zebrafish']:
        species_dir = data_dir / species
        if not species_dir.exists():
            continue

        pair_dirs = sorted([d for d in species_dir.iterdir()
                           if d.is_dir() and d.name.startswith('pair_')])

        for pair_dir in pair_dirs:
            tasks.append((pair_dir, pair_dir.name, species))

    return tasks


def main():
    """Main validation entry point with parallel execution."""
    # Assume script is in project/phase2/src/
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / 'data_phase1'

    print("=" * 80)
    print("Phase 2A - Input Validation (Parallelized)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"CPU cores available: {cpu_count()}")

    if not data_dir.exists():
        print(f"\nFATAL ERROR: data_phase1 directory not found at {data_dir}")
        print("Expected structure:")
        print("  project/phase2/data_phase1/")
        print("    mouse/")
        print("      pair_0/")
        print("        X_pca.npy")
        print("        Y_pca.npy")
        print("        ...")
        print("    zebrafish/")
        print("      pair_0/")
        print("        ...")
        sys.exit(1)

    # Collect all validation tasks
    tasks = collect_validation_tasks(data_dir)

    if len(tasks) == 0:
        print("\nERROR: No validation tasks found")
        sys.exit(1)

    print(f"Found {len(tasks)} pairs to validate")
    print(f"Validating in parallel using {min(cpu_count(), len(tasks))} workers...")
    print()

    # Parallel validation using all available cores
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(validate_pair, tasks)

    # Process results
    all_valid = True
    mouse_count = 0
    zebrafish_count = 0

    # Group by species for organized output
    mouse_results = []
    zebrafish_results = []

    for species, pair_name, errors, summary in results:
        if species == 'mouse':
            mouse_results.append((pair_name, errors, summary))
            mouse_count += 1
        else:
            zebrafish_results.append((pair_name, errors, summary))
            zebrafish_count += 1

        if errors:
            all_valid = False

    # Print results
    print(f"Mouse ({mouse_count} pairs):")
    for pair_name, errors, summary in sorted(mouse_results):
        if errors:
            print(f"  ✗ {pair_name}:")
            for err in errors:
                print(f"      - {err}")
        else:
            print(f"  ✓ {summary}")

    print(f"\nZebrafish ({zebrafish_count} pairs):")
    for pair_name, errors, summary in sorted(zebrafish_results):
        if errors:
            print(f"  ✗ {pair_name}:")
            for err in errors:
                print(f"      - {err}")
        else:
            print(f"  ✓ {summary}")

    print("\n" + "=" * 80)
    if all_valid:
        print(f"✓ All {len(tasks)} validation checks passed")
        print("=" * 80)
        sys.exit(0)
    else:
        print("✗ Validation FAILED - see errors above")
        print("=" * 80)
        sys.exit(1)


if __name__ == '__main__':
    main()
