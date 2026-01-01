#!/usr/bin/env python3
"""
04_phi.py

Compute Φ₁, Φ₂, Φ₃ from transition kernels T (Section VII)

These are the three discovered constraint laws:
- Φ₃ (Reachability): entropy of T(x,·) → collapses at commitment
- Φ₂ (Stability): log trace ratio of covariances → stabilizes after collapse
- Φ₁ (Propagation compatibility): diversity difference → spikes at lineage segregation

All formulas are FIXED by the plan.
"""

import numpy as np
import scanpy as sc
import pandas as pd
import yaml
from pathlib import Path
import pickle
from scipy.stats import entropy as scipy_entropy
import warnings

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_phi3_reachability(T):
    """
    Φ₃ (Reachability) - Section VII (VECTORIZED)

    Φ₃(x) = -Σ_j T_{ij} log T_{ij}

    This is the Shannon entropy of the row distribution T(x,·).

    Low entropy = near-deterministic = collapsed future = commitment.

    Returns:
        phi3: array of entropy values per source cell
    """
    # Convert to dense if sparse
    if hasattr(T, 'toarray'):
        T_dense = T.toarray()
    else:
        T_dense = T

    # Vectorized entropy: H = -Σ p log(p)
    # Handle zeros by adding small epsilon
    T_safe = T_dense + 1e-12

    # Compute -T * log2(T) element-wise, then sum over rows
    phi3 = -np.sum(T_safe * np.log2(T_safe), axis=1)

    return phi3

def compute_phi2_stability(adata_t, adata_tp1, T, pca_key='X_pca', k=30):
    """
    Φ₂ (Stability) - Section VII (GPU-OPTIMIZED)

    Φ₂(x) = -log(trace(Σ_{t+Δt}) / trace(Σ_t))

    where:
    - Σ_t: covariance of k-NN of x in PCA space at time t
    - Σ_{t+Δt}: pushforward covariance via T(x,·)

    Positive Φ₂ = compression = stabilization.
    Negative Φ₂ = expansion = amplification.

    Returns:
        phi2: array of stability values per source cell
    """
    try:
        import cupy as cp
        USE_GPU = True
    except ImportError:
        USE_GPU = False
        cp = np  # Use already-imported numpy

    X_t = adata_t.obsm[pca_key]
    X_tp1 = adata_tp1.obsm[pca_key]

    n_source = X_t.shape[0]
    n_dims = X_t.shape[1]

    from sklearn.neighbors import NearestNeighbors

    # Build kNN for source
    nbrs_t = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_t)
    _, indices_t = nbrs_t.kneighbors(X_t)

    # Compute all local covariances at once (CPU - small operation)
    trace_t = np.zeros(n_source)
    for i in range(n_source):
        neighbors_t = indices_t[i, 1:]
        X_local = X_t[neighbors_t, :]
        cov_t = np.cov(X_local, rowvar=False)
        trace_t[i] = np.trace(cov_t)

    # Convert T to dense if sparse
    if hasattr(T, 'toarray'):
        T_dense = T.toarray()
    else:
        T_dense = T

    if USE_GPU:
        # Pushforward covariance computation
        cp.get_default_memory_pool().free_all_blocks()

        # Move to GPU with float32 for memory efficiency
        T_gpu = cp.asarray(T_dense, dtype=cp.float32)
        X_tp1_gpu = cp.asarray(X_tp1, dtype=cp.float32)

        # Weighted means: (n_source, n_dims)
        means_tp1_gpu = T_gpu @ X_tp1_gpu

        # Compute traces using vectorized operations
        # trace(Cov) = Σ_j T_ij ||x_j - μ||²

        trace_tp1 = cp.zeros(n_source, dtype=cp.float32)

        # Process in chunks to avoid OOM
        chunk_size = 500
        for start in range(0, n_source, chunk_size):
            end = min(start + chunk_size, n_source)

            # Broadcast: (chunk_size, 1, n_dims) - (1, n_target, n_dims) = (chunk_size, n_target, n_dims)
            means_chunk = means_tp1_gpu[start:end, cp.newaxis, :]
            X_centered = X_tp1_gpu[cp.newaxis, :, :] - means_chunk

            # Squared norms: (chunk_size, n_target)
            sq_norms = cp.sum(X_centered ** 2, axis=2)

            # Weighted sum: (chunk_size,)
            T_chunk = T_gpu[start:end]
            trace_tp1[start:end] = cp.sum(T_chunk * sq_norms, axis=1)

        # Move back to CPU
        trace_tp1 = cp.asnumpy(trace_tp1)

        # Cleanup GPU memory
        del T_gpu, X_tp1_gpu, means_tp1_gpu
        cp.get_default_memory_pool().free_all_blocks()

    else:
        # CPU fallback (same logic but with numpy)
        means_tp1 = T_dense @ X_tp1
        trace_tp1 = np.zeros(n_source)

        chunk_size = 500
        for start in range(0, n_source, chunk_size):
            end = min(start + chunk_size, n_source)

            means_chunk = means_tp1[start:end, np.newaxis, :]
            X_centered = X_tp1[np.newaxis, :, :] - means_chunk
            sq_norms = np.sum(X_centered ** 2, axis=2)

            T_chunk = T_dense[start:end]
            trace_tp1[start:end] = np.sum(T_chunk * sq_norms, axis=1)

    # Φ₂
    ratio = trace_tp1 / (trace_t + 1e-12)
    phi2 = -np.log(ratio + 1e-12)

    return phi2

def compute_phi1_propagation_compatibility(T, k=30):
    """
    Φ₁ (Propagation Compatibility) - Section VII (VECTORIZED)

    Φ₁(x) = H(p_global) - H(p_S)

    where:
    - p_global = Σ_i π_i T(x_i, ·) (global reachable distribution)
    - p_S = Σ_{i ∈ S(x)} π_i T(x_i, ·) (local neighborhood reachable distribution)
    - S(x) = k-NN of x in source space

    High Φ₁ = local neighborhood has less diverse futures than global
           = crowding / competitive exclusion.

    Returns:
        phi1: array of propagation compatibility values per source cell
    """
    n_source, n_target = T.shape

    # Uniform source weights
    pi = np.ones(n_source) / n_source

    # Convert T to dense
    if hasattr(T, 'toarray'):
        T_dense = T.toarray()
    else:
        T_dense = T

    # Global reachable distribution
    p_global = pi @ T_dense

    # Compute local neighborhoods (k-NN in coupling space - use T as features)
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(T_dense)
    _, indices = nbrs.kneighbors(T_dense)

    # Build neighborhood weight matrix (n_source × n_source)
    # N[i, j] = 1/k if j in kNN(i), else 0
    from scipy.sparse import lil_matrix
    N = lil_matrix((n_source, n_source))

    for i in range(n_source):
        neighbors = indices[i, 1:]  # Exclude self
        N[i, neighbors] = 1.0 / k

    N = N.tocsr()  # Convert to efficient format

    # Vectorized computation of all local reachable distributions
    # P_local[i, :] = local distribution for cell i
    P_local = N @ T_dense  # (n_source × n_target)

    # Vectorized entropy computation
    P_local_safe = P_local + 1e-12
    H_local = -np.sum(P_local_safe * np.log2(P_local_safe), axis=1)

    # Global entropy
    p_global_safe = p_global + 1e-12
    H_global = -np.sum(p_global_safe * np.log2(p_global_safe))

    # Φ₁ = H_global - H_local (broadcasted)
    phi1 = H_global - H_local

    return phi1

def process_pair(pair_file, config):
    """
    Load a timepoint pair and compute Φ₁, Φ₂, Φ₃

    Returns:
        dict with phi values and metadata
    """
    print(f"\nProcessing {pair_file.name}...")

    with open(pair_file, 'rb') as f:
        pair_data = pickle.load(f)

    T = pair_data['T']
    t_source = pair_data['t_source']
    t_target = pair_data['t_target']

    print(f"  {t_source} → {t_target}")
    print(f"  T shape: {T.shape}")

    # Create minimal AnnData objects for Φ₂ computation
    # (we need PCA coordinates, which should be in adata_t_obs/adata_tp1_obs)

    # Check if PCA is stored
    pca_key = config['preprocessing']['pca_key']

    # Reconstruct from stored obs (if available) or load from preprocessed file
    # For now, assume we need to reload the preprocessed data

    # This is a design choice: either store X_pca in pair files, or reload here
    # To keep pair files small, we'll reload

    project_root = Path(__file__).parent.parent
    data_proc_dir = project_root / "data_proc"

    # Determine dataset
    if 'E6' in str(t_source) or 'E7' in str(t_source) or 'E8' in str(t_source):
        dataset = 'mouse'
        file_name = 'mouse_preprocessed.h5ad'
        timepoint_col = config['datasets']['mouse']['timepoint_column']
    elif 'hpf' in str(t_source):
        dataset = 'zebrafish'
        file_name = 'zebrafish_preprocessed.h5ad'
        timepoint_col = config['datasets']['zebrafish']['timepoint_column']
    else:
        raise ValueError(f"Cannot determine dataset from timepoint {t_source}")

    adata_full = sc.read_h5ad(data_proc_dir / file_name)

    # Slice to get source and target
    adata_t = adata_full[adata_full.obs[timepoint_col] == t_source].copy()
    adata_tp1 = adata_full[adata_full.obs[timepoint_col] == t_target].copy()

    # Compute Φ
    k = config['phi']['k_neighbors']

    print("  Computing Φ₃ (reachability)...")
    phi3 = compute_phi3_reachability(T)

    print("  Computing Φ₂ (stability)...")
    phi2 = compute_phi2_stability(adata_t, adata_tp1, T, pca_key=pca_key, k=k)

    print("  Computing Φ₁ (propagation compatibility)...")
    phi1 = compute_phi1_propagation_compatibility(T, k=k)

    print(f"  Φ₃ range: [{phi3.min():.3f}, {phi3.max():.3f}]")
    print(f"  Φ₂ range: [{phi2.min():.3f}, {phi2.max():.3f}]")
    print(f"  Φ₁ range: [{phi1.min():.3f}, {phi1.max():.3f}]")

    result = {
        't_source': t_source,
        't_target': t_target,
        'phi1': phi1,
        'phi2': phi2,
        'phi3': phi3,
        'source_cells': pair_data['source_cells'],
        'target_cells': pair_data['target_cells']
    }

    return result

def process_and_save_pair(pair_file, config):
    """Wrapper to process and save a single pair (for parallel execution)"""
    # Check if already completed
    phi_file = pair_file.parent / pair_file.name.replace('pair_', 'phi_')
    if phi_file.exists():
        print(f"⊙ Skipping {pair_file.name} (already computed)")
        return None

    result = process_pair(pair_file, config)

    # Save individual result
    with open(phi_file, 'wb') as f:
        pickle.dump(result, f)

    return result

def main():
    """
    Main Φ computation pipeline (PARALLELIZED)
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    config = load_config()
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    # Get all pair files
    mouse_pairs_dir = results_dir / "mouse_pairs"
    mouse_pair_files = sorted(mouse_pairs_dir.glob("pair_*.pkl"))

    zfish_pairs_dir = results_dir / "zebrafish_pairs"
    zfish_pair_files = sorted(zfish_pairs_dir.glob("pair_*.pkl"))

    all_pair_files = list(mouse_pair_files) + list(zfish_pair_files)

    print("\n" + "="*80)
    print(f"PARALLEL Φ COMPUTATION: {len(all_pair_files)} pairs")
    print("="*80)
    print(f"Mouse: {len(mouse_pair_files)} pairs")
    print(f"Zebrafish: {len(zfish_pair_files)} pairs")
    print(f"Using {multiprocessing.cpu_count()} CPU cores")
    print("="*80)

    # Process ALL pairs in parallel (maximize hardware utilization)
    n_workers = min(14, multiprocessing.cpu_count())  # 14 pairs max

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        futures = {executor.submit(process_and_save_pair, pf, config): pf for pf in all_pair_files}

        # Collect results
        completed = 0
        for future in futures:
            pair_file = futures[future]
            try:
                result = future.result()
                completed += 1
                print(f"✓ [{completed}/{len(all_pair_files)}] Completed: {pair_file.name}")
            except Exception as e:
                print(f"✗ FAILED: {pair_file.name} - {e}")

    print("\n" + "="*80)
    print("Φ COMPUTATION COMPLETE")
    print("="*80)
    print(f"Total: {completed}/{len(all_pair_files)} pairs")
    print("\nNext step: 05_locking.py")

if __name__ == "__main__":
    main()
