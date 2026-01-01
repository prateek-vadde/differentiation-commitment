#!/usr/bin/env python3
"""
03_uot.py

Unbalanced Optimal Transport (Section VI)

ARCHITECTURE:
- Processes ADJACENT TIMEPOINT PAIRS sequentially
- Two-pass deterministic scheme per pair:
  * Pass 1: UOT with g_poe_pre → T^(1)
  * Derive Δρ from T^(1)
  * Pass 2: UOT with g_poe_full → T^(2) [final]

NO TUNING. All hyperparameters are FIXED or scale-normalized.

Uses Python Optimal Transport (POT) library for UOT solver.
"""

import numpy as np
import scanpy as sc
import pandas as pd
import yaml
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
import os
import gc

# GPU detection
USE_GPU = False
try:
    import cupy as cp
    import cupyx
    USE_GPU = True
    print("✓ GPU (CuPy) available - will use GPU acceleration")
except ImportError:
    print("⚠ CuPy not available - using CPU (slower)")
    cp = np  # Fallback to numpy

# Import expert computation functions
import importlib.util
spec = importlib.util.spec_from_file_location("experts", Path(__file__).parent / "02_experts.py")
experts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experts)

load_config = experts.load_config
compute_poe_pre = experts.compute_poe_pre
compute_poe_full = experts.compute_poe_full

def solve_uot(adata_t, adata_tp1, g_poe, config):
    """
    Solve Unbalanced Optimal Transport (Section VI)

    minimize_P: <P, C> + ε·H(P) + λ_s·KL(P1||μ̂) + λ_t·KL(P^T1||ν)

    where:
    - C: cost matrix (Euclidean squared in PCA)
    - μ̂ = μ · g_poe (biased source mass)
    - ν: uniform target mass
    - ε, λ_s, λ_t: FIXED hyperparameters

    Returns:
        P: coupling matrix (n_t × n_tp1)
        T: transition kernel (row-normalized P)
    """
    pca_key = config['preprocessing']['pca_key']
    uot_config = config['uot']

    X_t = adata_t.obsm[pca_key]
    X_tp1 = adata_tp1.obsm[pca_key]

    n_t = X_t.shape[0]
    n_tp1 = X_tp1.shape[0]

    print(f"\n[UOT Solver]")
    print(f"  Source: {n_t} cells")
    print(f"  Target: {n_tp1} cells")

    # Clear GPU cache before computation
    if USE_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        mempool = cp.get_default_memory_pool()
        print(f"  GPU memory cleared: {mempool.used_bytes() / 1e9:.2f} GB used")

    # VI.1: Cost matrix C (Euclidean squared)
    print("  Computing cost matrix...")

    if USE_GPU:
        # GPU computation with float32 for memory efficiency
        X_t_gpu = cp.asarray(X_t, dtype=cp.float32)
        X_tp1_gpu = cp.asarray(X_tp1, dtype=cp.float32)

        # Compute squared Euclidean distance on GPU
        C = cp.sum(X_t_gpu**2, axis=1)[:, None] + cp.sum(X_tp1_gpu**2, axis=1)[None, :] - 2 * X_t_gpu @ X_tp1_gpu.T
        C = cp.maximum(C, 0)  # Numerical stability

        # Free intermediate arrays
        del X_t_gpu, X_tp1_gpu
        cp.get_default_memory_pool().free_all_blocks()

        median_cost = float(cp.median(C))
        print(f"  Cost range (GPU): [{float(C.min()):.2f}, {float(C.max()):.2f}], median: {median_cost:.2f}")
        print(f"  Cost matrix size: {C.nbytes / 1e9:.2f} GB")
    else:
        # CPU fallback
        from scipy.spatial.distance import cdist
        C = cdist(X_t, X_tp1, metric='sqeuclidean')
        median_cost = np.median(C)
        print(f"  Cost range (CPU): [{C.min():.2f}, {C.max():.2f}], median: {median_cost:.2f}")

    # VI.2: Source mass (biased by PoE)
    mu_biased = g_poe / g_poe.sum()  # Normalize to sum=1

    # Target mass (uniform)
    nu_np = np.ones(n_tp1) / n_tp1

    if USE_GPU:
        mu_biased = cp.asarray(mu_biased, dtype=cp.float32)
        nu = cp.asarray(nu_np, dtype=cp.float32)
    else:
        nu = nu_np

    print(f"  μ̂ (biased source): sum={float(mu_biased.sum()) if USE_GPU else mu_biased.sum():.6f}")
    print(f"  ν (target): sum={float(nu.sum()) if USE_GPU else nu.sum():.6f}")

    # UOT hyperparameters (scale-normalized)
    epsilon = uot_config['epsilon_scale'] * median_cost
    lambda_s = uot_config['lambda_s']
    lambda_t = uot_config['lambda_t']

    print(f"\n  Hyperparameters (FIXED):")
    print(f"    ε = {uot_config['epsilon_scale']} × median(C) = {epsilon:.3f}")
    print(f"    λ_s = {lambda_s}")
    print(f"    λ_t = {lambda_t}")
    print(f"    Device: {'GPU (CuPy)' if USE_GPU else 'CPU'}")

    # VI.4: Solve UOT using POT
    print("\n  Solving UOT (Sinkhorn)...")

    try:
        import ot

        # POT automatically uses GPU if arrays are CuPy
        P = ot.unbalanced.sinkhorn_knopp_unbalanced(
            mu_biased, nu, C,
            reg=epsilon,
            reg_m=(lambda_s, lambda_t),  # (source, target) KL divergence weights
            numItermax=uot_config['max_iter'],
            stopThr=uot_config['tol'],
            verbose=False
        )

        print(f"  ✓ UOT converged")

    except Exception as e:
        raise RuntimeError(f"UOT solver failed: {e}")

    # VI.5: Transition kernel T (row-normalize P)
    print("\n  Computing transition kernel T...")

    xp = cp if USE_GPU else np
    row_sums = xp.sum(P, axis=1, keepdims=True)
    T = P / (row_sums + 1e-12)  # Avoid division by zero

    # Verification
    row_sum_check = xp.sum(T, axis=1)
    max_row_sum_error = float(xp.max(xp.abs(row_sum_check - 1.0)))

    print(f"  T shape: {T.shape}")
    print(f"  Row sum error: max |Σ_j T_ij - 1| = {max_row_sum_error:.2e}")

    if max_row_sum_error > 1e-6:
        warnings.warn(f"Row normalization error {max_row_sum_error:.2e} > 1e-6")

    # Effective support (sparsity check)
    effective_support = 1.0 / xp.sum(T**2, axis=1)
    median_support = float(xp.median(effective_support))

    print(f"  Effective support (median): {median_support:.1f} targets per source cell")

    if median_support > n_tp1 * 0.5:
        warnings.warn(
            f"T is very diffuse (median support = {median_support:.1f} / {n_tp1}). "
            "Consider decreasing ε or increasing locality."
        )

    # Convert back to CPU for saving/compatibility
    if USE_GPU:
        P = cp.asnumpy(P)
        T = cp.asnumpy(T)

        # Free all GPU memory
        del C, mu_biased, nu
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  GPU memory freed")

    return P, T

def process_timepoint_pair(adata, timepoint_col, t_curr, t_next, config, output_dir):
    """
    Process one timepoint pair through the two-pass UOT scheme.

    Returns:
        dict with T, P, and metadata
    """
    print("\n" + "="*80)
    print(f"TIMEPOINT PAIR: {t_curr} → {t_next}")
    print("="*80)

    # Slice data
    adata_t = adata[adata.obs[timepoint_col] == t_curr].copy()
    adata_tp1 = adata[adata.obs[timepoint_col] == t_next].copy()

    print(f"Source ({t_curr}): {adata_t.n_obs} cells")
    print(f"Target ({t_next}): {adata_tp1.n_obs} cells")

    # PASS 1: Pre-OT
    print("\n" + "-"*80)
    print("PASS 1: Pre-OT (ρ_local only)")
    print("-"*80)

    g_poe_pre = compute_poe_pre(adata_t, adata_tp1, config)

    P1, T1 = solve_uot(adata_t, adata_tp1, g_poe_pre, config)

    # PASS 2: Post-OT (with Δρ)
    print("\n" + "-"*80)
    print("PASS 2: Post-OT (ρ_local + Δρ from T^(1))")
    print("-"*80)

    g_poe_full = compute_poe_full(adata_t, adata_tp1, T1, config)

    P2, T2 = solve_uot(adata_t, adata_tp1, g_poe_full, config)

    # Final T
    T_final = T2
    P_final = P2

    print("\n" + "-"*80)
    print("TWO-PASS COMPLETE")
    print("-"*80)
    print(f"Final T: {T_final.shape}")

    # Save outputs for this pair
    pair_name = f"{t_curr}_to_{t_next}".replace(".", "_").replace(" ", "_")

    # Save T as sparse matrix
    from scipy.sparse import csr_matrix
    import pickle

    T_sparse = csr_matrix(T_final)

    pair_output = {
        'T': T_sparse,
        'P': csr_matrix(P_final),
        't_source': t_curr,
        't_target': t_next,
        'n_source': adata_t.n_obs,
        'n_target': adata_tp1.n_obs,
        'source_cells': adata_t.obs_names.tolist(),
        'target_cells': adata_tp1.obs_names.tolist(),
        'adata_t_obs': adata_t.obs.copy(),  # Contains all expert values
        'adata_tp1_obs': adata_tp1.obs.copy()
    }

    output_file = output_dir / f"pair_{pair_name}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(pair_output, f)

    print(f"\nSaved: {output_file}")

    # Cleanup memory
    del adata_t, adata_tp1, T_final, P_final, T_sparse, pair_output
    gc.collect()
    if USE_GPU:
        cp.get_default_memory_pool().free_all_blocks()

    return None  # Don't return large objects

def main():
    """
    Main UOT pipeline: pairwise iteration over adjacent timepoints
    """
    config = load_config()

    project_root = Path(__file__).parent.parent
    data_proc_dir = project_root / "data_proc"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Process mouse
    print("\n" + "="*80)
    print("MOUSE: PAIRWISE UOT")
    print("="*80)

    mouse_file = data_proc_dir / "mouse_preprocessed.h5ad"
    if not mouse_file.exists():
        raise FileNotFoundError(f"{mouse_file} not found. Run 01_preprocess.py first.")

    mouse_adata = sc.read_h5ad(mouse_file)
    mouse_timepoint_col = config['datasets']['mouse']['timepoint_column']

    timepoints_mouse = sorted(mouse_adata.obs[mouse_timepoint_col].unique())
    print(f"\nTimepoints: {timepoints_mouse}")

    mouse_output_dir = results_dir / "mouse_pairs"
    mouse_output_dir.mkdir(exist_ok=True)

    n_mouse_pairs = len(timepoints_mouse) - 1
    for i in range(n_mouse_pairs):
        t_curr = timepoints_mouse[i]
        t_next = timepoints_mouse[i + 1]

        process_timepoint_pair(
            mouse_adata, mouse_timepoint_col,
            t_curr, t_next,
            config, mouse_output_dir
        )

    print(f"\n✓ Mouse: processed {n_mouse_pairs} pairs")

    # Process zebrafish
    print("\n" + "="*80)
    print("ZEBRAFISH: PAIRWISE UOT")
    print("="*80)

    zfish_file = data_proc_dir / "zebrafish_preprocessed.h5ad"
    if not zfish_file.exists():
        raise FileNotFoundError(f"{zfish_file} not found. Run 01_preprocess.py first.")

    zfish_adata = sc.read_h5ad(zfish_file)
    zfish_timepoint_col = config['datasets']['zebrafish']['timepoint_column']

    timepoints_zfish = sorted(zfish_adata.obs[zfish_timepoint_col].unique())
    print(f"\nTimepoints: {timepoints_zfish}")

    zfish_output_dir = results_dir / "zebrafish_pairs"
    zfish_output_dir.mkdir(exist_ok=True)

    n_zfish_pairs = len(timepoints_zfish) - 1
    for i in range(n_zfish_pairs):
        t_curr = timepoints_zfish[i]
        t_next = timepoints_zfish[i + 1]

        process_timepoint_pair(
            zfish_adata, zfish_timepoint_col,
            t_curr, t_next,
            config, zfish_output_dir
        )

    print(f"\n✓ Zebrafish: processed {n_zfish_pairs} pairs")

    print("\n" + "="*80)
    print("UOT COMPLETE")
    print("="*80)
    print(f"Mouse pairs: {n_mouse_pairs} saved in {mouse_output_dir}")
    print(f"Zebrafish pairs: {n_zfish_pairs} saved in {zfish_output_dir}")
    print("\nNext step: 04_phi.py")

if __name__ == "__main__":
    main()
