#!/usr/bin/env python3
"""
05_locking.py

Identify "Locking Surfaces" (Section VIII) - PARALLELIZED

A locking surface is where:
- Φ₃ < 25th percentile (low reachability = near-deterministic)
- Φ₂ > 0 (stable, not chaotic)
- Steep Φ₃ gradient (sharp transition)
- Persists across ≥2 timepoints

This answers the key question: where do small perturbations fail to reopen futures?
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_phi3_gradient(phi3, X_pca, k=30):
    """
    Compute local gradient of Φ₃ in PCA space (VECTORIZED)

    ∇Φ₃(x) ≈ mean difference to k-NN

    Returns:
        gradient_magnitude: array of gradient magnitudes
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_pca)
    _, indices = nbrs.kneighbors(X_pca)

    # Vectorized computation using advanced indexing
    # indices[:, 1:] has shape (n, k) - neighbors excluding self
    # phi3[indices[:, 1:]] broadcasts to (n, k)
    neighbors_phi3 = phi3[indices[:, 1:]]  # (n, k)
    phi3_expanded = phi3[:, np.newaxis]  # (n, 1)

    # Compute mean absolute difference for all cells at once
    gradient_mag = np.mean(np.abs(phi3_expanded - neighbors_phi3), axis=1)

    return gradient_mag

def identify_locking_regions(phi_result, X_pca, config):
    """
    Identify near-deterministic locking regions

    Criteria (Section VIII):
    - Φ₃ < 25th percentile
    - Φ₂ > 0
    - High Φ₃ gradient

    Returns:
        is_locking: boolean array
        locking_score: continuous score
    """
    phi1 = phi_result['phi1']
    phi2 = phi_result['phi2']
    phi3 = phi_result['phi3']

    locking_config = config['locking']

    # Criterion 1: Low Φ₃ (collapsed futures)
    phi3_threshold = np.percentile(phi3, locking_config['phi3_percentile'])
    low_phi3 = phi3 < phi3_threshold

    # Criterion 2: Positive Φ₂ (stable)
    stable = phi2 > locking_config['phi2_threshold']

    # Criterion 3: High Φ₃ gradient (steep transition)
    phi3_grad = compute_phi3_gradient(phi3, X_pca)
    phi3_grad_threshold = np.percentile(phi3_grad, 75)  # Top 25%
    steep_gradient = phi3_grad > phi3_grad_threshold

    # Combined locking criterion
    is_locking = low_phi3 & stable & steep_gradient

    # Continuous locking score (for visualization)
    # Normalize each component to [0, 1]
    phi3_norm = 1 - (phi3 - phi3.min()) / (phi3.max() - phi3.min() + 1e-12)
    phi2_norm = np.clip(phi2, 0, None)  # Already centered at 0
    phi2_norm = phi2_norm / (phi2_norm.max() + 1e-12)
    phi3_grad_norm = phi3_grad / (phi3_grad.max() + 1e-12)

    locking_score = phi3_norm * phi2_norm * phi3_grad_norm

    n_locking = is_locking.sum()
    pct_locking = 100 * n_locking / len(is_locking)

    print(f"  Locking cells: {n_locking} / {len(is_locking)} ({pct_locking:.1f}%)")
    print(f"  Φ₃ threshold (25th percentile): {phi3_threshold:.3f}")

    return is_locking, locking_score

def visualize_locking(dataset_name, phi_results, config, output_dir):
    """
    Create visualizations of locking surfaces across timepoints
    """
    print(f"\n[Visualization: {dataset_name}]")

    fig, axes = plt.subplots(3, len(phi_results), figsize=(5*len(phi_results), 12))

    if len(phi_results) == 1:
        axes = axes[:, None]

    for idx, phi_result in enumerate(phi_results):
        t_source = phi_result['t_source']
        t_target = phi_result['t_target']

        phi1 = phi_result['phi1']
        phi2 = phi_result['phi2']
        phi3 = phi_result['phi3']
        is_locking = phi_result['is_locking']

        # Plot Φ₃ trajectory
        axes[0, idx].hist(phi3, bins=50, alpha=0.7, edgecolor='black')
        axes[0, idx].axvline(np.percentile(phi3, 25), color='red', linestyle='--',
                            label='25th percentile')
        axes[0, idx].set_xlabel('Φ₃ (Reachability)', fontsize=10)
        axes[0, idx].set_ylabel('Cells', fontsize=10)
        axes[0, idx].set_title(f'{t_source} → {t_target}\nΦ₃ distribution', fontsize=11)
        axes[0, idx].legend(fontsize=8)

        # Plot Φ₂ trajectory
        axes[1, idx].hist(phi2, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1, idx].axvline(0, color='red', linestyle='--', label='Φ₂ = 0')
        axes[1, idx].set_xlabel('Φ₂ (Stability)', fontsize=10)
        axes[1, idx].set_ylabel('Cells', fontsize=10)
        axes[1, idx].set_title(f'Φ₂ distribution', fontsize=11)
        axes[1, idx].legend(fontsize=8)

        # Plot locking regions
        axes[2, idx].scatter(phi3, phi2, c=is_locking, cmap='RdYlBu_r',
                           s=5, alpha=0.6)
        axes[2, idx].axvline(np.percentile(phi3, 25), color='gray', linestyle='--', alpha=0.5)
        axes[2, idx].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[2, idx].set_xlabel('Φ₃ (Reachability)', fontsize=10)
        axes[2, idx].set_ylabel('Φ₂ (Stability)', fontsize=10)
        axes[2, idx].set_title(f'Locking regions\n({is_locking.sum()} cells)', fontsize=11)

    plt.tight_layout()
    fig_path = output_dir / f'{dataset_name}_locking_analysis.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {fig_path}")

def analyze_locking_persistence(phi_results, dataset_name):
    """
    Check if locking regions persist across multiple timepoints

    Section VIII criterion: persist across ≥2 timepoints
    """
    print(f"\n[Locking Persistence: {dataset_name}]")

    # This requires tracking cells across timepoints via OT
    # For now, report per-timepoint statistics

    for idx, phi_result in enumerate(phi_results):
        t_source = phi_result['t_source']
        t_target = phi_result['t_target']
        is_locking = phi_result['is_locking']
        n_locking = is_locking.sum()
        pct = 100 * n_locking / len(is_locking)

        print(f"  {t_source} → {t_target}: {n_locking} locking cells ({pct:.1f}%)")

    # TODO: Track cells forward via T to check persistence
    # This requires loading T matrices and following trajectories

def process_locking_pair(phi_file, X_pca, config):
    """
    Worker function: Compute locking for one pair (PARALLELIZABLE)

    Args:
        phi_file: Path to phi pickle file
        X_pca: PCA coordinates for source timepoint
        config: Configuration dict

    Returns:
        phi_result with locking annotations
    """
    with open(phi_file, 'rb') as f:
        phi_result = pickle.load(f)

    # Identify locking regions
    is_locking, locking_score = identify_locking_regions(phi_result, X_pca, config)

    # Store
    phi_result['is_locking'] = is_locking
    phi_result['locking_score'] = locking_score

    # Save updated result
    with open(phi_file, 'wb') as f:
        pickle.dump(phi_result, f)

    return phi_result

def main():
    """
    Main locking surface identification pipeline (OPTIMIZED)
    """
    config = load_config()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    data_proc_dir = project_root / "data_proc"

    import scanpy as sc
    pca_key = config['preprocessing']['pca_key']

    # Process mouse
    print("\n" + "="*80)
    print("MOUSE: Identifying Locking Surfaces")
    print("="*80)

    mouse_pairs_dir = results_dir / "mouse_pairs"
    mouse_phi_files = sorted(mouse_pairs_dir.glob("phi_*.pkl"))

    # Load h5ad and build timepoint -> PCA cache
    print("  Loading mouse data and building PCA cache...")
    mouse_adata = sc.read_h5ad(data_proc_dir / "mouse_preprocessed.h5ad")
    mouse_timepoint_col = config['datasets']['mouse']['timepoint_column']

    # Build cache: {timepoint: X_pca array}
    mouse_pca_cache = {}
    for tp in mouse_adata.obs[mouse_timepoint_col].unique():
        mask = mouse_adata.obs[mouse_timepoint_col] == tp
        mouse_pca_cache[tp] = mouse_adata.obsm[pca_key][mask]

    del mouse_adata  # Free memory
    print(f"  Cached PCA for {len(mouse_pca_cache)} timepoints")

    # PARALLEL: Prepare (phi_file, X_pca, config) tuples
    print("  Preparing parallel tasks...")
    tasks = []
    for phi_file in mouse_phi_files:
        # Pre-load to get t_source
        with open(phi_file, 'rb') as f:
            phi_result = pickle.load(f)
        t_source = phi_result['t_source']
        X_pca = mouse_pca_cache[t_source]
        tasks.append((phi_file, X_pca, config))

    # PARALLEL EXECUTION using all cores
    n_workers = min(len(tasks), multiprocessing.cpu_count())
    print(f"  Processing {len(tasks)} pairs in parallel using {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_locking_pair, *task) for task in tasks]
        mouse_phi_results = [f.result() for f in futures]

    # Visualize
    visualize_locking('mouse', mouse_phi_results, config, results_dir)

    # Analyze persistence
    analyze_locking_persistence(mouse_phi_results, 'mouse')

    # Process zebrafish
    print("\n" + "="*80)
    print("ZEBRAFISH: Identifying Locking Surfaces")
    print("="*80)

    zfish_pairs_dir = results_dir / "zebrafish_pairs"
    zfish_phi_files = sorted(zfish_pairs_dir.glob("phi_*.pkl"))

    # Load h5ad and build timepoint -> PCA cache
    print("  Loading zebrafish data and building PCA cache...")
    zfish_adata = sc.read_h5ad(data_proc_dir / "zebrafish_preprocessed.h5ad")
    zfish_timepoint_col = config['datasets']['zebrafish']['timepoint_column']

    # Build cache: {timepoint: X_pca array}
    zfish_pca_cache = {}
    for tp in zfish_adata.obs[zfish_timepoint_col].unique():
        mask = zfish_adata.obs[zfish_timepoint_col] == tp
        zfish_pca_cache[tp] = zfish_adata.obsm[pca_key][mask]

    del zfish_adata  # Free memory
    print(f"  Cached PCA for {len(zfish_pca_cache)} timepoints")

    # PARALLEL: Prepare (phi_file, X_pca, config) tuples
    print("  Preparing parallel tasks...")
    tasks = []
    for phi_file in zfish_phi_files:
        # Pre-load to get t_source
        with open(phi_file, 'rb') as f:
            phi_result = pickle.load(f)
        t_source = phi_result['t_source']
        X_pca = zfish_pca_cache[t_source]
        tasks.append((phi_file, X_pca, config))

    # PARALLEL EXECUTION using all cores
    n_workers = min(len(tasks), multiprocessing.cpu_count())
    print(f"  Processing {len(tasks)} pairs in parallel using {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_locking_pair, *task) for task in tasks]
        zfish_phi_results = [f.result() for f in futures]

    # Visualize
    visualize_locking('zebrafish', zfish_phi_results, config, results_dir)

    # Analyze persistence
    analyze_locking_persistence(zfish_phi_results, 'zebrafish')

    print("\n" + "="*80)
    print("LOCKING SURFACE IDENTIFICATION COMPLETE")
    print("="*80)

    # Final summary
    print("\nSUMMARY:")
    print(f"  Mouse: {len(mouse_phi_results)} timepoint pairs analyzed")
    print(f"  Zebrafish: {len(zfish_phi_results)} timepoint pairs analyzed")
    print("\nVisualizations saved in results/")
    print("\nNext: Review locking_analysis.png plots to assess constraint laws")

if __name__ == "__main__":
    main()
