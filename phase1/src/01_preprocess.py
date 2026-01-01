#!/usr/bin/env python3
"""
01_preprocess.py

Preprocessing pipeline (III. REPRESENTATION - NO FLEXIBILITY)

Fixed preprocessing steps:
- Filter cells: min_genes=500, min_cells_per_gene=10
- Normalize: library size to 10,000, log1p
- Feature selection: exactly 3,000 HVGs
- PCA: exactly 50 dimensions
- kNN graph: k=30

No tuning. No latent models. No flexibility.
"""

import numpy as np
import scanpy as sc
import pandas as pd
import yaml
from pathlib import Path
import sys

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_mouse_data(data_raw_dir):
    """
    Load mouse gastrulation atlas data (Pijuan-Sala et al.)

    Data structure:
    - raw_counts.mtx: sparse gene × cell matrix (MatrixMarket format)
    - genes.tsv: gene names
    - barcodes.tsv: cell barcodes
    - meta.tab: cell metadata with 'stage' column (E6.5-E8.5)
    """
    print("Loading mouse gastrulation data...")

    atlas_dir = data_raw_dir / "atlas"

    if not atlas_dir.exists():
        raise FileNotFoundError(
            f"{atlas_dir} not found. "
            f"Run: cd {data_raw_dir} && tar -xzf mouse_atlas_data.tar.gz"
        )

    # Load count matrix (gene × cell, need to transpose)
    from scipy.io import mmread
    counts = mmread(atlas_dir / "raw_counts.mtx").T.tocsr()  # Now cell × gene

    # Load gene names
    genes = pd.read_csv(atlas_dir / "genes.tsv", header=None, names=['gene_id'])

    # Load cell barcodes
    barcodes = pd.read_csv(atlas_dir / "barcodes.tsv", header=None, names=['barcode'])

    # Load metadata
    meta = pd.read_csv(atlas_dir / "meta.tab", sep='\t')

    print(f"  Raw matrix: {counts.shape[0]} cells × {counts.shape[1]} genes")
    print(f"  Metadata: {meta.shape[0]} rows")

    # Verify dimensions match
    assert counts.shape[0] == len(barcodes), f"Cells mismatch: {counts.shape[0]} vs {len(barcodes)}"
    assert counts.shape[0] == len(meta), f"Cells mismatch: {counts.shape[0]} vs {len(meta)}"
    assert counts.shape[1] == len(genes), f"Genes mismatch: {counts.shape[1]} vs {len(genes)}"

    # Create AnnData object
    adata = sc.AnnData(
        X=counts,
        obs=meta,
        var=genes
    )
    adata.var_names = genes['gene_id'].values
    adata.var_names_make_unique()

    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  Timepoints in data: {sorted(adata.obs['stage'].unique())}")

    return adata

def load_zebrafish_data(data_raw_dir):
    """Load zebrafish Farrell/URD dataset (h5ad format)"""
    print("Loading zebrafish embryogenesis data...")
    h5ad_path = data_raw_dir / "zebrafish_farrell_urd.h5ad"
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded from {h5ad_path}")
    return adata

def preprocess_dataset(adata, config, dataset_name, timepoint_col):
    """
    Apply fixed preprocessing pipeline (Section III).

    CRITICAL: All parameters are FIXED by the plan.
    NO TUNING ALLOWED.
    """
    prep_config = config['preprocessing']

    print(f"\n{'='*60}")
    print(f"Preprocessing {dataset_name}")
    print(f"{'='*60}")

    # Initial stats
    print(f"Initial: {adata.n_obs} cells × {adata.n_vars} genes")

    # Check for timepoint column
    if timepoint_col not in adata.obs.columns:
        raise ValueError(
            f"Timepoint column '{timepoint_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Filter excluded timepoints if specified
    dataset_config = config['datasets'][dataset_name]
    if 'exclude_timepoints' in dataset_config:
        exclude = dataset_config['exclude_timepoints']
        n_before = adata.n_obs
        adata = adata[~adata.obs[timepoint_col].isin(exclude)]
        print(f"Excluded timepoints {exclude}: {n_before} → {adata.n_obs} cells")

    # Count timepoints
    timepoints = sorted(adata.obs[timepoint_col].unique())
    print(f"Timepoints: {len(timepoints)} - {timepoints}")

    # QC Filter
    print("\n[QC Filter]")
    print(f"  min_genes = {prep_config['min_genes']}")
    print(f"  min_cells_per_gene = {prep_config['min_cells_per_gene']}")

    sc.pp.filter_cells(adata, min_genes=prep_config['min_genes'])
    sc.pp.filter_genes(adata, min_cells=prep_config['min_cells_per_gene'])

    print(f"  After QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # Check min cells per timepoint requirement
    cells_per_timepoint = adata.obs[timepoint_col].value_counts()
    min_cells = cells_per_timepoint.min()
    min_timepoints = config['datasets'][dataset_name]['min_timepoints']
    min_cells_required = config['datasets'][dataset_name]['min_cells_per_timepoint']

    print(f"\nCells per timepoint:")
    for tp in sorted(timepoints):
        if tp in cells_per_timepoint.index:
            print(f"  {tp}: {cells_per_timepoint[tp]} cells")

    if len(timepoints) < min_timepoints:
        raise ValueError(
            f"STOP: Dataset has {len(timepoints)} timepoints, "
            f"but requires ≥{min_timepoints}"
        )

    if min_cells < min_cells_required:
        print(f"\nWARNING: Some timepoints have <{min_cells_required} cells.")
        print(f"  Minimum: {min_cells} cells")
        print("  Proceeding, but results may be affected.")

    # Normalization
    print(f"\n[Normalization]")
    print(f"  Target counts: {prep_config['normalize_to']}")
    print(f"  Log transform: {prep_config['log_transform']}")

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    # Normalize
    sc.pp.normalize_total(adata, target_sum=prep_config['normalize_to'])

    if prep_config['log_transform']:
        sc.pp.log1p(adata)

    # Feature Selection
    print(f"\n[Feature Selection]")
    print(f"  HVGs: exactly {prep_config['n_hvgs']}")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=prep_config['n_hvgs'],
        flavor='seurat_v3',
        layer='counts'  # Use raw counts for HVG selection
    )

    n_hvg = adata.var['highly_variable'].sum()
    print(f"  Selected: {n_hvg} HVGs")

    # Subset to HVGs for downstream
    adata_hvg = adata[:, adata.var['highly_variable']].copy()

    # PCA
    print(f"\n[PCA]")
    print(f"  n_components: {prep_config['n_pca']}")

    sc.pp.scale(adata_hvg, max_value=10)  # Standard scaling
    sc.tl.pca(adata_hvg, n_comps=prep_config['n_pca'], svd_solver='arpack')

    # Store PCA in original adata
    adata.obsm[prep_config['pca_key']] = adata_hvg.obsm['X_pca']
    adata.varm['PCs'] = np.zeros((adata.n_vars, prep_config['n_pca']))
    adata.varm['PCs'][adata.var['highly_variable'], :] = adata_hvg.varm['PCs']
    adata.uns['pca'] = adata_hvg.uns['pca']

    variance_ratio = adata.uns['pca']['variance_ratio']
    print(f"  Variance explained (first 5 PCs): {variance_ratio[:5]}")
    print(f"  Total variance (50 PCs): {variance_ratio.sum():.3f}")

    # kNN Graph
    print(f"\n[kNN Graph]")
    print(f"  k = {prep_config['k_neighbors']}")

    sc.pp.neighbors(
        adata,
        n_neighbors=prep_config['k_neighbors'],
        use_rep=prep_config['pca_key'],
        metric='euclidean'
    )

    print(f"  Graph constructed: {adata.n_obs} nodes")

    # Sanity checks (Section III fallbacks)
    print(f"\n[Sanity Checks]")

    # Check: no single timepoint dominates PC1/PC2
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Simple visual check: plot PC1 vs PC2 colored by timepoint
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for tp in sorted(timepoints):
        mask = adata.obs[timepoint_col] == tp
        pca_coords = adata.obsm[prep_config['pca_key']][mask]
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                  label=str(tp), alpha=0.5, s=1)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(markerscale=5, title='Timepoint')
    ax.set_title(f'{dataset_name}: PC1 vs PC2')

    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    fig_path = output_dir / f"{dataset_name}_pc_sanity_check.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Sanity check plot saved: {fig_path}")
    print(f"  Visual inspection: ensure timepoints mix in PC space")
    print(f"  (pure separation by timepoint = batch effect)")

    print(f"\n{'='*60}")
    print(f"Preprocessing complete: {dataset_name}")
    print(f"Final: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"PCA: {adata.obsm[prep_config['pca_key']].shape}")
    print(f"{'='*60}\n")

    return adata

def main():
    """Main preprocessing pipeline"""

    # Load config
    config = load_config()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_raw_dir = project_root / "data_raw"
    data_proc_dir = project_root / "data_proc"
    data_proc_dir.mkdir(exist_ok=True)

    # Process mouse dataset
    print("\n" + "="*60)
    print("MOUSE GASTRULATION DATASET")
    print("="*60)

    mouse_adata = load_mouse_data(data_raw_dir)
    mouse_timepoint_col = config['datasets']['mouse']['timepoint_column']

    mouse_adata = preprocess_dataset(
        mouse_adata,
        config,
        dataset_name='mouse',
        timepoint_col=mouse_timepoint_col
    )

    # Save processed mouse data
    mouse_output = data_proc_dir / "mouse_preprocessed.h5ad"
    mouse_adata.write_h5ad(mouse_output)
    print(f"Saved: {mouse_output}")

    # Process zebrafish dataset
    print("\n" + "="*60)
    print("ZEBRAFISH EMBRYOGENESIS DATASET")
    print("="*60)

    zfish_adata = load_zebrafish_data(data_raw_dir)
    zfish_timepoint_col = config['datasets']['zebrafish']['timepoint_column']

    zfish_adata = preprocess_dataset(
        zfish_adata,
        config,
        dataset_name='zebrafish',
        timepoint_col=zfish_timepoint_col
    )

    # Save processed zebrafish data
    zfish_output = data_proc_dir / "zebrafish_preprocessed.h5ad"
    zfish_adata.write_h5ad(zfish_output)
    print(f"Saved: {zfish_output}")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Mouse: {mouse_output}")
    print(f"Zebrafish: {zfish_output}")
    print("\nNext step: 02_experts.py")

if __name__ == "__main__":
    main()
