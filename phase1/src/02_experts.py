#!/usr/bin/env python3
"""
02_experts.py

Product-of-Experts (PoE) Growth/Death Prior (Section V)

CORRECTED ARCHITECTURE:
- Computes experts PER TIMEPOINT PAIR (t_k, t_{k+1})
- V.4 crowding has two stages:
  * Pre-OT: g_crowding_pre using ρ_local only
  * Post-OT: g_crowding_full using ρ_local + Δρ (derived from T)
- NO CIRCULARITY: Δρ is computed AFTER first OT pass

This implements the two-pass deterministic scheme:
  Pass 1: PoE with ρ_local only → OT → T^(1)
  Derive: Δρ(x) = Σ_j T_{ij} ρ_{t+Δt}(y_j) - ρ_t(x)
  Pass 2: PoE with ρ_local + Δρ → OT → T^(2) [final]
"""

import numpy as np
import scanpy as sc
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import rankdata
import warnings

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_global_growth_expert(adata_t, adata_tp1):
    """
    V.1 Global Growth Expert (FIXED)

    g_global(t) = N_{t+Δt} / N_t

    Applied uniformly to all cells at time t.
    """
    n_t = adata_t.n_obs
    n_tp1 = adata_tp1.n_obs

    growth_ratio = n_tp1 / n_t

    print(f"  N_t = {n_t}, N_t+1 = {n_tp1}")
    print(f"  g_global = {growth_ratio:.3f}")

    return np.full(n_t, growth_ratio)

def compute_cell_cycle_expert(adata):
    """
    V.2 Cell-Cycle Expert (FIXED)

    g_cycle(x) = exp(z_S+G2M(x))

    Uses canonical S-phase and G2M gene lists from Tirosh et al. 2016.
    """
    # Canonical cell cycle genes (Tirosh et al. 2016, mouse homologs)
    # S phase genes
    s_genes = ['Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2', 'Mcm6',
               'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'Mlf1ip', 'Hells', 'Rfc2', 'Rpa2', 'Nasp',
               'Rad51ap1', 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7', 'Pold3', 'Msh2', 'Atad2',
               'Rad51', 'Rrm2', 'Cdc45', 'Cdc6', 'Exo1', 'Tipin', 'Dscc1', 'Blm', 'Casp8ap2',
               'Usp1', 'Clspn', 'Pola1', 'Chaf1b', 'Brip1', 'E2f8']

    # G2M phase genes
    g2m_genes = ['Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Birc5', 'Tpx2', 'Top2a', 'Ndc80', 'Cks2',
                 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'Fam64a', 'Smc4', 'Ccnb2',
                 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e', 'Tubb4b', 'Gtse1', 'Kif20b',
                 'Hjurp', 'Cdca3', 'Hn1', 'Cdc20', 'Ttk', 'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2',
                 'Dlgap5', 'Cdca2', 'Cdca8', 'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln',
                 'Lbr', 'Ckap5', 'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa']

    # Map to actual gene names in dataset
    # Gene names may be in format "ENSEMBL_ID\tGene_Name" or just "Gene_Name"
    def find_gene(gene_name):
        gene_upper = gene_name.upper()
        for var_name in adata.var_names:
            # Check if gene name contains a tab (Ensembl ID format)
            if '\t' in var_name:
                gene_symbol = var_name.split('\t')[1]
                if gene_symbol.upper() == gene_upper:
                    return var_name
            else:
                if var_name.upper() == gene_upper:
                    return var_name
        return None

    s_genes_matched = [g for g in [find_gene(gene) for gene in s_genes] if g is not None]
    g2m_genes_matched = [g for g in [find_gene(gene) for gene in g2m_genes] if g is not None]

    print(f"  Cell cycle genes: {len(s_genes_matched)} S-phase, {len(g2m_genes_matched)} G2M")

    if len(s_genes_matched) < 5 or len(g2m_genes_matched) < 5:
        warnings.warn(f"Too few cell cycle genes found (S: {len(s_genes_matched)}, G2M: {len(g2m_genes_matched)}). Using g_cycle = 1.0")
        return np.ones(adata.n_obs)

    # Score
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes_matched, g2m_genes=g2m_genes_matched)

    # Combined S+G2M score
    cycle_score = adata.obs['S_score'] + adata.obs['G2M_score']

    # Z-score
    z_cycle = (cycle_score - cycle_score.mean()) / (cycle_score.std() + 1e-10)

    # Exponentiate
    g_cycle = np.exp(z_cycle.values)

    print(f"  g_cycle range: [{g_cycle.min():.3f}, {g_cycle.max():.3f}]")

    return g_cycle

def compute_velocity_expert(adata):
    """
    V.3 Velocity Expert (DETERMINISTIC FALLBACK)

    If spliced/unspliced available: compute RNA velocity
    If not: g_velocity = 1.0 (neutral)
    """
    has_spliced = 'spliced' in adata.layers
    has_unspliced = 'unspliced' in adata.layers

    if not (has_spliced and has_unspliced):
        print("  No spliced/unspliced → g_velocity = 1.0")
        return np.ones(adata.n_obs)

    try:
        import scvelo as scv
        scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(adata)

        v_magnitude = np.linalg.norm(adata.layers['velocity'], axis=1)
        z_v = (v_magnitude - v_magnitude.mean()) / (v_magnitude.std() + 1e-10)
        g_velocity = np.exp(z_v)

        print(f"  g_velocity range: [{g_velocity.min():.3f}, {g_velocity.max():.3f}]")
        return g_velocity

    except Exception as e:
        warnings.warn(f"Velocity failed: {e}. Setting g_velocity = 1.0")
        return np.ones(adata.n_obs)

def compute_local_density(X_pca, k=30):
    """
    V.4.1: Compute local density ρ_local(x) = 1 / mean_kNN_distance

    Returns:
        rho_local: array of local densities
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X_pca)
    distances, _ = nbrs.kneighbors(X_pca)

    # distances[:, 0] is self, use [:, 1:] for neighbors
    mean_knn_dist = distances[:, 1:].mean(axis=1)
    rho_local = 1.0 / (mean_knn_dist + 1e-10)

    return rho_local

def compute_crowding_expert_pre(adata, pca_key='X_pca', k=30):
    """
    V.4 Crowding Expert - PRE-OT (only ρ_local)

    g_crowding_pre(x) = exp(-r_ρ)

    where r_ρ is percentile rank of ρ_local.

    NO Δρ yet (that requires OT coupling).
    """
    if pca_key not in adata.obsm:
        raise ValueError(f"{pca_key} not found. Run preprocessing first.")

    X_pca = adata.obsm[pca_key]

    # Compute local density
    rho_local = compute_local_density(X_pca, k=k)

    # Percentile rank normalization
    r_rho = (rankdata(rho_local) - 1) / (len(rho_local) - 1)

    # Pre-OT crowding expert (no Δρ term yet)
    g_crowding = np.exp(-r_rho)

    print(f"  ρ_local range: [{rho_local.min():.3e}, {rho_local.max():.3e}]")
    print(f"  g_crowding_pre range: [{g_crowding.min():.3f}, {g_crowding.max():.3f}]")

    # Store for later use
    adata.obs['rho_local'] = rho_local

    return g_crowding

def compute_delta_rho(adata_t, adata_tp1, T, pca_key='X_pca', k=30):
    """
    V.4.2 (CORRECTED): Compute Δρ using OT coupling T

    Δρ(x_i) = Σ_j T_{ij} ρ_{t+Δt}(y_j) - ρ_t(x_i)

    This is the expected future density minus current density.
    NON-CIRCULAR: requires T from first OT pass.

    Args:
        adata_t: source timepoint
        adata_tp1: target timepoint
        T: transition kernel (n_t × n_tp1), rows sum to 1

    Returns:
        delta_rho: array of density changes
    """
    # Compute densities at both timepoints
    if 'rho_local' not in adata_t.obs:
        rho_t = compute_local_density(adata_t.obsm[pca_key], k=k)
    else:
        rho_t = adata_t.obs['rho_local'].values

    if 'rho_local' not in adata_tp1.obs:
        rho_tp1 = compute_local_density(adata_tp1.obsm[pca_key], k=k)
        adata_tp1.obs['rho_local'] = rho_tp1
    else:
        rho_tp1 = adata_tp1.obs['rho_local'].values

    # Expected future density under T
    expected_rho_future = T @ rho_tp1  # matrix-vector product

    # Change
    delta_rho = expected_rho_future - rho_t

    print(f"  Δρ range: [{delta_rho.min():.3e}, {delta_rho.max():.3e}]")

    return delta_rho

def compute_crowding_expert_full(adata_t, adata_tp1, T, pca_key='X_pca', k=30):
    """
    V.4 Crowding Expert - POST-OT (ρ_local + Δρ)

    g_crowding_full(x) = exp(-r_ρ) · exp(-r_Δρ)

    where both r_ρ and r_Δρ are percentile ranks.

    Requires T from first OT pass.
    """
    # Get ρ_local (already computed in pre-OT)
    if 'rho_local' not in adata_t.obs:
        rho_local = compute_local_density(adata_t.obsm[pca_key], k=k)
        adata_t.obs['rho_local'] = rho_local
    else:
        rho_local = adata_t.obs['rho_local'].values

    # Compute Δρ using T
    delta_rho = compute_delta_rho(adata_t, adata_tp1, T, pca_key=pca_key, k=k)

    # Percentile ranks
    r_rho = (rankdata(rho_local) - 1) / (len(rho_local) - 1)
    r_delta_rho = (rankdata(delta_rho) - 1) / (len(delta_rho) - 1)

    # Full crowding expert
    g_crowding = np.exp(-r_rho) * np.exp(-r_delta_rho)

    print(f"  g_crowding_full range: [{g_crowding.min():.3f}, {g_crowding.max():.3f}]")

    # Store
    adata_t.obs['delta_rho'] = delta_rho

    return g_crowding

def compute_poe_pre(adata_t, adata_tp1, config):
    """
    Compute PRE-OT Product-of-Experts

    g^{pre}(x) = g_global · g_cycle · g_velocity · g_crowding_pre

    Normalized to median = 1.
    """
    print("\n[Pre-OT PoE]")

    expert_config = config['experts']
    k = config['preprocessing']['k_neighbors']
    pca_key = config['preprocessing']['pca_key']

    # Compute experts
    print("  Computing g_global...")
    g_global = compute_global_growth_expert(adata_t, adata_tp1)

    print("  Computing g_cycle...")
    g_cycle = compute_cell_cycle_expert(adata_t)

    print("  Computing g_velocity...")
    g_velocity = compute_velocity_expert(adata_t)

    print("  Computing g_crowding_pre...")
    g_crowding = compute_crowding_expert_pre(adata_t, pca_key=pca_key, k=k)

    # Multiply
    g_poe = g_global * g_cycle * g_velocity * g_crowding

    # Normalize to median = 1
    g_poe = g_poe / np.median(g_poe)

    print(f"  g_poe^(pre): median={np.median(g_poe):.3f}, range=[{g_poe.min():.3f}, {g_poe.max():.3f}]")

    # Store individual experts
    adata_t.obs['g_global'] = g_global
    adata_t.obs['g_cycle'] = g_cycle
    adata_t.obs['g_velocity'] = g_velocity
    adata_t.obs['g_crowding_pre'] = g_crowding
    adata_t.obs['g_poe_pre'] = g_poe

    return g_poe

def compute_poe_full(adata_t, adata_tp1, T, config):
    """
    Compute POST-OT Product-of-Experts

    g^{full}(x) = g_global · g_cycle · g_velocity · g_crowding_full

    Uses full crowding (ρ + Δρ) computed from T.
    Normalized to median = 1.
    """
    print("\n[Post-OT PoE]")

    k = config['preprocessing']['k_neighbors']
    pca_key = config['preprocessing']['pca_key']

    # Reuse pre-computed experts (they don't change)
    g_global = adata_t.obs['g_global'].values
    g_cycle = adata_t.obs['g_cycle'].values
    g_velocity = adata_t.obs['g_velocity'].values

    # Compute full crowding using T
    print("  Computing g_crowding_full (using T)...")
    g_crowding = compute_crowding_expert_full(adata_t, adata_tp1, T, pca_key=pca_key, k=k)

    # Multiply
    g_poe = g_global * g_cycle * g_velocity * g_crowding

    # Normalize to median = 1
    g_poe = g_poe / np.median(g_poe)

    print(f"  g_poe^(full): median={np.median(g_poe):.3f}, range=[{g_poe.min():.3f}, {g_poe.max():.3f}]")

    # Store
    adata_t.obs['g_crowding_full'] = g_crowding
    adata_t.obs['g_poe_full'] = g_poe

    return g_poe

# NOTE: This module now provides functions for use by 03_uot.py
# It does NOT run a main loop - the pairwise iteration happens in 03_uot.py
