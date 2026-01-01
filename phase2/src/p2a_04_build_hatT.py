"""
Phase 2A - Build Compressed Operator \hat{T}
GPU-accelerated TopM CSR sparse matrix construction
Optimized for GH200
"""
import torch
import numpy as np
from scipy.sparse import csr_matrix
import json
from pathlib import Path
from tqdm import tqdm

from p2a_02_model import Phase2Model


def build_hatT_for_pair(
    model: Phase2Model,
    pair,
    pair_idx: int,
    device: str,
    batch_size: int = 1024
) -> csr_matrix:
    """
    Build compressed transition operator \hat{T} for a single pair.

    Uses TopM softmax construction:
    1. Encode all source cells (with source time embedding)
    2. Encode all target cells (with target time embedding)
    3. Compute similarities: s_ij = e_i^T f_j / tau
    4. For each row, take TopM and apply softmax
    5. Return as CSR sparse matrix

    Args:
        model: Trained Phase2Model
        pair: TransitionPairData object
        pair_idx: Index of this pair
        topM: Number of top targets to keep per source
        device: 'cuda' or 'cpu'
        batch_size: Batch size for encoding

    Returns:
        T_hat: (n_source, n_target) CSR sparse matrix
    """
    model.eval()

    n_source = pair.n_source
    n_target = pair.n_target

    print(f"  Building \hat{{T}} for pair {pair_idx}: {n_source} → {n_target} (TopM={pair.topM})")

    # Clamp topM to actual number of targets
    effective_topM = min(pair.topM, n_target)

    # Encode all source cells in batches
    print(f"    Encoding {n_source} source cells...")
    source_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, n_source, batch_size), desc="    Source", leave=False):
            end_i = min(i + batch_size, n_source)
            X_batch = torch.from_numpy(pair.X[i:end_i]).to(device)
            e_batch = model.encode_source(X_batch, pair.source_time)
            source_embeds.append(e_batch.cpu())

    source_embeds = torch.cat(source_embeds, dim=0)  # (n_source, embed_dim)

    # Encode all target cells in batches
    print(f"    Encoding {n_target} target cells...")
    target_embeds = []
    with torch.no_grad():
        for j in tqdm(range(0, n_target, batch_size), desc="    Target", leave=False):
            end_j = min(j + batch_size, n_target)
            Y_batch = torch.from_numpy(pair.Y[j:end_j]).to(device)
            f_batch = model.encode_target(Y_batch, pair.target_time)
            target_embeds.append(f_batch.cpu())

    target_embeds = torch.cat(target_embeds, dim=0)  # (n_target, embed_dim)

    # Move to GPU for similarity computation
    if device == 'cuda' and torch.cuda.is_available():
        source_embeds = source_embeds.cuda()
        target_embeds = target_embeds.cuda()

    # Compute TopM per source
    print(f"    Computing TopM={effective_topM} similarities...")

    # CSR components (preallocate on GPU)
    row_indices_list = []
    col_indices_list = []
    data_values_list = []

    # Aggressive batching for GH200 (96GB VRAM)
    sim_batch_size = 4096 if device == 'cuda' else 128

    with torch.no_grad():
        for i in tqdm(range(0, n_source, sim_batch_size), desc="    TopM", leave=False):
            end_i = min(i + sim_batch_size, n_source)
            batch_size_actual = end_i - i
            e_batch = source_embeds[i:end_i]  # (batch, embed_dim)

            # Compute similarities to ALL targets
            # sim: (batch, n_target)
            sim = torch.matmul(e_batch, target_embeds.T) / model.tau

            # Get TopM per row
            topM_vals, topM_idx = torch.topk(sim, k=effective_topM, dim=1)  # (batch, topM)

            # Apply softmax to get probabilities
            T_hat_batch = torch.softmax(topM_vals, dim=1)  # (batch, topM)

            # Vectorized CSR triplet extraction (no Python loops!)
            batch_row_base = torch.arange(batch_size_actual, device=device).unsqueeze(1)  # (batch, 1)
            batch_row_indices = (batch_row_base + i).expand(-1, effective_topM).flatten()  # (batch * topM,)

            row_indices_list.append(batch_row_indices.cpu())
            col_indices_list.append(topM_idx.flatten().cpu())
            data_values_list.append(T_hat_batch.flatten().cpu())

    # Build CSR matrix from collected batches
    row_indices = torch.cat(row_indices_list).numpy().astype(np.int32)
    col_indices = torch.cat(col_indices_list).numpy().astype(np.int32)
    data_values = torch.cat(data_values_list).numpy().astype(np.float32)

    T_hat = csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(n_source, n_target)
    )

    print(f"    \hat{{T}} nnz: {T_hat.nnz} ({100*T_hat.nnz/(n_source*n_target):.2f}% density)")

    return T_hat


def build_all_hatT(
    model: Phase2Model,
    pairs,
    species: str,
    config: dict,
    device: str,
    save_dir: Path
):
    """
    Build \hat{T} for all pairs of a species and save.

    Args:
        model: Trained Phase2Model
        pairs: List of TransitionPairData objects
        species: 'mouse' or 'zebrafish'
        config: Configuration dict
        device: 'cuda' or 'cpu'
        save_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Building \hat{{T}} for {species}")
    print(f"{'='*80}")

    output_dir = save_dir / species
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_idx, pair in enumerate(pairs):
        T_hat = build_hatT_for_pair(
            model, pair, pair_idx, device, batch_size=1024
        )

        # Save
        save_path = output_dir / f'pair_{pair_idx}_That.npz'
        from scipy.sparse import save_npz
        save_npz(save_path, T_hat)
        print(f"    Saved to {save_path}")

    print(f"\n✓ All \hat{{T}} built for {species}")


if __name__ == '__main__':
    """Test hatT building."""
    import sys
    from p2a_01_dataset import load_species_data

    # Load config
    config_path = Path(__file__).parent.parent / 'config_phase2A.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(__file__).parent.parent / 'data_phase1'
    pairs, splits = load_species_data(data_dir / 'mouse', 'mouse', config, device)

    # Load trained model (for testing, create a random one)
    from p2a_02_model import build_time_vocabulary
    time_vocab = build_time_vocabulary(pairs)
    model = Phase2Model(config, time_vocab)

    # Build hatT for first pair only (for testing)
    print("\nTesting hatT construction on first pair...")
    T_hat = build_hatT_for_pair(
        model, pairs[0], 0, device, batch_size=512
    )

    print(f"\nTest complete:")
    print(f"  T_hat shape: {T_hat.shape}")
    print(f"  T_hat nnz: {T_hat.nnz}")
    print(f"  Row sums (first 10): {np.array(T_hat.sum(axis=1)).flatten()[:10]}")
    print("  ✓ All rows should sum to ~1.0")
