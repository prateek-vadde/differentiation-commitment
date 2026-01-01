"""
Phase 2A - Dataset with Hard Negative Mining
Optimized for GH200: 64 cores, 96GB VRAM
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.sparse import load_npz, csr_matrix
from scipy.stats import rankdata
from typing import List, Tuple, Dict
import warnings

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not available - falling back to CPU (will be slower)")


class TransitionPairData:
    """Container for a single transition pair with precomputed hard negatives."""

    def __init__(self, pair_dir: Path, pair_idx: int, device: str, hard_pool_size: int, topM: int):
        """Load and prepare data for one pair."""
        self.pair_idx = pair_idx
        self.pair_dir = pair_dir
        self.device = device
        self.topM = topM  # Precomputed TopM constant for this pair

        # Load data
        self.X = np.load(pair_dir / 'X_pca.npy').astype(np.float32)
        self.Y = np.load(pair_dir / 'Y_pca.npy').astype(np.float32)
        self.T = load_npz(pair_dir / 'T_sparse.npz').astype(np.float32)
        self.phi3 = np.load(pair_dir / 'phi3_bits.npy').astype(np.float32)
        self.phi2 = np.load(pair_dir / 'phi2.npy').astype(np.float32)
        self.lock = np.load(pair_dir / 'lock_label.npy')

        with open(pair_dir / 'time_id.json', 'r') as f:
            time_info = json.load(f)
        self.source_time = time_info['source_time']
        self.target_time = time_info['target_time']

        self.n_source = self.X.shape[0]
        self.n_target = self.Y.shape[0]

        # Compute phi3 percentile ranks (scale-free)
        self.phi3_rank = rankdata(self.phi3, method='average') / self.n_source

        # Precompute hard negative pools
        print(f"  Precomputing hard negatives for pair {pair_idx} "
              f"({self.n_source} → {self.n_target}, TopM={topM})...", end=' ', flush=True)
        self.hard_pools = self._precompute_hard_pools(hard_pool_size)
        print("✓")

    def _precompute_hard_pools(self, pool_size: int) -> np.ndarray:
        """
        Precompute hard negative pools: for each source, find K nearest targets in PCA space.
        GPU-accelerated for massive speedup.

        Returns: (n_source, pool_size) array of target indices
        """
        # Use GPU if available
        if HAS_CUPY and self.device == 'cuda':
            return self._precompute_hard_pools_gpu(pool_size)
        else:
            return self._precompute_hard_pools_cpu(pool_size)

    def _precompute_hard_pools_gpu(self, pool_size: int) -> np.ndarray:
        """GPU implementation using CuPy - vectorized distance computation."""
        # Clamp pool size to actual number of targets
        effective_pool_size = min(pool_size, self.n_target)

        # If pool size >= n_target, just return all targets for each source
        if effective_pool_size == self.n_target:
            all_targets = np.arange(self.n_target, dtype=np.int32)
            return np.tile(all_targets, (self.n_source, 1))

        X_gpu = cp.asarray(self.X)  # (n_source, pca_dim)
        Y_gpu = cp.asarray(self.Y)  # (n_target, pca_dim)

        # Compute squared distances in batches to handle large matrices
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        batch_size = min(2048, self.n_source)
        hard_pools = np.zeros((self.n_source, effective_pool_size), dtype=np.int32)

        X_norm_sq = cp.sum(X_gpu ** 2, axis=1, keepdims=True)  # (n_source, 1)
        Y_norm_sq = cp.sum(Y_gpu ** 2, axis=1)  # (n_target,)

        for i in range(0, self.n_source, batch_size):
            end_i = min(i + batch_size, self.n_source)
            X_batch = X_gpu[i:end_i]  # (batch, pca_dim)

            # Compute distances for this batch
            # D[i,j] = ||X[i] - Y[j]||^2
            XY = cp.matmul(X_batch, Y_gpu.T)  # (batch, n_target)
            D_sq = X_norm_sq[i:end_i] + Y_norm_sq - 2 * XY  # (batch, n_target)

            # Get top-K smallest distances (nearest neighbors)
            # Use argpartition for O(n) performance instead of full sort
            topk_indices = cp.argpartition(D_sq, effective_pool_size - 1, axis=1)[:, :effective_pool_size]

            hard_pools[i:end_i] = cp.asnumpy(topk_indices)

        return hard_pools

    def _precompute_hard_pools_cpu(self, pool_size: int) -> np.ndarray:
        """CPU fallback using numpy."""
        from sklearn.metrics import pairwise_distances

        # Clamp pool size to actual number of targets
        effective_pool_size = min(pool_size, self.n_target)

        # If pool size >= n_target, just return all targets for each source
        if effective_pool_size == self.n_target:
            all_targets = np.arange(self.n_target, dtype=np.int32)
            return np.tile(all_targets, (self.n_source, 1))

        # Compute pairwise squared distances
        # For very large matrices, compute in chunks
        hard_pools = np.zeros((self.n_source, effective_pool_size), dtype=np.int32)
        batch_size = min(1024, self.n_source)

        for i in range(0, self.n_source, batch_size):
            end_i = min(i + batch_size, self.n_source)
            X_batch = self.X[i:end_i]

            D_sq = pairwise_distances(X_batch, self.Y, metric='sqeuclidean', n_jobs=-1)

            # Get top-K smallest
            topk_indices = np.argpartition(D_sq, effective_pool_size - 1, axis=1)[:, :effective_pool_size]
            hard_pools[i:end_i] = topk_indices

        return hard_pools


class Phase2Dataset(Dataset):
    """
    PyTorch Dataset for Phase 2A training.
    Handles multiple transition pairs jointly.
    """

    def __init__(
        self,
        pairs: List[TransitionPairData],
        indices: np.ndarray,
        config: dict,
        is_training: bool = True
    ):
        """
        Args:
            pairs: List of TransitionPairData objects
            indices: Global indices for this split (train/val/test)
            config: Configuration dict
            is_training: If True, sample positives/negatives. If False, deterministic.
        """
        self.pairs = pairs
        self.indices = indices
        self.config = config
        self.is_training = is_training

        # Build index mapping: global_idx -> (pair_idx, local_idx)
        self.idx_to_pair = []
        cumsum = 0
        for pair_idx, pair in enumerate(pairs):
            for local_idx in range(pair.n_source):
                self.idx_to_pair.append((pair_idx, local_idx))
            cumsum += pair.n_source

        # Filter indices for this split
        self.valid_indices = indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        Returns dict with:
            - source_emb: (pca_dim,) source cell embedding
            - target_embs: (1+P+N, pca_dim) [positive samples + negative samples]
            - T_weights: (P,) weights from T matrix for positives
            - phi3: scalar phi3 value
            - phi3_rank: scalar phi3 percentile rank
            - phi2: scalar phi2 value
            - lock: bool lock label
            - pair_idx: int pair index
            - source_idx: int source cell index within pair
        """
        global_idx = self.valid_indices[idx]
        pair_idx, source_idx = self.idx_to_pair[global_idx]
        pair = self.pairs[pair_idx]

        # Source features
        x = pair.X[source_idx]  # (pca_dim,)

        # Get T row (sparse)
        T_row = pair.T.getrow(source_idx).tocoo()
        target_indices = T_row.col
        T_values = T_row.data

        # Sample positives
        P = self.config['positives_per_source']

        # Handle edge case: source cell has zero mass (dying cell in UOT)
        if len(target_indices) == 0:
            # Sample uniformly from all targets and assign equal weights
            pos_targets = np.random.choice(pair.n_target, size=P, replace=False)
            pos_weights = np.ones(P) / P
        elif len(target_indices) >= P:
            # Sample P targets weighted by T
            probs = T_values / T_values.sum()
            pos_sample_idx = np.random.choice(len(target_indices), size=P, replace=False, p=probs)
            pos_targets = target_indices[pos_sample_idx]
            pos_weights = T_values[pos_sample_idx]
        else:
            # If fewer than P nonzero entries, sample with replacement
            probs = T_values / T_values.sum()
            pos_sample_idx = np.random.choice(len(target_indices), size=P, replace=True, p=probs)
            pos_targets = target_indices[pos_sample_idx]
            pos_weights = T_values[pos_sample_idx]

        # Sample negatives (hard + uniform mix)
        N = self.config['negatives_gpu'] if torch.cuda.is_available() else self.config['negatives_cpu']
        N_hard = N // 2
        N_uniform = N - N_hard

        # Hard negatives from precomputed pool
        hard_pool = pair.hard_pools[source_idx]
        # Clamp to available hard pool size
        N_hard_actual = min(N_hard, len(hard_pool))
        neg_hard = np.random.choice(hard_pool, size=N_hard_actual, replace=False)

        # Uniform negatives (adjust to compensate if hard pool was too small)
        N_uniform_actual = N - N_hard_actual
        neg_uniform = np.random.choice(pair.n_target, size=N_uniform_actual, replace=False)

        # Combine and remove any positives
        neg_candidates = np.concatenate([neg_hard, neg_uniform])
        pos_set = set(pos_targets)
        neg_targets = np.array([n for n in neg_candidates if n not in pos_set])

        # If we filtered out too many, resample
        while len(neg_targets) < N:
            extra_needed = N - len(neg_targets)
            extra = np.random.choice(pair.n_target, size=extra_needed, replace=False)
            for e in extra:
                if e not in pos_set and e not in set(neg_targets):
                    neg_targets = np.append(neg_targets, e)
                    if len(neg_targets) >= N:
                        break

        neg_targets = neg_targets[:N]

        # Gather target embeddings
        y_pos = pair.Y[pos_targets]  # (P, pca_dim)
        y_neg = pair.Y[neg_targets]  # (N, pca_dim)

        return {
            'source_emb': torch.from_numpy(x),
            'target_pos': torch.from_numpy(y_pos),
            'target_neg': torch.from_numpy(y_neg),
            'T_weights': torch.from_numpy(pos_weights),
            'phi3': torch.tensor(pair.phi3[source_idx]),
            'phi3_rank': torch.tensor(pair.phi3_rank[source_idx]),
            'phi2': torch.tensor(pair.phi2[source_idx]),
            'lock': torch.tensor(pair.lock[source_idx], dtype=torch.float32),
            'pair_idx': torch.tensor(pair_idx, dtype=torch.long),
            'source_idx': torch.tensor(source_idx, dtype=torch.long),
            'source_time': pair.source_time,
            'target_time': pair.target_time,
        }


def load_species_data(
    species_dir: Path,
    species: str,
    config: dict,
    device: str = 'cuda'
) -> Tuple[List[TransitionPairData], Dict[str, np.ndarray]]:
    """
    Load all pairs for a species and create train/val/test splits.

    Returns:
        pairs: List of TransitionPairData
        splits: Dict with 'train', 'val', 'test' keys containing global indices
    """
    print(f"\nLoading {species} data:")

    # Load TopM values per pair
    topM_path = species_dir.parent.parent / 'topM_per_pair.json'
    with open(topM_path, 'r') as f:
        topM_dict = json.load(f)
    topM_for_species = topM_dict[species]

    # Find all pairs
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    print(f"  Found {len(pair_dirs)} pairs")

    # Load pairs with hard negative precomputation
    pairs = []
    for i, pair_dir in enumerate(pair_dirs):
        # Get TopM for this pair (JSON keys are strings)
        topM = topM_for_species[str(i)]

        pair = TransitionPairData(
            pair_dir,
            pair_idx=i,
            device=device,
            hard_pool_size=config['hard_negative_pool'],
            topM=topM
        )
        pairs.append(pair)

    # Create deterministic splits
    total_cells = sum(p.n_source for p in pairs)
    print(f"  Total source cells: {total_cells}")

    # Global indices
    all_indices = np.arange(total_cells)

    # Deterministic shuffle
    rng = np.random.RandomState(config['seed'])
    rng.shuffle(all_indices)

    # Split by integer rule
    n_train = int(total_cells * config['split_train'])
    n_val = int(total_cells * config['split_val'])

    splits = {
        'train': all_indices[:n_train],
        'val': all_indices[n_train:n_train + n_val],
        'test': all_indices[n_train + n_val:]
    }

    print(f"  Splits: train={len(splits['train'])}, "
          f"val={len(splits['val'])}, test={len(splits['test'])}")

    return pairs, splits


def create_dataloaders(
    pairs: List[TransitionPairData],
    splits: Dict[str, np.ndarray],
    config: dict,
    num_workers: int = 8
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders."""

    batch_size = config['batch_gpu'] if torch.cuda.is_available() else config['batch_cpu']

    datasets = {
        'train': Phase2Dataset(pairs, splits['train'], config, is_training=True),
        'val': Phase2Dataset(pairs, splits['val'], config, is_training=True),
        'test': Phase2Dataset(pairs, splits['test'], config, is_training=False),
    }

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        ),
    }

    return dataloaders


if __name__ == '__main__':
    """Test dataset loading."""
    import sys

    # Load config
    config_path = Path(__file__).parent.parent / 'config_phase2A.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Test loading
    data_dir = Path(__file__).parent.parent / 'data_phase1'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"CuPy available: {HAS_CUPY}")

    # Load mouse
    mouse_pairs, mouse_splits = load_species_data(
        data_dir / 'mouse', 'mouse', config, device
    )

    # Create dataloaders
    mouse_loaders = create_dataloaders(mouse_pairs, mouse_splits, config, num_workers=4)

    # Test iteration
    print("\nTesting dataloader iteration:")
    batch = next(iter(mouse_loaders['train']))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  source_emb shape: {batch['source_emb'].shape}")
    print(f"  target_pos shape: {batch['target_pos'].shape}")
    print(f"  target_neg shape: {batch['target_neg'].shape}")
    print("✓ Dataset test passed")
