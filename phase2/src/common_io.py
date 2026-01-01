"""
Common I/O utilities for Phase 2B/3.

Handles:
- Loading per-pair bundles (enforce data contract)
- Validating dimensional consistency
- Building alignment across pairs for composition
- Composing sparse transition operators T^(h)
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
import torch


def log(msg: str):
    """Simple logging function."""
    print(msg, flush=True)


@dataclass
class PairData:
    """Data bundle for one transition pair k."""
    pair_idx: int
    species: str

    # Core data (required)
    T_hat: csr_matrix          # Compressed transition operator
    topM: int                  # Pair-specific TopM
    source_ids: List[str]      # Ordered source cell IDs
    target_ids: List[str]      # Ordered target cell IDs
    X_pca: np.ndarray          # Source features (N_src x d)
    Y_pca: np.ndarray          # Target features (N_tgt x d)

    # Phase 1 data (optional, evaluation only)
    T_true: Optional[csr_matrix] = None
    phi3_bits: Optional[np.ndarray] = None
    phi2: Optional[np.ndarray] = None
    lock_label: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate dimensional consistency."""
        n_src, n_tgt = self.T_hat.shape

        if len(self.source_ids) != n_src:
            raise ValueError(
                f"Pair {self.pair_idx}: source_ids length ({len(self.source_ids)}) "
                f"!= T_hat sources ({n_src})"
            )

        if len(self.target_ids) != n_tgt:
            raise ValueError(
                f"Pair {self.pair_idx}: target_ids length ({len(self.target_ids)}) "
                f"!= T_hat targets ({n_tgt})"
            )

        if self.X_pca.shape[0] != n_src:
            raise ValueError(
                f"Pair {self.pair_idx}: X_pca rows ({self.X_pca.shape[0]}) "
                f"!= T_hat sources ({n_src})"
            )

        if self.Y_pca.shape[0] != n_tgt:
            raise ValueError(
                f"Pair {self.pair_idx}: Y_pca rows ({self.Y_pca.shape[0]}) "
                f"!= T_hat targets ({n_tgt})"
            )

        # Validate Phase 1 data if present
        if self.phi3_bits is not None and len(self.phi3_bits) != n_src:
            raise ValueError(f"Pair {self.pair_idx}: phi3_bits length mismatch")

        if self.phi2 is not None and len(self.phi2) != n_src:
            raise ValueError(f"Pair {self.pair_idx}: phi2 length mismatch")

        if self.lock_label is not None and len(self.lock_label) != n_src:
            raise ValueError(f"Pair {self.pair_idx}: lock_label length mismatch")


def load_pair_bundle(species: str, pair_idx: int, data_root: Path) -> PairData:
    """
    Load data bundle for one pair according to Phase 2B/3 contract.

    Args:
        species: 'mouse' or 'zebrafish'
        pair_idx: Pair index
        data_root: Path to data_phase2A directory

    Returns:
        PairData object with all required and optional data

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If dimensional consistency checks fail
    """
    pair_dir = data_root / species / f'pair_{pair_idx}'

    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

    # Load required files
    T_hat = load_npz(pair_dir / 'T_hat_sparse.npz')

    with open(pair_dir / 'topM.json', 'r') as f:
        topM = json.load(f)

    with open(pair_dir / 'source_ids.txt', 'r') as f:
        source_ids = [line.strip() for line in f]

    with open(pair_dir / 'target_ids.txt', 'r') as f:
        target_ids = [line.strip() for line in f]

    X_pca = np.load(pair_dir / 'X_pca.npy')
    Y_pca = np.load(pair_dir / 'Y_pca.npy')

    # Load optional Phase 1 files
    T_true = None
    phi3_bits = None
    phi2 = None
    lock_label = None

    if (pair_dir / 'T_sparse.npz').exists():
        T_true = load_npz(pair_dir / 'T_sparse.npz')

    if (pair_dir / 'phi3_bits.npy').exists():
        phi3_bits = np.load(pair_dir / 'phi3_bits.npy')

    if (pair_dir / 'phi2.npy').exists():
        phi2 = np.load(pair_dir / 'phi2.npy')

    if (pair_dir / 'lock_label.npy').exists():
        lock_label = np.load(pair_dir / 'lock_label.npy')

    return PairData(
        pair_idx=pair_idx,
        species=species,
        T_hat=T_hat,
        topM=topM,
        source_ids=source_ids,
        target_ids=target_ids,
        X_pca=X_pca,
        Y_pca=Y_pca,
        T_true=T_true,
        phi3_bits=phi3_bits,
        phi2=phi2,
        lock_label=lock_label
    )


def validate_pairdata(pair: PairData, row_sum_atol: float = 1e-5):
    """
    Validate PairData meets Phase 2B/3 requirements.

    Args:
        pair: PairData object
        row_sum_atol: Absolute tolerance for row-stochasticity check

    Raises:
        ValueError: If validation fails
    """
    # Check row-stochasticity
    row_sums = np.array(pair.T_hat.sum(axis=1)).flatten()
    bad_rows = np.where(np.abs(row_sums - 1.0) > row_sum_atol)[0]

    if len(bad_rows) > 0:
        worst_idx = bad_rows[np.argmax(np.abs(row_sums[bad_rows] - 1.0))]
        worst_sum = row_sums[worst_idx]

        raise ValueError(
            f"Pair {pair.pair_idx}: Row-stochasticity violated. "
            f"{len(bad_rows)} / {len(row_sums)} rows exceed tolerance {row_sum_atol}. "
            f"Worst: row {worst_idx} sum = {worst_sum:.10f}"
        )


@dataclass
class AlignmentIndex:
    """Alignment mapping for composing operators across pairs."""
    species: str
    num_pairs: int
    pairs: List[PairData]

    def __post_init__(self):
        """Validate alignment across consecutive pairs."""
        # Alignment check results
        # aligned[k] = True if pair k targets align with pair k+1 sources
        self.aligned: List[bool] = []

        for k in range(self.num_pairs - 1):
            pair_k = self.pairs[k]
            pair_k1 = self.pairs[k + 1]

            # Check if target IDs of pair k match source IDs of pair k+1
            targets_k = pair_k.target_ids
            sources_k1 = pair_k1.source_ids

            # Exact element-wise comparison
            if len(targets_k) != len(sources_k1):
                is_aligned = False
            else:
                is_aligned = all(t == s for t, s in zip(targets_k, sources_k1))

            self.aligned.append(is_aligned)

            if not is_aligned:
                # Find first mismatch for debugging
                mismatch_idx = None
                if len(targets_k) == len(sources_k1):
                    for i, (t, s) in enumerate(zip(targets_k, sources_k1)):
                        if t != s:
                            mismatch_idx = i
                            break

                error_msg = (
                    f"Alignment failure between pair {k} and {k+1}:\n"
                    f"  Pair {k} has {len(targets_k)} targets\n"
                    f"  Pair {k+1} has {len(sources_k1)} sources\n"
                )

                if mismatch_idx is not None:
                    error_msg += (
                        f"  First mismatch at index {mismatch_idx}:\n"
                        f"    pair {k} target: '{targets_k[mismatch_idx]}'\n"
                        f"    pair {k+1} source: '{sources_k1[mismatch_idx]}'"
                    )

                raise ValueError(error_msg)

    def can_compose(self, k: int, horizon: int) -> bool:
        """Check if we can compose T_k through T_{k+horizon-1}."""
        if k + horizon > self.num_pairs:
            return False

        # Check all intermediate alignments
        for i in range(k, k + horizon - 1):
            if not self.aligned[i]:
                return False

        return True


def build_alignment(species: str, data_root: Path) -> AlignmentIndex:
    """
    Build alignment index for one species.

    Args:
        species: 'mouse' or 'zebrafish'
        data_root: Path to data_phase2A directory

    Returns:
        AlignmentIndex with all pairs loaded and validated

    Raises:
        ValueError: If alignment check fails
    """
    species_dir = data_root / species

    # Find all pairs
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])

    num_pairs = len(pair_dirs)

    # Load all pairs
    pairs = []
    for pair_dir in pair_dirs:
        pair_idx = int(pair_dir.name.split('_')[1])
        pair = load_pair_bundle(species, pair_idx, data_root)
        validate_pairdata(pair)
        pairs.append(pair)

    # Build alignment index (validates alignment in __post_init__)
    alignment = AlignmentIndex(
        species=species,
        num_pairs=num_pairs,
        pairs=pairs
    )

    return alignment


def compose_T_hat_gpu(
    alignment: AlignmentIndex,
    k: int,
    horizon: int,
    device: str = 'cuda'
) -> csr_matrix:
    """
    GPU-accelerated composition of transition operators.

    Performs sparse matrix multiplication on GPU for massive speedup.

    Args:
        alignment: AlignmentIndex for the species
        k: Starting pair index
        horizon: Number of steps forward (h)
        device: 'cuda' for GPU

    Returns:
        Composed sparse CSR matrix T^(h)_k (on CPU)
    """
    if horizon == 1:
        return alignment.pairs[k].T_hat

    # Convert first matrix to PyTorch sparse
    T_current = alignment.pairs[k].T_hat.tocoo()
    indices = torch.LongTensor(np.vstack([T_current.row, T_current.col]))
    values = torch.FloatTensor(T_current.data)
    shape = T_current.shape

    result = torch.sparse_coo_tensor(indices, values, shape, device=device)

    # Iteratively multiply on GPU
    for i in range(1, horizon):
        T_next = alignment.pairs[k + i].T_hat.tocoo()
        indices_next = torch.LongTensor(np.vstack([T_next.row, T_next.col]))
        values_next = torch.FloatTensor(T_next.data)
        shape_next = T_next.shape

        T_next_sparse = torch.sparse_coo_tensor(
            indices_next, values_next, shape_next, device=device
        )

        # Sparse @ sparse on GPU
        result = torch.sparse.mm(result, T_next_sparse)

    # Convert back to scipy CSR on CPU
    result_coo = result.cpu().coalesce()
    indices_np = result_coo.indices().numpy()
    values_np = result_coo.values().numpy()
    shape_np = result_coo.shape

    from scipy.sparse import coo_matrix
    result_scipy = coo_matrix(
        (values_np, (indices_np[0], indices_np[1])),
        shape=shape_np
    )

    return result_scipy.tocsr()


def compose_T_hat(
    alignment: AlignmentIndex,
    k: int,
    horizon: int,
    cache_dir: Optional[Path] = None,
    use_gpu: bool = True
) -> csr_matrix:
    """
    Compose transition operators: T^(h)_k = T_k · T_{k+1} · ... · T_{k+h-1}

    Uses GPU acceleration and caching for maximum performance.

    Args:
        alignment: AlignmentIndex for the species
        k: Starting pair index
        horizon: Number of steps forward (h)
        cache_dir: Optional directory to cache composed operators
        use_gpu: If True and CUDA available, use GPU acceleration

    Returns:
        Composed sparse CSR matrix T^(h)_k

    Raises:
        ValueError: If composition is not possible (alignment failure or out of bounds)
    """
    if not alignment.can_compose(k, horizon):
        raise ValueError(
            f"Cannot compose from pair {k} with horizon {horizon}. "
            f"Either out of bounds (max pair: {alignment.num_pairs - 1}) "
            f"or alignment failure."
        )

    # Check cache first
    if cache_dir is not None:
        cache_path = cache_dir / alignment.species / f'T_composed_k{k}_h{horizon}.npz'
        if cache_path.exists():
            return load_npz(cache_path)

    # Sparse matrix multiplication is better on CPU with scipy
    # GPU is reserved for dense operations (entropy, etc.)
    if horizon == 1:
        result = alignment.pairs[k].T_hat
    else:
        # Iteratively multiply on CPU (scipy sparse @ is very efficient)
        result = alignment.pairs[k].T_hat
        for i in range(1, horizon):
            result = result @ alignment.pairs[k + i].T_hat

    # Cache result
    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_npz(cache_path, result, compressed=True)

    return result
