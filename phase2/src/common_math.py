"""
Common mathematical utilities for Phase 2B/3.

Handles:
- Entropy computation for sparse rows (GPU-accelerated)
- Percentile rank transforms
- Permutation tests (e.g., Spearman with null distribution)
- Bootstrap confidence intervals
- Paired statistical tests
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, wilcoxon
from typing import Tuple, Optional
import torch


def row_entropy_csr_gpu(
    T: csr_matrix,
    clip: float = 1e-12,
    base: str = 'e',
    device: str = 'cuda',
    batch_size: int = 4096
) -> np.ndarray:
    """
    GPU-accelerated entropy computation - STAYS SPARSE, processes per-row on GPU.

    Args:
        T: Sparse CSR matrix (N x M), rows are probability distributions
        clip: Minimum probability value (to avoid log(0))
        base: 'e' for nats (natural log) or '2' for bits
        device: 'cuda' for GPU
        batch_size: Number of rows to process per batch

    Returns:
        Array of shape (N,) with entropy for each row
    """
    if base not in ['e', '2']:
        raise ValueError(f"Invalid base: {base}. Use 'e' or '2'")

    N = T.shape[0]
    entropies = np.zeros(N, dtype=np.float32)

    # Process in batches - each batch processes multiple rows in parallel
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)

        # Extract all non-zero values for this batch of rows
        batch_data = []
        batch_lengths = []

        for i in range(start_idx, end_idx):
            row_start = T.indptr[i]
            row_end = T.indptr[i + 1]
            row_data = T.data[row_start:row_end]

            if len(row_data) > 0:
                batch_data.append(row_data)
                batch_lengths.append(len(row_data))
            else:
                batch_data.append(np.array([]))
                batch_lengths.append(0)

        # Concatenate all data and process on GPU
        if sum(batch_lengths) > 0:
            all_data = np.concatenate([d for d in batch_data if len(d) > 0])
            data_gpu = torch.from_numpy(all_data).to(device)

            # Clip and compute -p * log(p) for ALL values at once
            data_gpu = torch.clamp(data_gpu, min=clip)

            if base == 'e':
                log_data = torch.log(data_gpu)
            else:
                log_data = torch.log2(data_gpu)

            contributions = -data_gpu * log_data

            # Move back to CPU for segmented sum
            contributions_cpu = contributions.cpu().numpy()

            # Sum contributions per row
            offset = 0
            for local_idx, length in enumerate(batch_lengths):
                if length > 0:
                    entropies[start_idx + local_idx] = contributions_cpu[offset:offset + length].sum()
                    offset += length

            del data_gpu, log_data, contributions
            torch.cuda.empty_cache()

    return entropies


def row_entropy_csr(
    T: csr_matrix,
    clip: float = 1e-12,
    base: str = 'e',
    use_gpu: bool = True
) -> np.ndarray:
    """
    Compute Shannon entropy for each row of a sparse CSR matrix.

    H(p) = -Σ p_j log(p_j)

    Automatically uses GPU acceleration if available and beneficial.

    Args:
        T: Sparse CSR matrix (N x M), rows are probability distributions
        clip: Minimum probability value (to avoid log(0))
        base: 'e' for nats (natural log) or '2' for bits
        use_gpu: If True and CUDA available, use GPU acceleration

    Returns:
        Array of shape (N,) with entropy for each row

    Raises:
        ValueError: If base is not 'e' or '2'
    """
    if base not in ['e', '2']:
        raise ValueError(f"Invalid base: {base}. Use 'e' or '2'")

    # Use GPU if requested and available
    if use_gpu and torch.cuda.is_available():
        return row_entropy_csr_gpu(T, clip, base, device='cuda', batch_size=4096)

    # Vectorized CPU implementation
    N = T.shape[0]

    # Clip all non-zero values at once
    data_clipped = np.maximum(T.data, clip)

    # Compute log once for all data
    if base == 'e':
        log_data = np.log(data_clipped)
    else:
        log_data = np.log2(data_clipped)

    # Element-wise: -p * log(p)
    entropy_contributions = -data_clipped * log_data

    # Use np.add.reduceat to sum segments (vectorized over indptr)
    # reduceat sums from indptr[i] to indptr[i+1]-1 for each i
    entropies = np.add.reduceat(entropy_contributions, T.indptr[:-1])

    return entropies


def percentile_ranks(x: np.ndarray) -> np.ndarray:
    """
    Convert values to percentile ranks in [0, 1].

    Uses rank-based transform with average handling for ties.

    Args:
        x: Array of values (N,)

    Returns:
        Array of percentile ranks (N,), where:
        - 0.0 = minimum value
        - 1.0 = maximum value
        - Ties get average rank

    Raises:
        ValueError: If x is empty or contains NaN/Inf
    """
    from scipy.stats import rankdata

    if len(x) == 0:
        raise ValueError("Input array is empty")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains NaN or Inf values")

    if len(x) == 1:
        return np.array([0.5])

    ranks = rankdata(x, method='average')  # Average rank for ties
    percentiles = (ranks - 1) / (len(x) - 1)

    return percentiles


def _compute_perm_spearman(args):
    """Worker function for parallel permutation test."""
    x, y, perm_indices = args
    y_perm = y[perm_indices]
    rho, _ = spearmanr(x, y_perm)
    return rho


def perm_test_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
    n_jobs: int = 8
) -> Tuple[float, float, np.ndarray]:
    """
    Permutation test for Spearman correlation (PARALLELIZED).

    Args:
        x: First variable (N,)
        y: Second variable (N,)
        n_perm: Number of permutations for null distribution
        seed: Random seed for reproducibility
        n_jobs: Number of parallel workers

    Returns:
        (observed_rho, p_value, null_distribution)
        - observed_rho: Observed Spearman correlation
        - p_value: Two-tailed p-value
        - null_distribution: Array of null correlations (n_perm,)

    Raises:
        ValueError: If arrays have different lengths or contain invalid values
    """
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: x={len(x)}, y={len(y)}")

    if len(x) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(x)}")

    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or Inf values")

    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN or Inf values")

    # Compute observed correlation
    obs_rho, _ = spearmanr(x, y)

    # Generate permutation indices
    rng = np.random.RandomState(seed)
    perm_indices_list = [rng.permutation(len(y)) for _ in range(n_perm)]

    # Parallel computation
    from concurrent.futures import ProcessPoolExecutor
    args_list = [(x, y, perm_idx) for perm_idx in perm_indices_list]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        null_dist = np.array(list(executor.map(_compute_perm_spearman, args_list)))

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_dist) >= np.abs(obs_rho))

    return obs_rho, p_value, null_dist


def bootstrap_ci_median(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for median.

    Args:
        values: Array of values (N,)
        n_boot: Number of bootstrap samples
        seed: Random seed
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (median, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)

    medians = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[i] = np.median(sample)

    median = np.median(values)
    lower = np.percentile(medians, 100 * alpha / 2)
    upper = np.percentile(medians, 100 * (1 - alpha / 2))

    return median, lower, upper


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean.

    Args:
        values: Array of values (N,)
        n_boot: Number of bootstrap samples
        seed: Random seed
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)

    means = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.mean(sample)

    mean = np.mean(values)
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return mean, lower, upper


def paired_wilcoxon(
    deltas: np.ndarray,
    deltas_null: np.ndarray,
    alternative: str = 'two-sided'
) -> float:
    """
    Paired Wilcoxon signed-rank test.

    Tests whether the distribution of (deltas - deltas_null) is centered at 0.

    Args:
        deltas: Directional perturbation effects (N,)
        deltas_null: Matched null perturbation effects (N,)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        p_value
    """
    differences = deltas - deltas_null

    # Remove zeros (ties at zero)
    differences = differences[differences != 0]

    if len(differences) == 0:
        return 1.0  # No evidence of difference

    _, p_value = wilcoxon(differences, alternative=alternative)

    return p_value


def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute AUROC for binary classification.

    Args:
        scores: Predicted scores (N,)
        labels: Binary labels (N,), 0 or 1

    Returns:
        AUROC value in [0, 1]
    """
    from sklearn.metrics import roc_auc_score

    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return 0.5  # Undefined, return random guess

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    return auroc


def compute_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Args:
        x: First variable (N,)
        y: Second variable (N,)

    Returns:
        (correlation, p_value)
    """
    from scipy.stats import pearsonr

    if len(x) < 2:
        return 0.0, 1.0

    r, p = pearsonr(x, y)
    return r, p


def compute_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        x: First variable (N,)
        y: Second variable (N,)

    Returns:
        (correlation, p_value)
    """
    if len(x) < 2:
        return 0.0, 1.0

    rho, p = spearmanr(x, y)
    return rho, p


def _compute_boot_auroc(args):
    """Worker function for parallel bootstrap AUROC."""
    from sklearn.metrics import roc_auc_score
    scores, labels, boot_indices = args

    scores_boot = scores[boot_indices]
    labels_boot = labels[boot_indices]

    # Check if bootstrap has both classes
    if len(np.unique(labels_boot)) < 2:
        return 0.5
    else:
        try:
            return roc_auc_score(labels_boot, scores_boot)
        except ValueError:
            return 0.5


def bootstrap_ci_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    n_jobs: int = 8
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for AUROC (PARALLELIZED).

    Args:
        scores: Predicted scores (N,)
        labels: Binary labels (N,), 0 or 1
        n_boot: Number of bootstrap samples
        seed: Random seed
        alpha: Significance level (default 0.05 for 95% CI)
        n_jobs: Number of parallel workers

    Returns:
        (auroc, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(scores)

    # Generate bootstrap indices
    boot_indices_list = [rng.choice(n, size=n, replace=True) for _ in range(n_boot)]

    # Parallel computation
    from concurrent.futures import ProcessPoolExecutor
    args_list = [(scores, labels, boot_idx) for boot_idx in boot_indices_list]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        aurocs = np.array(list(executor.map(_compute_boot_auroc, args_list)))

    auroc = compute_auroc(scores, labels)
    lower = np.percentile(aurocs, 100 * alpha / 2)
    upper = np.percentile(aurocs, 100 * (1 - alpha / 2))

    return auroc, lower, upper


def shuffle_labels(
    labels: np.ndarray,
    n_shuffles: int = 1000,
    seed: int = 0
) -> np.ndarray:
    """
    Generate shuffled versions of labels for null distribution.

    Args:
        labels: Original labels (N,)
        n_shuffles: Number of shuffles
        seed: Random seed

    Returns:
        Array of shape (n_shuffles, N) with shuffled labels
    """
    rng = np.random.RandomState(seed)

    shuffled = np.zeros((n_shuffles, len(labels)))
    for i in range(n_shuffles):
        shuffled[i] = rng.permutation(labels)

    return shuffled
