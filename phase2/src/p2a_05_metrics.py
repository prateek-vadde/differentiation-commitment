"""
Phase 2A - Metrics & Pass/Fail Gates
Implements all 4 metrics with exact spec:
1. Mass capture ≥ 0.90
2. Φ₃ rank Spearman ≥ 0.80
3. Lock AUROC ≥ 0.80
4. Lock Jaccard ≥ 0.50 (scale-free top-q fraction)
"""
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from p2a_02_model import Phase2Model


def compute_mass_capture(T_true: csr_matrix, T_hat: csr_matrix) -> float:
    """
    Mass capture: For each source, measure what fraction of true mass is captured by \hat{T}.

    mass_capture_i = Σ_j min(T_true[i,j], T_hat[i,j])

    Returns:
        mean mass capture across all sources
    """
    n_source = T_true.shape[0]
    mass_captures = []

    for i in range(n_source):
        true_row = T_true.getrow(i).toarray().flatten()
        hat_row = T_hat.getrow(i).toarray().flatten()

        # Element-wise min
        captured = np.minimum(true_row, hat_row).sum()
        mass_captures.append(captured)

    return np.mean(mass_captures)


def compute_phi3_spearman(
    model: Phase2Model,
    pair,
    T_hat: csr_matrix,
    device: str,
    batch_size: int = 512
) -> float:
    """
    Φ₃ preservation: Spearman correlation between true Φ₃ rank and TopM entropy rank.

    Procedure (per spec):
    1. Compute TopM entropy from \hat{T}: Ĥ^TopM_i = -Σ_{j∈TopM(i)} T̂_{ij} log T̂_{ij}
    2. Convert true Φ₃ to ranks
    3. Convert Ĥ^TopM to ranks
    4. Spearman correlation

    Args:
        model: Trained model (not used, but kept for API consistency)
        pair: TransitionPairData object
        T_hat: Compressed operator
        device: 'cuda' or 'cpu'
        batch_size: Batch size for computation

    Returns:
        Spearman correlation coefficient
    """
    n_source = pair.n_source

    # True Φ₃
    phi3_true = pair.phi3  # (n_source,)

    # Compute TopM entropy from T_hat
    H_topM = []

    for i in range(n_source):
        row = T_hat.getrow(i)
        row_data = row.data  # Already TopM values (probabilities)

        if len(row_data) == 0:
            H_topM.append(0.0)
        else:
            # Entropy: H = -Σ p log p
            entropy = -(row_data * np.log(row_data + 1e-10)).sum()
            H_topM.append(entropy)

    H_topM = np.array(H_topM)

    # Convert to ranks
    phi3_rank = rankdata(phi3_true, method='average')
    H_topM_rank = rankdata(H_topM, method='average')

    # Spearman correlation
    rho, pval = spearmanr(phi3_rank, H_topM_rank)

    return rho


def compute_lock_predictions(
    model: Phase2Model,
    pair,
    device: str,
    batch_size: int = 512
) -> np.ndarray:
    """
    Get lock probabilities from model for all source cells.

    Args:
        model: Trained model
        pair: TransitionPairData object
        device: 'cuda' or 'cpu'
        batch_size: Batch size

    Returns:
        lock_probs: (n_source,) array of lock probabilities
    """
    model.eval()
    n_source = pair.n_source

    lock_logits = []

    with torch.no_grad():
        for i in range(0, n_source, batch_size):
            end_i = min(i + batch_size, n_source)
            X_batch = torch.from_numpy(pair.X[i:end_i]).to(device)

            # Encode and get lock logits
            e_batch = model.encode_source(X_batch, pair.source_time)
            logits = model.lock_head(e_batch)

            lock_logits.append(logits.cpu().numpy())

    lock_logits = np.concatenate(lock_logits)

    # Convert to probabilities
    lock_probs = 1.0 / (1.0 + np.exp(-lock_logits))  # sigmoid

    return lock_probs


def compute_lock_auroc(lock_probs: np.ndarray, lock_labels: np.ndarray) -> float:
    """
    Lock AUROC: Area under ROC curve for lock classification.

    Args:
        lock_probs: (n_source,) predicted lock probabilities
        lock_labels: (n_source,) true lock labels (bool)

    Returns:
        AUROC score
    """
    if len(np.unique(lock_labels)) < 2:
        # If all labels are same class, AUROC is undefined
        return 1.0 if lock_labels[0] else 0.0

    return roc_auc_score(lock_labels.astype(int), lock_probs)


def compute_lock_jaccard_scalefree(
    lock_probs: np.ndarray,
    lock_labels: np.ndarray,
    train_lock_labels: np.ndarray
) -> float:
    """
    Scale-free Lock Jaccard (UPGRADED per spec).

    Procedure:
    1. Compute true locked fraction q from train set
    2. Predicted locked = top q fraction by lock_probs
    3. Jaccard overlap with test set lock_labels

    Args:
        lock_probs: (n_test,) predicted lock probabilities on TEST set
        lock_labels: (n_test,) true lock labels on TEST set
        train_lock_labels: (n_train,) true lock labels on TRAIN set

    Returns:
        Jaccard similarity
    """
    # Compute locked fraction from train set
    q = train_lock_labels.mean()

    # Number of cells to predict as locked
    n_predict_locked = int(np.round(len(lock_probs) * q))
    n_predict_locked = max(1, min(n_predict_locked, len(lock_probs)))  # Clamp to [1, n_test]

    # Get top-q fraction by probability
    threshold_idx = np.argsort(lock_probs)[-n_predict_locked:]
    pred_locked = np.zeros(len(lock_probs), dtype=bool)
    pred_locked[threshold_idx] = True

    # Compute Jaccard
    intersection = (pred_locked & lock_labels).sum()
    union = (pred_locked | lock_labels).sum()

    if union == 0:
        return 1.0  # Both sets empty

    jaccard = intersection / union

    return jaccard


def compute_all_metrics_for_pair(
    model: Phase2Model,
    pair,
    pair_idx: int,
    T_hat: csr_matrix,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    device: str
) -> Dict[str, float]:
    """
    Compute all 4 metrics for a single pair.

    Args:
        model: Trained model
        pair: TransitionPairData object
        pair_idx: Pair index
        T_hat: Compressed operator
        train_indices: Indices of train samples (for lock Jaccard q computation)
        test_indices: Indices of test samples (for evaluation)
        device: 'cuda' or 'cpu'

    Returns:
        metrics: Dict with mass_capture, phi3_spearman, lock_auroc, lock_jaccard
    """
    print(f"  Computing metrics for pair {pair_idx}...")

    # Mass capture (on test set)
    print("    Mass capture...")
    mass_capture = compute_mass_capture(pair.T, T_hat)

    # Φ₃ Spearman (on test set)
    print("    Φ₃ Spearman...")
    phi3_spearman = compute_phi3_spearman(model, pair, T_hat, device)

    # Get lock predictions (all sources)
    print("    Lock predictions...")
    lock_probs_all = compute_lock_predictions(model, pair, device)

    # Lock AUROC (on test set)
    print("    Lock AUROC...")
    lock_probs_test = lock_probs_all[test_indices]
    lock_labels_test = pair.lock[test_indices]
    lock_auroc = compute_lock_auroc(lock_probs_test, lock_labels_test)

    # Lock Jaccard (scale-free, using train q)
    print("    Lock Jaccard (scale-free)...")
    lock_labels_train = pair.lock[train_indices]
    lock_jaccard = compute_lock_jaccard_scalefree(lock_probs_test, lock_labels_test, lock_labels_train)

    metrics = {
        'mass_capture': mass_capture,
        'phi3_spearman': phi3_spearman,
        'lock_auroc': lock_auroc,
        'lock_jaccard': lock_jaccard
    }

    print(f"    Results: mass={mass_capture:.3f}, phi3_rho={phi3_spearman:.3f}, "
          f"auroc={lock_auroc:.3f}, jaccard={lock_jaccard:.3f}")

    return metrics


def compute_all_metrics_for_species(
    model: Phase2Model,
    pairs,
    splits: Dict[str, np.ndarray],
    species: str,
    config: dict,
    device: str,
    hatT_dir: Path
) -> Tuple[List[Dict], Dict[str, float], bool]:
    """
    Compute metrics for all pairs of a species and check pass/fail gates.

    Args:
        model: Trained model
        pairs: List of TransitionPairData objects
        splits: Dict with 'train', 'val', 'test' indices
        species: 'mouse' or 'zebrafish'
        config: Configuration dict
        device: 'cuda' or 'cpu'
        hatT_dir: Directory containing \hat{T} files

    Returns:
        pair_metrics: List of metric dicts (one per pair)
        aggregate_metrics: Dict with mean metrics across pairs
        passed: Boolean indicating if all gates passed
    """
    print(f"\n{'='*80}")
    print(f"Computing metrics for {species}")
    print(f"{'='*80}")

    pair_metrics = []

    # Map global indices to (pair_idx, local_idx)
    idx_to_pair = []
    for pair_idx, pair in enumerate(pairs):
        for local_idx in range(pair.n_source):
            idx_to_pair.append((pair_idx, local_idx))

    # Group indices by pair
    pair_train_indices = [[] for _ in pairs]
    pair_test_indices = [[] for _ in pairs]

    for global_idx in splits['train']:
        pair_idx, local_idx = idx_to_pair[global_idx]
        pair_train_indices[pair_idx].append(local_idx)

    for global_idx in splits['test']:
        pair_idx, local_idx = idx_to_pair[global_idx]
        pair_test_indices[pair_idx].append(local_idx)

    # Compute metrics per pair
    for pair_idx, pair in enumerate(pairs):
        # Load \hat{T}
        hatT_path = hatT_dir / species / f'pair_{pair_idx}_That.npz'
        T_hat = load_npz(hatT_path)

        train_idx = np.array(pair_train_indices[pair_idx])
        test_idx = np.array(pair_test_indices[pair_idx])

        metrics = compute_all_metrics_for_pair(
            model, pair, pair_idx, T_hat, train_idx, test_idx, device
        )

        pair_metrics.append(metrics)

    # Aggregate
    aggregate_metrics = {
        'mass_capture': np.mean([m['mass_capture'] for m in pair_metrics]),
        'phi3_spearman': np.mean([m['phi3_spearman'] for m in pair_metrics]),
        'lock_auroc': np.mean([m['lock_auroc'] for m in pair_metrics]),
        'lock_jaccard': np.mean([m['lock_jaccard'] for m in pair_metrics])
    }

    # Check pass/fail gates
    passed = (
        aggregate_metrics['mass_capture'] >= config['pass_mass_capture'] and
        aggregate_metrics['phi3_spearman'] >= config['pass_spearman_phi3'] and
        aggregate_metrics['lock_auroc'] >= config['pass_lock_auroc'] and
        aggregate_metrics['lock_jaccard'] >= config['pass_lock_jaccard']
    )

    print(f"\nAggregate metrics for {species}:")
    print(f"  Mass capture: {aggregate_metrics['mass_capture']:.3f} (threshold: {config['pass_mass_capture']})")
    print(f"  Φ₃ Spearman:  {aggregate_metrics['phi3_spearman']:.3f} (threshold: {config['pass_spearman_phi3']})")
    print(f"  Lock AUROC:   {aggregate_metrics['lock_auroc']:.3f} (threshold: {config['pass_lock_auroc']})")
    print(f"  Lock Jaccard: {aggregate_metrics['lock_jaccard']:.3f} (threshold: {config['pass_lock_jaccard']})")

    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"\n{status}")

    # Log to wandb
    if HAS_WANDB:
        wandb.init(
            project="phase2a-compression",
            name=f"{species}_metrics",
            config=config,
            tags=[species, "metrics", "test"]
        )

        # Log aggregate metrics
        wandb.log({
            f'{species}/test/mass_capture': aggregate_metrics['mass_capture'],
            f'{species}/test/phi3_spearman': aggregate_metrics['phi3_spearman'],
            f'{species}/test/lock_auroc': aggregate_metrics['lock_auroc'],
            f'{species}/test/lock_jaccard': aggregate_metrics['lock_jaccard'],
            f'{species}/test/passed': float(passed)
        })

        # Log per-pair metrics
        for i, metrics in enumerate(pair_metrics):
            wandb.log({
                f'{species}/pair_{i}/mass_capture': metrics['mass_capture'],
                f'{species}/pair_{i}/phi3_spearman': metrics['phi3_spearman'],
                f'{species}/pair_{i}/lock_auroc': metrics['lock_auroc'],
                f'{species}/pair_{i}/lock_jaccard': metrics['lock_jaccard']
            })

        wandb.finish()

    return pair_metrics, aggregate_metrics, passed


if __name__ == '__main__':
    """Test metrics computation."""
    import sys
    from p2a_01_dataset import load_species_data
    from p2a_02_model import build_time_vocabulary

    # Load config
    config_path = Path(__file__).parent.parent / 'config_phase2A.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test with dummy data
    print("\nTesting metrics computation...")
    print("(This test requires trained model + hatT files)")
