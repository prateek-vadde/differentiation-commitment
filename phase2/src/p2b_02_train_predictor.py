"""
Phase 2B: Train Commitment Predictor

Train MLP regressor: X_pca → C

Critical rules:
- Split by PAIRS, never mix cells from same pair across splits
- Normalize on train set only, apply to val/test
- Train separate models per species
- Report Pearson + Spearman on test set
- Generate calibration plot data

Outputs per species:
- models/<species>/p2b_regressor.pt
- models/<species>/p2b_regressor_normalization.json
- results/<species>/p2b_regressor_metrics.json
- results/<species>/p2b_regressor_history.json
- results/<species>/p2b_regressor_split.json
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


class CommitmentMLP(nn.Module):
    """MLP regressor for commitment score prediction."""

    def __init__(self, input_dim, hidden_dim, num_layers, activation='gelu', layernorm=True):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self._get_activation(activation))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(activation))

        # Output layer (no activation - regression)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == 'gelu':
            return nn.GELU()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x):
        return self.network(x)


def split_pairs(num_pairs, train_frac, val_frac, test_frac, seed):
    """
    Split pair indices into train/val/test deterministically.

    Args:
        num_pairs: Total number of pairs
        train_frac, val_frac, test_frac: Split fractions (must sum to 1.0)
        seed: Random seed for reproducibility

    Returns:
        train_pairs, val_pairs, test_pairs: Lists of pair indices
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    rng = np.random.RandomState(seed)

    # Shuffle pair indices
    pair_indices = np.arange(num_pairs)
    rng.shuffle(pair_indices)

    # Split - ensure at least 1 in val and test if possible
    n_train = int(num_pairs * train_frac)
    n_val = int(num_pairs * val_frac)
    n_test = num_pairs - n_train - n_val

    # Edge case: if val or test is 0 due to rounding, steal from train
    if n_val == 0 and num_pairs >= 3:
        n_val = 1
        n_train -= 1
    if n_test == 0 and num_pairs >= 3:
        n_test = 1
        n_train -= 1

    train_pairs = pair_indices[:n_train].tolist()
    val_pairs = pair_indices[n_train:n_train + n_val].tolist()
    test_pairs = pair_indices[n_train + n_val:].tolist()

    return train_pairs, val_pairs, test_pairs


def load_dataset(species, pair_indices, data_root, results_root):
    """
    Load X_pca and C for all cells in specified pairs.

    Args:
        species: 'mouse' or 'zebrafish'
        pair_indices: List of pair indices to load
        data_root: Path to data_phase2A
        results_root: Path to results_p2b

    Returns:
        X: Features (N x d)
        C: Commitment scores (N,)
        pair_labels: Pair index for each cell (N,)
    """
    X_list = []
    C_list = []
    pair_labels_list = []

    for pair_idx in pair_indices:
        # Load X_pca
        X_path = data_root / species / f'pair_{pair_idx}' / 'X_pca.npy'
        X_pair = np.load(X_path)

        # Load C
        C_path = results_root / species / f'pair_{pair_idx}' / 'C.npy'
        C_pair = np.load(C_path)

        # Validate dimensions
        assert len(X_pair) == len(C_pair), \
            f"Pair {pair_idx}: X_pca ({len(X_pair)}) != C ({len(C_pair)})"

        X_list.append(X_pair)
        C_list.append(C_pair)
        pair_labels_list.append(np.full(len(C_pair), pair_idx))

    X = np.vstack(X_list)
    C = np.concatenate(C_list)
    pair_labels = np.concatenate(pair_labels_list)

    return X, C, pair_labels


def normalize_features(X_train, X_val, X_test):
    """
    Compute normalization stats on train, apply to all splits.

    Returns:
        X_train_norm, X_val_norm, X_test_norm, mean, std
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # Avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


def train_epoch(model, loader, optimizer, criterion, grad_clip, device):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for X_batch, C_batch in loader:
        X_batch = X_batch.to(device)
        C_batch = C_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch).squeeze()
        loss = criterion(pred, C_batch)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, C_batch in loader:
            X_batch = X_batch.to(device)
            C_batch = C_batch.to(device)

            pred = model(X_batch).squeeze()
            loss = criterion(pred, C_batch)
            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def compute_test_metrics(model, X_test, C_test, device):
    """
    Compute Pearson, Spearman, and calibration on test set.

    Returns:
        metrics dict with pearson_r, pearson_p, spearman_r, spearman_p, calibration
    """
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        C_pred = model(X_test_tensor).squeeze().cpu().numpy()

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(C_test, C_pred)

    # Spearman correlation
    spearman_r, spearman_p = spearmanr(C_test, C_pred)

    # Calibration: bin by predicted C deciles
    deciles = np.percentile(C_pred, np.arange(0, 101, 10))
    decile_bins = np.digitize(C_pred, deciles[1:-1])

    mean_pred_per_bin = []
    mean_true_per_bin = []

    for bin_idx in range(10):
        mask = decile_bins == bin_idx
        if mask.sum() > 0:
            mean_pred_per_bin.append(float(C_pred[mask].mean()))
            mean_true_per_bin.append(float(C_test[mask].mean()))
        else:
            mean_pred_per_bin.append(None)
            mean_true_per_bin.append(None)

    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'calibration': {
            'decile_edges': deciles.tolist(),
            'mean_predicted': mean_pred_per_bin,
            'mean_true': mean_true_per_bin
        }
    }


def train_species(species, config, data_root, results_root, models_root):
    """
    Train commitment predictor for one species.

    Args:
        species: 'mouse' or 'zebrafish'
        config: Configuration dict
        data_root: Path to data_phase2A
        results_root: Path to results_p2b
        models_root: Path to models directory
    """
    print(f"\n{'='*60}")
    print(f"Training regressor for {species}")
    print(f"{'='*60}")

    # Find all pairs
    species_dir = results_root / species
    pair_dirs = sorted([d for d in species_dir.iterdir()
                       if d.is_dir() and d.name.startswith('pair_')])
    num_pairs = len(pair_dirs)

    print(f"Found {num_pairs} pairs")

    # Split pairs
    seed = config['seed']
    split_config = config['split_by_pairs']

    train_pairs, val_pairs, test_pairs = split_pairs(
        num_pairs,
        split_config['train_frac'],
        split_config['val_frac'],
        split_config['test_frac'],
        seed
    )

    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    print(f"  Train pairs: {train_pairs}")
    print(f"  Val pairs: {val_pairs}")
    print(f"  Test pairs: {test_pairs}")

    # Load datasets
    print("\nLoading datasets...")
    X_train, C_train, train_labels = load_dataset(species, train_pairs, data_root, results_root)
    X_val, C_val, val_labels = load_dataset(species, val_pairs, data_root, results_root)
    X_test, C_test, test_labels = load_dataset(species, test_pairs, data_root, results_root)

    print(f"  Train: {len(X_train)} cells")
    print(f"  Val: {len(X_val)} cells")
    print(f"  Test: {len(X_test)} cells")
    print(f"  Feature dim: {X_train.shape[1]}")

    # Normalize
    print("\nNormalizing features (stats from train only)...")
    X_train_norm, X_val_norm, X_test_norm, mean, std = normalize_features(
        X_train, X_val, X_test
    )

    # Create DataLoaders
    reg_config = config['regressor']
    batch_size = reg_config['batch_size']

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_norm).float(),
        torch.from_numpy(C_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_norm).float(),
        torch.from_numpy(C_val).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    input_dim = X_train.shape[1]
    model = CommitmentMLP(
        input_dim=input_dim,
        hidden_dim=reg_config['mlp_hidden'],
        num_layers=reg_config['mlp_layers'],
        activation=reg_config['activation'],
        layernorm=reg_config['layernorm']
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=reg_config['lr'],
        weight_decay=reg_config['weight_decay']
    )
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for {reg_config['epochs']} epochs...")

    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(reg_config['epochs']):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            reg_config['grad_clip'], device
        )
        val_loss = eval_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f} "
                  f"{'*' if val_loss == best_val_loss else ''}")

    print(f"\nBest val loss: {best_val_loss:.6f}")

    # Load best model for test evaluation
    model.load_state_dict(best_model_state)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = compute_test_metrics(model, X_test_norm, C_test, device)

    print(f"  Pearson r: {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p']:.2e})")
    print(f"  Spearman r: {test_metrics['spearman_r']:.4f} (p={test_metrics['spearman_p']:.2e})")

    # Save outputs
    models_species_dir = models_root / species
    results_species_dir = results_root / species
    models_species_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_species_dir / 'p2b_regressor.pt'
    torch.save({
        'model_state_dict': best_model_state,
        'input_dim': input_dim,
        'config': reg_config
    }, model_path)
    print(f"\n✓ Model saved: {model_path}")

    # Save normalization stats
    norm_path = models_species_dir / 'p2b_regressor_normalization.json'
    with open(norm_path, 'w') as f:
        json.dump({
            'mean': mean.tolist(),
            'std': std.tolist()
        }, f, indent=2)
    print(f"✓ Normalization stats saved: {norm_path}")

    # Save split info
    split_path = results_species_dir / 'p2b_regressor_split.json'
    with open(split_path, 'w') as f:
        json.dump({
            'train_pairs': train_pairs,
            'val_pairs': val_pairs,
            'test_pairs': test_pairs,
            'train_cells': int(len(X_train)),
            'val_cells': int(len(X_val)),
            'test_cells': int(len(X_test))
        }, f, indent=2)
    print(f"✓ Split info saved: {split_path}")

    # Save training history
    history_path = results_species_dir / 'p2b_regressor_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")

    # Save test metrics
    metrics_path = results_species_dir / 'p2b_regressor_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"✓ Test metrics saved: {metrics_path}")

    print(f"\n[{species}] ✓ DONE\n")


def main():
    """Train regressors for all species."""
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')
    data_root = project_root / 'data_phase2A'
    results_root = project_root / 'results_p2b'
    models_root = project_root / 'models'

    config_path = project_root / 'config_p2b_p3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Phase 2B: Train Commitment Predictor")
    print(f"Seed: {config['seed']}")
    print(f"Split: {config['split_by_pairs']}")
    print(f"Model: MLP {config['regressor']}")

    # Train for each species
    for species in ['mouse', 'zebrafish']:
        train_species(species, config, data_root, results_root, models_root)

    print("✓ ALL SPECIES COMPLETE\n")


if __name__ == '__main__':
    main()
