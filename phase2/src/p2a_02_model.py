"""
Phase 2A - Model Architecture
Encoder + Time Embeddings + Lock Head
Optimized for GH200 GPU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SharedEncoder(nn.Module):
    """
    Shared encoder for source and target cells.

    Architecture (from spec):
      Input: 50 (PCA)
      Hidden: 256 → 256
      Output: 64
      Activation: GELU
      LayerNorm after each hidden layer
      Dropout: 0.0
      Output: L2-normalized
    """

    def __init__(self, config: dict):
        super().__init__()

        pca_dim = config['pca_dim']
        hidden_dim = config['mlp_hidden']
        embed_dim = config['embed_dim']
        n_layers = config['mlp_layers']
        dropout = config['dropout']
        use_layernorm = config['layernorm']

        # Build MLP layers
        layers = []

        # Input layer
        layers.append(nn.Linear(pca_dim, hidden_dim))
        layers.append(nn.GELU())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, embed_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, pca_dim) or (batch, n_samples, pca_dim)

        Returns:
            embeddings: (batch, embed_dim) or (batch, n_samples, embed_dim), NOT normalized
                        (normalization happens after time embedding is added)
        """
        # Handle both 2D and 3D inputs
        input_shape = x.shape
        if x.ndim == 3:
            # Flatten batch and sample dimensions
            batch, n_samples, pca_dim = x.shape
            x = x.reshape(batch * n_samples, pca_dim)

        # Encode (no normalization yet - spec says "Added before normalization")
        z = self.mlp(x)  # (batch [* n_samples], embed_dim)

        # Restore original shape if needed
        if len(input_shape) == 3:
            z = z.reshape(input_shape[0], input_shape[1], -1)

        return z


class TimeEmbedding(nn.Module):
    """
    Learned time embeddings for each unique timepoint.
    Added to cell embeddings before normalization.
    """

    def __init__(self, num_timepoints: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_timepoints, embed_dim) * 0.01)

    def forward(self, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_indices: (batch,) integer indices

        Returns:
            time_embeds: (batch, embed_dim)
        """
        return self.embeddings[time_indices]


class LockHead(nn.Module):
    """
    Linear head for predicting lock probability from source embedding.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, source_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source_embed: (batch, embed_dim)

        Returns:
            logits: (batch,) scalar logits
        """
        return self.fc(source_embed).squeeze(-1)


class Phase2Model(nn.Module):
    """
    Complete Phase 2A model:
      - Shared encoder for source and target cells
      - Time embeddings
      - Lock prediction head
    """

    def __init__(self, config: dict, time_vocab: Dict[str, int]):
        """
        Args:
            config: Configuration dict
            time_vocab: Mapping from time string to integer index
        """
        super().__init__()

        self.config = config
        self.time_vocab = time_vocab
        self.tau = config['temperature_tau']

        # Components
        self.encoder = SharedEncoder(config)
        self.time_embed = TimeEmbedding(len(time_vocab), config['embed_dim'])
        self.lock_head = LockHead(config['embed_dim'])

    def encode_source(self, x: torch.Tensor, time_str: str) -> torch.Tensor:
        """
        Encode source cells with time embedding.

        Args:
            x: (batch, pca_dim)
            time_str: time identifier string

        Returns:
            embeddings: (batch, embed_dim), L2-normalized
        """
        # Encode PCA features (NOT normalized yet)
        z = self.encoder(x)  # (batch, embed_dim)

        # Add time embedding BEFORE normalization (per spec)
        time_idx = self.time_vocab[time_str]
        time_indices = torch.full((x.shape[0],), time_idx, dtype=torch.long, device=x.device)
        t_emb = self.time_embed(time_indices)  # (batch, embed_dim)

        z = z + t_emb

        # NOW normalize
        z = F.normalize(z, p=2, dim=-1)

        return z

    def encode_target(self, y: torch.Tensor, time_str: str) -> torch.Tensor:
        """
        Encode target cells with time embedding.

        Args:
            y: (batch, n_targets, pca_dim) or (batch, pca_dim)
            time_str: time identifier string

        Returns:
            embeddings: (batch, n_targets, embed_dim) or (batch, embed_dim), L2-normalized
        """
        # Encode (NOT normalized yet)
        f = self.encoder(y)

        # Add time embedding BEFORE normalization (per spec)
        time_idx = self.time_vocab[time_str]
        if y.ndim == 3:
            batch, n_targets, _ = y.shape
            time_indices = torch.full((batch * n_targets,), time_idx, dtype=torch.long, device=y.device)
            t_emb = self.time_embed(time_indices)  # (batch * n_targets, embed_dim)
            t_emb = t_emb.reshape(batch, n_targets, -1)

            f = f + t_emb
            # NOW normalize
            f = F.normalize(f, p=2, dim=-1)
        else:
            time_indices = torch.full((y.shape[0],), time_idx, dtype=torch.long, device=y.device)
            t_emb = self.time_embed(time_indices)
            f = f + t_emb
            # NOW normalize
            f = F.normalize(f, p=2, dim=-1)

        return f

    def compute_similarity(self, source_embed: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product similarity.

        Args:
            source_embed: (batch, embed_dim)
            target_embed: (batch, n_targets, embed_dim)

        Returns:
            similarity: (batch, n_targets) scaled by temperature
        """
        # s(i,j) = e_i^T f_j / tau
        # source_embed: (batch, embed_dim)
        # target_embed: (batch, n_targets, embed_dim)

        sim = torch.einsum('be,bne->bn', source_embed, target_embed)  # (batch, n_targets)
        sim = sim / self.tau

        return sim

    def forward(
        self,
        source_pca: torch.Tensor,
        target_pos_pca: torch.Tensor,
        target_neg_pca: torch.Tensor,
        source_time: str,
        target_time: str
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            source_pca: (batch, pca_dim)
            target_pos_pca: (batch, n_pos, pca_dim)
            target_neg_pca: (batch, n_neg, pca_dim)
            source_time: source timepoint string
            target_time: target timepoint string

        Returns:
            dict with:
                - source_embed: (batch, embed_dim)
                - target_pos_embed: (batch, n_pos, embed_dim)
                - target_neg_embed: (batch, n_neg, embed_dim)
                - sim_pos: (batch, n_pos)
                - sim_neg: (batch, n_neg)
                - lock_logits: (batch,)
        """
        # Encode source
        e = self.encode_source(source_pca, source_time)  # (batch, embed_dim)

        # Encode targets
        f_pos = self.encode_target(target_pos_pca, target_time)  # (batch, n_pos, embed_dim)
        f_neg = self.encode_target(target_neg_pca, target_time)  # (batch, n_neg, embed_dim)

        # Compute similarities
        sim_pos = self.compute_similarity(e, f_pos)  # (batch, n_pos)
        sim_neg = self.compute_similarity(e, f_neg)  # (batch, n_neg)

        # Lock prediction
        lock_logits = self.lock_head(e)  # (batch,)

        return {
            'source_embed': e,
            'target_pos_embed': f_pos,
            'target_neg_embed': f_neg,
            'sim_pos': sim_pos,
            'sim_neg': sim_neg,
            'lock_logits': lock_logits
        }

    def encode_batch_for_inference(
        self,
        pca: torch.Tensor,
        time_str: str,
        is_source: bool = True
    ) -> torch.Tensor:
        """
        Encode a batch of cells for inference (building hatT).

        Args:
            pca: (batch, pca_dim)
            time_str: time identifier
            is_source: whether these are source or target cells

        Returns:
            embeddings: (batch, embed_dim)
        """
        if is_source:
            return self.encode_source(pca, time_str)
        else:
            return self.encode_target(pca, time_str)


def build_time_vocabulary(pairs) -> Dict[str, int]:
    """
    Build vocabulary mapping time strings to integer indices.

    Args:
        pairs: List of TransitionPairData objects

    Returns:
        time_vocab: Dict mapping time string to integer index
    """
    unique_times = set()
    for pair in pairs:
        unique_times.add(pair.source_time)
        unique_times.add(pair.target_time)

    time_vocab = {t: i for i, t in enumerate(sorted(unique_times))}
    return time_vocab


if __name__ == '__main__':
    """Test model architecture."""
    import json
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent.parent / 'config_phase2A.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create dummy time vocabulary
    time_vocab = {'E6_5': 0, 'E6_75': 1, 'E7_0': 2}

    # Create model
    model = Phase2Model(config, time_vocab)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 32
    n_pos = config['positives_per_source']
    n_neg = config['negatives_gpu']
    pca_dim = config['pca_dim']

    source_pca = torch.randn(batch_size, pca_dim)
    target_pos_pca = torch.randn(batch_size, n_pos, pca_dim)
    target_neg_pca = torch.randn(batch_size, n_neg, pca_dim)

    # Forward
    outputs = model(source_pca, target_pos_pca, target_neg_pca, 'E6_5', 'E6_75')

    print("\nForward pass outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # Check normalization
    source_norms = torch.norm(outputs['source_embed'], p=2, dim=-1)
    print(f"\nSource embedding L2 norms: min={source_norms.min():.4f}, max={source_norms.max():.4f}")

    print("\n✓ Model test passed")
