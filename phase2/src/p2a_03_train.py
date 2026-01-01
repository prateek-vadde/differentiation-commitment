"""
Phase 2A - Training Loop
Implements multi-positive InfoNCE + upgraded Φ₃ loss + lock loss
Optimized for GH200
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.stats import rankdata
import numpy as np
from tqdm import tqdm
import sys

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not available, skipping logging")

from p2a_01_dataset import load_species_data, create_dataloaders
from p2a_02_model import Phase2Model, build_time_vocabulary
from sklearn.metrics import roc_auc_score


class Phase2Trainer:
    """Trainer for Phase 2A model."""

    def __init__(self, model: Phase2Model, config: dict, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # Loss weights
        self.w_phi3 = config['loss_w_phi3']
        self.w_lock = config['loss_w_lock']

        # Gradient clipping
        self.grad_clip = config['grad_clip']

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def compute_multi_positive_infonce(
        self,
        sim_pos: torch.Tensor,
        sim_neg: torch.Tensor,
        T_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-positive InfoNCE loss.

        Args:
            sim_pos: (batch, P) similarities to positives
            sim_neg: (batch, N) similarities to negatives
            T_weights: (batch, P) weights from T matrix

        Returns:
            loss: scalar
        """
        # Normalize T weights per source
        T_weights = T_weights / (T_weights.sum(dim=1, keepdim=True) + 1e-10)

        # Concatenate positive and negative similarities
        # sim_all: (batch, P + N)
        sim_all = torch.cat([sim_pos, sim_neg], dim=1)

        # Compute log-sum-exp denominator
        # log_sum_exp: (batch,)
        log_sum_exp = torch.logsumexp(sim_all, dim=1)

        # Weighted log probabilities for positives
        # log_probs: (batch, P)
        log_probs = sim_pos - log_sum_exp.unsqueeze(1)

        # Weighted loss
        loss = -(T_weights * log_probs).sum(dim=1).mean()

        return loss

    def compute_phi3_loss_upgraded(
        self,
        source_embed: torch.Tensor,
        target_embed: torch.Tensor,
        phi3_true: torch.Tensor,
        pair_indices: torch.Tensor,
        pairs
    ) -> torch.Tensor:
        """
        UPGRADED Φ₃ loss: compute from TopM entropy, not sampled-candidate entropy.

        Procedure:
        1. For each source, compute TopM targets using current embeddings
        2. Compute predicted entropy: Ĥ^TopM_i = -Σ_{j∈TopM(i)} T̂_{ij} log T̂_{ij}
        3. Convert both true Φ₃ and Ĥ^TopM to percentile ranks within pair
        4. Loss: MSE(rank_true, rank_pred)

        Args:
            source_embed: (batch, embed_dim)
            target_embed: (batch, n_targets, embed_dim) - ALL targets for each source
            phi3_true: (batch,) true Φ₃ values
            pair_indices: (batch,) pair index for each sample
            pairs: List of TransitionPairData objects

        Returns:
            loss: scalar
        """
        batch_size = source_embed.shape[0]

        # We need to compute TopM for each source in the batch
        # Problem: each source may belong to a different pair with different targets
        # Solution: process pair by pair

        unique_pairs = torch.unique(pair_indices)
        phi3_pred_list = []
        phi3_true_list = []

        for pair_idx in unique_pairs:
            # Get samples from this pair
            mask = (pair_indices == pair_idx)
            if not mask.any():
                continue

            pair = pairs[pair_idx.item()]
            src_embed = source_embed[mask]  # (n_samples, embed_dim)
            n_samples = src_embed.shape[0]

            # Get ALL target embeddings for this pair (need to encode them)
            # Actually, we can't do this efficiently in the batch...
            # We need access to all Y for this pair.
            # This is where we need to be careful.

            # For now, let's use a simplified approach: use the targets that were sampled
            # But this defeats the purpose of the upgrade...

            # CORRECT approach: we need to pass ALL target embeddings per pair
            # This means target_embed should be (batch, n_targets_for_pair, embed_dim)
            # But n_targets varies per pair... this is tricky.

            # For research-grade implementation, we need to handle this properly.
            # Let's compute similarities to ALL targets for this pair.

            # Get all target PCA for this pair
            Y_all = torch.from_numpy(pair.Y).to(self.device)  # (n_target, pca_dim)

            # Encode all targets (with proper time embedding)
            # We need the target time for this pair
            target_time = pair.target_time
            f_all = self.model.encode_target(Y_all, target_time)  # (n_target, embed_dim)

            # Compute similarities: (n_samples, n_target)
            sim_all = torch.matmul(src_embed, f_all.T) / self.model.tau

            # Get TopM per source (using pair-specific precomputed TopM)
            topM = min(pair.topM, f_all.shape[0])
            topM_sim, topM_indices = torch.topk(sim_all, k=topM, dim=1)  # (n_samples, topM)

            # Convert to probabilities (softmax over TopM)
            T_hat_topM = F.softmax(topM_sim, dim=1)  # (n_samples, topM)

            # Compute entropy
            # Ĥ^TopM_i = -Σ T̂_{ij} log T̂_{ij}
            H_pred = -(T_hat_topM * torch.log(T_hat_topM + 1e-10)).sum(dim=1)  # (n_samples,)

            # Get true Φ₃ for these samples
            phi3_pair = phi3_true[mask]  # (n_samples,)

            # Convert to percentile ranks within this pair
            phi3_true_rank = torch.from_numpy(
                rankdata(phi3_pair.cpu().numpy(), method='average') / len(phi3_pair)
            ).float().to(self.device)

            H_pred_rank = torch.from_numpy(
                rankdata(H_pred.detach().cpu().numpy(), method='average') / len(H_pred)
            ).float().to(self.device)

            phi3_pred_list.append(H_pred_rank)
            phi3_true_list.append(phi3_true_rank)

        if len(phi3_pred_list) == 0:
            return torch.tensor(0.0, device=self.device)

        # Concatenate all ranks
        phi3_pred_all = torch.cat(phi3_pred_list)
        phi3_true_all = torch.cat(phi3_true_list)

        # MSE loss on ranks
        loss = F.mse_loss(phi3_pred_all, phi3_true_all)

        return loss

    def compute_lock_loss(self, lock_logits: torch.Tensor, lock_labels: torch.Tensor) -> torch.Tensor:
        """
        Binary cross-entropy loss for lock prediction.

        Args:
            lock_logits: (batch,) predicted logits
            lock_labels: (batch,) true labels (0 or 1)

        Returns:
            loss: scalar
        """
        return F.binary_cross_entropy_with_logits(lock_logits, lock_labels)

    def train_step(self, batch, pairs) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        source_pca = batch['source_emb'].to(self.device)
        target_pos_pca = batch['target_pos'].to(self.device)
        target_neg_pca = batch['target_neg'].to(self.device)
        T_weights = batch['T_weights'].to(self.device)
        phi3_true = batch['phi3'].to(self.device)
        lock_labels = batch['lock'].to(self.device)
        pair_indices = batch['pair_idx'].to(self.device)

        # Get time strings (assume all in batch have same source/target times for now)
        # For multi-pair batches, we need to handle this differently
        # For simplicity, process unique pair/time combinations

        # Actually, let's batch by pair to make this cleaner
        # But the dataloader mixes pairs...

        # For now, let's assume batch has mixed pairs and handle it properly
        # We need to group by (source_time, target_time) and forward separately

        # Get unique time combinations in batch
        source_times = batch['source_time']
        target_times = batch['target_time']

        # Since batch can have multiple pairs with different times, we need to process in groups
        # This is getting complex. Let me simplify: forward for each unique time combination

        unique_time_pairs = set(zip(source_times, target_times))

        all_sim_pos = []
        all_sim_neg = []
        all_lock_logits = []
        all_source_embeds = []
        all_T_weights = []
        all_phi3 = []
        all_pair_idx = []
        all_lock = []

        for src_time, tgt_time in unique_time_pairs:
            # Get mask for this time combination
            mask = [(s == src_time and t == tgt_time) for s, t in zip(source_times, target_times)]
            mask = torch.tensor(mask, device=self.device)

            if not mask.any():
                continue

            # Forward pass for this group
            outputs = self.model(
                source_pca[mask],
                target_pos_pca[mask],
                target_neg_pca[mask],
                src_time,
                tgt_time
            )

            all_sim_pos.append(outputs['sim_pos'])
            all_sim_neg.append(outputs['sim_neg'])
            all_lock_logits.append(outputs['lock_logits'])
            all_source_embeds.append(outputs['source_embed'])
            all_T_weights.append(T_weights[mask])
            all_phi3.append(phi3_true[mask])
            all_pair_idx.append(pair_indices[mask])
            all_lock.append(lock_labels[mask])

        # Concatenate all outputs
        sim_pos = torch.cat(all_sim_pos, dim=0)
        sim_neg = torch.cat(all_sim_neg, dim=0)
        lock_logits = torch.cat(all_lock_logits, dim=0)
        source_embeds = torch.cat(all_source_embeds, dim=0)
        T_weights_all = torch.cat(all_T_weights, dim=0)
        phi3_all = torch.cat(all_phi3, dim=0)
        pair_idx_all = torch.cat(all_pair_idx, dim=0)
        lock_all = torch.cat(all_lock, dim=0)

        # Compute losses
        loss_nce = self.compute_multi_positive_infonce(sim_pos, sim_neg, T_weights_all)
        loss_phi3 = self.compute_phi3_loss_upgraded(source_embeds, None, phi3_all, pair_idx_all, pairs)
        loss_lock = self.compute_lock_loss(lock_logits, lock_all)

        # Total loss
        loss = loss_nce + self.w_phi3 * loss_phi3 + self.w_lock * loss_lock

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'loss_nce': loss_nce.item(),
            'loss_phi3': loss_phi3.item(),
            'loss_lock': loss_lock.item()
        }

    @torch.no_grad()
    def val_step(self, batch, pairs) -> dict:
        """Single validation step."""
        self.model.eval()

        # Same as train_step but no gradients
        source_pca = batch['source_emb'].to(self.device)
        target_pos_pca = batch['target_pos'].to(self.device)
        target_neg_pca = batch['target_neg'].to(self.device)
        T_weights = batch['T_weights'].to(self.device)
        phi3_true = batch['phi3'].to(self.device)
        lock_labels = batch['lock'].to(self.device)
        pair_indices = batch['pair_idx'].to(self.device)

        source_times = batch['source_time']
        target_times = batch['target_time']
        unique_time_pairs = set(zip(source_times, target_times))

        all_sim_pos = []
        all_sim_neg = []
        all_lock_logits = []
        all_source_embeds = []
        all_T_weights = []
        all_phi3 = []
        all_pair_idx = []
        all_lock = []

        for src_time, tgt_time in unique_time_pairs:
            mask = [(s == src_time and t == tgt_time) for s, t in zip(source_times, target_times)]
            mask = torch.tensor(mask, device=self.device)

            if not mask.any():
                continue

            outputs = self.model(
                source_pca[mask],
                target_pos_pca[mask],
                target_neg_pca[mask],
                src_time,
                tgt_time
            )

            all_sim_pos.append(outputs['sim_pos'])
            all_sim_neg.append(outputs['sim_neg'])
            all_lock_logits.append(outputs['lock_logits'])
            all_source_embeds.append(outputs['source_embed'])
            all_T_weights.append(T_weights[mask])
            all_phi3.append(phi3_true[mask])
            all_pair_idx.append(pair_indices[mask])
            all_lock.append(lock_labels[mask])

        sim_pos = torch.cat(all_sim_pos, dim=0)
        sim_neg = torch.cat(all_sim_neg, dim=0)
        lock_logits = torch.cat(all_lock_logits, dim=0)
        source_embeds = torch.cat(all_source_embeds, dim=0)
        T_weights_all = torch.cat(all_T_weights, dim=0)
        phi3_all = torch.cat(all_phi3, dim=0)
        pair_idx_all = torch.cat(all_pair_idx, dim=0)
        lock_all = torch.cat(all_lock, dim=0)

        loss_nce = self.compute_multi_positive_infonce(sim_pos, sim_neg, T_weights_all)
        loss_phi3 = self.compute_phi3_loss_upgraded(source_embeds, None, phi3_all, pair_idx_all, pairs)
        loss_lock = self.compute_lock_loss(lock_logits, lock_all)

        loss = loss_nce + self.w_phi3 * loss_phi3 + self.w_lock * loss_lock

        # Compute validation AUROC for lock prediction
        lock_probs = torch.sigmoid(lock_logits).cpu().numpy()
        lock_labels_np = lock_all.cpu().numpy()

        # Handle edge case: all same class
        if len(np.unique(lock_labels_np)) < 2:
            lock_auroc = 1.0 if lock_labels_np[0] else 0.0
        else:
            lock_auroc = roc_auc_score(lock_labels_np.astype(int), lock_probs)

        return {
            'loss': loss.item(),
            'loss_nce': loss_nce.item(),
            'loss_phi3': loss_phi3.item(),
            'loss_lock': loss_lock.item(),
            'lock_auroc': lock_auroc
        }

    def train_epoch(self, train_loader, pairs) -> dict:
        """Train for one epoch."""
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config['epochs']}")
        for step, batch in enumerate(pbar):
            metrics = self.train_step(batch, pairs)
            epoch_losses.append(metrics)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'nce': f"{metrics['loss_nce']:.4f}",
                'phi3': f"{metrics['loss_phi3']:.4f}",
                'lock': f"{metrics['loss_lock']:.4f}"
            })

            # Log to wandb every 10 steps
            if HAS_WANDB and step % 10 == 0:
                wandb.log({
                    'train/step_loss': metrics['loss'],
                    'train/step_nce': metrics['loss_nce'],
                    'train/step_phi3': metrics['loss_phi3'],
                    'train/step_lock': metrics['loss_lock'],
                    'epoch': self.current_epoch,
                    'step': self.current_epoch * len(train_loader) + step
                })

        # Average metrics
        avg_metrics = {k: np.mean([m[k] for m in epoch_losses]) for k in epoch_losses[0].keys()}
        return avg_metrics

    @torch.no_grad()
    def validate(self, val_loader, pairs) -> dict:
        """Validate."""
        epoch_losses = []

        for batch in val_loader:
            metrics = self.val_step(batch, pairs)
            epoch_losses.append(metrics)

        avg_metrics = {k: np.mean([m[k] for m in epoch_losses]) for k in epoch_losses[0].keys()}
        return avg_metrics

    def save_checkpoint(self, save_path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, save_path)
        if is_best:
            torch.save(checkpoint, save_path.parent / 'best_model.pt')


def train_species(species: str, config: dict, device: str, save_dir: Path):
    """
    Train model for one species.

    Args:
        species: 'mouse' or 'zebrafish'
        config: Configuration dict
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
    """
    print(f"\n{'='*80}")
    print(f"Training {species} model")
    print(f"{'='*80}")

    # Initialize wandb
    if HAS_WANDB:
        wandb.init(
            project="phase2a-compression",
            name=f"{species}_seed{config['seed']}",
            config=config,
            tags=[species, f"epochs_{config['epochs']}", f"batch_{config['batch_gpu']}"]
        )

    # Load data
    data_dir = save_dir.parent / 'data_phase1'
    pairs, splits = load_species_data(data_dir / species, species, config, device)

    # Build time vocabulary
    time_vocab = build_time_vocabulary(pairs)
    print(f"Time vocabulary: {time_vocab}")

    # Create dataloaders
    loaders = create_dataloaders(pairs, splits, config, num_workers=8)

    # Create model
    model = Phase2Model(config, time_vocab)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Log model architecture to wandb
    if HAS_WANDB:
        wandb.watch(model, log='all', log_freq=100)

    # Create trainer
    trainer = Phase2Trainer(model, config, device)

    # Training loop
    for epoch in range(config['epochs']):
        trainer.current_epoch = epoch

        # Train
        train_metrics = trainer.train_epoch(loaders['train'], pairs)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, NCE: {train_metrics['loss_nce']:.4f}, "
              f"Phi3: {train_metrics['loss_phi3']:.4f}, Lock: {train_metrics['loss_lock']:.4f}")

        # Validate
        val_metrics = trainer.validate(loaders['val'], pairs)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, NCE: {val_metrics['loss_nce']:.4f}, "
              f"Phi3: {val_metrics['loss_phi3']:.4f}, Lock: {val_metrics['loss_lock']:.4f}, "
              f"AUROC: {val_metrics['lock_auroc']:.4f}")

        # Log epoch metrics to wandb
        if HAS_WANDB:
            wandb.log({
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_nce': train_metrics['loss_nce'],
                'train/epoch_phi3': train_metrics['loss_phi3'],
                'train/epoch_lock': train_metrics['loss_lock'],
                'val/loss': val_metrics['loss'],
                'val/nce': val_metrics['loss_nce'],
                'val/phi3': val_metrics['loss_phi3'],
                'val/lock': val_metrics['loss_lock'],
                'val/lock_auroc': val_metrics['lock_auroc'],
                'val/best_nce': trainer.best_val_loss,
                'epoch': epoch
            })

        # Save checkpoint
        is_best = val_metrics['loss_nce'] < trainer.best_val_loss  # Use NCE as primary metric per spec
        if is_best:
            trainer.best_val_loss = val_metrics['loss_nce']
            if HAS_WANDB:
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_val_nce'] = trainer.best_val_loss

        save_path = save_dir / species / f'checkpoint_epoch{epoch+1}.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(save_path, is_best=is_best)

    print(f"\n✓ Training complete for {species}")
    print(f"Best validation NCE: {trainer.best_val_loss:.4f}")

    if HAS_WANDB:
        wandb.finish()

    return trainer.model, time_vocab


if __name__ == '__main__':
    # Load config
    config_path = Path(__file__).parent.parent / 'config_phase2A.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    save_dir = Path(__file__).parent.parent / 'models'

    # Train mouse
    mouse_model, mouse_time_vocab = train_species('mouse', config, device, save_dir)

    # Train zebrafish
    # zfish_model, zfish_time_vocab = train_species('zebrafish', config, device, save_dir)

    print("\n" + "="*80)
    print("✓ All training complete")
    print("="*80)
