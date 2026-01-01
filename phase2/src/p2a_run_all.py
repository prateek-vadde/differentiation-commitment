"""
Phase 2A - Main Orchestrator
Runs complete pipeline: validate → train → build hatT → compute metrics → generate report
"""
import json
import sys
import torch
import numpy as np
from pathlib import Path

# Import all pipeline components
from p2a_00_validate_inputs import main as validate_inputs
from p2a_01_dataset import load_species_data, create_dataloaders
from p2a_02_model import Phase2Model, build_time_vocabulary
from p2a_03_train import train_species
from p2a_04_build_hatT import build_all_hatT
from p2a_05_metrics import compute_all_metrics_for_species
from p2a_06_report import generate_report


def load_or_train_species(species: str, config: dict, device: str, models_dir: Path, data_dir: Path):
    """Load existing best model or train from scratch."""
    best_model_path = models_dir / species / 'best_model.pt'

    if best_model_path.exists():
        print(f"\n✓ Found existing best model at {best_model_path}")
        print(f"  Skipping training, loading checkpoint...")

        # Load data to build time vocab
        pairs, splits = load_species_data(data_dir / species, species, config, device)
        time_vocab = build_time_vocabulary(pairs)

        # Create model and load checkpoint
        model = Phase2Model(config, time_vocab).to(device)
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"  Loaded model from epoch {checkpoint['epoch']} (val loss: {checkpoint['best_val_loss']:.4f})")
        return model, time_vocab
    else:
        print(f"\n  No existing model found, training from scratch...")
        return train_species(species, config, device, models_dir)


def main():
    """Main pipeline execution with robust error handling."""

    import time
    start_time = time.time()

    print("="*80)
    print("PHASE 2A - COMPLETE PIPELINE")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    config_path = project_dir / 'config_phase2A.json'
    data_dir = project_dir / 'data_phase1'
    models_dir = project_dir / 'models'
    results_dir = project_dir / 'results'

    # Create output directories
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"\nLoading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1

    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Step 0: Validate inputs
    print("\n" + "="*80)
    print("STEP 0: Validate Inputs")
    print("="*80)

    try:
        validate_inputs()
    except SystemExit as e:
        if e.code != 0:
            print("\n✗ Input validation FAILED")
            return 1
        print("✓ Input validation passed")

    # Step 1: Train Mouse Model
    print("\n" + "="*80)
    print("STEP 1: Train Mouse Model")
    print("="*80)

    try:
        mouse_model, mouse_time_vocab = load_or_train_species('mouse', config, device, models_dir, data_dir)
    except Exception as e:
        print(f"\n✗ Mouse training FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Build Mouse hatT
    print("\n" + "="*80)
    print("STEP 2: Build Mouse \hat{T}")
    print("="*80)

    try:
        mouse_pairs, mouse_splits = load_species_data(data_dir / 'mouse', 'mouse', config, device)
        build_all_hatT(mouse_model, mouse_pairs, 'mouse', config, device, results_dir)
    except Exception as e:
        print(f"\n✗ Mouse hatT construction FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Compute Mouse Metrics
    print("\n" + "="*80)
    print("STEP 3: Compute Mouse Metrics")
    print("="*80)

    try:
        mouse_pair_metrics, mouse_aggregate, mouse_passed = compute_all_metrics_for_species(
            mouse_model, mouse_pairs, mouse_splits, 'mouse', config, device, results_dir
        )
    except Exception as e:
        print(f"\n✗ Mouse metrics computation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Train Zebrafish Model
    print("\n" + "="*80)
    print("STEP 4: Train Zebrafish Model")
    print("="*80)

    try:
        zfish_model, zfish_time_vocab = load_or_train_species('zebrafish', config, device, models_dir, data_dir)
    except Exception as e:
        print(f"\n✗ Zebrafish training FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Build Zebrafish hatT
    print("\n" + "="*80)
    print("STEP 5: Build Zebrafish \hat{T}")
    print("="*80)

    try:
        zfish_pairs, zfish_splits = load_species_data(data_dir / 'zebrafish', 'zebrafish', config, device)
        build_all_hatT(zfish_model, zfish_pairs, 'zebrafish', config, device, results_dir)
    except Exception as e:
        print(f"\n✗ Zebrafish hatT construction FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 6: Compute Zebrafish Metrics
    print("\n" + "="*80)
    print("STEP 6: Compute Zebrafish Metrics")
    print("="*80)

    try:
        zfish_pair_metrics, zfish_aggregate, zfish_passed = compute_all_metrics_for_species(
            zfish_model, zfish_pairs, zfish_splits, 'zebrafish', config, device, results_dir
        )
    except Exception as e:
        print(f"\n✗ Zebrafish metrics computation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 7: Generate Report
    print("\n" + "="*80)
    print("STEP 7: Generate Summary Report")
    print("="*80)

    try:
        overall_passed = generate_report(
            mouse_pair_metrics, mouse_aggregate, mouse_passed,
            zfish_pair_metrics, zfish_aggregate, zfish_passed,
            config, results_dir
        )
    except Exception as e:
        print(f"\n✗ Report generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Final status
    elapsed_time = time.time() - start_time
    elapsed_mins = elapsed_time / 60

    print("\n" + "="*80)
    print("PHASE 2A COMPLETE")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed_mins:.1f} minutes")
    print("="*80)

    if overall_passed:
        print("\n✓✓✓ ALL GATES PASSED ✓✓✓")
        print(f"\nResults saved to: {results_dir}")
        print(f"Models saved to: {models_dir}")
        print(f"Report: {results_dir / 'summary_report.md'}")
        return 0
    else:
        print("\n✗✗✗ SOME GATES FAILED ✗✗✗")
        print(f"\nSee report for details: {results_dir / 'summary_report.md'}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
