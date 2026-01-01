"""
Prepare Phase 1 data for Phase 2A training.
Copies data from project/results/phase2_data/ to project/phase2/data_phase1/
with proper naming conventions.
"""
import json
import pickle
import shutil
from pathlib import Path
import numpy as np


def prepare_species_data(source_dir: Path, target_dir: Path, species: str):
    """Prepare data for one species."""

    # Find all transition directories
    transition_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])

    print(f"\n{species}: Found {len(transition_dirs)} transitions")

    for idx, trans_dir in enumerate(transition_dirs):
        pair_name = f"pair_{idx}"
        pair_target = target_dir / pair_name
        pair_target.mkdir(parents=True, exist_ok=True)

        print(f"  {pair_name}: {trans_dir.name}")

        # Copy files with proper names
        files_to_copy = {
            'X_pca.npy': 'X_pca.npy',
            'Y_pca.npy': 'Y_pca.npy',
            'T_sparse.npz': 'T_sparse.npz',
            'phi3.npy': 'phi3_bits.npy',  # Rename phi3 to phi3_bits
            'phi2.npy': 'phi2.npy',
            'lock_label.npy': 'lock_label.npy',
        }

        for src_name, dst_name in files_to_copy.items():
            src = trans_dir / src_name
            dst = pair_target / dst_name
            if src.exists():
                shutil.copy2(src, dst)
            else:
                print(f"    WARNING: Missing {src_name}")

        # Convert metadata.pkl to time_id.json
        metadata_path = trans_dir / 'metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Extract time information from directory name
            # e.g., "E6_5_to_E6_75" or "4hpf_to_6hpf"
            trans_name = trans_dir.name
            parts = trans_name.split('_to_')
            if len(parts) == 2:
                source_time = parts[0]
                target_time = parts[1]
            else:
                # Fallback
                source_time = metadata.get('source_time', 'unknown')
                target_time = metadata.get('target_time', 'unknown')

            time_id = {
                'source_time': source_time,
                'target_time': target_time,
                'transition_name': trans_name
            }

            with open(pair_target / 'time_id.json', 'w') as f:
                json.dump(time_id, f, indent=2)
        else:
            print(f"    WARNING: Missing metadata.pkl")


def main():
    """Main preparation entry point."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent  # Go up to project/

    source_base = project_dir / 'results' / 'phase2_data'
    target_base = project_dir / 'phase2' / 'data_phase1'

    print("=" * 80)
    print("Preparing Phase 1 Data for Phase 2A")
    print("=" * 80)
    print(f"Source: {source_base}")
    print(f"Target: {target_base}")

    if not source_base.exists():
        print(f"\nERROR: Source directory not found: {source_base}")
        return 1

    # Create target directories
    target_base.mkdir(parents=True, exist_ok=True)

    # Prepare mouse data
    mouse_src = source_base / 'mouse'
    mouse_dst = target_base / 'mouse'
    if mouse_src.exists():
        prepare_species_data(mouse_src, mouse_dst, 'mouse')
    else:
        print(f"\nWARNING: Mouse data not found at {mouse_src}")

    # Prepare zebrafish data
    zfish_src = source_base / 'zebrafish'
    zfish_dst = target_base / 'zebrafish'
    if zfish_src.exists():
        prepare_species_data(zfish_src, zfish_dst, 'zebrafish')
    else:
        print(f"\nWARNING: Zebrafish data not found at {zfish_src}")

    print("\n" + "=" * 80)
    print("✓ Data preparation complete")
    print("=" * 80)
    return 0


if __name__ == '__main__':
    exit(main())
