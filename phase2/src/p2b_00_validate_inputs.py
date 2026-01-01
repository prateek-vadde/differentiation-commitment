"""
Phase 2B: Input Validation

Validates all input data according to strict Phase 2B/3 contract:
1. All required files present for each pair
2. Dimensional consistency within pairs
3. Row-stochasticity of T_hat matrices
4. Alignment across consecutive pairs for composition
5. TopM values are valid scalars

Outputs validation report to results/<species>/p2b_validation.json
"""

import json
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from common_io import build_alignment, validate_pairdata


def validate_species(species: str, data_root: Path, config: dict, results_root: Path):
    """
    Run all validation checks for one species.

    Args:
        species: 'mouse' or 'zebrafish'
        data_root: Path to data_phase2A directory
        config: Configuration dict
        results_root: Path to results directory

    Returns:
        validation_report: Dict with validation results
    """
    print(f"\n{'='*70}")
    print(f"Validating {species}")
    print(f"{'='*70}\n")

    report = {
        'species': species,
        'checks': [],
        'errors': [],
        'warnings': [],
        'passed': True
    }

    try:
        # Load and validate alignment (this loads all pairs and validates them)
        alignment = build_alignment(species, data_root)

        # Check 1: Number of pairs
        num_pairs = alignment.num_pairs
        report['num_pairs'] = num_pairs
        report['checks'].append(f"✓ Found {num_pairs} pairs")

        # Check 2: Per-pair validation (already done in build_alignment)
        report['checks'].append(f"✓ All {num_pairs} pairs passed dimensional consistency checks")

        # Check 3: Row-stochasticity (already validated in build_alignment via validate_pairdata)
        row_sum_atol = config['numerical_tolerance']['row_sum_atol']
        report['checks'].append(f"✓ All T_hat matrices are row-stochastic (tol={row_sum_atol})")

        # Check 4: Alignment for composition
        num_aligned = len([a for a in alignment.aligned if a])
        report['num_aligned_transitions'] = num_aligned
        report['num_total_transitions'] = len(alignment.aligned)

        if num_aligned == len(alignment.aligned):
            report['checks'].append(
                f"✓ All {num_aligned} consecutive pair transitions are aligned for composition"
            )
        else:
            num_broken = len(alignment.aligned) - num_aligned
            report['errors'].append(
                f"✗ {num_broken} pair transitions are NOT aligned (cannot compose)"
            )
            report['passed'] = False

        # Check 5: TopM values
        topM_values = [pair.topM for pair in alignment.pairs]
        report['topM_values'] = topM_values
        report['topM_range'] = [min(topM_values), max(topM_values)]

        for i, topM in enumerate(topM_values):
            if not isinstance(topM, (int, np.integer)):
                report['errors'].append(f"✗ Pair {i}: topM is not an integer: {topM}")
                report['passed'] = False
            elif topM <= 0:
                report['errors'].append(f"✗ Pair {i}: topM must be positive: {topM}")
                report['passed'] = False

        if all(isinstance(t, (int, np.integer)) and t > 0 for t in topM_values):
            report['checks'].append(
                f"✓ All topM values are valid integers in range {report['topM_range']}"
            )

        # Check 6: Feature dimensions consistency
        feature_dims = [pair.X_pca.shape[1] for pair in alignment.pairs]
        if len(set(feature_dims)) == 1:
            report['feature_dim'] = feature_dims[0]
            report['checks'].append(f"✓ Feature dimension consistent across all pairs: {feature_dims[0]}")
        else:
            report['errors'].append(
                f"✗ Inconsistent feature dimensions: {set(feature_dims)}"
            )
            report['passed'] = False

        # Check 7: Optional Phase 1 data availability
        phase1_available = {
            'T_true': sum(1 for p in alignment.pairs if p.T_true is not None),
            'phi3_bits': sum(1 for p in alignment.pairs if p.phi3_bits is not None),
            'phi2': sum(1 for p in alignment.pairs if p.phi2 is not None),
            'lock_label': sum(1 for p in alignment.pairs if p.lock_label is not None)
        }

        report['phase1_availability'] = phase1_available

        for key, count in phase1_available.items():
            if count == num_pairs:
                report['checks'].append(f"✓ Phase 1 {key} available for all {num_pairs} pairs")
            elif count == 0:
                report['warnings'].append(f"⚠ Phase 1 {key} not available (evaluation will be limited)")
            else:
                report['warnings'].append(
                    f"⚠ Phase 1 {key} available for only {count}/{num_pairs} pairs"
                )

        # Check 8: Composition horizons feasibility
        horizons = config['horizons']
        max_horizon = max(horizons)

        composable_pairs = []
        for k in range(num_pairs):
            can_compose_max = alignment.can_compose(k, max_horizon)
            composable_pairs.append(can_compose_max)

        num_composable = sum(composable_pairs)

        report['max_horizon'] = max_horizon
        report['num_pairs_composable_to_max_horizon'] = num_composable

        if num_composable > 0:
            report['checks'].append(
                f"✓ {num_composable} pairs can compose to maximum horizon {max_horizon}"
            )
        else:
            report['errors'].append(
                f"✗ No pairs can compose to maximum horizon {max_horizon}"
            )
            report['passed'] = False

    except Exception as e:
        report['errors'].append(f"✗ Fatal error during validation: {str(e)}")
        report['passed'] = False
        import traceback
        report['traceback'] = traceback.format_exc()

    # Summary
    print("\nValidation Summary:")
    print("-" * 70)

    for check in report['checks']:
        print(check)

    for warning in report['warnings']:
        print(warning)

    for error in report['errors']:
        print(error)

    print("-" * 70)

    if report['passed']:
        print(f"✓ {species} validation PASSED\n")
    else:
        print(f"✗ {species} validation FAILED\n")

    return report


def main():
    """Run validation for all species."""
    project_root = Path('/lambda/nfs/prateek/diff/project/phase2')
    data_root = project_root / 'data_phase2A'
    results_root = project_root / 'results_p2b'

    # Load config
    config_path = project_root / 'config_p2b_p3.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\n" + "="*70)
    print("Phase 2B: Input Validation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  Results root: {results_root}")
    print(f"  Max horizon: {max(config['horizons'])}")
    print(f"  Row-sum tolerance: {config['numerical_tolerance']['row_sum_atol']}")

    # Validate each species in parallel
    species_list = ['mouse', 'zebrafish']
    all_reports = {}
    all_passed = True

    # Create result directories
    for species in species_list:
        species_results_dir = results_root / species
        species_results_dir.mkdir(parents=True, exist_ok=True)

    # Parallelize validation across species
    with ProcessPoolExecutor(max_workers=len(species_list)) as executor:
        futures = {
            executor.submit(validate_species, species, data_root, config, results_root): species
            for species in species_list
        }

        for future in as_completed(futures):
            species = futures[future]
            report = future.result()
            all_reports[species] = report

            if not report['passed']:
                all_passed = False

            # Save species report
            species_results_dir = results_root / species
            report_path = species_results_dir / 'p2b_validation.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Saved validation report: {report_path}")

    # Overall summary
    print("\n" + "="*70)
    print("Overall Validation Summary")
    print("="*70 + "\n")

    for species, report in all_reports.items():
        status = "PASSED" if report['passed'] else "FAILED"
        print(f"  {species}: {status}")

    print()

    if all_passed:
        print("✓ All species passed validation. Ready for Phase 2B.")
        sys.exit(0)
    else:
        print("✗ Some species failed validation. Fix errors before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    main()
