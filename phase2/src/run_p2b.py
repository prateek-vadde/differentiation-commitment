#!/usr/bin/env python3
"""
run_p2b.py

Master script for Phase 2B (Prediction) pipeline.

Orchestrates:
1. p2b_00_validate_inputs.py - Validate data contracts
2. p2b_01_build_commitment_score.py - Compute commitment scores C
3. p2b_02_train_predictor.py - Train MLP regressor X_pca → C
4. p2b_03_eval_prospective.py - Validate with 4 biological benchmarks
5. p2b_04_report.py - Generate summary report

Standards:
- Cell-quality rigor
- GPU acceleration where appropriate
- Full parallelization
- Exact adherence to plan
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_io import log


def run_script(script_name, args_dict):
    """
    Run a Python script with given arguments.

    Args:
        script_name: Name of script (e.g., 'p2b_00_validate_inputs.py')
        args_dict: Dictionary of argument name -> value

    Returns:
        True if successful, False otherwise
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        log(f"ERROR: Script {script_path} does not exist")
        return False

    # Build command
    cmd = [sys.executable, str(script_path)]

    for arg_name, arg_value in args_dict.items():
        if isinstance(arg_value, list):
            cmd.append(f'--{arg_name}')
            cmd.extend([str(v) for v in arg_value])
        elif isinstance(arg_value, bool):
            if arg_value:
                cmd.append(f'--{arg_name}')
        else:
            cmd.append(f'--{arg_name}')
            cmd.append(str(arg_value))

    log(f"Running: {' '.join(cmd)}")

    # Execute
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        log(f"ERROR: {script_name} failed with return code {result.returncode}")
        return False

    log(f"SUCCESS: {script_name} completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2B pipeline")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for prepared data (data_phase2A)')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory for results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config_p2b_p3.json')
    parser.add_argument('--species', type=str, nargs='+', default=['mouse', 'zebrafish'],
                        help='List of species to process')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip input validation step')
    parser.add_argument('--skip_commitment', action='store_true',
                        help='Skip commitment score computation')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip predictor training')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip prospective evaluation')
    parser.add_argument('--skip_report', action='store_true',
                        help='Skip report generation')

    args = parser.parse_args()

    log("=" * 80)
    log("Phase 2B Pipeline - Master Script")
    log("=" * 80)
    log(f"Data root: {args.data_root}")
    log(f"Results root: {args.results_root}")
    log(f"Config: {args.config}")
    log(f"Species: {args.species}")
    log("=" * 80)

    # Common arguments for all scripts
    common_args = {
        'data_root': args.data_root,
        'results_root': args.results_root,
        'config': args.config,
        'species': args.species
    }

    # Step 1: Validate inputs
    if not args.skip_validation:
        log("\n[Step 1/5] Validating inputs...")
        success = run_script('p2b_00_validate_inputs.py', common_args)
        if not success:
            log("ERROR: Input validation failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 1/5] Skipping input validation")

    # Step 2: Build commitment scores
    if not args.skip_commitment:
        log("\n[Step 2/5] Building commitment scores...")
        success = run_script('p2b_01_build_commitment_score.py', common_args)
        if not success:
            log("ERROR: Commitment score computation failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 2/5] Skipping commitment score computation")

    # Step 3: Train predictor
    if not args.skip_training:
        log("\n[Step 3/5] Training predictor...")
        success = run_script('p2b_02_train_predictor.py', common_args)
        if not success:
            log("ERROR: Predictor training failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 3/5] Skipping predictor training")

    # Step 4: Prospective evaluation
    if not args.skip_eval:
        log("\n[Step 4/5] Running prospective evaluation...")
        success = run_script('p2b_03_eval_prospective.py', common_args)
        if not success:
            log("ERROR: Prospective evaluation failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 4/5] Skipping prospective evaluation")

    # Step 5: Generate report
    if not args.skip_report:
        log("\n[Step 5/5] Generating report...")
        success = run_script('p2b_04_report.py', common_args)
        if not success:
            log("ERROR: Report generation failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 5/5] Skipping report generation")

    log("\n" + "=" * 80)
    log("Phase 2B Pipeline Complete!")
    log("=" * 80)

    # Print summary
    report_path = Path(args.results_root) / "p2b_summary.md"
    if report_path.exists():
        log(f"\nSummary report: {report_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
