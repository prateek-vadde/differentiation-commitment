#!/usr/bin/env python3
"""
run_p3.py

Master script for Phase 3 (Control/Perturbation) pipeline.

Orchestrates:
1. p3_00_perturb_defs.py - Compute epsilon_k and gradients
2. p3_01_apply_perturb.py - Generate perturbations + matched nulls
3. p3_02_eval_perturb.py - Evaluate ΔC with Mode A (exact recomputation)
4. p3_03_report.py - Generate summary report

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
        script_name: Name of script (e.g., 'p3_00_perturb_defs.py')
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
    parser = argparse.ArgumentParser(description="Run Phase 3 pipeline")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for prepared data (data_phase2A)')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory for results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config_p2b_p3.json')
    parser.add_argument('--species', type=str, nargs='+', default=['mouse', 'zebrafish'],
                        help='List of species to process')
    parser.add_argument('--skip_perturb_defs', action='store_true',
                        help='Skip perturbation definitions step')
    parser.add_argument('--skip_apply_perturb', action='store_true',
                        help='Skip perturbation application')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip perturbation evaluation')
    parser.add_argument('--skip_report', action='store_true',
                        help='Skip report generation')

    args = parser.parse_args()

    log("=" * 80)
    log("Phase 3 Pipeline - Master Script")
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

    # Step 1: Compute perturbation definitions
    if not args.skip_perturb_defs:
        log("\n[Step 1/4] Computing perturbation definitions (epsilon_k, gradients)...")
        success = run_script('p3_00_perturb_defs.py', common_args)
        if not success:
            log("ERROR: Perturbation definitions failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 1/4] Skipping perturbation definitions")

    # Step 2: Apply perturbations
    if not args.skip_apply_perturb:
        log("\n[Step 2/4] Applying perturbations and generating matched nulls...")
        success = run_script('p3_01_apply_perturb.py', common_args)
        if not success:
            log("ERROR: Perturbation application failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 2/4] Skipping perturbation application")

    # Step 3: Evaluate perturbations
    if not args.skip_eval:
        log("\n[Step 3/4] Evaluating perturbations with Mode A (exact recomputation)...")
        success = run_script('p3_02_eval_perturb.py', common_args)
        if not success:
            log("ERROR: Perturbation evaluation failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 3/4] Skipping perturbation evaluation")

    # Step 4: Generate report
    if not args.skip_report:
        log("\n[Step 4/4] Generating report...")
        success = run_script('p3_03_report.py', common_args)
        if not success:
            log("ERROR: Report generation failed. Aborting pipeline.")
            return 1
    else:
        log("\n[Step 4/4] Skipping report generation")

    log("\n" + "=" * 80)
    log("Phase 3 Pipeline Complete!")
    log("=" * 80)

    # Print summary
    report_path = Path(args.results_root) / "p3_summary.md"
    if report_path.exists():
        log(f"\nSummary report: {report_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
