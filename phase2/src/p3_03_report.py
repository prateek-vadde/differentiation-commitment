#!/usr/bin/env python3
"""
p3_03_report.py

Phase 3: Control/Perturbation Analysis - Report Generation

Aggregates per-pair perturbation results and generates publication-quality
markdown summary with:
- Per-decile effect sizes (median ΔC_directional - ΔC_null) with bootstrap 95% CIs
- Statistical significance (paired Wilcoxon p-values)
- Interpretation of control "windows" where perturbations matter vs locked regions
- Species comparison (sharp vs diffuse commitment landscapes)

Standards:
- Cell-quality rigor
- No arbitrary thresholds
- All conclusions backed by nulls/CIs/effect sizes
- Exact adherence to plan
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common_io import log
from common_math import bootstrap_median_ci


def load_p3_pair_results(results_root, species):
    """Load all Phase 3 pair-level results for a species."""
    species_results_root = Path(results_root) / "p3" / species

    all_pair_results = []
    pair_dirs = sorted([d for d in species_results_root.iterdir() if d.is_dir()])

    for pair_dir in pair_dirs:
        effects_path = pair_dir / "p3_pair_effects.json"
        if not effects_path.exists():
            log(f"WARNING: {effects_path} does not exist, skipping pair")
            continue

        with open(effects_path, 'r') as f:
            pair_data = json.load(f)

        all_pair_results.append(pair_data)

    return all_pair_results


def aggregate_by_decile(all_pair_results, families):
    """
    Aggregate results across pairs by decile.

    For each decile and family:
    - Collect all paired differences D = ΔC_directional - ΔC_null
    - Compute: median(D), bootstrap 95% CI, paired Wilcoxon p-value

    Returns:
    {
        family: {
            decile: {
                'median_effect': float,
                'ci_lower': float,
                'ci_upper': float,
                'p_value': float,
                'n_cells': int
            }
        }
    }
    """
    # Collect all paired differences per decile per family
    # paired_diffs[family][decile] = list of D values across all pairs
    paired_diffs = {family: defaultdict(list) for family in families}

    for pair_result in all_pair_results:
        for family in families:
            if family not in pair_result:
                continue

            family_data = pair_result[family]

            for decile_key, decile_data in family_data.items():
                if not decile_key.startswith('decile_'):
                    continue

                decile_idx = int(decile_key.split('_')[1])

                # Get paired differences for this decile
                if 'paired_differences' in decile_data:
                    D_values = decile_data['paired_differences']
                    paired_diffs[family][decile_idx].extend(D_values)

    # Compute aggregate statistics per decile per family
    aggregated = {}

    for family in families:
        aggregated[family] = {}

        for decile_idx in range(10):
            if decile_idx not in paired_diffs[family]:
                continue

            D = np.array(paired_diffs[family][decile_idx])

            if len(D) == 0:
                continue

            # Median effect size
            median_effect = float(np.median(D))

            # Bootstrap 95% CI
            ci_lower, ci_upper = bootstrap_median_ci(
                D,
                n_bootstrap=2000,
                seed=0,
                alpha=0.05
            )

            # Paired Wilcoxon test (H0: median(D) = 0)
            # Only valid if n >= 10
            if len(D) >= 10:
                try:
                    stat, p_value = wilcoxon(D, alternative='two-sided')
                    p_value = float(p_value)
                except:
                    p_value = 1.0
            else:
                p_value = 1.0

            aggregated[family][decile_idx] = {
                'median_effect': median_effect,
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'p_value': p_value,
                'n_cells': len(D)
            }

    return aggregated


def interpret_control_windows(aggregated, family):
    """
    Identify commitment deciles where perturbations have significant effects.

    A decile shows "controllability" if:
    - 95% CI does not contain zero
    - p < 0.05
    - |median_effect| > 0.01 (practical significance: 1 percentile point)

    Returns list of tuples: (decile_idx, median_effect, p_value)
    """
    controllable = []

    family_data = aggregated.get(family, {})

    for decile_idx in range(10):
        if decile_idx not in family_data:
            continue

        stats = family_data[decile_idx]

        median_effect = stats['median_effect']
        ci_lower = stats['ci_lower']
        ci_upper = stats['ci_upper']
        p_value = stats['p_value']

        # Check significance criteria
        ci_excludes_zero = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
        p_significant = p_value < 0.05
        effect_practical = abs(median_effect) > 0.01

        if ci_excludes_zero and p_significant and effect_practical:
            controllable.append((decile_idx, median_effect, p_value))

    return controllable


def generate_p3_report(results_root, species_list, config):
    """Generate Phase 3 markdown summary report."""

    families = config['perturb']['families']

    # Load results for all species
    species_aggregated = {}

    for species in species_list:
        log(f"Loading Phase 3 results for {species}...")
        all_pair_results = load_p3_pair_results(results_root, species)

        if len(all_pair_results) == 0:
            log(f"No Phase 3 results found for {species}, skipping")
            continue

        log(f"  Loaded {len(all_pair_results)} pairs")

        # Aggregate by decile
        aggregated = aggregate_by_decile(all_pair_results, families)
        species_aggregated[species] = aggregated

    # Generate markdown report
    lines = []
    lines.append("# Phase 3: Control/Perturbation Analysis - Summary Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This report summarizes results from **Phase 3**, which evaluates:")
    lines.append("- **Whether commitment scores C can be perturbed**")
    lines.append("- **Where in the commitment landscape perturbations matter** (vs deeply locked regions)")
    lines.append("- **Matched null controls** (orthogonal random directions with same magnitude)")
    lines.append("")
    lines.append("### Perturbation Families")
    lines.append("")
    lines.append("1. **local_random**: Random direction, scaled by epsilon_k")
    lines.append("2. **directional_downC**: Along -∇C (reopen futures)")
    lines.append("3. **directional_upC**: Along +∇C (commit further)")
    lines.append("")
    lines.append("### Step Size epsilon_k")
    lines.append("")
    lines.append("- Computed per-pair as **median distance to 10th nearest neighbor** in X_pca space")
    lines.append("- Biologically scaled: represents local PCA-space step")
    lines.append("")
    lines.append("### Evaluation Method: Mode A (Exact Recomputation)")
    lines.append("")
    lines.append("For each perturbed state x':")
    lines.append("1. Embed x' using Phase 2A encoder")
    lines.append("2. Compute cosine similarities to all target states Y")
    lines.append("3. Apply TopM (pair-specific) + softmax → new row distribution")
    lines.append("4. Compose forward for horizons h=1,2,3")
    lines.append("5. Compute C'(x') exactly as in Phase 2B")
    lines.append("")
    lines.append("ΔC = C'(x') - C(x) measures commitment change.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-species results
    for species in species_list:
        if species not in species_aggregated:
            continue

        aggregated = species_aggregated[species]

        lines.append(f"## Species: {species.capitalize()}")
        lines.append("")

        # Per-family analysis
        for family in families:
            if family not in aggregated or len(aggregated[family]) == 0:
                lines.append(f"### Family: {family}")
                lines.append("")
                lines.append("*No results available*")
                lines.append("")
                continue

            lines.append(f"### Family: {family}")
            lines.append("")

            # Decile table
            lines.append("#### Per-Decile Effects")
            lines.append("")
            lines.append("| Decile | Median ΔC | 95% CI | p-value | n_cells | Significant? |")
            lines.append("|--------|-----------|--------|---------|---------|--------------|")

            family_data = aggregated[family]

            for decile_idx in range(10):
                if decile_idx not in family_data:
                    lines.append(f"| {decile_idx} | - | - | - | - | - |")
                    continue

                stats = family_data[decile_idx]
                median_effect = stats['median_effect']
                ci_lower = stats['ci_lower']
                ci_upper = stats['ci_upper']
                p_value = stats['p_value']
                n_cells = stats['n_cells']

                # Check significance
                ci_excludes_zero = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
                p_significant = p_value < 0.05
                effect_practical = abs(median_effect) > 0.01

                is_significant = ci_excludes_zero and p_significant and effect_practical
                sig_marker = "✓" if is_significant else ""

                lines.append(
                    f"| {decile_idx} | {median_effect:.4f} | "
                    f"[{ci_lower:.4f}, {ci_upper:.4f}] | "
                    f"{p_value:.4f} | {n_cells} | {sig_marker} |"
                )

            lines.append("")

            # Interpretation: control windows
            controllable = interpret_control_windows(aggregated, family)

            lines.append("#### Interpretation")
            lines.append("")

            if len(controllable) == 0:
                lines.append(f"**No significant perturbation effects detected** for {family}.")
                lines.append("")
                lines.append("This suggests:")
                lines.append("- States are deeply locked across all commitment levels")
                lines.append("- Or effect sizes are below detection threshold (< 1 percentile point)")
                lines.append("")
            else:
                lines.append(f"**Significant perturbation effects detected** in {len(controllable)} decile(s):")
                lines.append("")

                for decile_idx, median_effect, p_value in controllable:
                    direction = "increased" if median_effect > 0 else "decreased"
                    lines.append(
                        f"- **Decile {decile_idx}** (C ∈ [{decile_idx*10}-{(decile_idx+1)*10}%]): "
                        f"median ΔC = {median_effect:.4f} ({direction} commitment, p={p_value:.4f})"
                    )

                lines.append("")
                lines.append("**Control Window Analysis:**")
                lines.append("")

                # Identify low vs high commitment regions
                low_deciles = [d for d, _, _ in controllable if d < 5]
                high_deciles = [d for d, _, _ in controllable if d >= 5]

                if len(low_deciles) > 0 and len(high_deciles) == 0:
                    lines.append("- Perturbations matter primarily in **low-commitment regions** (C < 50%)")
                    lines.append("- High-commitment cells are **locked** and resistant to perturbation")
                elif len(low_deciles) == 0 and len(high_deciles) > 0:
                    lines.append("- Perturbations matter primarily in **high-commitment regions** (C ≥ 50%)")
                    lines.append("- Low-commitment cells may already be exploring futures")
                elif len(low_deciles) > 0 and len(high_deciles) > 0:
                    lines.append("- Perturbations matter across **both low and high commitment** regions")
                    lines.append("- Suggests continuous controllability throughout differentiation")

                lines.append("")

        lines.append("---")
        lines.append("")

    # Cross-species comparison
    if len(species_aggregated) > 1:
        lines.append("## Cross-Species Comparison")
        lines.append("")

        # For each family, compare controllability patterns across species
        for family in families:
            lines.append(f"### {family}")
            lines.append("")

            for species in species_list:
                if species not in species_aggregated:
                    continue

                aggregated = species_aggregated[species]
                controllable = interpret_control_windows(aggregated, family)

                if len(controllable) == 0:
                    lines.append(f"- **{species.capitalize()}**: No significant effects")
                else:
                    decile_str = ", ".join([str(d) for d, _, _ in controllable])
                    lines.append(f"- **{species.capitalize()}**: Controllable in deciles {decile_str}")

            lines.append("")

        lines.append("**Interpretation:**")
        lines.append("")
        lines.append("Species with sharp commitment landscapes (e.g., mouse) may show:")
        lines.append("- Narrow control windows")
        lines.append("- Strong lockdown in high-C regions")
        lines.append("")
        lines.append("Species with diffuse commitment (e.g., zebrafish) may show:")
        lines.append("- Broader controllability")
        lines.append("- Less pronounced lockdown")
        lines.append("")

    # Methods summary
    lines.append("---")
    lines.append("")
    lines.append("## Methods Summary")
    lines.append("")
    lines.append("### Perturbation Generation")
    lines.append("")
    lines.append("- **Sampling**: Up to 5000 cells uniformly per pair")
    lines.append("- **Step size epsilon_k**: Median distance to 10th nearest neighbor (per pair)")
    lines.append("- **Gradient estimation**: kNN regression (k=50) to estimate ∇C locally")
    lines.append("- **Matched nulls**: 5 orthogonal random directions per cell (same magnitude as directional)")
    lines.append("")
    lines.append("### Statistical Analysis")
    lines.append("")
    lines.append("- **Commitment binning**: 10 deciles based on baseline C")
    lines.append("- **Effect size**: Median of paired differences D = ΔC_directional - ΔC_null")
    lines.append("- **Uncertainty**: Bootstrap 95% CI (2000 resamples)")
    lines.append("- **Significance**: Paired Wilcoxon test (H0: median(D) = 0)")
    lines.append("- **Practical significance threshold**: |median ΔC| > 0.01 (1 percentile point)")
    lines.append("")
    lines.append("### Significance Criteria")
    lines.append("")
    lines.append("A decile is considered **controllable** if ALL of the following hold:")
    lines.append("1. 95% CI excludes zero")
    lines.append("2. p < 0.05 (Wilcoxon)")
    lines.append("3. |median ΔC| > 0.01 (practical significance)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by p3_03_report.py*")

    # Write report
    report_path = Path(results_root) / "p3_summary.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

    log(f"Phase 3 report written to {report_path}")

    return report_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Phase 3 summary report")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for prepared data (data_phase2A)')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Root directory for results')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config_p2b_p3.json')
    parser.add_argument('--species', type=str, nargs='+', default=['mouse', 'zebrafish'],
                        help='List of species to process')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    log("=" * 80)
    log("Phase 3: Report Generation")
    log("=" * 80)

    # Generate report
    report_path = generate_p3_report(
        results_root=args.results_root,
        species_list=args.species,
        config=config
    )

    log("=" * 80)
    log("Phase 3 report generation complete!")
    log(f"Report: {report_path}")
    log("=" * 80)


if __name__ == '__main__':
    main()
