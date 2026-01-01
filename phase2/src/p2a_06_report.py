"""
Phase 2A - Report Generation
Creates summary_report.md with tables and figures
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def create_metrics_table(
    mouse_pair_metrics: List[Dict],
    mouse_aggregate: Dict,
    zfish_pair_metrics: List[Dict],
    zfish_aggregate: Dict,
    config: Dict
) -> str:
    """Create markdown table of metrics."""

    table = "## Metrics Summary\n\n"

    # Per-species aggregate table
    table += "### Aggregate Metrics (averaged across pairs)\n\n"
    table += "| Species | Mass Capture | Φ₃ Spearman | Lock AUROC | Lock Jaccard | Status |\n"
    table += "|---------|--------------|-------------|------------|--------------|--------|\n"

    # Mouse
    mouse_pass = (
        mouse_aggregate['mass_capture'] >= config['pass_mass_capture'] and
        mouse_aggregate['phi3_spearman'] >= config['pass_spearman_phi3'] and
        mouse_aggregate['lock_auroc'] >= config['pass_lock_auroc'] and
        mouse_aggregate['lock_jaccard'] >= config['pass_lock_jaccard']
    )
    mouse_status = "✓ PASS" if mouse_pass else "✗ FAIL"

    table += f"| Mouse | {mouse_aggregate['mass_capture']:.3f} | "
    table += f"{mouse_aggregate['phi3_spearman']:.3f} | "
    table += f"{mouse_aggregate['lock_auroc']:.3f} | "
    table += f"{mouse_aggregate['lock_jaccard']:.3f} | "
    table += f"{mouse_status} |\n"

    # Zebrafish
    zfish_pass = (
        zfish_aggregate['mass_capture'] >= config['pass_mass_capture'] and
        zfish_aggregate['phi3_spearman'] >= config['pass_spearman_phi3'] and
        zfish_aggregate['lock_auroc'] >= config['pass_lock_auroc'] and
        zfish_aggregate['lock_jaccard'] >= config['pass_lock_jaccard']
    )
    zfish_status = "✓ PASS" if zfish_pass else "✗ FAIL"

    table += f"| Zebrafish | {zfish_aggregate['mass_capture']:.3f} | "
    table += f"{zfish_aggregate['phi3_spearman']:.3f} | "
    table += f"{zfish_aggregate['lock_auroc']:.3f} | "
    table += f"{zfish_aggregate['lock_jaccard']:.3f} | "
    table += f"{zfish_status} |\n\n"

    # Thresholds
    table += "**Pass Thresholds:**\n"
    table += f"- Mass Capture ≥ {config['pass_mass_capture']}\n"
    table += f"- Φ₃ Spearman ≥ {config['pass_spearman_phi3']}\n"
    table += f"- Lock AUROC ≥ {config['pass_lock_auroc']}\n"
    table += f"- Lock Jaccard ≥ {config['pass_lock_jaccard']}\n\n"

    # Per-pair table
    table += "### Per-Pair Metrics\n\n"
    table += "#### Mouse\n\n"
    table += "| Pair | Mass Capture | Φ₃ Spearman | Lock AUROC | Lock Jaccard |\n"
    table += "|------|--------------|-------------|------------|------------|\n"

    for i, m in enumerate(mouse_pair_metrics):
        table += f"| {i} | {m['mass_capture']:.3f} | {m['phi3_spearman']:.3f} | "
        table += f"{m['lock_auroc']:.3f} | {m['lock_jaccard']:.3f} |\n"

    table += "\n#### Zebrafish\n\n"
    table += "| Pair | Mass Capture | Φ₃ Spearman | Lock AUROC | Lock Jaccard |\n"
    table += "|------|--------------|-------------|------------|------------|\n"

    for i, m in enumerate(zfish_pair_metrics):
        table += f"| {i} | {m['mass_capture']:.3f} | {m['phi3_spearman']:.3f} | "
        table += f"{m['lock_auroc']:.3f} | {m['lock_jaccard']:.3f} |\n"

    return table


def create_figures(
    mouse_pair_metrics: List[Dict],
    zfish_pair_metrics: List[Dict],
    config: Dict,
    output_dir: Path
):
    """Create visualization figures."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase 2A Metrics', fontsize=16)

    metrics_names = ['mass_capture', 'phi3_spearman', 'lock_auroc', 'lock_jaccard']
    metric_labels = ['Mass Capture', 'Φ₃ Spearman ρ', 'Lock AUROC', 'Lock Jaccard']
    thresholds = [
        config['pass_mass_capture'],
        config['pass_spearman_phi3'],
        config['pass_lock_auroc'],
        config['pass_lock_jaccard']
    ]

    for idx, (metric_name, label, threshold) in enumerate(zip(metrics_names, metric_labels, thresholds)):
        ax = axes[idx // 2, idx % 2]

        mouse_vals = [m[metric_name] for m in mouse_pair_metrics]
        zfish_vals = [m[metric_name] for m in zfish_pair_metrics]

        x_mouse = np.arange(len(mouse_vals))
        x_zfish = np.arange(len(zfish_vals))

        ax.scatter(x_mouse, mouse_vals, label='Mouse', alpha=0.7, s=80)
        ax.scatter(x_zfish, zfish_vals, label='Zebrafish', alpha=0.7, s=80)

        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})', alpha=0.5)

        # Add mean lines
        ax.axhline(y=np.mean(mouse_vals), color='blue', linestyle=':', alpha=0.5, label=f'Mouse mean')
        ax.axhline(y=np.mean(zfish_vals), color='orange', linestyle=':', alpha=0.5, label=f'Zfish mean')

        ax.set_xlabel('Pair Index')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150)
    plt.close()

    print(f"  Saved figure: {output_dir / 'metrics_summary.png'}")


def generate_report(
    mouse_pair_metrics: List[Dict],
    mouse_aggregate: Dict,
    mouse_passed: bool,
    zfish_pair_metrics: List[Dict],
    zfish_aggregate: Dict,
    zfish_passed: bool,
    config: Dict,
    output_dir: Path
):
    """Generate final summary report."""

    print(f"\n{'='*80}")
    print("Generating Summary Report")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create markdown report
    report = "# Phase 2A Summary Report\n\n"

    # Overall status
    overall_passed = mouse_passed and zfish_passed
    status_text = "**✓ ALL GATES PASSED**" if overall_passed else "**✗ SOME GATES FAILED**"
    report += f"{status_text}\n\n"

    # Configuration
    report += "## Configuration\n\n"
    report += f"- Seed: {config['seed']}\n"
    report += f"- Epochs: {config['epochs']}\n"
    report += f"- Batch size (GPU): {config['batch_gpu']}\n"
    report += f"- Learning rate: {config['lr']}\n"
    report += f"- Embed dim: {config['embed_dim']}\n"
    report += f"- TopM: Per-pair (data-driven, see topM_per_pair.json)\n"
    report += f"- Hard negative pool: {config['hard_negative_pool']}\n\n"

    # Metrics table
    report += create_metrics_table(
        mouse_pair_metrics, mouse_aggregate,
        zfish_pair_metrics, zfish_aggregate,
        config
    )

    # Figures
    report += "\n## Figures\n\n"
    report += "![Metrics Summary](metrics_summary.png)\n\n"

    # Create figures
    create_figures(
        mouse_pair_metrics, zfish_pair_metrics, config, output_dir
    )

    # Conclusions
    report += "## Conclusions\n\n"

    if overall_passed:
        report += "All pass/fail gates satisfied for both species. The compressed operators "
        report += "\hat{T} successfully preserve:\n\n"
        report += "1. **Mass capture**: Majority of transition probability mass retained\n"
        report += "2. **Φ₃ structure**: Entropy rankings (commitment signatures) preserved\n"
        report += "3. **Lock geometry**: AUROC demonstrates discriminative power\n"
        report += "4. **Lock overlap**: Scale-free Jaccard validates structural consistency\n\n"
        report += "Phase 2A compression is **successful**.\n"
    else:
        report += "Some gates failed. Investigation needed:\n\n"

        if not mouse_passed:
            report += "- **Mouse**: "
            fails = []
            if mouse_aggregate['mass_capture'] < config['pass_mass_capture']:
                fails.append(f"Mass capture ({mouse_aggregate['mass_capture']:.3f})")
            if mouse_aggregate['phi3_spearman'] < config['pass_spearman_phi3']:
                fails.append(f"Φ₃ Spearman ({mouse_aggregate['phi3_spearman']:.3f})")
            if mouse_aggregate['lock_auroc'] < config['pass_lock_auroc']:
                fails.append(f"Lock AUROC ({mouse_aggregate['lock_auroc']:.3f})")
            if mouse_aggregate['lock_jaccard'] < config['pass_lock_jaccard']:
                fails.append(f"Lock Jaccard ({mouse_aggregate['lock_jaccard']:.3f})")
            report += ", ".join(fails) + "\n"

        if not zfish_passed:
            report += "- **Zebrafish**: "
            fails = []
            if zfish_aggregate['mass_capture'] < config['pass_mass_capture']:
                fails.append(f"Mass capture ({zfish_aggregate['mass_capture']:.3f})")
            if zfish_aggregate['phi3_spearman'] < config['pass_spearman_phi3']:
                fails.append(f"Φ₃ Spearman ({zfish_aggregate['phi3_spearman']:.3f})")
            if zfish_aggregate['lock_auroc'] < config['pass_lock_auroc']:
                fails.append(f"Lock AUROC ({zfish_aggregate['lock_auroc']:.3f})")
            if zfish_aggregate['lock_jaccard'] < config['pass_lock_jaccard']:
                fails.append(f"Lock Jaccard ({zfish_aggregate['lock_jaccard']:.3f})")
            report += ", ".join(fails) + "\n"

        report += "\nRecommendations: Increase epochs, adjust TopM, or inspect model training curves.\n"

    # Save report
    report_path = output_dir / 'summary_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"  Saved report: {report_path}")
    print("\n" + "="*80)
    print(status_text)
    print("="*80)

    return overall_passed


if __name__ == '__main__':
    """Test report generation with dummy data."""

    # Dummy data
    mouse_pair_metrics = [
        {'mass_capture': 0.92, 'phi3_spearman': 0.85, 'lock_auroc': 0.88, 'lock_jaccard': 0.65},
        {'mass_capture': 0.91, 'phi3_spearman': 0.83, 'lock_auroc': 0.86, 'lock_jaccard': 0.62},
    ]
    mouse_aggregate = {k: np.mean([m[k] for m in mouse_pair_metrics]) for k in mouse_pair_metrics[0].keys()}

    zfish_pair_metrics = [
        {'mass_capture': 0.93, 'phi3_spearman': 0.87, 'lock_auroc': 0.89, 'lock_jaccard': 0.67},
    ]
    zfish_aggregate = {k: np.mean([m[k] for m in zfish_pair_metrics]) for k in zfish_pair_metrics[0].keys()}

    config = {
        'seed': 0,
        'epochs': 40,
        'batch_gpu': 256,
        'lr': 1e-3,
        'embed_dim': 64,
        'topM_gpu': 2048,
        'hard_negative_pool': 4096,
        'pass_mass_capture': 0.90,
        'pass_spearman_phi3': 0.80,
        'pass_lock_auroc': 0.80,
        'pass_lock_jaccard': 0.50
    }

    output_dir = Path(__file__).parent.parent / 'results' / 'test_report'

    passed = generate_report(
        mouse_pair_metrics, mouse_aggregate, True,
        zfish_pair_metrics, zfish_aggregate, True,
        config, output_dir
    )

    print(f"\n✓ Test report generated: {output_dir / 'summary_report.md'}")
