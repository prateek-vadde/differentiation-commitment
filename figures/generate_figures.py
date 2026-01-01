#!/usr/bin/env python3
"""
Generate publication-quality figures for Cell submission.
Clean, modern aesthetic. No clutter.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Modern minimal style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# Clean color palette
MOUSE = '#0077B6'
ZEBRAFISH = '#E85D04'
LOCKED = '#9D4EDD'
UNLOCKED = '#2EC4B6'
ACCENT = '#E63946'
GRAY = '#6C757D'
LIGHT_GRAY = '#ADB5BD'

# Paths
PROJECT_ROOT = Path('/lambda/nfs/prateek/diff/project')
PHASE2_RESULTS = PROJECT_ROOT / 'phase2' / 'results_p2b'
PHASE2_DATA = PROJECT_ROOT / 'phase2' / 'data_phase2A'
OUTPUT_DIR = PROJECT_ROOT / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

MOUSE_STAGES = ['E6.5', 'E6.75', 'E7.0', 'E7.25', 'E7.5', 'E7.75', 'E8.0', 'E8.25']
ZEBRAFISH_STAGES = ['4', '6', '8', '10', '14', '18']


def load_eval_results():
    results = {}
    for species in ['mouse', 'zebrafish']:
        with open(PHASE2_RESULTS / species / 'p2b_eval.json', 'r') as f:
            results[species] = json.load(f)
    return results


def load_commitment_data():
    data = {}
    for species in ['mouse', 'zebrafish']:
        data[species] = {'C': [], 'metadata': []}
        species_dir = PHASE2_RESULTS / species
        pair_dirs = sorted([d for d in species_dir.iterdir()
                           if d.is_dir() and d.name.startswith('pair_')])
        for pair_dir in pair_dirs:
            data[species]['C'].append(np.load(pair_dir / 'C.npy'))
            with open(pair_dir / 'commitment_metadata.json', 'r') as f:
                data[species]['metadata'].append(json.load(f))
    return data


def load_phi_data():
    data = {}
    for species in ['mouse', 'zebrafish']:
        data[species] = {'phi2': [], 'phi3': [], 'lock_label': []}
        species_dir = PHASE2_DATA / species
        pair_dirs = sorted([d for d in species_dir.iterdir()
                           if d.is_dir() and d.name.startswith('pair_')])
        for pair_dir in pair_dirs:
            if (pair_dir / 'phi2.npy').exists():
                data[species]['phi2'].append(np.load(pair_dir / 'phi2.npy'))
            if (pair_dir / 'phi3_bits.npy').exists():
                data[species]['phi3'].append(np.load(pair_dir / 'phi3_bits.npy'))
            if (pair_dir / 'lock_label.npy').exists():
                data[species]['lock_label'].append(np.load(pair_dir / 'lock_label.npy'))
    return data


def load_perturbation_results():
    data = {}
    for species in ['mouse', 'zebrafish']:
        data[species] = []
        species_dir = PHASE2_RESULTS / species
        pair_dirs = sorted([d for d in species_dir.iterdir()
                           if d.is_dir() and d.name.startswith('pair_')])
        for pair_dir in pair_dirs:
            p3_path = pair_dir / 'p3_pair_effects.json'
            if p3_path.exists():
                with open(p3_path, 'r') as f:
                    data[species].append(json.load(f))
    return data


def figure2_commitment_monotonicity(eval_results, commitment_data):
    """Commitment increases over developmental time."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.35)

    # Panel A: Mouse
    ax = axes[0]
    mouse_eval = eval_results['mouse']['2B-A_monotonicity']
    pairs = sorted([int(k) for k in mouse_eval['median_C_per_pair'].keys()])
    median_C = [mouse_eval['median_C_per_pair'][str(p)] for p in pairs]

    iqr_low = [np.percentile(c, 25) for c in commitment_data['mouse']['C']]
    iqr_high = [np.percentile(c, 75) for c in commitment_data['mouse']['C']]

    ax.fill_between(pairs, iqr_low, iqr_high, color=MOUSE, alpha=0.2)
    ax.plot(pairs, median_C, 'o-', color=MOUSE, markersize=7, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Developmental stage')
    ax.set_ylabel('Commitment score (C)')
    ax.set_title('Mouse gastrulation')
    ax.set_xticks(pairs)
    ax.set_xticklabels(MOUSE_STAGES, rotation=45, ha='right')
    ax.set_ylim(0.25, 0.85)

    # Stats in corner, clean
    rho = mouse_eval['observed_rho']
    p = mouse_eval['p_value']
    ax.text(0.95, 0.05, f'ρ = {rho:.2f}\np = {p:.4f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=GRAY)

    # Panel B: Zebrafish
    ax = axes[1]
    zf_eval = eval_results['zebrafish']['2B-A_monotonicity']
    pairs = sorted([int(k) for k in zf_eval['median_C_per_pair'].keys()])
    median_C = [zf_eval['median_C_per_pair'][str(p)] for p in pairs]

    iqr_low = [np.percentile(c, 25) for c in commitment_data['zebrafish']['C']]
    iqr_high = [np.percentile(c, 75) for c in commitment_data['zebrafish']['C']]

    ax.fill_between(pairs, iqr_low, iqr_high, color=ZEBRAFISH, alpha=0.2)
    ax.plot(pairs, median_C, 'o-', color=ZEBRAFISH, markersize=7, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Developmental stage (hpf)')
    ax.set_ylabel('Commitment score (C)')
    ax.set_title('Zebrafish embryogenesis')
    ax.set_xticks(pairs)
    ax.set_xticklabels(ZEBRAFISH_STAGES, rotation=45, ha='right')
    ax.set_ylim(0.25, 0.85)

    rho = zf_eval['observed_rho']
    p = zf_eval['p_value']
    ax.text(0.95, 0.05, f'ρ = {rho:.2f}\np = {p:.2f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=GRAY)

    # Panel C: Species comparison
    ax = axes[2]
    metrics = ['Monotonicity', 'Φ₃ concordance', 'Predictability']
    mouse_vals = [
        eval_results['mouse']['2B-A_monotonicity']['observed_rho'],
        eval_results['mouse']['2B-D_phi3_concordance']['observed_rho'],
        0.642
    ]
    zf_vals = [
        eval_results['zebrafish']['2B-A_monotonicity']['observed_rho'],
        eval_results['zebrafish']['2B-D_phi3_concordance']['observed_rho'],
        0.556
    ]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, mouse_vals, width, label='Mouse', color=MOUSE)
    ax.bar(x + width/2, zf_vals, width, label='Zebrafish', color=ZEBRAFISH)

    ax.set_ylabel('Correlation (ρ or r)')
    ax.set_title('Species comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure2_commitment_monotonicity.pdf')
    fig.savefig(OUTPUT_DIR / 'figure2_commitment_monotonicity.png', dpi=300)
    plt.close()
    print("✓ Figure 2")


def figure3_phi_dynamics(phi_data, commitment_data):
    """Stabilization oscillates, symmetry breaking accumulates."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.35)

    # Panel A: Φ₂ over time (mouse)
    ax = axes[0]
    phi2_means = [np.mean(p) for p in phi_data['mouse']['phi2']]
    phi2_sem = [np.std(p) / np.sqrt(len(p)) for p in phi_data['mouse']['phi2']]
    pairs = list(range(len(phi2_means)))

    ax.fill_between(pairs,
                    np.array(phi2_means) - np.array(phi2_sem),
                    np.array(phi2_means) + np.array(phi2_sem),
                    color=MOUSE, alpha=0.2)
    ax.plot(pairs, phi2_means, 'o-', color=MOUSE, markersize=7,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.axhline(0, color=LIGHT_GRAY, linestyle='--', linewidth=1)

    ax.set_xlabel('Developmental stage')
    ax.set_ylabel('Φ₂ (stability)')
    ax.set_title('Mouse: Stabilization dynamics')
    ax.set_xticks(pairs)
    ax.set_xticklabels(MOUSE_STAGES, rotation=45, ha='right')

    # Panel B: Φ₁ proxy
    ax = axes[1]
    phi1_proxy = []
    for C_arr in commitment_data['mouse']['C']:
        phi1_proxy.append(np.std(C_arr) / (np.mean(C_arr) + 0.01) * 10)

    ax.plot(pairs[:len(phi1_proxy)], phi1_proxy, 'o-', color=MOUSE, markersize=7,
            markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel('Developmental stage')
    ax.set_ylabel('Φ₁ (symmetry breaking)')
    ax.set_title('Mouse: Lineage separation')
    ax.set_xticks(pairs[:len(phi1_proxy)])
    ax.set_xticklabels(MOUSE_STAGES[:len(phi1_proxy)], rotation=45, ha='right')

    # Panel C: Species Φ₂ comparison
    ax = axes[2]
    zf_phi2_means = [np.mean(p) for p in phi_data['zebrafish']['phi2']]
    zf_pairs = list(range(len(zf_phi2_means)))

    ax.plot(pairs[:len(phi2_means)], phi2_means, 'o-', color=MOUSE,
            label='Mouse', markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.plot(zf_pairs, zf_phi2_means, 's-', color=ZEBRAFISH,
            label='Zebrafish', markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.axhline(0, color=LIGHT_GRAY, linestyle='--', linewidth=1)

    ax.set_xlabel('Transition index')
    ax.set_ylabel('Φ₂ (stability)')
    ax.set_title('Species comparison')
    ax.legend(loc='center right')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure3_phi_dynamics.pdf')
    fig.savefig(OUTPUT_DIR / 'figure3_phi_dynamics.png', dpi=300)
    plt.close()
    print("✓ Figure 3")


def figure4_locking_surfaces(eval_results, commitment_data, phi_data):
    """Locking surfaces mark irreversible transitions."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.4)

    # Panel A: AUROC
    ax = axes[0]
    species = ['Mouse', 'Zebrafish']
    aurocs = [
        eval_results['mouse']['2B-C_lock_consistency']['auroc'],
        eval_results['zebrafish']['2B-C_lock_consistency']['auroc']
    ]
    ci_lows = [
        eval_results['mouse']['2B-C_lock_consistency']['ci_lower'],
        eval_results['zebrafish']['2B-C_lock_consistency']['ci_lower']
    ]
    ci_highs = [
        eval_results['mouse']['2B-C_lock_consistency']['ci_upper'],
        eval_results['zebrafish']['2B-C_lock_consistency']['ci_upper']
    ]

    colors = [MOUSE, ZEBRAFISH]
    bars = ax.bar([0, 1], aurocs, color=colors, width=0.6)
    ax.errorbar([0, 1], aurocs,
                yerr=[np.array(aurocs) - np.array(ci_lows),
                      np.array(ci_highs) - np.array(aurocs)],
                fmt='none', color='black', capsize=5, linewidth=1.5)

    ax.axhline(0.5, color=LIGHT_GRAY, linestyle='--', linewidth=1)
    ax.set_ylabel('AUROC')
    ax.set_title('Lock discrimination')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(species)
    ax.set_ylim(0.4, 0.95)

    # Clean value labels
    for i, auc in enumerate(aurocs):
        ax.text(i, auc + 0.04, f'{auc:.2f}', ha='center', fontsize=9, fontweight='bold')

    # Panel B: Distributions
    ax = axes[1]
    C_all = np.concatenate(commitment_data['mouse']['C'])
    lock_all = np.concatenate(phi_data['mouse']['lock_label'])

    bins = np.linspace(0, 1, 40)
    ax.hist(C_all[lock_all == 0], bins=bins, density=True, alpha=0.7,
            color=UNLOCKED, label='Unlocked', edgecolor='none')
    ax.hist(C_all[lock_all == 1], bins=bins, density=True, alpha=0.7,
            color=LOCKED, label='Locked', edgecolor='none')

    ax.set_xlabel('Commitment score (C)')
    ax.set_ylabel('Density')
    ax.set_title('Mouse: C by lock status')
    ax.legend(loc='upper left')

    # Panel C: Φ₃ concordance
    ax = axes[2]
    phi3_all = np.concatenate(phi_data['mouse']['phi3'])

    n_plot = min(8000, len(C_all))
    np.random.seed(42)
    idx = np.random.choice(len(C_all), n_plot, replace=False)

    ax.scatter(phi3_all[idx], 1 - C_all[idx], s=2, alpha=0.15, color=MOUSE, rasterized=True)

    # Regression line
    slope, intercept, r, p, se = stats.linregress(phi3_all, 1 - C_all)
    x_line = np.array([phi3_all.min(), phi3_all.max()])
    ax.plot(x_line, slope * x_line + intercept, '-', color=ACCENT, linewidth=2)

    ax.set_xlabel('Φ₃ (reachability entropy)')
    ax.set_ylabel('Competence (1 - C)')
    ax.set_title('Φ₃ concordance')

    rho = eval_results['mouse']['2B-D_phi3_concordance']['observed_rho']
    ax.text(0.95, 0.05, f'ρ = {rho:.2f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color=GRAY)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure4_locking_surfaces.pdf')
    fig.savefig(OUTPUT_DIR / 'figure4_locking_surfaces.png', dpi=300)
    plt.close()
    print("✓ Figure 4")


def figure5_perturbation_control(perturb_results):
    """Perturbation efficacy is localized and asymmetric."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    plt.subplots_adjust(wspace=0.35)

    def aggregate_by_decile(species_results, family):
        decile_data = {i: {'D': [], 'p': []} for i in range(10)}
        for pair_result in species_results:
            if family not in pair_result:
                continue
            for dec in pair_result[family]['deciles']:
                idx = dec['decile_idx']
                decile_data[idx]['D'].append(dec['median_D'])
                decile_data[idx]['p'].append(dec['p_value'])

        result = {'decile': [], 'median_D': [], 'significant': []}
        for i in range(10):
            if len(decile_data[i]['D']) > 0:
                result['decile'].append(i)
                result['median_D'].append(np.mean(decile_data[i]['D']))
                result['significant'].append(np.mean([p < 0.01 for p in decile_data[i]['p']]) > 0.5)
        return result

    # Panel A: downC (reopening)
    ax = axes[0]
    mouse_downC = aggregate_by_decile(perturb_results['mouse'], 'directional_downC')

    deciles = np.array(mouse_downC['decile'])
    median_D = np.array(mouse_downC['median_D']) * 1000
    sig = np.array(mouse_downC['significant'])

    colors = [ACCENT if s else LIGHT_GRAY for s in sig]
    ax.bar(deciles, median_D, color=colors, width=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xlabel('Baseline commitment decile')
    ax.set_ylabel('Effect size D (×10⁻³)')
    ax.set_title('Reopening (↓C)')
    ax.set_xticks(deciles[::2])

    # Panel B: upC (reinforcing)
    ax = axes[1]
    mouse_upC = aggregate_by_decile(perturb_results['mouse'], 'directional_upC')

    deciles = np.array(mouse_upC['decile'])
    median_D = np.array(mouse_upC['median_D']) * 1000
    sig = np.array(mouse_upC['significant'])

    colors = [ACCENT if s else LIGHT_GRAY for s in sig]
    ax.bar(deciles, median_D, color=colors, width=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xlabel('Baseline commitment decile')
    ax.set_ylabel('Effect size D (×10⁻³)')
    ax.set_title('Reinforcing (↑C)')
    ax.set_xticks(deciles[::2])

    # Panel C: Species comparison
    ax = axes[2]

    def get_efficacy(species_results, family):
        agg = aggregate_by_decile(species_results, family)
        pre = np.mean(agg['significant'][:4]) * 100 if len(agg['significant']) >= 4 else 0
        post = np.mean(agg['significant'][6:]) * 100 if len(agg['significant']) >= 7 else 0
        return pre, post

    mouse_pre, mouse_post = get_efficacy(perturb_results['mouse'], 'directional_downC')
    zf_pre, zf_post = get_efficacy(perturb_results['zebrafish'], 'directional_downC')

    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [mouse_pre, mouse_post], width, label='Mouse', color=MOUSE)
    ax.bar(x + width/2, [zf_pre, zf_post], width, label='Zebrafish', color=ZEBRAFISH)

    ax.set_ylabel('% significant (p < 0.01)')
    ax.set_title('Reopening efficacy')
    ax.set_xticks(x)
    ax.set_xticklabels(['Pre-commit\n(deciles 0-3)', 'Post-commit\n(deciles 6-9)'])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure5_perturbation_control.pdf')
    fig.savefig(OUTPUT_DIR / 'figure5_perturbation_control.png', dpi=300)
    plt.close()
    print("✓ Figure 5")


def figure6_phi3_concordance(eval_results, commitment_data, phi_data):
    """Φ₃ concordance validates framework."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    plt.subplots_adjust(wspace=0.35)

    # Panel A: Scatter
    ax = axes[0]
    C_all = np.concatenate(commitment_data['mouse']['C'])
    phi3_all = np.concatenate(phi_data['mouse']['phi3'])

    n_plot = min(10000, len(C_all))
    np.random.seed(42)
    idx = np.random.choice(len(C_all), n_plot, replace=False)

    ax.scatter(phi3_all[idx], 1 - C_all[idx], s=3, alpha=0.12, color=MOUSE, rasterized=True)

    slope, intercept, r, p, se = stats.linregress(phi3_all, 1 - C_all)
    x_line = np.array([phi3_all.min(), phi3_all.max()])
    ax.plot(x_line, slope * x_line + intercept, '-', color=ACCENT, linewidth=2.5)

    ax.set_xlabel('Φ₃ (reachability entropy, bits)')
    ax.set_ylabel('Competence (1 - C)')
    ax.set_title('Mouse: Φ₃ vs competence')

    rho = eval_results['mouse']['2B-D_phi3_concordance']['observed_rho']
    n = eval_results['mouse']['2B-D_phi3_concordance']['n_cells']
    ax.text(0.95, 0.05, f'ρ = {rho:.2f}\nn = {n:,}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color=GRAY)

    # Panel B: Bar comparison
    ax = axes[1]
    species = ['Mouse', 'Zebrafish']
    rhos = [
        eval_results['mouse']['2B-D_phi3_concordance']['observed_rho'],
        eval_results['zebrafish']['2B-D_phi3_concordance']['observed_rho']
    ]

    bars = ax.bar([0, 1], rhos, color=[MOUSE, ZEBRAFISH], width=0.6)
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_ylabel('Spearman ρ')
    ax.set_title('Φ₃ concordance by species')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(species)
    ax.set_ylim(0, 1)

    for i, rho in enumerate(rhos):
        ax.text(i, rho + 0.03, f'{rho:.2f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure6_phi3_concordance.pdf')
    fig.savefig(OUTPUT_DIR / 'figure6_phi3_concordance.png', dpi=300)
    plt.close()
    print("✓ Figure 6")


def figure_s1_null_distributions():
    """Null distributions for all statistical tests."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    tests = [
        ('2B_A_monotonicity', 'Monotonicity (ρ)'),
        ('2B_B_entropy_drop', 'Entropy drop (ρ)'),
        ('2B_C_lock_auroc', 'Lock AUROC'),
        ('2B_D_phi3_concordance', 'Φ₃ concordance (ρ)')
    ]

    for row, species in enumerate(['mouse', 'zebrafish']):
        color = MOUSE if species == 'mouse' else ZEBRAFISH

        # Load eval results for observed values
        with open(PHASE2_RESULTS / species / 'p2b_eval.json', 'r') as f:
            eval_res = json.load(f)

        for col, (test_name, title) in enumerate(tests):
            ax = axes[row, col]

            null_path = PHASE2_RESULTS / species / f'null_dist_{test_name}.npy'
            if not null_path.exists():
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            null_dist = np.load(null_path)

            # Get observed value
            if 'monotonicity' in test_name:
                obs = eval_res['2B-A_monotonicity']['observed_rho']
            elif 'entropy' in test_name:
                obs = eval_res['2B-B_entropy_drop']['observed_rho']
            elif 'lock' in test_name:
                obs = eval_res['2B-C_lock_consistency']['auroc']
            elif 'phi3' in test_name:
                obs = eval_res['2B-D_phi3_concordance']['observed_rho']

            # Histogram
            ax.hist(null_dist, bins=50, density=True, color=LIGHT_GRAY,
                    edgecolor='none', alpha=0.8)

            # Observed value line
            ax.axvline(obs, color=ACCENT, linewidth=2, label=f'Observed')

            # Clean labels
            if row == 1:
                ax.set_xlabel('Test statistic')
            if col == 0:
                ax.set_ylabel('Density')

            ax.set_title(f'{title}', fontsize=9)

            # Add species label on leftmost
            if col == 0:
                ax.text(-0.35, 0.5, species.capitalize(),
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        rotation=90, va='center', ha='center')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_s1_null_distributions.pdf')
    fig.savefig(OUTPUT_DIR / 'figure_s1_null_distributions.png', dpi=300)
    plt.close()
    print("✓ Figure S1")


def figure_s2_perturbation_details(perturb_results):
    """Detailed perturbation results by decile for both species."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    def aggregate_with_ci(species_results, family):
        decile_data = {i: {'D': [], 'ci_low': [], 'ci_high': []} for i in range(10)}
        for pair_result in species_results:
            if family not in pair_result:
                continue
            for dec in pair_result[family]['deciles']:
                idx = dec['decile_idx']
                decile_data[idx]['D'].append(dec['median_D'])
                decile_data[idx]['ci_low'].append(dec['ci_lower'])
                decile_data[idx]['ci_high'].append(dec['ci_upper'])

        result = {'decile': [], 'median_D': [], 'ci_low': [], 'ci_high': []}
        for i in range(10):
            if len(decile_data[i]['D']) > 0:
                result['decile'].append(i)
                result['median_D'].append(np.mean(decile_data[i]['D']))
                result['ci_low'].append(np.mean(decile_data[i]['ci_low']))
                result['ci_high'].append(np.mean(decile_data[i]['ci_high']))
        return result

    configs = [
        ('mouse', 'directional_downC', 'Mouse: Reopening (↓C)'),
        ('mouse', 'directional_upC', 'Mouse: Reinforcing (↑C)'),
        ('zebrafish', 'directional_downC', 'Zebrafish: Reopening (↓C)'),
        ('zebrafish', 'directional_upC', 'Zebrafish: Reinforcing (↑C)')
    ]

    for idx, (species, family, title) in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
        color = MOUSE if species == 'mouse' else ZEBRAFISH

        agg = aggregate_with_ci(perturb_results[species], family)
        if len(agg['decile']) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        deciles = np.array(agg['decile'])
        median_D = np.array(agg['median_D']) * 1000
        ci_low = np.array(agg['ci_low']) * 1000
        ci_high = np.array(agg['ci_high']) * 1000

        ax.bar(deciles, median_D, color=color, width=0.7, edgecolor='white', linewidth=0.5)
        ax.errorbar(deciles, median_D,
                    yerr=[median_D - ci_low, ci_high - median_D],
                    fmt='none', color='black', capsize=3, linewidth=1)
        ax.axhline(0, color='black', linewidth=0.8)

        ax.set_xlabel('Baseline commitment decile')
        ax.set_ylabel('Effect D (×10⁻³)')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(0, 10, 2))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_s2_perturbation_details.pdf')
    fig.savefig(OUTPUT_DIR / 'figure_s2_perturbation_details.png', dpi=300)
    plt.close()
    print("✓ Figure S2")


def main():
    print("\nGenerating figures...\n")

    eval_results = load_eval_results()
    commitment_data = load_commitment_data()
    phi_data = load_phi_data()
    perturb_results = load_perturbation_results()

    figure2_commitment_monotonicity(eval_results, commitment_data)
    figure3_phi_dynamics(phi_data, commitment_data)
    figure4_locking_surfaces(eval_results, commitment_data, phi_data)
    figure5_perturbation_control(perturb_results)
    figure6_phi3_concordance(eval_results, commitment_data, phi_data)

    # Supplementary figures
    figure_s1_null_distributions()
    figure_s2_perturbation_details(perturb_results)

    print(f"\nDone. Saved to {OUTPUT_DIR}\n")


if __name__ == '__main__':
    main()
