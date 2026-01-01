# Differentiation as Collapse of Reachable Futures

Code and analysis for the paper: *"Differentiation as progressive collapse of reachable futures: A quantitative framework for developmental commitment"*

## Overview

This repository implements a mathematical framework characterizing cell differentiation as the progressive reduction of accessible future states. Using optimal transport to map transitions between developmental timepoints, we define three constraint observables (Φ₁, Φ₂, Φ₃) and a continuous commitment score C that quantifies developmental restriction.

## Repository Structure

```
├── paper.tex                 # Main manuscript
├── supplementary.tex         # Supplementary materials
├── references.bib            # Bibliography
├── figures/                  # Figure generation
│   └── generate_figures.py   # Publication figures
├── phase1/                   # Phase 1: Transition inference
│   ├── config.yaml
│   └── src/
│       ├── 01_preprocess.py  # Data preprocessing
│       ├── 02_experts.py     # Expert ensemble
│       ├── 03_uot.py         # Unbalanced optimal transport
│       ├── 04_phi.py         # Φ₁, Φ₂, Φ₃ computation
│       └── 05_locking.py     # Locking surface detection
└── phase2/                   # Phase 2-3: Commitment & perturbation
    ├── config_phase2A.json
    ├── config_p2b_p3.json
    └── src/
        ├── common_io.py      # I/O utilities
        ├── common_math.py    # Statistical functions
        ├── p2a_*.py          # Phase 2A: Encoder training
        ├── p2b_*.py          # Phase 2B: Commitment score
        └── p3_*.py           # Phase 3: Perturbation analysis
```

## Key Concepts

- **Commitment Score C**: Percentile of collapsed future uncertainty, computed from multi-horizon entropy
- **Φ₃ (Reachability)**: Shannon entropy of transition distribution — collapses at commitment
- **Φ₂ (Stability)**: Log trace ratio of covariances — stabilizes after collapse
- **Φ₁ (Propagation)**: Diversity difference — spikes at lineage segregation

## Requirements

- Python 3.8+
- PyTorch 1.12+
- scanpy, anndata
- scipy, numpy
- ot (Python Optimal Transport)
- cupy (optional, for GPU acceleration)

## Data

Analysis performed on:
- Mouse gastrulation (Pijuan-Sala et al., 2019)
- Zebrafish embryogenesis (Wagner et al., 2018)

Raw data should be placed in `phase1/data_raw/` following the structure in `phase1/config.yaml`.

## Citation

```bibtex
@article{vadde2025differentiation,
  title={Differentiation as progressive collapse of reachable futures:
         A quantitative framework for developmental commitment},
  author={Vadde, Prateek},
  journal={},
  year={2025}
}
```

## License

MIT License
