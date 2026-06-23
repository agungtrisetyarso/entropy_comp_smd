# entropy_comp_smd

**Companion code and analysis repository** for the manuscript:

> **Entropy computing is stochastic mirror descent: a photon-number-resolved identification of a dissipative photonic optimizer**  
> Agung Trisetyarso  
> School of Computer Science, Bina Nusantara University  
> *(Target journal: npj Quantum Information)*

This repository provides the Python scripts used to generate the key quantitative results, statistical claims, and figures in the paper. It rigorously demonstrates that the measurement-and-feedback dynamics of the **Dirac-3** entropy-computing photonic optimizer is a physical realization of **stochastic entropic mirror descent** on the probability simplex.

## Key Scientific Claims Reproduced

The code hardens three independent signatures that identify the device dynamics with stochastic mirror descent (SMD):

1. **Multiplicative updates** — The device's per-iteration steps are exponentiated-gradient (replicator) updates, not Euclidean projected gradient descent. One-step prediction error is ~14% lower for mirror descent.

2. **Multiplicative (geometric) noise** — Residual fluctuations after subtracting the deterministic mirror drift obey `Var(δx_i) ∝ x_i^β` with `β = 1.96 ± 0.01`. A photon-number-resolved, zero-landscape calibration shows this is the classical `β = 2` law in the high-photon regime where the device operates (`> 5σ` separation from instrumental floor). No resolvable Poisson (`β = 1`) component exists.

3. **Average-case Bregman advantage** — On synthetic nonconvex simplex-constrained quadratics, entropic mirror descent reaches statistically significantly better average solutions than Euclidean PGD, with the gap increasing with nonconvexity.

Additional verification:
- Momentum-accelerated mirror descent converges ~30% faster (implementable in the existing FPGA feedback loop).
- Open-system Lindblad simulation qualitatively reproduces the observed multiplicative flow on the simplex.

## Repository Structure

```
entropy_comp_smd/
├── README.md
├── analysis.py          # Core analysis script (one-step mismatch, noise exponent β, average-case advantage)
├── acc_mirr.py          # Momentum-accelerated SMD experiment (Fig. 7 verification)
├── quant_traj.py        # Lindblad master-equation simulation of entropy computing (Fig. 5)
├── strong-conv.tex      # Short note on 1-strong convexity of negative entropy + Pinsker inequality
└── requirements.txt     # Python dependencies
```

## Data Requirements

The main analysis (`analysis.py`) requires the public trajectory dataset released by the entropy-computing authors:

```bash
git clone https://github.com/qci-github/eqc-studies.git
cd eqc-studies
# Unzip the relevant result archives (results_500_sched_2_*.zip)
```

**Required files** (place in a directory referenced by `EC_DATA_DIR` environment variable, defaults to `.`):

- `QPLIB_0018_OBJ.csv` — cost vector `c` and quadratic matrix `J` for the benchmark problem
- `results_500_sched_2_0.json`, `results_500_sched_2_1.json`, ... (five trajectory files)

These contain 100 runs × 2001 iterations × 50 variables each.

## Installation

```bash
git clone https://github.com/agungtrisetyarso/entropy_comp_smd.git
cd entropy_comp_smd

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

**`requirements.txt`** (create this file):

```
numpy
matplotlib
scipy
qutip          # only needed for quant_traj.py (Lindblad simulation)
```

## Reproducing the Paper Results

### 1. Main quantitative claims (Sections 3 & 5 of the paper)

```bash
python analysis.py
```

This script:
- Loads the public Dirac-3 trajectories
- Computes the one-step prediction mismatch ratio (PGD vs. MD)
- Estimates the noise scaling exponent `β`
- Evaluates the average-case energy gap on synthetic nonconvex instances
- Prints the three hardened numerical results that appear in the manuscript

Expected output includes statements such as:
- `PGD/MD mismatch = 1.14 ± 0.01`
- `β = 1.96 ± 0.01`
- Table of mean energy gaps with increasing nonconvexity

### 2. Momentum acceleration verification (predicted improvement)

```bash
python acc_mirr.py
```

Generates the plot `accelerated_smd_prediction.png` showing that Nesterov-style momentum in mirror space reaches target performance ~30% faster than standard SMD.

### 3. Open-system Lindblad simulation (qualitative support for Fig. 5)

```bash
python quant_traj.py
```

Simulates cost-dependent photon loss using the Lindblad master equation on a few bosonic modes. The resulting population flow on the simplex qualitatively matches the multiplicative mirror-descent behavior observed on the physical device.

## What Each Script Does

| Script          | Purpose                                                                 | Key Output                          | Paper Relevance                  |
|-----------------|-------------------------------------------------------------------------|-------------------------------------|----------------------------------|
| `analysis.py`   | Trajectory fitting, noise scaling, synthetic advantage experiments     | Printed statistics + `HARDENED_RESULTS.txt` | Sections 3 (Evidence) & 5       |
| `acc_mirr.py`   | Compares standard vs. momentum-accelerated stochastic mirror descent   | `accelerated_smd_prediction.png`   | Discussion: predicted accelerations |
| `quant_traj.py` | Lindblad simulation of photon-dependent loss on the simplex            | Population trajectories            | Fig. 5 (qualitative support)    |

## Citation

If you use this code, data, or the scientific results, please cite:

```bibtex
@article{Trisetyarso2025EntropyMirror,
  title   = {Entropy computing is stochastic mirror descent: a photon-number-resolved identification of a dissipative photonic optimizer},
  author  = {Trisetyarso, Agung},
  journal = {npj Quantum Information},
  year    = {2025},
  note    = {Companion code: https://github.com/agungtrisetyarso/entropy_comp_smd}
}
```

The original hardware demonstration paper is:

```bibtex
@article{Nguyen2025EntropyComputing,
  title   = {Entropy Computing, A Paradigm for Optimization in Open Photonic Systems},
  author  = {Nguyen, Lac and others},
  journal = {Communications Physics},
  year    = {2025},
  arxiv   = {2407.04512}
}
```

## License

This repository is released under the MIT License (see `LICENSE` file).

## Contact & Acknowledgments

**Author**: Agung Trisetyarso  
**Email**: trisetyarso@binus.ac.id  
**Affiliation**: School of Computer Science, Bina Nusantara University, Jakarta, Indonesia

The device, public trajectory data, and Euclidean baseline are due to Quantum Computing Inc. (QCi) and the authors of the entropy-computing hardware paper. The dynamical identification, noise-law analysis, photon-number-resolved floor calibration, and average-case advantage results are the contribution of this work.

## Related Links

- Original entropy-computing hardware paper: [arXiv:2407.04512](https://arxiv.org/abs/2407.04512)
- Public trajectory data: [github.com/qci-github/eqc-studies](https://github.com/qci-github/eqc-studies)
- Target journal: *npj Quantum Information*

---

*This README was prepared to accompany the manuscript submission. Feedback and issues are welcome.*