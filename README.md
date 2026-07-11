<div align="center">

# ADSV Protocol

### Automated Doping Stability Validation for Rare-Earth-Substituted Hosts

[![Method](https://img.shields.io/badge/method-Legacy%20ADSV-4C78A8)](./Automated_Doping_Stability_Validation.py)
[![Models](https://img.shields.io/badge/models-CHGNet%20%2B%20MACE-59A14F)](#method-overview)
[![Materials Project](https://img.shields.io/badge/data-Materials%20Project-F28E2B)](https://materialsproject.org/)
[![License](https://img.shields.io/badge/license-MIT-8C8C8C)](./LICENSE)

A reproducible screening workflow for evaluating candidate substitution sites,
aligned energies, and residual-force consistency in doped inorganic hosts.

</div>

---

## Overview

This repository contains the **Legacy ADSV workflow** used to generate the original screening dataset.

The workflow combines:

- **Materials Project** host structures and reference formation energies;
- **CHGNet** structural relaxation and substitution-site screening;
- **O-rich chemical-potential references** for substitution-energy ranking;
- **MACE** single-point energy and residual-force validation.

> **Important:** MACE is used as a single-point cross-check in this Legacy workflow. It does not perform a second structural relaxation.

---

## Method Overview

```text
Materials Project host structure
        |
        v
CHGNet host-supercell relaxation
        |
        v
CHGNet candidate-site screening
        |
        v
Full doped-supercell construction
        |
        v
Final CHGNet relaxation
        |
        v
Energy alignment
        |
        v
MACE single-point energy and residual-force check
```

### Historical stability criterion

The Legacy report labels a structure as `Stable` when:

```text
MACE residual max force < 0.5 eV/Angstrom
```

This label means that the CHGNet-relaxed structure passed the historical MACE single-point force check. It is not equivalent to full convergence on the MACE potential-energy surface.

---

## Materials Project Compatibility Patch

The current script includes a minimal compatibility patch for recent `mp-api`, `emmet-core`, and `pydantic` environments.

Earlier versions requested complete Materials Project Summary documents. Unused DOS and band-structure fields could then trigger local schema-validation errors before ADSV reached the actual calculation.

The patched script requests only the fields it uses, for example:

```python
fields=["structure", "formation_energy_per_atom"]
```

### What the patch changes

- Materials Project query field selection only.

### What the patch does not change

- CHGNet or MACE models;
- CHGNet relaxation thresholds;
- MACE single-point validation;
- the historical `0.5 eV/Angstrom` criterion;
- O-rich chemical-potential equations;
- aligned-energy equations;
- dopant counting;
- random dopant placement.

Therefore, the patched script preserves the numerical method used for the original Legacy dataset.

---

## Repository Contents

```text
ADSV-Protocol/
├── Automated_Doping_Stability_Validation.py
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Quick Start

### 1. Prepare the environment

Use an environment with CUDA-enabled PyTorch, CHGNet, MACE, pymatgen, ASE, and `mp-api`.

```bash
pip install -r requirements.txt
```

### 2. Set the Materials Project API key

```bash
export MP_API_KEY='YOUR_32_CHARACTER_MP_API_KEY'
```

Do not place the API key directly in the Python source file, and do not commit it to GitHub.

### 3. Run a calculation

```bash
python -u Automated_Doping_Stability_Validation.py \
  --mp_id mp-1213396 \
  --doping "Yb3+:0.035, Tm3+:0.005"
```

When prompted for the supercell, enter the required dimensions, for example:

```text
1 1 2
```

Another example:

```bash
python -u Automated_Doping_Stability_Validation.py \
  --mp_id mp-5516 \
  --doping "Yb3+:0.18, Er3+:0.02"
```

Supercell input:

```text
2 2 2
```

---

## Output Files

Results are written under:

```text
results/
```

A completed run typically contains:

```text
*_Report.txt
*_Optimized.cif
```

The CIF file is the final **CHGNet-relaxed** structure. MACE evaluates that structure without further relaxation.

---

## Energy Definition

The reported aligned energy is:

```text
E_aligned = E_MP,host + (E_ML,doped - E_ML,host)
```

where all terms are expressed per atom.

This quantity is intended for comparison within the same Legacy ADSV workflow. It should not be interpreted as a complete charged-defect formation energy.

---

## Reproducibility Notes

- Use the same host MP ID, supercell, dopant ratios, software environment, and model versions when comparing with the original dataset.
- Dopant positions are selected randomly in the Legacy implementation. Runs without a fixed random seed may produce different atomic arrangements and slightly different energies or forces.
- Nominal dopant fractions are converted to integer substitution counts inside a finite supercell. The actual composition may therefore differ from the nominal percentage.
- Do not mix Legacy ADSV results with results from a later workflow that performs full MACE relaxation unless the methods are clearly separated.
- Changing a later MACE relaxation target to `0.5 eV/Angstrom` does not reproduce this Legacy workflow, because the Legacy method uses a MACE single-point check rather than MACE relaxation.

---

## Recommended Result Wording

For reports and manuscripts, the following description is accurate:

> Structures were relaxed with CHGNet and subsequently cross-validated by a MACE single-point energy and residual-force calculation. Candidates with a maximum absolute Cartesian force component below 0.5 eV/Angstrom were classified as stable under the Legacy ADSV criterion.

---

## Reference

The workflow was developed for rare-earth substitution screening in glass-ceramic host systems and is associated with the study:

> *Thermodynamic Criterion for Interfacial Engineering of Ultra-Stable Phosphor-in-Glass Composites.*

Please cite the corresponding publication when it becomes available.

---

## License

This project is distributed under the terms of the [MIT License](./LICENSE).
