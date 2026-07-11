# ADSV Protocol

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](./Automated_Doping_Stability_Validation.py)
[![License](https://img.shields.io/badge/license-MIT-8C8C8C)](./LICENSE)
[![CHGNet](https://img.shields.io/badge/model-CHGNet-59A14F)](https://github.com/CederGroupHub/chgnet)
[![MACE](https://img.shields.io/badge/model-MACE-4C78A8)](https://github.com/ACEsuit/mace)
[![Materials Project](https://img.shields.io/badge/data-Materials_Project-F28E2B)](https://materialsproject.org/)

ADSV is a machine-learning-assisted screening workflow for rare-earth substitution in crystalline host materials.

---

## 1. Overview

This repository implements the **Legacy ADSV workflow** used to generate the original screening dataset.

The workflow combines Materials Project host structures and reference formation energies with CHGNet structural relaxation and MACE single-point cross-validation, using O-rich chemical-potential references for substitution-energy ranking.

---

## 2. Workflow

```
Materials Project host structure
        |
        v
CHGNet host-supercell relaxation
        |
        v
O-rich substitution-site screening
        |
        v
Full-composition doped-supercell construction
        |
        v
CHGNet doped-structure relaxation
        |
        v
MP-aligned energy calculation
        |
        v
MACE single-point energy and residual-force cross-check
```

---

## 3. Important Method Note

- **CHGNet** performs structural relaxation.
- **MACE does not** perform structural relaxation in this Legacy workflow.
- MACE is used only for a single-point energy and residual-force cross-check.
- The historical criterion is: **maximum absolute Cartesian force component < 0.5 eV/A**.
- "Stable" means passing this Legacy screening criterion, not rigorous proof of thermodynamic or dynamical stability.

---

## 4. Materials Project Compatibility Fix

The current version requests only the Summary fields that are actually used by the calculation, avoiding data-model validation errors triggered by unused DOS and band-structure fields in recent `mp-api`, `emmet-core`, and `pydantic` environments.

This compatibility fix does **not** change the scientific workflow or historical numerical thresholds.

---

## 5. Installation

```bash
pip install -r requirements.txt
```

Set your Materials Project API key:

```bash
export MP_API_KEY='YOUR_32_CHARACTER_MP_API_KEY'
```

Do not place the API key directly in the Python source file or commit it to version control.

---

## 6. Usage

```bash
python Automated_Doping_Stability_Validation.py \
  --mp_id mp-1213396 \
  --doping "Yb3+:0.035, Tm3+:0.005"
```

- `MP_API_KEY` can be supplied via the environment variable shown above.
- The program will interactively prompt for a supercell; for example:

```
1 1 2
```

---

## 7. Outputs

A completed run produces, under `results/`:

- `*_Report.txt` — text report with aligned formation energy, lattice parameters, and MACE cross-check.
- `*_Optimized.cif` — CHGNet-relaxed structure in CIF format.

The exported CIF is **CHGNet-relaxed**; MACE does not further relax it.

---

## 8. Interpretation and Reproducibility

- Nominal concentrations are rounded to integer substitutions.
- Dopant sites are randomly assigned.
- The historical script does not record a fixed random seed.
- The aligned formation energy is a host-anchored ML energy-shift metric, not a complete defect formation energy.

---

## 9. License

This project is distributed under the terms of the [MIT License](./LICENSE).
