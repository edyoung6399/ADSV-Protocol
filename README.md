```markdown
# ADSV Protocol

**Automated Doping Stability Validation for Glass-Ceramic Systems**

Computational workflow for validating the thermodynamic stability of doped ceramic phosphors during glass-ceramic co-sintering processes.

## Overview

ADSV is a computational workflow that validates whether rare-earth doping preserves the thermodynamic stability of a phosphor host lattice. It uses the CHGNet and MACE universal machine learning force fields to relax doped supercells (≥1000 atoms) and compute an aligned formation energy referenced to the Materials Project DFT baseline.

The core idea: the lattice formation energy (FE) potential well depth governs a ceramic phase's intrinsic resistance to interfacial thermal erosion during glass-ceramic co-sintering. ADSV checks that functional doping does not degrade the host into a less stable thermodynamic tier.

### Key features

- **ValenceGate** — automatic bond-valence analysis to classify substitution sites as isovalent (Class A, full calculation) or aliovalent (Class B, blocked with geometric compatibility report)
- **Hybrid relaxation** — CHGNet native `StructOptimizer` for pre-conditioning, MACE-MP for high-fidelity refinement
- **O-rich thermodynamics** — chemical potentials derived under the oxygen-rich limit relevant to ceramic sintering
- **Aligned formation energy** — `E_f,aligned = E_DFT,host + (E_ML,doped − E_ML,host)`, anchoring ML perturbations to established DFT references
- **Hardware-aware** — automatic GPU profiling with degradation modes for different VRAM tiers

## Requirements

- Python ≥ 3.9
- PyTorch with CUDA support (recommended)
- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [MACE](https://github.com/ACEsuit/mace) (optional but recommended)
- [Pymatgen](https://pymatgen.org/)
- [mp-api](https://github.com/materialsproject/api)
- ASE
- NumPy

Install dependencies:

```bash
pip install chgnet mace-torch pymatgen mp-api ase numpy torch
```

A MACE pre-trained model file is expected at `./models/2023-12-03-mace-128-L1_epoch-199.model`. If unavailable, the protocol runs in CHGNet-only mode.

## Usage

```bash
python ADSV_V2.7.py \
  --mp_id mp-5732 \
  --doping "Yb:0.18, Er:0.02" \
  --api_key YOUR_MP_API_KEY
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--mp_id` | Materials Project ID of the host | `mp-542724` |
| `--doping` | Doping scheme (`Element:fraction`, comma-separated) | `Yb3+:0.18, Er3+:0.02` |
| `--relaxed` | Use relaxed energy window (0.20 eV instead of 0.10 eV) | `False` |
| `--api_key` | Materials Project API key (or set `MP_API_KEY` env variable) | — |

The script will prompt for supercell dimensions interactively. Recommended sizes are printed based on detected hardware.

## Output

Results are saved to a timestamped folder under `results/` (local) or `/root/autodl-tmp/ADSV_Results/` (AutoDL cloud):

- `*_Report.txt` — full calculation report including aligned formation energies, valence analysis, and stability assessment
- `*_Optimized.cif` — relaxed doped supercell structure
- `*_Metadata.json` — machine-readable results and parameters
- `BLOCKED_*_aliovalent_report.txt` — geometric compatibility reports for aliovalent sites
- `Terminal_Log_*.txt` — complete terminal log

## How it works

1. **Fetch host structure** from the Materials Project and identify symmetry-inequivalent cation sites
2. **ValenceGate** determines host-site oxidation states (BVAnalyzer or composition fallback) and classifies each site–dopant pair as isovalent or aliovalent
3. **Build supercell** (≥1000 atoms) and relax the undoped host with CHGNet
4. **Screen isovalent sites** via single-point MACE force evaluation with valence-aware thresholds
5. **Evaluate substitution energies** for passing sites using O-rich chemical potentials
6. **Deep validation** with dual-engine (CHGNet → MACE) relaxation on two independent random-seed samples
7. **Compute aligned formation energy** by referencing ML energy differences to the Materials Project DFT baseline

## Citation

If you use this code in your research, please cite the related work (to be published) and include a link to this repository.

## License

This project is provided for academic use. Please contact the authors for commercial licensing inquiries.
```
