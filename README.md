***

# ADSV Protocol

**Automated Doping Stability Validation for Glass-Ceramic Systems**

A computational workflow for validating the thermodynamic stability of doped ceramic phosphors during glass-ceramic co-sintering processes.

## Overview

ADSV is a computational workflow that validates whether rare-earth doping preserves the thermodynamic stability of a phosphor host lattice. It uses the CHGNet and MACE universal machine learning force fields to relax doped supercells (≥1000 atoms) and compute an aligned formation energy referenced to the Materials Project DFT baseline.

The core idea: the lattice formation energy (FE) potential well depth governs a ceramic phase's intrinsic resistance to interfacial thermal erosion. ADSV checks that functional doping does not degrade the host into a less stable thermodynamic tier.

### Key Features

- **ValenceGate**: Automatic bond-valence analysis to classify substitution sites as isovalent (Class A, full calculation) or aliovalent (Class B, blocked with geometric compatibility report).
- **Hybrid Relaxation**: CHGNet native `StructOptimizer` for pre-conditioning, followed by MACE-MP for high-fidelity refinement.
- **O-Rich Thermodynamics**: Chemical potentials derived under the oxygen-rich limit relevant to high-temperature ceramic sintering.
- **Aligned Formation Energy**: Anchoring ML perturbations to established DFT references (`E_f,aligned = E_DFT,host + (E_ML,doped − E_ML,host)`).
- **Hardware-Aware**: Automatic GPU profiling with degradation modes for different VRAM tiers (supports Cloud HPC to local CPU).

## Requirements

- Python ≥ 3.9
- PyTorch with CUDA support (recommended)
- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [MACE](https://github.com/ACEsuit/mace) (optional but recommended)
- [Pymatgen](https://pymatgen.org/)
- [mp-api](https://github.com/materialsproject/api)
- ASE, NumPy

Install the required dependencies:

```bash
pip install chgnet mace-torch pymatgen mp-api ase numpy torch
```

*Note: A MACE pre-trained model file is expected at `./models/2023-12-03-mace-128-L1_epoch-199.model`. If unavailable, the protocol will automatically run in CHGNet-only mode.*

## Usage

Run the pipeline from the command line:

```bash
python ADSV_V2.7.py \
  --mp_id mp-5732 \
  --doping "Yb:0.18, Er:0.02" \
  --api_key YOUR_MP_API_KEY
```

### Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--mp_id` | Materials Project ID of the host lattice | `mp-542724` |
| `--doping` | Doping scheme (`Element:fraction`, comma-separated) | `Yb3+:0.18, Er3+:0.02` |
| `--relaxed` | Flag to use relaxed energy window (0.20 eV instead of 0.10 eV) | `False` |
| `--api_key` | Materials Project API key (or set `MP_API_KEY` env variable) | None |

*During execution, the script will interactively prompt for supercell dimensions. Recommended sizes will be printed based on detected hardware capabilities.*

## Output Structure

Results are saved to a timestamped folder under `results/` (local) or `/root/autodl-tmp/ADSV_Results/` (AutoDL cloud environment):

- `*_Report.txt` — Full calculation report including aligned formation energies, valence analysis, and stability assessment.
- `*_Optimized.cif` — Relaxed doped supercell structure.
- `*_Metadata.json` — Machine-readable results and physical parameters.
- `BLOCKED_*_aliovalent_report.txt` — Detailed geometric compatibility reports for blocked aliovalent sites.
- `Terminal_Log_*.txt` — Complete intercepted terminal log.

## Workflow Pipeline

1. **Fetch Host**: Retrieves the unmodified structure from the Materials Project and identifies symmetry-inequivalent cation sites.
2. **ValenceGate**: Determines host-site oxidation states (via BVAnalyzer or composition fallback) and categorizes each site–dopant pair.
3. **Establish Baseline**: Builds the supercell (≥1000 atoms) and relaxes the undoped host using CHGNet.
4. **Force Screening**: Screens isovalent sites via single-point MACE force evaluation using valence-aware thresholds.
5. **Thermodynamic Engine**: Evaluates substitution energies using pre-computed O-rich chemical potentials.
6. **Deep Validation**: Performs dual-engine (CHGNet → MACE) relaxation on two independent random-seed samples.
7. **Energy Alignment**: Computes the final aligned formation energy to benchmark against the original thermodynamic database.

## Citation

If this protocol assists your research, please cite our related publication (details to be updated upon publication).

## License

This project and its codebase are provided for academic research use.
