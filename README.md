# 🧪 Automated Doping Stability Validation (ADSV)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust Python pipeline for evaluating the thermodynamic stability of rare-earth doped materials (e.g., phosphor-in-glass hosts) at realistic concentrations. This tool serves as the computational companion to the manuscript:  
**"Thermodynamic Criterion for Interfacial Engineering of Ultra-Stable Phosphor-in-Glass Composites"** (Submitted for publication).

## 🔬 Physics Context
Simulating high-concentration doping in traditional DFT requires massive supercells (300+ atoms) to minimize artificial image interactions under Periodic Boundary Conditions (PBC). This tool utilizes **CHGNet** and **MACE** graph neural networks to enable rapid evaluation of these large-scale systems on local hardware.

## ⚖️ Thermodynamic Alignment (Delta Method)
To ensure physical accuracy and cancel out systematic potential residuals, the formation energy is aligned using:

$$E_{aligned} = E_{f, \text{host, MP}} + (E_{\text{total, doped, AI}} - E_{\text{total, host, AI}})$$

* **$E_{f, \text{host, MP}}$**: High-precision DFT baseline from Materials Project.
* **$E_{AI}$**: Total energy calculated from Machine Learning interatomic potentials.

## ✨ Key Features
* **Automated Supercell Construction**: Intelligently suggests optimal supercell dimensions based on VRAM.
* **Hybrid Site Screening**: Identifies thermodynamically favorable substitution sites using CHGNet.
* **Dynamic Thermo Engine**: Calculates dynamic chemical potentials for elements under O-rich limits.
* **Physical Validation**: Secondary stability cross-validation using MACE to evaluate residual forces.

## ⚙️ Requirements & Performance
* **📦 Core Deps**: `chgnet`, `mp-api`, `pymatgen`, `torch`.
* **🧠 Memory**: ~24GB RAM for 800-atom supercells.
* **⚡ Speed**: < 60s for a 324-atom system on CPU.

## 🚀 Installation & Usage

1. **Setup Environment**:
   ```bash
   conda create -n adsv_env python=3.10
   conda activate adsv_env
   pip install -r requirements.txt
   ```

2. **Run Prediction**:
   ```bash
   export MP_API_KEY="your_api_key"
   python Automated_Doping_Stability_Validation.py --mp_id "mp-542724" --doping "Yb3+:0.18, Er3+:0.02"
   ```

## 📖 Citation
```bibtex
@article{Yang2026Thermodynamic,
  title={Thermodynamic Criterion for Interfacial Engineering of Ultra-Stable Phosphor-in-Glass Composites},
  author={Yang, Yuanming and Huang, Ling et al.},
  journal={Submitted for publication},
  year={2026}
}
```
