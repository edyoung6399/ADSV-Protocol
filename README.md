# Automated Doping Stability Validation (ADSV) Protocol

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the **Automated Doping Stability Validation (ADSV)** protocol.

The ADSV protocol is an AI-accelerated thermodynamic screening workflow designed to evaluate the interfacial survivability of rare-earth doped functional ceramics (such as phosphors) within glass matrices. By leveraging universal machine learning interatomic potentials (CHGNet and MACE), this tool maps high-fidelity energy landscapes for doped supercells, ensuring that functional substitutions do not compromise the intrinsic thermodynamic stability of the host lattice.

This code serves as the computational companion to the manuscript:
**"Thermodynamic Criterion for Interfacial Engineering of Ultra-Stable Phosphor-in-Glass Composites"** (Submitted for publication).

## 📝 Features

* **Automated Supercell Construction:** Intelligently suggests optimal supercell dimensions based on available GPU VRAM.
* **Hybrid Site Screening:** Rapidly identifies the most thermodynamically favorable substitution sites using CHGNet.
* **Dynamic Thermodynamic Engine:** Automatically calculates dynamic chemical potentials for elements under O-rich limits using the Materials Project database.
* **Physical Validation Check:** Performs secondary stability cross-validation using MACE (Higher Order Equivariant Message Passing Neural Networks) to evaluate residual max forces.

## 🧮 Theoretical Background

The protocol validates the thermodynamic stability by calculating the aligned formation energy ($E_{aligned}$) of the doped supercell. The calculation bridges DFT-level accuracy with Machine Learning interatomic potentials:

$$E_{aligned}=E_{MP}+(M_{AI-Doped}-M_{AI-Host})$$

Where:
* $E_{MP}$ is the formation energy of the pristine host lattice from the Materials Project (DFT).
* $M_{AI-Doped}$ is the energy of the relaxed doped supercell calculated via CHGNet.
* $M_{AI-Host}$ is the energy of the relaxed pristine supercell calculated via CHGNet.

Sites are screened based on the substitution energy ($E_{sub}$) relative to the thermodynamic threshold ($\Delta E \le 0.5$ eV).

## ⚙️ Prerequisites and Installation

This script requires Python 3.9+ and a CUDA-enabled GPU (highly recommended) for efficient structural relaxation. 

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ADSV-Protocol.git](https://github.com/yourusername/ADSV-Protocol.git)
   cd ADSV-Protocol
   ```

2. **Create a Conda environment:**
   ```bash
   conda create -n adsv_env python=3.10
   conda activate adsv_env
   ```

3. **Install dependencies:**
   ```bash
   pip install chgnet pymatgen mp-api
   pip install mace-torch
   ```

## 🔑 Configuration

The script pulls reference thermodynamic and structural data directly from the Materials Project. You must set your Materials Project API key as an environment variable before running the script.

**For Linux/macOS:**
```bash
export MP_API_KEY="your_personal_mp_api_key_here"
```

**For Windows (Command Prompt):**
```cmd
set MP_API_KEY="your_personal_mp_api_key_here"
```

## 🚀 Usage

Run the script via the command line, specifying the Materials Project ID (`--mp_id`) of your host lattice and your desired doping scheme (`--doping`).

```bash
python Automated_Doping_Stability_Validation.py --mp_id "mp-542724" --doping "Yb3+:0.18, Er3+:0.02"
```

### Arguments:
* `--mp_id`: The Materials Project ID for the host crystal structure (e.g., `mp-542724` for La2Si2O7).
* `--doping`: A comma-separated string of dopant elements and their target substitution ratios (e.g., `Yb3+:0.18, Er3+:0.02`). The script automatically parses the elements and valence states.

## 📂 Output

Upon completion, the script generates a time-stamped directory in the `results/` folder containing:
1. **`*_Optimized.cif`**: The final, relaxed crystal structure of the doped supercell.
2. **`*_Report.txt`**: A detailed summary including crystallographic metrics, the aligned formation energy calculation, and the MACE physical stability criterion check.

## 📖 Citation

If you use this code in your research, please cite our corresponding paper:

```bibtex
@article{Yang2026Thermodynamic,
  title={Thermodynamic Criterion for Interfacial Engineering of Ultra-Stable Phosphor-in-Glass Composites},
  author={Yang, Yuanming and Sun, Chao and Lin, Jidong and Yao, Hanwen and Huang, Feng and Song, Hao and Long, Tengxiang and Peng, Yongli and Wu, Shasha and Liu, Jianhua and Wang, Hui and Bai, Jiashun and Wang, Yilin and Xi, Guoyu and Qiu, Huaxiang and Xie, Ke and Hu, Ziyi and Lin, Shisheng and Chen, Daqin and Zhu, Jinpei and Wang, Yihao and Zhang, Zhitang and Wang, Qiang and Luo, Changlong and Liu, Chengguo and Wei, Yang and Huang, Ling},
  journal={Submitted for publication},
  year={2026}
}
```
