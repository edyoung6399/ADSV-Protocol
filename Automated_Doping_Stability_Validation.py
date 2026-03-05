"""
Automated Doping Stability Validation (ADSV) Protocol
---------------------------------------------------------
This script implements the ADSV protocol to verify thermodynamic 
stability for rare-earth substitution in glass-ceramic hosts, 
utilizing universal machine learning force fields (CHGNet and MACE).

Reference: "Thermodynamic Criterion for Interfacial Engineering of 
Ultra-Stable Phosphor-in-Glass Composites" (Submitted for publication)
Authors: Yuanming Yang, Chao Sun, Jidong Lin, Hanwen Yao, et al.
"""

import os, numpy as np, time, psutil, csv, warnings, torch, gc, sys, json, argparse, re
from pathlib import Path
from datetime import datetime
from chgnet.model import CHGNet, StructOptimizer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester
import requests
from urllib3.exceptions import InsecureRequestWarning

# ==========================================
# 1. Environment & Resource Management
# ==========================================
os.environ.update({
    'http_proxy': '', 'https_proxy': '', 'HTTP_PROXY': '', 'HTTPS_PROXY': '',
    'CURL_CA_BUNDLE': '', 'NO_PROXY': '*',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'
})
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
warnings.filterwarnings('ignore')

try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0

def format_time(seconds):
    return f"{seconds:.1f}s"

# ==========================================
# 2. Dynamic Thermodynamic Engine (O-rich Limit)
# ==========================================
CACHE_FILE = "chgnet_thermo_cache.json"

def load_mu_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: return {}
    return {}

def save_mu_cache(cache_dict):
    clean_dict = {k: float(v) for k, v in cache_dict.items()}
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_dict, f, indent=4)

def get_dynamic_mu_chgnet(el, model, relaxer, mpr, local_cache, env='O-rich'):
    if el in local_cache:
        return local_cache[el]
    
    print(f"  [Thermodynamics] No local cache found. Calculating ref potential for {el} ({env} limit)")
    try:
        if el == 'O':
            print("      ├─ Retrieving standard O reference phase (mp-12957)...")
            doc = mpr.summary.get_data_by_id("mp-12957")
            res = relaxer.relax(doc.structure, verbose=False)
            mu = float(res['trajectory'].energies[-1] / len(doc.structure))
            local_cache[el] = mu
            save_mu_cache(local_cache)
            print(f"      └─ O relaxation complete | μ_O = {mu:.4f} eV")
            return mu
            
        if env == 'O-rich':
            mu_O = get_dynamic_mu_chgnet('O', model, relaxer, mpr, local_cache, env='metal-rich')
            with SuppressStdout():
                docs = mpr.summary.search(chemsys=f"{el}-O", is_stable=True)
            
            if docs:
                docs = sorted(docs, key=lambda x: x.formation_energy_per_atom)
                oxide_doc = docs[0]
                struct = oxide_doc.structure
                comp = struct.composition
                n_el, n_O = comp.get(el, 0), comp.get('O', 0)
                
                print(f"      ├─ Found stable precursor: {oxide_doc.formula_pretty} ({oxide_doc.material_id})")
                res = relaxer.relax(struct, verbose=False)
                e_total = float(res['trajectory'].energies[-1])
                
                mu = (e_total - n_O * mu_O) / n_el
                local_cache[el] = mu
                save_mu_cache(local_cache)
                print(f"      └─ Deduction complete | Excluding O, μ_{el} = {mu:.4f} eV")
                return mu
            else:
                print(f"      ├─ No {el}-O stable phase found. Defaulting to elemental limit...")
                
        with SuppressStdout():
            docs = mpr.summary.search(chemsys=el, is_stable=True)
            if not docs: raise ValueError(f"Elemental structure for {el} not found in MP.")
            struct = sorted(docs, key=lambda x: x.formation_energy_per_atom)[0].structure
        
        print(f"      ├─ Found stable elemental phase ({docs[0].material_id})")
        res = relaxer.relax(struct, verbose=False)
        mu = float(res['trajectory'].energies[-1] / len(struct))
        local_cache[el] = mu
        save_mu_cache(local_cache)
        print(f"      └─ Elemental phase relaxation complete | μ_{el} = {mu:.4f} eV")
        return mu
    except Exception as e:
        raise ValueError(f"\n[Fatal Error] Unable to dynamically calculate chemical potential for {el}: {str(e)}")

def calculate_aligned_formation_energy(mp_ef_host, e_ml_doped, e_ml_host):
    return mp_ef_host + (e_ml_doped - e_ml_host)

# ==========================================
# 3. Supercell Suggestion
# ==========================================
def suggest_supercells(structure, vram_gb):
    lattice = structure.lattice
    abc = lattice.abc
    num_atoms = len(structure)
    safe_atom_limit = int(vram_gb * 120) if vram_gb > 0 else 200
    
    options = []
    for k in [1, 2, 3, 4]:
        dim_str = f"{k}x{k}x{k}"
        min_len = min([x * k for x in abc])
        total_atoms = num_atoms * (k**3)
        est_min = (0.6 * total_atoms) / 60
        
        score = 0
        if min_len >= 10.0: score += 2
        elif min_len >= 8.0: score += 1
            
        if total_atoms <= safe_atom_limit: score += 2
        elif total_atoms <= safe_atom_limit * 1.5: score += 1
        else: score -= 10
        
        if est_min > 15: score -= 1

        if score >= 4: star = "Highly Recommended"
        elif score == 3: star = "Recommended"
        elif score == 2: star = "Marginal"
        else: star = "Not Recommended"
        
        options.append({
            'dim': dim_str, 'atoms': total_atoms, 'min_len': min_len,
            'time': f"{est_min:.1f} min", 'star': star
        })
    return options

# ==========================================
# 4. ADSV Site Screening Engine
# ==========================================
def hybrid_site_screening(model, relaxer, host_struct, main_dopant, candidate_sites, mpr, sc_dims, local_cache):
    SCREEN_FMAX = 0.2 
    print(f"\n[Step 1] Establishing Host Reference")
    t0 = time.time()
    base_sc_host = host_struct.copy()
    base_sc_host.make_supercell(sc_dims)
    
    clear_memory()
    res_host = relaxer.relax(base_sc_host, fmax=SCREEN_FMAX, verbose=False)
    e_host_total = res_host['trajectory'].energies[-1]
    relaxed_host_sc = res_host['final_structure']
    print(f"  └─ Host relaxation complete | Time: {format_time(time.time()-t0)} | M_AI_Host: {e_host_total:.4f} eV")

    top_candidates = list(candidate_sites.keys())

    # --- UI Fix: Pre-load thermodynamic potentials to prevent table breaking ---
    print(f"\n[Step 1.5] Thermodynamic Reference Pre-loading")
    mu_add = get_dynamic_mu_chgnet(main_dopant, model, relaxer, mpr, local_cache, env='O-rich')
    for el in top_candidates:
        get_dynamic_mu_chgnet(el, model, relaxer, mpr, local_cache, env='O-rich')
    
    fine_results = {}
    
    print(f"\n[Step 2] Parallel Site Evaluation ({len(top_candidates)} candidates)")
    print(f"  {'-'*65}")
    print(f"  | Site   | Time       | M_AI_Doped     | E_sub (eV)   |")
    print(f"  {'-'*65}")
    
    for idx, el in enumerate(top_candidates):
        clear_memory()
        t_site_start = time.time()
        
        # Safe hit from cache, no console output overlap
        mu_rem = get_dynamic_mu_chgnet(el, model, relaxer, mpr, local_cache, env='O-rich')
        
        test_sc_doped = relaxed_host_sc.copy()
        for i, s in enumerate(test_sc_doped):
            if s.specie.symbol == el:
                test_sc_doped.replace(i, main_dopant)
                break
        
        print(f"  | {el:<6} | {'Relaxing...':<38} |", end='\r', flush=True)
                
        res_doped = relaxer.relax(test_sc_doped, fmax=SCREEN_FMAX, verbose=False)
        e_doped_total = res_doped['trajectory'].energies[-1]
        
        e_sub = e_doped_total - e_host_total - mu_add + mu_rem
        duration = time.time() - t_site_start
        fine_results[el] = {
            'e_sub': e_sub, 'e_per_atom': e_doped_total / len(test_sc_doped),
            'structure': res_doped['final_structure']
        }
        
        print(f"  | {el:<6} | {format_time(duration):<10} | {e_doped_total:<14.2f} | {e_sub:<12.4f} |")
        
    print(f"  {'-'*65}")
    return fine_results, e_host_total/len(base_sc_host), relaxed_host_sc

# ==========================================
# 5. MACE Validation & Report Generation
# ==========================================
def mace_validation_hardcore(structure, device='cuda'):
    if not MACE_AVAILABLE: return {'error': 'MACE not installed'}, device
    clear_memory()
    try:
        ase_atoms = AseAtomsAdaptor.get_atoms(structure)
        calc = mace_mp(model="medium", device=device, default_dtype="float64")
        ase_atoms.calc = calc
        return {
            'e_per_atom': float(ase_atoms.get_potential_energy() / len(ase_atoms)),
            'max_force': float(np.max(np.abs(ase_atoms.get_forces()))),
        }, device
    except Exception as e: return {'error': str(e)}, device

def generate_rich_report(res):
    lat = res['struct'].lattice
    dims = res['sc_dims']
    n_atoms = len(res['struct'])
    vol_per_atom = lat.volume / n_atoms
    
    norm_a = lat.a / dims[0]
    norm_b = lat.b / dims[1]
    norm_c = lat.c / dims[2]
    norm_vol = lat.volume / (dims[0] * dims[1] * dims[2]) 
    
    mp_ef = res['mp_ef_host']
    e_host = res['e_ml_host']
    e_doped = res['chgnet_e']
    delta_e = e_doped - e_host
    final_ef = res['aligned_ef']
    
    lines = [
        "==================================================================",
        f" ADSV Report: {res['host_el']} -> {res['scheme_str']}",
        "==================================================================",
        f"  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Material ID : {res['mp_id']} ({res['formula']})",
        f"  Space Group : {res['spacegroup']}",
        f"  Supercell   : {dims[0]}x{dims[1]}x{dims[2]} (Total Atoms: {n_atoms})",
        f"  Device Used : {res['device'].upper()}",
        f"  Total Time  : {res['total_time']}",
        "",
        "[Crystallographic Metrics]",
        f"  Volume/Atom : {vol_per_atom:.3f} A^3",
        "",
        "[Lattice Parameters (Supercell / Normalized)]",
        f"  a: {lat.a:.4f} A  (Norm: {norm_a:.2f})",
        f"  b: {lat.b:.4f} A  (Norm: {norm_b:.2f})",
        f"  c: {lat.c:.4f} A  (Norm: {norm_c:.2f})",
        f"  alpha: {lat.alpha:.2f} deg",
        f"  beta:  {lat.beta:.2f} deg",
        f"  gamma: {lat.gamma:.2f} deg",
        f"  Total Vol: {lat.volume:.3f} A^3",
        f"  Norm Vol:  {norm_vol:.3f} A^3", 
        "",
        "[Aligned Formation Energy Calculation]",
        "  Formula: E_aligned = E_MP + (M_AI_Doped - M_AI_Host)",
        "  -------------------------------------------------------",
        f"  [A] MP Reference Energy : {mp_ef:.6f} eV/atom (DFT)",
        f"  [B] M_AI_Host           : {e_host:.6f} eV/atom (CHGNet)",
        f"  [C] M_AI_Doped          : {e_doped:.6f} eV/atom (CHGNet)",
        f"  [D] Delta Energy (C-B)  : {delta_e:+.6f} eV/atom",
        "  -------------------------------------------------------",
        f"  Final E_aligned (A+D)   : {final_ef:.6f} eV/atom",
        "",
        "[Physical Stability Criterion (MACE Check)]",
        f"  MACE Cross-val Energy   : {res['mace_e']} eV/atom",
        f"  MACE Residual Max Force : {res['mace_f']} eV/A",
    ]
    
    mace_f = res['mace_f']
    if isinstance(mace_f, (int, float)):
        status = "Stable" if mace_f < 0.5 else "Requires Attention (High Force)"
    else:
        status = "Failed or MACE uninstalled"
    lines.append(f"  Stability Status        : {status}")
    lines.append("==================================================================")
    return "\n".join(lines)

# ==========================================
# 6. Main Pipeline
# ==========================================
def run_adsv_pipeline(args):
    global_start = time.time()
    API_KEY = os.environ.get("MP_API_KEY")
    vram = get_gpu_memory()
    device = 'cuda' if vram > 0 else 'cpu'
    
    print(f"\n{'='*20} Automated Doping Stability Validation {'='*20}")
    print(f"  Hardware: {device.upper()} | Available VRAM: {vram:.2f} GB")
    
    mp_id = args.mp_id
    doping_input = args.doping
    
    scheme_raw = doping_input.replace(',', ',')
    doping_scheme = {}
    main_dopant = None
    
    print("\n[Input Parsing]")
    for part in scheme_raw.split(','):
        if not part.strip(): continue
        k, v = part.split(':')
        val = float(v.strip().replace('%',''))
        val = val/100 if val > 1 else val
        
        match = re.match(r'([A-Z][a-z]?)(\d*[\+-])?', k.strip())
        if match:
            el = match.group(1)
            valence = match.group(2) if match.group(2) else "Auto"
            doping_scheme[el] = val
            if main_dopant is None: main_dopant = el
            print(f"  ├─ Element: {el} | Valence: {valence} | Ratio: {val}")

    print("\n[Step 0] Initialization & Physical Reference Check...")
    local_cache = load_mu_cache()
    
    print(f"  Loading CHGNet...", end=" ")
    model = CHGNet.load()
    relaxer = StructOptimizer()
    print("Done")
    
    with MPRester(API_KEY) as mpr:
        summary = mpr.summary.get_data_by_id(mp_id)
        host_struct = summary.structure
        mp_ef_host = summary.formation_energy_per_atom
        sg_symbol = SpacegroupAnalyzer(host_struct).get_space_group_symbol()
        
        print(f"\n[Supercell Suggestions] (Host: {host_struct.formula}, SG: {sg_symbol})")
        print(f"{'-'*75}")
        print(f"| Size     | Atoms    | Min Dist   | Est. Time    | Recommendation     |")
        print(f"{'-'*75}")
        
        for o in suggest_supercells(host_struct, vram):
            print(f"| {o['dim']:<8} | {o['atoms']:<8} | {o.get('min_len', 0):<8.1f} A | {o['time']:<12} | {o['star']:<18} |")
        print(f"{'-'*75}")
        
        sc_str = input(f"\n  Input supercell dimensions [Default: 1 1 1]: ").strip() or "1 1 1"
        sc_dims = [int(x) for x in sc_str.replace('x',' ').split()]

        sga = SpacegroupAnalyzer(host_struct)
        sym_struct = sga.get_symmetrized_structure()
        candidate_sites = {}
        for i in [eq[0] for eq in sym_struct.equivalent_indices]:
            site_el = host_struct[i].specie.symbol
            if site_el not in ['O', 'F', 'S', 'Cl', 'N']: 
                candidate_sites[site_el] = i
        
        fine_results, e_ml_host, relaxed_host_sc = hybrid_site_screening(
            model, relaxer, host_struct, main_dopant, candidate_sites, mpr, sc_dims, local_cache
        )
    
    if not fine_results: return

    best_e_sub = min(d['e_sub'] for d in fine_results.values())
    ENERGY_CUTOFF = 0.5
    
    print(f"\n[Step 3] Thermodynamic Cutoff Check (Threshold: Delta E < {ENERGY_CUTOFF} eV)")
    valid_sites = {s: d for s, d in fine_results.items() if d['e_sub'] - best_e_sub <= ENERGY_CUTOFF}
    for site, data in sorted(fine_results.items(), key=lambda x: x[1]['e_sub']):
        delta_e = data['e_sub'] - best_e_sub
        status = "Pass" if delta_e <= ENERGY_CUTOFF else "Fail"
        print(f"  ├─ {site}: E_sub = {data['e_sub']:>8.4f} eV | Delta E = {delta_e:>6.4f} eV ({status})")

    run_folder = Path(f"results/Run_{datetime.now().strftime('%m%d_%H%M')}_{mp_id}")
    run_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Step 4] Deep Validation (Target Sites: {len(valid_sites)})")
    
    for i, (site_el, data) in enumerate(valid_sites.items()):
        print(f"  >>> [{i+1}/{len(valid_sites)}] Generating detailed report for site: {site_el} ...")
        clear_memory()
        
        final_sc = relaxed_host_sc.copy()
        target_indices = [idx for idx, s in enumerate(final_sc) if s.specie.symbol == site_el]
        np.random.shuffle(target_indices)
        
        curr_idx = 0
        for dop_el, ratio in doping_scheme.items():
            count = max(1, int(round(len(target_indices) * ratio)))
            actual_count = min(count, len(target_indices) - curr_idx)
            for _ in range(actual_count):
                final_sc.replace(target_indices[curr_idx], dop_el)
                curr_idx += 1
        
        print(f"      ├─ Final Structure Optimization (fmax=0.08)...", end=" ", flush=True)
        res_final = relaxer.relax(final_sc, fmax=0.08, verbose=False)
        f_struct = res_final['final_structure']
        e_ml_doped = res_final['trajectory'].energies[-1] / len(f_struct)
        print("Done")
        
        print("      ├─ MACE Physical Validation...", end=" ", flush=True)
        m_res, dev_used = mace_validation_hardcore(f_struct, device)
        print(f"Done ({dev_used.upper()})" if 'error' not in m_res else f"Failed ({m_res['error']})")
        
        formula_clean = f_struct.formula.replace(" ", "")
        file_prefix = f"{mp_id}_{formula_clean}_{site_el}-Site"
        
        report_data = {
            'mp_id': mp_id, 'formula': f_struct.formula, 'host_el': site_el, 
            'scheme_str': doping_input, 'struct': f_struct, 'sc_dims': sc_dims, 
            'spacegroup': sg_symbol, 'device': dev_used, 
            'total_time': format_time(time.time() - global_start),
            'mp_ef_host': mp_ef_host, 'e_ml_host': e_ml_host, 'chgnet_e': e_ml_doped,
            'aligned_ef': calculate_aligned_formation_energy(mp_ef_host, e_ml_doped, e_ml_host),
            'mace_e': m_res.get('e_per_atom', 'N/A'), 'mace_f': m_res.get('max_force', 'N/A')
        }
        
        report_path = run_folder / f"{file_prefix}_Report.txt"
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(generate_rich_report(report_data))
            
        cif_path = run_folder / f"{file_prefix}_Optimized.cif"
        f_struct.to(filename=str(cif_path))
        
        print(f"      └─ Saved: {file_prefix}_Report.txt")
        
    print(f"\n*** Process Completed! Results saved in: {run_folder} ***")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp_id", type=str, default="mp-542724")
    parser.add_argument("--doping", type=str, default="Yb3+:0.18, Er3+:0.02")
    try: args = parser.parse_args()
    except SystemExit: args = argparse.Namespace(mp_id="mp-542724", doping="Yb3+:0.18, Er3+:0.02")

    try: run_adsv_pipeline(args)
    except Exception as e:
        import traceback; traceback.print_exc()
        input("\nExecution error. Press Enter to exit...")
