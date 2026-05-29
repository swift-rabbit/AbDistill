"""
precompute_features.py
──────────────────────
Extracts per-complex features from Boltz2 outputs for teacher training.

Saved per npz (feature_cache/{seq_id}.npz):
  pocket        [N_p, 512]   ESM-IF1 per-residue embeddings (full antibody)
  ligand        [21,  768]   UniMol2 per-atom embeddings
  plddt         [N_p + 21]   per-token pLDDT (antibody residues + ligand atoms)
  pae_p2l       [N_p, 21]    pocket→ligand PAE  (low = antibody knows where ligand is)
  pae_l2p       [21,  N_p]   ligand→pocket PAE  (high, less useful — kept for completeness)
  pde_p2l       [N_p, 21]    pocket→ligand PDE  (distance error, orthogonal to PAE)
  pde_l2p       [21,  N_p]   ligand→pocket PDE
  pae_to_lig    [N_p]        per-residue mean PAE to ligand  (data-driven paratope map)

Chain layout assumed: chain A = heavy (N_heavy res), chain B = light (N_light res),
chain C = ligand (N_lig = 21 atoms). Total tokens = N_heavy + N_light + N_lig.
"""

import os
import sys
import contextlib
import torch
import numpy as np
import pandas as pd
from unimol_tools import UniMolRepr
import esm
import esm.inverse_folding.util
from tqdm import tqdm


# ─────────────────────────────────────────────
# PDB parsing
# ─────────────────────────────────────────────
def load_pdb_backbone(pdb_path):
    """
    Returns:
      bb_coords    torch.Tensor [N_p, 3, 3]   N / CA / C per protein residue
      ligand_atoms list[dict]                  elem + coord for ligand atoms
      n_heavy      int                         number of heavy-chain residues
      n_light      int                         number of light-chain residues
    """
    with open(pdb_path) as f:
        lines = f.readlines()

    res_dict   = {}   # (chain, res_id) → {atom_name: [x,y,z]}
    chain_order = []  # ordered list of chains seen
    ligand_atoms = []

    for line in lines:
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        chain     = line[21]
        x, y, z   = float(line[30:38]), float(line[38:46]), float(line[46:54])

        if chain == 'C':   # ligand
            elem = line[76:78].strip() or line[12:16].strip()[0]
            ligand_atoms.append({'elem': elem, 'coord': [x, y, z]})
            continue

        res_id    = line[22:26].strip()
        atom_name = line[12:16].strip()
        if atom_name in ('N', 'CA', 'C'):
            key = (chain, res_id)
            if key not in res_dict:
                res_dict[key] = {}
                if chain not in chain_order:
                    chain_order.append(chain)
            res_dict[key][atom_name] = [x, y, z]

    if not ligand_atoms:
        raise ValueError(f"No ligand atoms in {pdb_path}")

    # Build ordered coordinate list, track chain boundaries
    coords   = []
    n_heavy  = 0
    n_light  = 0
    for chain in chain_order:
        keys = [k for k in res_dict if k[0] == chain]
        for key in keys:
            atoms = res_dict[key]
            if 'N' in atoms and 'CA' in atoms and 'C' in atoms:
                coords.append([atoms['N'], atoms['CA'], atoms['C']])
            else:
                print(f"[WARN] Missing backbone atom at {key}, filling zeros")
                coords.append([[0,0,0],[0,0,0],[0,0,0]])
        if chain == 'A':
            n_heavy = len(keys)
        elif chain == 'B':
            n_light = len(keys)

    return torch.tensor(coords, dtype=torch.float32), ligand_atoms, n_heavy, n_light


# ─────────────────────────────────────────────
# Per-complex feature extraction
# ─────────────────────────────────────────────
def process_complex(seq_id, boltz_dir, esm_model, batch_converter, unimol_clf):
    pred_dir = os.path.join(boltz_dir, f"boltz_results_{seq_id}", "predictions", seq_id)
    pdb_path  = os.path.join(pred_dir, f"{seq_id}_model_0.pdb")

    bb_coords, ligand_atoms, n_heavy, n_light = load_pdb_backbone(pdb_path)
    N_p   = len(bb_coords)          # total protein residues
    N_lig = len(ligand_atoms)       # ligand atoms (21 for testosterone)

    # ── 1. ESM-IF1 pocket encoding ──────────────────────────────────────────
    seq = "A" * N_p
    batch = [(bb_coords, None, seq)]
    coords_batch, confidence_batch, _, _, padding_mask = batch_converter(batch)

    device = next(esm_model.parameters()).device
    coords_batch      = coords_batch.to(device)
    confidence_batch  = confidence_batch.to(device)
    padding_mask      = padding_mask.to(device)

    with torch.no_grad():
        encoder_out = esm_model.encoder(coords_batch, padding_mask, confidence_batch)
        features    = encoder_out['encoder_out'][0]
        pocket_repr = features[1:-1, 0, :].cpu().numpy()   # [N_p, 512]

    # ── 2. UniMol2 ligand encoding ──────────────────────────────────────────
    lig_dict = {
        'atoms':       [[a['elem'] for a in ligand_atoms]],
        'coordinates': [[[a['coord'][0], a['coord'][1], a['coord'][2]]
                          for a in ligand_atoms]],
    }
    
    # Silence UniMol's hardcoded internal tqdm spam
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            ligand_repr = unimol_clf.get_repr(
                lig_dict, return_atomic_reprs=True
            )['atomic_reprs'][0].astype(np.float32)                 # [N_lig, 768]

    # ── 3. PAE, PDE, pLDDT from Boltz2 outputs ─────────────────────────────
    plddt_full = np.load(
        os.path.join(pred_dir, f"plddt_{seq_id}_model_0.npz")
    )['plddt']                                              # [N_p + N_lig]

    pae_full = np.load(
        os.path.join(pred_dir, f"pae_{seq_id}_model_0.npz")
    )['pae']                                                # [N_p + N_lig, N_p + N_lig]

    pde_full = np.load(
        os.path.join(pred_dir, f"pde_{seq_id}_model_0.npz")
    )['pde']                                                # [N_p + N_lig, N_p + N_lig]

    # Slice out interface sub-matrices
    # pocket→ligand: antibody residues (rows) to ligand atoms (cols)
    #   row 0:N_p  = all antibody residues
    #   col N_p:   = ligand atoms
    pae_p2l = pae_full[:N_p, N_p:].astype(np.float32)     # [N_p, N_lig]
    pae_l2p = pae_full[N_p:, :N_p].astype(np.float32)     # [N_lig, N_p]
    pde_p2l = pde_full[:N_p, N_p:].astype(np.float32)     # [N_p, N_lig]
    pde_l2p = pde_full[N_p:, :N_p].astype(np.float32)     # [N_lig, N_p]

    # Per-residue paratope confidence: mean PAE from each antibody residue to all ligand atoms
    # Low value = that residue is confidently positioned relative to the ligand = likely paratope
    pae_to_lig = pae_p2l.mean(axis=1).astype(np.float32)  # [N_p]

    # pLDDT split: protein tokens then ligand tokens
    plddt_pocket = plddt_full[:N_p].astype(np.float32)    # [N_p]
    plddt_ligand = plddt_full[N_p:].astype(np.float32)    # [N_lig]
    plddt_combined = np.concatenate([plddt_pocket, plddt_ligand])  # [N_p + N_lig]

    return {
        'pocket':     pocket_repr,    # [N_p, 512]
        'ligand':     ligand_repr,    # [N_lig, 768]
        'plddt':      plddt_combined, # [N_p + N_lig]   (backward-compat key)
        'pae_p2l':    pae_p2l,        # [N_p, N_lig]   ← use this in teacher
        'pae_l2p':    pae_l2p,        # [N_lig, N_p]
        'pde_p2l':    pde_p2l,        # [N_p, N_lig]
        'pde_l2p':    pde_l2p,        # [N_lig, N_p]
        'pae_to_lig': pae_to_lig,     # [N_p]           data-driven paratope map
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(input_csv, boltz_dir, out_dir):
    df        = pd.read_csv(input_csv)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading UniMol2...")
    unimol_clf = UniMolRepr(data_type='molecule', model_name='unimolv2', model_size='84m')

    print("Loading ESM-IF1...")
    esm_model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    esm_model = esm_model.eval()
    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)

    skipped  = 0
    failed   = 0
    computed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq_id   = row['id']
        out_path = os.path.join(out_dir, f"{seq_id}.npz")

        if os.path.exists(out_path):
            # Upgrade existing cache files that are missing new keys
            existing = np.load(out_path)
            if 'pae_p2l' in existing:
                skipped += 1
                continue
            # Fall through to recompute

        try:
            data = process_complex(seq_id, boltz_dir, esm_model, batch_converter, unimol_clf)
            np.savez_compressed(
                out_path,
                pocket     = data['pocket'],
                ligand     = data['ligand'],
                plddt      = data['plddt'],
                pae_p2l    = data['pae_p2l'],
                pae_l2p    = data['pae_l2p'],
                pde_p2l    = data['pde_p2l'],
                pde_l2p    = data['pde_l2p'],
                pae_to_lig = data['pae_to_lig'],
            )
            computed += 1
        except Exception as e:
            print(f"[FAIL] {seq_id}: {e}")
            failed += 1

    print(f"\nDone. computed={computed}  skipped={skipped}  failed={failed}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extracts per-complex features from Boltz2 outputs for teacher training.")
    parser.add_argument("--input", type=str, default="train_set.csv", help="Input CSV (e.g. train_set.csv)")
    parser.add_argument("--boltz_dir", type=str, default="boltz_out", help="Directory containing boltz_results_* folders")
    parser.add_argument("--out_dir", type=str, default="feature_cache", help="Output directory for cached .npz files")
    args = parser.parse_args()
    
    main(args.input, args.boltz_dir, args.out_dir)
