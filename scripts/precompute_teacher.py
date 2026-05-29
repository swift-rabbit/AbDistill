"""
precompute_teacher.py
──────────────────────
Runs the trained BindNet Teacher model on the dataset to pre-compute
the 256-D complex representations (targets for Student distillation).

Run this AFTER train_teacher.py and BEFORE train.py.

Usage:
    python precompute_teacher.py
"""

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abdistill.models import TeacherModel

def precompute(model_weights, dataset_csv, cache_dir, out_embeddings, out_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load trained model
    model = TeacherModel(p_in_dim=512, l_in_dim=768, hidden_dim=256, num_heads=4, num_layers=3).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    df = pd.read_csv(dataset_csv)

    embeddings = []
    index = {}

    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=len(df)):
            seq_id = row['id']
            path = os.path.join(cache_dir, f"{seq_id}.npz")

            try:
                data     = np.load(path)
                pocket_t = torch.tensor(data['pocket'], dtype=torch.float32, device=device).unsqueeze(0)
                ligand_t = torch.tensor(data['ligand'], dtype=torch.float32, device=device).unsqueeze(0)
                N_p      = pocket_t.shape[1]

                plddt   = torch.tensor(data['plddt'], dtype=torch.float32, device=device)
                p_plddt = plddt[:N_p].unsqueeze(0)
                l_plddt = plddt[N_p:].unsqueeze(0)

                # New keys — fall back to full PAE if cache is old format
                if 'pae_p2l' in data:
                    pae_p2l    = torch.tensor(data['pae_p2l'],    dtype=torch.float32, device=device).unsqueeze(0)
                    pae_l2p    = torch.tensor(data['pae_l2p'],    dtype=torch.float32, device=device).unsqueeze(0)
                    pde_p2l    = torch.tensor(data['pde_p2l'],    dtype=torch.float32, device=device).unsqueeze(0)
                    pde_l2p    = torch.tensor(data['pde_l2p'],    dtype=torch.float32, device=device).unsqueeze(0)
                    pae_to_lig = torch.tensor(data['pae_to_lig'], dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    pae        = torch.tensor(data['pae'], dtype=torch.float32, device=device)
                    pae_l2p    = pae[N_p:, :N_p].unsqueeze(0)
                    pae_p2l    = pae[:N_p, N_p:].unsqueeze(0)
                    pde_l2p    = torch.zeros_like(pae_l2p)
                    pde_p2l    = torch.zeros_like(pae_p2l)
                    pae_to_lig = pae_p2l.squeeze(0).mean(dim=1).unsqueeze(0)

                _, complex_embed = model(
                    pocket_t, ligand_t, p_plddt, l_plddt,
                    pae_l2p, pae_p2l, pde_l2p, pde_p2l,
                    pae_to_lig, p_mask=None
                )

                index[seq_id] = len(embeddings)
                embeddings.append(complex_embed.cpu().numpy()[0])
            except Exception as e:
                print(f"[WARN] Failed to process {seq_id}: {e}")

    emb_matrix = np.array(embeddings, dtype=np.float32)
    np.save(out_embeddings, emb_matrix)
    # Save index for dataloader
    with open(out_index, 'w') as f:
        json.dump(index, f)

    print(f"Saved {out_embeddings} with shape {emb_matrix.shape}")
    print("Done.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Precompute the 256-D complex representations using the trained TeacherModel.")
    parser.add_argument("--model", type=str, default="best_teacher.pt", help="Path to teacher weights")
    parser.add_argument("--input", type=str, default="final_master_dataset.csv", help="Input dataset CSV")
    parser.add_argument("--cache_dir", type=str, default="feature_cache", help="Directory containing precomputed .npz features")
    parser.add_argument("--out_emb", type=str, default="teacher_embeddings.npy", help="Output path for embeddings")
    parser.add_argument("--out_idx", type=str, default="teacher_embedding_index.json", help="Output path for embedding index")
    args = parser.parse_args()
    
    precompute(args.model, args.input, args.cache_dir, args.out_emb, args.out_idx)
