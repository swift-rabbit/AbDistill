"""
train_teacher.py  (v2)
──────────────────────
Trains the BindNet TeacherModel on precomputed features from feature_cache/.
Loss: Asymmetric ListMLE on SMINA Vinardo scores.

New in v2:
  - Loads pae_p2l / pae_l2p directly (correctly-oriented slices, not full matrix)
  - Loads pde_p2l / pde_l2p (new PDE attention bias)
  - Loads pae_to_lig (per-residue paratope confidence map)
  - Backward-compatible: falls back to reconstructing from 'pae' if new keys absent
  - Two independent checkpoint files (EEF@1% and Spearman)

Run this AFTER precompute_features.py and prepare_dataset.py.

Usage:
    python train_teacher.py
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abdistill.models import TeacherModel
from abdistill.distillation import asymmetric_list_mle_loss, early_enrichment_factor


class FeatureTeacherDataset(Dataset):
    def __init__(self, csv_file: str, cache_dir: str = 'feature_cache'):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        seq_id = row['id']
        path   = os.path.join(self.cache_dir, f"{seq_id}.npz")

        try:
            data   = np.load(path)
            pocket = torch.tensor(data['pocket'], dtype=torch.float32)
            ligand = torch.tensor(data['ligand'], dtype=torch.float32)
            plddt  = torch.tensor(data['plddt'],  dtype=torch.float32)
            N_p    = pocket.shape[0]
            p_plddt = plddt[:N_p]
            l_plddt = plddt[N_p:]

            # New keys (precompute_features.py v2) — fall back to full PAE if absent
            if 'pae_p2l' in data:
                pae_p2l    = torch.tensor(data['pae_p2l'],    dtype=torch.float32)  # [N_p, 21]
                pae_l2p    = torch.tensor(data['pae_l2p'],    dtype=torch.float32)  # [21, N_p]
                pde_p2l    = torch.tensor(data['pde_p2l'],    dtype=torch.float32)  # [N_p, 21]
                pde_l2p    = torch.tensor(data['pde_l2p'],    dtype=torch.float32)  # [21, N_p]
                pae_to_lig = torch.tensor(data['pae_to_lig'], dtype=torch.float32)  # [N_p]
            else:
                # Legacy: reconstruct from full PAE matrix
                pae     = torch.tensor(data['pae'], dtype=torch.float32)
                pae_l2p = pae[N_p:, :N_p]
                pae_p2l = pae[:N_p, N_p:]
                pde_p2l = torch.zeros_like(pae_p2l)
                pde_l2p = torch.zeros_like(pae_l2p)
                pae_to_lig = pae_p2l.mean(dim=1)   # approximate from full PAE

        except Exception:
            return None

        return {
            'seq_id':         seq_id,
            'pocket':         pocket,
            'ligand':         ligand,
            'p_plddt':        p_plddt,
            'l_plddt':        l_plddt,
            'pae_l2p':        pae_l2p,
            'pae_p2l':        pae_p2l,
            'pde_l2p':        pde_l2p,
            'pde_p2l':        pde_p2l,
            'pae_to_lig':     pae_to_lig,
            'smina':          torch.tensor(row['smina_vinardo_affinity'], dtype=torch.float32),
            'ranking_weight': torch.tensor(row['ranking_weight'],         dtype=torch.float32),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    ligands  = torch.stack([b['ligand']  for b in batch])
    l_plddts = torch.stack([b['l_plddt'] for b in batch])
    sminas   = torch.stack([b['smina']   for b in batch])
    weights  = torch.stack([b['ranking_weight'] for b in batch])

    pockets         = [b['pocket']  for b in batch]
    pocket_padded   = pad_sequence(pockets, batch_first=True, padding_value=0.0)

    p_plddts_padded = pad_sequence(
        [b['p_plddt'] for b in batch], batch_first=True, padding_value=0.0
    )

    # pae_l2p / pde_l2p: [21, N_p] — transpose → pad on N_p dim → transpose back
    def pad_l2p(key):
        return pad_sequence(
            [b[key].transpose(0, 1) for b in batch],
            batch_first=True, padding_value=0.0
        ).transpose(1, 2)   # [B, 21, max_N_p]

    def pad_p2l(key):
        return pad_sequence(
            [b[key] for b in batch],
            batch_first=True, padding_value=0.0
        )   # [B, max_N_p, 21]

    def pad_np(key):
        return pad_sequence(
            [b[key] for b in batch],
            batch_first=True, padding_value=0.0
        )   # [B, max_N_p]

    pae_l2p_padded = pad_l2p('pae_l2p')
    pae_p2l_padded = pad_p2l('pae_p2l')
    pde_l2p_padded = pad_l2p('pde_l2p')
    pde_p2l_padded = pad_p2l('pde_p2l')
    pae_to_lig_pad = pad_np('pae_to_lig')

    lengths = torch.tensor([len(p) for p in pockets])
    max_len = pocket_padded.size(1)
    p_mask  = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    return {
        'ligand':         ligands,
        'pocket':         pocket_padded,
        'p_plddt':        p_plddts_padded,
        'l_plddt':        l_plddts,
        'pae_l2p':        pae_l2p_padded,
        'pae_p2l':        pae_p2l_padded,
        'pde_l2p':        pde_l2p_padded,
        'pde_p2l':        pde_p2l_padded,
        'pae_to_lig':     pae_to_lig_pad,
        'p_mask':         p_mask,
        'smina':          sminas,
        'ranking_weight': weights,
    }


def make_sampler(dataset: FeatureTeacherDataset) -> WeightedRandomSampler:
    """
    Over-sample strong binders so top-1% SMINA binders appear in ~every batch.

    Sample weight = inverse of SMINA percentile (clipped at 5th pct so mediocre
    binders aren't starved completely).  Samples with failed feature caches
    (None items) receive weight 0 — they are silently skipped by collate_fn anyway.
    """
    smina = dataset.df['smina_vinardo_affinity'].values
    # Rank from best (most negative) → rank 0
    ranks    = smina.argsort().argsort().astype(float)   # ascending rank
    N        = len(ranks)
    # Convert rank to weight: best binder → highest weight
    inv_rank = (N - ranks)          # best = N, worst ≈ 1
    # Soft-clip: ensure no sample has weight < 5 % of max
    inv_rank = np.clip(inv_rank, 0.05 * N, N)
    inv_rank = inv_rank / inv_rank.sum()
    return WeightedRandomSampler(
        weights=torch.from_numpy(inv_rank.astype(np.float32)),
        num_samples=len(dataset),
        replacement=True,
    )


def train_teacher(train_csv, val_csv, epochs, lr, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    EPOCHS     = epochs       # extra epochs — scheduler handles the late-stage refinement
    LR         = lr
    BATCH_SIZE = batch_size       # larger batches → better ListMLE ranking signal per step

    train_ds = FeatureTeacherDataset(train_csv)
    val_ds   = FeatureTeacherDataset(val_csv)

    train_sampler = make_sampler(train_ds)
    train_loader  = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,   # replaces shuffle=True
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    model = TeacherModel(
        p_in_dim=512, l_in_dim=768, hidden_dim=256, num_heads=4, num_layers=3
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine decay from LR → eta_min over all epochs.
    # The scheduler is the main fix: prevents the LR-oscillation plateau we saw before.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    best_eef      = 0.0
    best_spearman = 1.0   # most-negative = best; init to +1 so any real negative triggers save

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Teacher Train]")
        for batch in pbar:
            if batch is None:
                continue

            optimizer.zero_grad()

            h_p     = batch['pocket'].to(device)
            h_l     = batch['ligand'].to(device)
            p_plddt = batch['p_plddt'].to(device)
            l_plddt = batch['l_plddt'].to(device)
            pae_l2p = batch['pae_l2p'].to(device)
            pae_p2l = batch['pae_p2l'].to(device)
            p_mask  = batch['p_mask'].to(device)
            sminas  = batch['smina'].to(device)
            weights = batch['ranking_weight'].to(device)

            pde_l2p    = batch['pde_l2p'].to(device)
            pde_p2l    = batch['pde_p2l'].to(device)
            pae_to_lig = batch['pae_to_lig'].to(device)

            score, _ = model(
                h_p, h_l, p_plddt, l_plddt,
                pae_l2p, pae_p2l, pde_l2p, pde_p2l,
                pae_to_lig, p_mask=p_mask
            )
            score = score.squeeze(-1)   # [B]

            loss = asymmetric_list_mle_loss(score, sminas, weights, fn_penalty=2.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr':   f'{scheduler.get_last_lr()[0]:.2e}',
            })

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_smina_all = []
        val_pred_all  = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Teacher Val]", leave=False):
                if batch is None:
                    continue
                h_p        = batch['pocket'].to(device)
                h_l        = batch['ligand'].to(device)
                p_plddt    = batch['p_plddt'].to(device)
                l_plddt    = batch['l_plddt'].to(device)
                pae_l2p    = batch['pae_l2p'].to(device)
                pae_p2l    = batch['pae_p2l'].to(device)
                pde_l2p    = batch['pde_l2p'].to(device)
                pde_p2l    = batch['pde_p2l'].to(device)
                pae_to_lig = batch['pae_to_lig'].to(device)
                p_mask     = batch['p_mask'].to(device)

                score, _ = model(
                    h_p, h_l, p_plddt, l_plddt,
                    pae_l2p, pae_p2l, pde_l2p, pde_p2l,
                    pae_to_lig, p_mask=p_mask
                )
                score = score.squeeze(-1)

                val_smina_all.extend(batch['smina'].tolist())
                val_pred_all.extend(score.cpu().tolist())

        val_smina_all = np.array(val_smina_all)
        val_pred_all  = np.array(val_pred_all)

        spearman_rho, _ = spearmanr(val_smina_all, val_pred_all)
        eef_1 = early_enrichment_factor(val_smina_all, val_pred_all, frac=0.01)
        eef_5 = early_enrichment_factor(val_smina_all, val_pred_all, frac=0.05)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {total_loss / max(n_batches, 1):.4f} | "
            f"Spearman: {spearman_rho:.4f} | "
            f"EEF@1%: {eef_1:.2f} | "
            f"EEF@5%: {eef_5:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        # Two independent checkpoints — neither overwrites the other
        # EEF@5% is used (not @1%) because val only has ~445 rows → k=4 for @1%,
        # which is too noisy to use for checkpoint selection.
        if eef_5 > best_eef:
            best_eef = eef_5
            torch.save(model.state_dict(), 'best_teacher.pt')
            print(f"  >>> Saved best_teacher.pt  (EEF@5%={eef_5:.2f})")

        if spearman_rho < best_spearman:
            best_spearman = spearman_rho
            torch.save(model.state_dict(), 'best_teacher_spearman.pt')
            print(f"  >>> Saved best_teacher_spearman.pt  (Spearman={spearman_rho:.4f})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Trains the BindNet TeacherModel.")
    parser.add_argument("--train", type=str, default="train_set.csv", help="Training dataset CSV")
    parser.add_argument("--val", type=str, default="val_set.csv", help="Validation dataset CSV")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    train_teacher(args.train, args.val, args.epochs, args.lr, args.batch_size)
