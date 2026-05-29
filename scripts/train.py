"""
train.py  (Student v2)
──────────────────────
Trains the StudentModel using:
  1. Distillation MSE  — student complex_embed → teacher complex_embed (pre-computed)
  2. Asymmetric ListMLE — vs SMINA Vinardo scores
  3. Boltz Regression  — vs boltz_affinity_pred_value (weighted MSE)
  4. Pose Quality BCE  — vs ligand_iptm > 0.75  (auxiliary)
  5. Paratope BCE      — vs IMGT CDR labels (structural self-supervision, new)

New in v2:
  - StudentModel returns 5 outputs (added pred_paratope)
  - pae_weights loaded from feature_cache per sample (per-residue PAE-to-ligand)
    and passed to model.forward() for CDR-biased cross-attention
  - IMGT CDR labels generated on-the-fly from sequence lengths
  - LAMBDA_PARATOPE loss term added

Staged AbLang2 unfreezing:
  Epochs  0– 5 : backbone frozen  | heads lr=1e-3  → cosine → 1e-4
  Epochs  6–15 : last 2 blocks    | all   lr=1e-4  → cosine → 1e-5
  Epochs 16–39 : full backbone    | all   lr=1e-5  → cosine → 1e-6

Run AFTER train_teacher.py and precompute_teacher.py.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from scipy.stats import spearmanr

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abdistill.models import StudentModel
from abdistill.distillation import (
    distillation_mse_loss,
    asymmetric_list_mle_loss,
    boltz_regression_loss,
    pose_quality_loss,
    linear_cka,
    early_enrichment_factor,
)



# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class AbLigDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        embed_npy: str = 'teacher_embeddings.npy',
        embed_idx: str = 'teacher_embedding_index.json',
    ):
        self.df = pd.read_csv(csv_file)

        self.teacher_embeds = np.load(embed_npy)
        with open(embed_idx) as f:
            self.embed_idx = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        seq_id = row['id']
        heavy  = str(row['heavy_sequence'])
        light  = str(row['light_sequence'])
        paired = f"{heavy}|{light}"

        # Teacher embedding
        embed_row = self.embed_idx.get(seq_id, -1)
        if embed_row == -1 or np.all(self.teacher_embeds[embed_row] == 0):
            teacher_embed  = torch.zeros(256, dtype=torch.float32)
            distill_weight = 0.0
        else:
            teacher_embed  = torch.tensor(self.teacher_embeds[embed_row], dtype=torch.float32)
            distill_weight = float(row['latent_weight'])


        n_heavy    = len(heavy)
        n_light    = len(light)

        smina          = torch.tensor(row['smina_vinardo_affinity'],    dtype=torch.float32)
        boltz_affinity = torch.tensor(row['boltz_affinity_pred_value'], dtype=torch.float32)
        ligand_iptm    = torch.tensor(row['boltz_conf_ligand_iptm'],    dtype=torch.float32)
        ranking_weight = torch.tensor(row['ranking_weight'],            dtype=torch.float32)
        distill_weight = torch.tensor(distill_weight,                   dtype=torch.float32)

        return {
            'sequence':       paired,
            'teacher_embed':  teacher_embed,
            'smina':          smina,
            'boltz_affinity': boltz_affinity,
            'ligand_iptm':    ligand_iptm,
            'ranking_weight': ranking_weight,
            'distill_weight': distill_weight,
            'n_heavy':        n_heavy,
            'n_light':        n_light,
        }



def collate_fn(batch):
    sequences      = [b['sequence']       for b in batch]
    teacher_embeds = torch.stack([b['teacher_embed']  for b in batch])
    sminas         = torch.stack([b['smina']          for b in batch])
    boltz_affs     = torch.stack([b['boltz_affinity'] for b in batch])
    ligand_iptms   = torch.stack([b['ligand_iptm']    for b in batch])
    rank_weights   = torch.stack([b['ranking_weight'] for b in batch])
    dist_weights   = torch.stack([b['distill_weight'] for b in batch])
    return (sequences, teacher_embeds, sminas, boltz_affs, ligand_iptms,
            rank_weights, dist_weights)



# ─────────────────────────────────────────────
# Staged AbLang2 Unfreezing
# ─────────────────────────────────────────────
def set_ablang_freeze_state(model, state):
    backbone = model.prot_enc.ablang_model.AbLang
    if state == 'frozen':
        for p in backbone.parameters():
            p.requires_grad = False
    elif state == 'unfrozen':
        for p in backbone.parameters():
            p.requires_grad = True
    elif state == 'partial':
        for p in backbone.parameters():
            p.requires_grad = False
        for name, p in backbone.named_parameters():
            if any(f'encoder_blocks.{i}' in name for i in [10, 11]):
                p.requires_grad = True


def set_lr(optimizer, backbone_lr: float, other_lr: float) -> None:
    optimizer.param_groups[0]['lr'] = backbone_lr
    optimizer.param_groups[1]['lr'] = other_lr


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train(train_csv, val_csv, teacher_npy, teacher_idx, epochs, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── GPU speed settings ───────────────────────────────────────────────────
    use_amp = device.type == 'cuda'
    if use_amp:
        # Let cuDNN auto-tune conv algorithms for fixed kernel sizes.
        # Has no downside here since attention shapes are stable within a batch.
        torch.backends.cudnn.benchmark = True
    # GradScaler keeps FP32 master weights while running forward in FP16.
    # enabled=False on CPU so the same code path works both places.
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    BATCH_SIZE = batch_size
    EPOCHS     = epochs

    LAMBDA_DISTILL  = 1.0   # v1 teacher embeddings — matched geometry, restore to 1.0
    LAMBDA_RANK     = 1.0
    LAMBDA_BOLTZ    = 0.5
    LAMBDA_AUX      = 0.1

    PHASE2_START = 6
    PHASE3_START = 16
    PHASE1_LEN   = PHASE2_START
    PHASE2_LEN   = PHASE3_START - PHASE2_START
    PHASE3_LEN   = EPOCHS - PHASE3_START

    train_ds = AbLigDataset(train_csv,
                            embed_npy=teacher_npy,
                            embed_idx=teacher_idx)
    val_ds   = AbLigDataset(val_csv,
                            embed_npy=teacher_npy,
                            embed_idx=teacher_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    model = StudentModel(hidden_dim=256, teacher_dim=256).to(device)

    matched = [n for n, _ in model.prot_enc.ablang_model.AbLang.named_parameters()
               if any(f'encoder_blocks.{i}' in n for i in [10, 11])]
    print(f"[INFO] Partial freeze will unfreeze {len(matched)} params in blocks 10–11")
    assert len(matched) > 0, "No params matched — check AbLang2 layer naming"

    set_ablang_freeze_state(model, 'frozen')

    ablang_params = set(model.prot_enc.ablang_model.AbLang.parameters())
    other_params  = [p for p in model.parameters() if p not in ablang_params]
    optimizer = optim.AdamW([
        {'params': list(ablang_params), 'lr': 0.0},
        {'params': other_params,        'lr': 1e-3},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE1_LEN, eta_min=1e-4,
    )

    best_eef      = 0.0
    best_spearman = 0.0
    training_logs = []

    for epoch in range(EPOCHS):

        # ── Phase transitions ────────────────────────────────────────────────
        if epoch == PHASE2_START:
            set_ablang_freeze_state(model, 'partial')
            set_lr(optimizer, backbone_lr=1e-4, other_lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PHASE2_LEN, eta_min=1e-5,
            )
            print("--- Partially Unfrozen AbLang2 (blocks 10–11, LR=1e-4 → 1e-5) ---")

        elif epoch == PHASE3_START:
            set_ablang_freeze_state(model, 'unfrozen')
            set_lr(optimizer, backbone_lr=1e-5, other_lr=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PHASE3_LEN, eta_min=1e-6,
            )
            print("--- Fully Unfrozen AbLang2 (LR=1e-5 → 1e-6) ---")

        current_lr = optimizer.param_groups[1]['lr']

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for batch in pbar:
            (sequences, teacher_embeds, sminas, boltz_affs,
             ligand_iptms, rank_weights, dist_weights) = batch

            # non_blocking=True overlaps H→D transfer with GPU compute
            teacher_embeds = teacher_embeds.to(device, non_blocking=True)
            sminas         = sminas.to(device, non_blocking=True)
            boltz_affs     = boltz_affs.to(device, non_blocking=True)
            ligand_iptms   = ligand_iptms.to(device, non_blocking=True)
            rank_weights   = rank_weights.to(device, non_blocking=True)
            dist_weights   = dist_weights.to(device, non_blocking=True)

            # set_to_none frees gradient buffers entirely instead of zeroing them
            optimizer.zero_grad(set_to_none=True)

            # autocast runs eligible ops in FP16; numerically sensitive ops
            # (logsumexp, softmax, BCE) stay in FP32 automatically
            with torch.amp.autocast('cuda', enabled=use_amp):
                (pred_teacher, pred_affinity, pred_boltz,
                 pred_quality) = model(sequences)

                loss_distill  = distillation_mse_loss(pred_teacher, teacher_embeds, dist_weights)
                loss_rank     = asymmetric_list_mle_loss(pred_affinity, sminas, rank_weights, fn_penalty=2.0)
                loss_boltz    = boltz_regression_loss(pred_boltz, boltz_affs, dist_weights)
                loss_aux      = pose_quality_loss(pred_quality, ligand_iptms, threshold=0.75)

                loss = (LAMBDA_DISTILL  * loss_distill
                        + LAMBDA_RANK    * loss_rank
                        + LAMBDA_BOLTZ   * loss_boltz
                        + LAMBDA_AUX     * loss_aux)

            # scaler: backward in FP16, unscale before grad clip, step in FP32
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'dst':  f'{loss_distill.item():.3f}',
                'rnk':  f'{loss_rank.item():.3f}',
            })

        scheduler.step()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        val_smina_all    = []
        val_pred_all     = []
        val_teacher_true = []
        val_teacher_pred = []

        with torch.no_grad():
            for batch in val_loader:
                (sequences, teacher_embeds, sminas, boltz_affs,
                 ligand_iptms, rank_weights, dist_weights) = batch

                sminas         = sminas.to(device, non_blocking=True)
                teacher_embeds = teacher_embeds.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    (pred_teacher, pred_affinity, pred_boltz,
                     pred_quality) = model(sequences)

                val_smina_all.extend(sminas.cpu().tolist())
                val_pred_all.extend(pred_affinity.cpu().tolist())
                val_teacher_true.append(teacher_embeds.cpu())
                val_teacher_pred.append(pred_teacher.cpu())

        val_smina_all = np.array(val_smina_all)
        val_pred_all  = np.array(val_pred_all)

        spearman_rho, _ = spearmanr(val_smina_all, val_pred_all)
        eef_1  = early_enrichment_factor(val_smina_all, val_pred_all, frac=0.01)
        eef_5  = early_enrichment_factor(val_smina_all, val_pred_all, frac=0.05)

        teacher_true_cat = torch.cat(val_teacher_true, dim=0)
        teacher_pred_cat = torch.cat(val_teacher_pred, dim=0)
        cka_score = linear_cka(teacher_pred_cat, teacher_true_cat)

        print(
            f"Epoch {epoch+1:02d} | "
            f"Spearman: {spearman_rho:.4f} | "
            f"EEF@1%: {eef_1:.2f} | "
            f"EEF@5%: {eef_5:.2f} | "
            f"CKA: {cka_score:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        training_logs.append({
            'epoch':      epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'spearman':   spearman_rho,
            'eef_1':      eef_1,
            'eef_5':      eef_5,
            'cka':        cka_score,
            'lr':         current_lr,
        })
        pd.DataFrame(training_logs).to_csv('student_training_logs.csv', index=False)

        # Two independent checkpoints
        # EEF@5% is used (not @1%) because val only has ~445 rows → k=4 for @1%,
        # which is too noisy to use for checkpoint selection.
        if eef_5 > best_eef:
            best_eef = eef_5
            torch.save(model.state_dict(), 'best_student_model.pt')
            print(f"  >>> Saved best_student_model.pt  (EEF@5%={eef_5:.2f})")

        if spearman_rho < best_spearman:
            best_spearman = spearman_rho
            torch.save(model.state_dict(), 'best_student_model_spearman.pt')
            print(f"  >>> Saved best_student_model_spearman.pt  (Spearman={spearman_rho:.4f})")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Trains the StudentModel via Knowledge Distillation.")
    parser.add_argument("--train", type=str, default="train_set.csv", help="Training dataset CSV")
    parser.add_argument("--val", type=str, default="val_set.csv", help="Validation dataset CSV")
    parser.add_argument("--teacher_emb", type=str, default="teacher_embeddings.npy", help="Precomputed teacher embeddings")
    parser.add_argument("--teacher_idx", type=str, default="teacher_embedding_index.json", help="Teacher embedding index mapping")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    train(args.train, args.val, args.teacher_emb, args.teacher_idx, args.epochs, args.batch_size)
