import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abdistill.models import StudentModel, TeacherModel
from abdistill.distillation import early_enrichment_factor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_student(model_path, df):
    print(f"\n--- Evaluating Student Model ---")
    print(f"Loading weights from {model_path}...")
    
    model = StudentModel(hidden_dim=256, teacher_dim=256).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    predictions = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Student Inference"):
            batch = df.iloc[i:i+batch_size]
            seqs = [f"{r['heavy_sequence']}|{r['light_sequence']}" for _, r in batch.iterrows()]
            
            # Forward pass
            _, pred_affinity, _, _ = model(seqs)
            predictions.extend(pred_affinity.cpu().numpy())
            
    return np.array(predictions)

def evaluate_teacher(model_path, df, cache_dir):
    print(f"\n--- Evaluating Teacher Model ---")
    print(f"Loading weights from {model_path}...")
    
    if not cache_dir or not os.path.exists(cache_dir):
        raise ValueError(f"Teacher evaluation requires a valid --cache_dir containing .npz files. Provided: {cache_dir}")
        
    model = TeacherModel(p_in_dim=512, l_in_dim=768, hidden_dim=256, num_heads=4, num_layers=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    predictions = []
    valid_indices = []
    
    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Teacher Inference"):
            seq_id = row['id']
            path = os.path.join(cache_dir, f"{seq_id}.npz")
            
            if not os.path.exists(path):
                print(f"[WARN] Cache file missing for {seq_id}. Skipping.")
                continue
                
            data = np.load(path)
            pocket_t = torch.tensor(data['pocket'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            ligand_t = torch.tensor(data['ligand'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            N_p = pocket_t.shape[1]

            plddt = torch.tensor(data['plddt'], dtype=torch.float32, device=DEVICE)
            p_plddt = plddt[:N_p].unsqueeze(0)
            l_plddt = plddt[N_p:].unsqueeze(0)

            if 'pae_p2l' in data:
                pae_p2l = torch.tensor(data['pae_p2l'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                pae_l2p = torch.tensor(data['pae_l2p'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                pde_p2l = torch.tensor(data['pde_p2l'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                pde_l2p = torch.tensor(data['pde_l2p'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                pae_to_lig = torch.tensor(data['pae_to_lig'], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            else:
                pae = torch.tensor(data['pae'], dtype=torch.float32, device=DEVICE)
                pae_l2p = pae[N_p:, :N_p].unsqueeze(0)
                pae_p2l = pae[:N_p, N_p:].unsqueeze(0)
                pde_l2p = torch.zeros_like(pae_l2p)
                pde_p2l = torch.zeros_like(pae_p2l)
                pae_to_lig = pae_p2l.squeeze(0).mean(dim=1).unsqueeze(0)

            score, _ = model(
                pocket_t, ligand_t, p_plddt, l_plddt,
                pae_l2p, pae_p2l, pde_l2p, pde_p2l,
                pae_to_lig, p_mask=None
            )
            
            predictions.append(score.item())
            valid_indices.append(i)
            
    return np.array(predictions), valid_indices

def main(args):
    print(f"Loading test set from {args.input}...")
    df = pd.read_csv(args.input)
    
    if args.type == 'student':
        preds = evaluate_student(args.model, df)
        df_eval = df.copy()
    else:
        preds, valid_idx = evaluate_teacher(args.model, df, args.cache_dir)
        df_eval = df.iloc[valid_idx].copy()
        
    df_eval['predicted_affinity'] = preds
    smina_true = df_eval['smina_vinardo_affinity'].values
    
    # ── Calculate Metrics ──
    spearman_rho, _ = spearmanr(smina_true, preds)
    eef_1 = early_enrichment_factor(smina_true, preds, frac=0.01)
    eef_5 = early_enrichment_factor(smina_true, preds, frac=0.05)
    
    print("\n" + "="*40)
    print(f" {args.type.upper()} MODEL PERFORMANCE")
    print("="*40)
    print(f" Test Set Size: {len(df_eval)} samples")
    print(f" Spearman Corr: {spearman_rho:.4f}")
    print(f" EEF@1%       : {eef_1:.2f}x enrichment")
    print(f" EEF@5%       : {eef_5:.2f}x enrichment")
    print("="*40)
    
    # ── Save Results ──
    df_eval.to_csv(args.out, index=False)
    print(f"\nRaw predictions saved to -> {args.out}")
    
    # ── Plot Scatter ──
    if args.plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(preds, smina_true, alpha=0.5, color='blue', edgecolor='k')
        
        # Line of best fit
        m, b = np.polyfit(preds, smina_true, 1)
        plt.plot(preds, m*preds + b, color='red', linestyle='--', label=f'Trend (Spearman: {spearman_rho:.2f})')
        
        plt.title(f"{args.type.capitalize()} Model Evaluation on Test Set")
        plt.xlabel("Predicted Score (More Negative = Better)")
        plt.ylabel("True SMINA Vinardo Score (More Negative = Better)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(args.plot)
        print(f"Scatter plot saved to -> {args.plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Student or Teacher models on a holdout test dataset.")
    parser.add_argument("--type", choices=['student', 'teacher'], required=True, help="Which model architecture to evaluate")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pt model weights")
    parser.add_argument("--input", type=str, default="data/test_set.csv", help="Test dataset CSV")
    parser.add_argument("--cache_dir", type=str, default=None, help="Feature cache directory (Required for Teacher)")
    parser.add_argument("--out", type=str, default="test_predictions.csv", help="Output CSV with predictions")
    parser.add_argument("--plot", type=str, default="test_scatter.png", help="Output path for scatter plot image")
    args = parser.parse_args()
    
    main(args)
