import torch
import torch.nn.functional as F
import math
import numpy as np


def distillation_mse_loss(pred_embed, teacher_embed, weights):
    """
    Weighted MSE between student's predicted embedding and teacher's complex_embed.
    pred_embed:    [B, D]  student distillation head output
    teacher_embed: [B, D]  pre-computed teacher complex embedding
    weights:       [B]     distill_weight (0.0 for failed/missing structures)
    """
    mse = F.mse_loss(pred_embed, teacher_embed, reduction='none').mean(dim=1)  # [B]
    return (mse * weights).mean()


def boltz_regression_loss(pred, target, weights):
    """
    Weighted MSE against the Boltz-2 ensemble affinity scalar (boltz_affinity_pred_value).
    pred:    [B]  - predicted boltz affinity (log10 IC50)
    target:  [B]  - true boltz affinity from CSV
    weights: [B]  - sample weights (ligand_iptm^2 / (1+disagreement))
    """
    mse = F.mse_loss(pred, target, reduction='none')  # [B]
    return (mse * weights).mean()


def latent_mse_loss(pred_latent, latents, weights):
    """
    DEPRECATED: pre_affinity.npz contains structural geometry, not latent vectors.
    Kept for backward compatibility but not used in training.
    """
    mse = F.mse_loss(pred_latent, latents, reduction='none').mean(dim=1)
    return (mse * weights).mean()

def asymmetric_list_mle_loss(y_pred, y_true, weights, fn_penalty=2.0):
    """
    Computes a ListMLE ranking loss across the batch.
    y_pred: [B] - Predicted affinity (higher is better)
    y_true: [B] - Ground truth Smina score (more negative is better)
    weights: [B] - Sample weights
    """
    list_size = len(y_pred)
    if list_size <= 1:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
    # Smina score: lower is better. So we sort ascending to put best binders at the top.
    sorted_indices = torch.argsort(y_true, descending=False)
    pred_sorted = y_pred[sorted_indices]
    weights_sorted = weights[sorted_indices]
    
    loss = 0
    for i in range(list_size - 1):
        score_i = pred_sorted[i]
        scores_j = pred_sorted[i:]
        
        # log(exp(score_i) / sum(exp(scores_j)))
        mle_step = - (score_i - torch.logsumexp(scores_j, dim=0))
        
        # Asymmetric Penalty: exponentially larger for true positive errors (top of the list)
        weight = 1.0 + (fn_penalty - 1.0) * math.exp(-i)
        
        loss += mle_step * weight * weights_sorted[i]
        
    return loss / list_size

def pose_quality_loss(pred_quality, ligand_iptms, threshold=0.75):
    """
    Binary Cross Entropy for predicting if the Boltz pose is high quality.
    pred_quality: [B, 1] logits
    ligand_iptms: [B] raw iptm values
    """
    target = (ligand_iptms > threshold).float().unsqueeze(1)
    return F.binary_cross_entropy_with_logits(pred_quality, target)

def linear_cka(X, Y):
    """
    Computes Linear Centered Kernel Alignment between two feature spaces.
    X: [N, D1], Y: [N, D2]
    """
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    
    dot_x = torch.mm(X, X.t())
    dot_y = torch.mm(Y, Y.t())
    
    norm_x = torch.linalg.norm(dot_x, 'fro')
    norm_y = torch.linalg.norm(dot_y, 'fro')
    dot_xy = torch.sum(dot_x * dot_y)
    
    return (dot_xy / (norm_x * norm_y + 1e-8)).item()

def early_enrichment_factor(y_true, y_pred, frac=0.01):
    """
    Calculates how many of the top true binders the model correctly placed in its top % frac.
    y_true: Smina scores (lower is better)
    y_pred: Predicted scores (higher is better)
    
    # NOTE: Expected Spearman correlation is negative — high pred score = strong binder = negative SMINA
    """
    N = len(y_true)
    top_k = max(1, int(N * frac))
    
    # Ground truth top K (lowest Smina scores)
    true_top_idx = np.argsort(y_true)[:top_k]
    true_binders = set(true_top_idx)
    
    # Predicted top K (highest pred scores)
    pred_top_idx = np.argsort(y_pred)[::-1][:top_k]
    
    hits = sum(1 for idx in pred_top_idx if idx in true_binders)
    
    # Random expected hits
    expected = top_k * (top_k / N)
    
    return hits / (expected + 1e-8)
