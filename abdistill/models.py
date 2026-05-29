import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class MockLigandEncoder(nn.Module):
    """
    Learned static embedding for Testosterone (same molecule across all complexes).
    The MSE distillation loss shapes this embedding to encode testosterone's
    binding properties, guided entirely by the teacher's 3D structural signal.
    """
    def __init__(self, hidden_dim=256, num_tokens=10):
        super().__init__()
        self.ligand_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))

    def forward(self, batch_size):
        return self.ligand_tokens.expand(batch_size, -1, -1)   # [B, 10, hidden_dim]


class CDRPositionEncoding(nn.Module):
    """
    IMGT-based CDR position bias injected into cross-attention.

    Assigns a learned additive vector to CDR residues so attention naturally
    focuses on the paratope without hard masking.

    IMGT CDR positions (1-based within each chain):
      Heavy: CDR-H1 27-38, CDR-H2 56-65, CDR-H3 105-117
      Light: CDR-L1 27-38, CDR-L2 56-65, CDR-L3 105-117
    """
    CDR_INTERVALS = {
        'heavy': [(27, 38), (56, 65), (105, 117)],
        'light': [(27, 38), (56, 65), (105, 117)],
    }

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.cdr_bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.cdr_bias, std=0.02)

    @staticmethod
    def _cdr_mask(seq_len: int, chain: str, device) -> torch.Tensor:
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for start, end in CDRPositionEncoding.CDR_INTERVALS[chain]:
            lo = start - 1
            hi = min(end, seq_len)
            if lo < seq_len:
                mask[lo:hi] = True
        return mask

    def forward(self, prot_features: torch.Tensor, n_heavy_list: list) -> torch.Tensor:
        """
        prot_features : [B, L, hidden_dim]
        n_heavy_list  : list[int]  number of heavy-chain residues per sample
        """
        B, L, D = prot_features.shape
        out = prot_features.clone()
        # Cast bias to match input dtype (e.g. FP16 under AMP autocast)
        bias = self.cdr_bias.to(prot_features.dtype)
        for b, n_h in enumerate(n_heavy_list):
            h_mask = self._cdr_mask(n_h, 'heavy', prot_features.device)
            out[b, :n_h][h_mask] = out[b, :n_h][h_mask] + bias

            n_l = L - n_h
            l_mask = self._cdr_mask(n_l, 'light', prot_features.device)
            out[b, n_h:n_h + n_l][l_mask] = (
                out[b, n_h:n_h + n_l][l_mask] + bias
            )
        return out


class FullCrossAttention(nn.Module):
    """
    Ligand attends to the full antibody sequence.

    Accepts optional:
      pae_weights : [B, L]  per-residue PAE-to-ligand — low = high confidence
                            used to scale protein keys before attention
      cdr_encoder : CDRPositionEncoding — applied to protein features first
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.lig_to_prot = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm      = nn.LayerNorm(hidden_dim)

    def forward(self, prot_features, lig_features, prot_pad_mask,
                n_heavy_list=None, cdr_encoder=None):
        if cdr_encoder is not None and n_heavy_list is not None:
            prot_features = cdr_encoder(prot_features, n_heavy_list)

        prot_keys = prot_features

        ignore_mask  = ~prot_pad_mask
        lig_attended, _ = self.lig_to_prot(
            query=lig_features,
            key=prot_keys,
            value=prot_features,
            key_padding_mask=ignore_mask
        )
        lig_out = self.norm(lig_features + lig_attended)
        return lig_out.mean(dim=1)   # [B, hidden_dim]


# ─────────────────────────────────────────────────────────────────────────────
# BindNet Teacher Model
# ─────────────────────────────────────────────────────────────────────────────

class BindNetLayer(nn.Module):
    """
    Cross-attention layer with PAE + PDE attention biases (both directions).
    Uses correctly-oriented PAE: pocket→ligand (p2l) has low PAE (~1.74 Å)
    and is the meaningful signal; ligand→pocket (l2p) has high PAE (~14 Å).
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn_l2p  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff_l      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm_l_q  = nn.LayerNorm(hidden_dim)
        self.norm_l_kv = nn.LayerNorm(hidden_dim)
        self.norm2_l   = nn.LayerNorm(hidden_dim)

        self.attn_p2l  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff_p      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm_p_q  = nn.LayerNorm(hidden_dim)
        self.norm_p_kv = nn.LayerNorm(hidden_dim)
        self.norm2_p   = nn.LayerNorm(hidden_dim)

        # PAE bias weights
        self.pae_w_l2p = nn.Parameter(torch.zeros(1))
        self.pae_w_p2l = nn.Parameter(torch.zeros(1))
        # PDE bias weights (new — orthogonal signal)
        self.pde_w_l2p = nn.Parameter(torch.zeros(1))
        self.pde_w_p2l = nn.Parameter(torch.zeros(1))

    def forward(self, h_l, h_p, pae_l2p, pae_p2l, pde_l2p, pde_p2l, p_mask=None):
        B, N_l, _ = h_l.shape
        _, N_p, _ = h_p.shape
        num_heads  = self.attn_l2p.num_heads

        # Ligand attends to Pocket
        bias_l2p = (self.pae_w_l2p * pae_l2p + self.pde_w_l2p * pde_l2p)
        bias_l2p = (bias_l2p.unsqueeze(1)
                    .expand(B, num_heads, N_l, N_p)
                    .reshape(B * num_heads, N_l, N_p))

        attn_out, attn_weights = self.attn_l2p(
            query=self.norm_l_q(h_l),
            key=self.norm_l_kv(h_p),
            value=self.norm_l_kv(h_p),
            key_padding_mask=p_mask, attn_mask=bias_l2p
        )
        h_l = h_l + attn_out
        h_l = h_l + self.ff_l(self.norm2_l(h_l))

        # Pocket attends to Ligand (p2l = low PAE direction — primary signal)
        bias_p2l = (self.pae_w_p2l * pae_p2l + self.pde_w_p2l * pde_p2l)
        bias_p2l = (bias_p2l.unsqueeze(1)
                    .expand(B, num_heads, N_p, N_l)
                    .reshape(B * num_heads, N_p, N_l))

        attn_out_p, _ = self.attn_p2l(
            query=self.norm_p_q(h_p),
            key=self.norm_p_kv(h_l),
            value=self.norm_p_kv(h_l),
            attn_mask=bias_p2l
        )
        h_p = h_p + attn_out_p
        h_p = h_p + self.ff_p(self.norm2_p(h_p))

        return h_l, h_p, attn_weights


class TeacherModel(nn.Module):
    """
    BindNet Teacher — ESM-IF1 pocket + UniMol2 ligand + interaction transformer.

    v2 improvements:
      - Correctly-oriented PAE slices (p2l primary, l2p secondary)
      - PDE as second learned attention bias per BindNetLayer
      - pae_to_lig: per-residue paratope confidence as additive pocket correction
    """
    def __init__(self, p_in_dim=512, l_in_dim=768, hidden_dim=256,
                 num_heads=4, num_layers=3):
        super().__init__()
        self.proj_l = nn.Linear(l_in_dim, hidden_dim)
        self.proj_p = nn.Linear(p_in_dim, hidden_dim)

        self.plddt_mlp_p = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, p_in_dim)
        )
        self.plddt_mlp_l = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, l_in_dim)
        )
        # New: per-residue PAE-to-ligand paratope correction
        self.pae_lig_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, p_in_dim)
        )

        self.layers = nn.ModuleList([
            BindNetLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(64, 1),
        )

    def forward(self, h_p, h_l, p_plddt, l_plddt,
                pae_l2p, pae_p2l, pde_l2p, pde_p2l,
                pae_to_lig, p_mask=None):
        """
        h_p        [B, N_p, p_in_dim]
        h_l        [B, N_l, l_in_dim]
        p_plddt    [B, N_p]
        l_plddt    [B, N_l]
        pae_l2p    [B, N_l, N_p]
        pae_p2l    [B, N_p, N_l]   PRIMARY — pocket→ligand (low PAE ~1.74 Å)
        pde_l2p    [B, N_l, N_p]
        pde_p2l    [B, N_p, N_l]
        pae_to_lig [B, N_p]        per-residue mean PAE to ligand
        p_mask     [B, N_p]        True = padding
        """
        p_plddt    = p_plddt.unsqueeze(-1)
        l_plddt    = l_plddt.unsqueeze(-1)
        pae_to_lig = pae_to_lig.unsqueeze(-1)

        # Pocket features: pLDDT + paratope confidence correction
        h_p = self.proj_p(
            h_p
            + self.plddt_mlp_p(p_plddt)
            + self.pae_lig_mlp(pae_to_lig)
        )
        h_l = self.proj_l(h_l + self.plddt_mlp_l(l_plddt))

        for layer in self.layers:
            h_l, h_p, attn_weights = layer(
                h_l, h_p,
                pae_l2p, pae_p2l,
                pde_l2p, pde_p2l,
                p_mask=p_mask
            )

        # Soft attention pooling
        p_attn_score = attn_weights.sum(dim=1)
        if p_mask is not None:
            p_attn_score = p_attn_score.masked_fill(p_mask, -1e9)
        p_attn_soft = F.softmax(p_attn_score, dim=-1)
        p_pool = torch.bmm(p_attn_soft.unsqueeze(1), h_p).squeeze(1)
        l_pool = h_l.mean(dim=1)

        complex_embed = self.fusion(torch.cat([l_pool, p_pool], dim=-1))
        score = self.ranking_head(complex_embed)
        return score, complex_embed


# ─────────────────────────────────────────────────────────────────────────────
# Student Model  (AbLang2 sequence encoder + CDR-aware cross-attention)
# ─────────────────────────────────────────────────────────────────────────────

class StudentProteinEncoder(nn.Module):
    """Encodes the full paired antibody sequence (Heavy|Light) using AbLang2."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        import ablang2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ablang_model = ablang2.pretrained("ablang2-paired", device=device)
        self.ablang_model.freeze()

        dummy          = self.ablang_model([["EVQLVESGG", "DIQMTQSPS"]], mode='rescoding')
        import numpy as np
        ablang_out_dim = dummy[0].shape[-1]
        print(f"[StudentProteinEncoder] AbLang2 output dim: {ablang_out_dim}")
        self.proj = nn.Linear(ablang_out_dim, hidden_dim)

    def forward(self, sequences):
        """sequences: list of 'VH|VL' strings."""
        device    = next(self.proj.parameters()).device
        seq_pairs = [seq.split('|') for seq in sequences]

        # Do NOT wrap in torch.no_grad() — gradient flow through AbLang2 is
        # controlled by requires_grad on each parameter (set by the staged
        # unfreezing schedule in train.py).  Using no_grad() here would silently
        # block gradients in phases 2 and 3 even after unfreezing.
        encodings = self.ablang_model(seq_pairs, mode='rescoding')

        from torch.nn.utils.rnn import pad_sequence
        tensors  = [torch.tensor(enc, dtype=torch.float32, device=device)
                    for enc in encodings]
        h_padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)

        lengths  = torch.tensor([len(t) for t in tensors], device=device)
        max_len  = h_padded.size(1)
        mask     = (
            torch.arange(max_len, device=device).expand(len(lengths), max_len)
            < lengths.unsqueeze(1)
        )   # [B, L]  True = real token

        # Use the actual heavy-chain length from the input sequences, not len//2.
        # CDRPositionEncoding applies IMGT-based biases at specific positions, so
        # the wrong n_heavy would mis-place all CDR intervals.
        n_heavy_list = [len(pair[0]) for pair in seq_pairs]

        h = self.proj(h_padded)   # [B, L, hidden_dim]
        return h, mask, n_heavy_list


class StudentModel(nn.Module):
    """
    Student model: AbLang2 sequence encoder + CDR-aware cross-attention.

    Five prediction heads:
      1. Distillation  — MSE vs teacher complex_embed
      2. Affinity      — Asymmetric ListMLE vs SMINA vinardo scores
      3. Boltz         — Weighted MSE vs boltz_affinity_pred_value
      4. Pose Quality  — BCE vs ligand_iptm > 0.75
      5. Paratope      — BCE vs IMGT CDR labels (structural self-supervision)

    Architecture improvements vs v1:
      - CDRPositionEncoding: IMGT-based bias on CDR query features
      - PAE-to-ligand key scaling: Boltz2 paratope confidence weights protein keys
    """
    def __init__(self, hidden_dim: int = 256, teacher_dim: int = 256):
        super().__init__()
        self.lig_enc    = MockLigandEncoder(hidden_dim)
        self.prot_enc   = StudentProteinEncoder(hidden_dim)
        self.cdr_enc    = CDRPositionEncoding(hidden_dim)
        self.cross_attn = FullCrossAttention(hidden_dim)

        # Head 1: Distillation
        self.distill_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, teacher_dim),
        )
        # Head 2: Affinity Ranking
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1),
        )
        # Head 3: Boltz Affinity Regression
        self.boltz_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1),
        )
        # Head 4: Pose Quality
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1),
        )

    def forward(self, sequences):
        """
        sequences   : list[str]  'VH|VL'

        Returns:
          pred_teacher   [B, teacher_dim]
          pred_affinity  [B]
          pred_boltz     [B]
          pred_quality   [B, 1]
        """
        batch_size = len(sequences)

        lig_features = self.lig_enc(batch_size)                        # [B, 10, hidden]
        prot_features, prot_pad_mask, n_heavy_list = self.prot_enc(sequences)

        # CDR-biased cross-attention
        complex_embed = self.cross_attn(
            prot_features, lig_features, prot_pad_mask,
            n_heavy_list = n_heavy_list,
            cdr_encoder  = self.cdr_enc,
        )   # [B, hidden]

        pred_teacher  = self.distill_head(complex_embed)
        pred_affinity = self.ranking_head(complex_embed).squeeze(-1)
        pred_boltz    = self.boltz_head(complex_embed).squeeze(-1)
        pred_quality  = self.quality_head(complex_embed)

        return pred_teacher, pred_affinity, pred_boltz, pred_quality
