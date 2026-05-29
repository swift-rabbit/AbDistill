"""
prepare_dataset.py
──────────────────
Adds derived columns (ranking_weight, latent_weight) and creates
cluster-based train / val / test splits from final_master_dataset.csv.

Columns added
─────────────
  ranking_weight = boltz_conf_ligand_iptm² / (1 + |pred_value1 - pred_value2|)
  latent_weight  = same formula  (identical — used by distillation head)

Splits  (cluster-based — no two sequences within Hamming ≤ 2 appear in
         different splits, so there is zero sequence-level leakage)
──────
  70 % → train_set.csv
  15 % → val_set.csv
  15 % → test_set.csv   ← held-out; only for final evaluation

Usage:
    python prepare_dataset.py
"""

import pandas as pd
import numpy as np

VAL_FRAC    = 0.15
TEST_FRAC   = 0.15
RANDOM_SEED = 42


def main(input_csv, train_csv, val_csv, test_csv):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # ── Derive ranking_weight / latent_weight ──────────────────────────────
    ligand_iptm   = df['boltz_conf_ligand_iptm'].fillna(0.0).clip(0.0, 1.0)
    pred1         = df['boltz_affinity_pred_value1'].fillna(0.0)
    pred2         = df['boltz_affinity_pred_value2'].fillna(0.0)
    disagreement  = (pred1 - pred2).abs()

    weight = (ligand_iptm ** 2) / (1.0 + disagreement)
    df['ranking_weight'] = weight.values
    df['latent_weight']  = weight.values

    print(f"ranking_weight  — mean: {weight.mean():.4f}  std: {weight.std():.4f}  "
          f"min: {weight.min():.4f}  max: {weight.max():.4f}")

    # ── Drop rows with missing mandatory fields ────────────────────────────
    required = [
        'smina_vinardo_affinity',
        'boltz_affinity_pred_value',
        'boltz_conf_ligand_iptm',
        'heavy_sequence',
        'light_sequence',
    ]
    before = len(df)
    df = df.dropna(subset=required)
    print(f"Dropped {before - len(df)} rows with missing mandatory fields; {len(df)} remain")

    # ── Cluster-based split to avoid leakage ──────────────────────────────
    # Two sequences that differ by ≤ 2 residues must land in the same split.
    # We cluster them via connected components, then greedily allocate whole
    # clusters to train / val / test so the exact proportions are respected
    # at the cluster level, not the row level.
    print("Clustering sequences by Hamming distance ≤ 2 to prevent leakage…")
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import pdist, squareform

    seqs = df['heavy_sequence'].values
    chars = np.array([list(s) for s in seqs])
    unique_chars = np.unique(chars)
    char_map = {c: i for i, c in enumerate(unique_chars)}
    int_chars = np.vectorize(char_map.get)(chars)           # [N, L]

    dist_condensed = pdist(int_chars, metric='hamming') * int_chars.shape[1]
    adj_matrix     = squareform(dist_condensed <= 2.0001)
    n_components, labels = connected_components(
        sp.csr_matrix(adj_matrix), directed=False
    )
    print(f"Found {n_components} independent sequence clusters.")

    # Group DataFrame indices by cluster label
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    # ── Stratified cluster allocation ─────────────────────────────────────
    # Problem with pure greedy: all strong-binder clusters land in train
    # (they're large/first) → val and test get 0% top binders, making
    # EEF metrics on those splits meaningless.
    #
    # Fix: split clusters into "enriched" (contain ≥1 top-1% binder) and
    # "plain", then distribute each group independently in 70/15/15 ratio.
    # This guarantees every split has strong binders to evaluate against.

    smina_col    = 'smina_vinardo_affinity'
    p1_threshold = np.percentile(df[smina_col], 1)
    top1_idx_set = set(np.where(df[smina_col].values <= p1_threshold)[0])

    enriched_clusters = []
    plain_clusters    = []
    for cluster in clusters.values():
        if any(i in top1_idx_set for i in cluster):
            enriched_clusters.append(cluster)
        else:
            plain_clusters.append(cluster)

    print(f"  Enriched clusters (≥1 top-1% binder): {len(enriched_clusters)}")
    print(f"  Plain clusters: {len(plain_clusters)}")

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(enriched_clusters)
    rng.shuffle(plain_clusters)

    TRAIN_FRAC = 1 - VAL_FRAC - TEST_FRAC   # 0.70

    def split_by_cluster_count(cluster_list, train_frac, val_frac):
        """
        Assign clusters proportionally by CLUSTER COUNT, not row count.
        Row-count greedy fails when enriched clusters are few — the 70% row
        quota can be reached before any cluster lands in val/test.
        Index-based assignment guarantees each split always gets clusters.
        """
        n  = len(cluster_list)
        tr, va, te = [], [], []
        for i, c in enumerate(cluster_list):
            frac = i / max(n, 1)
            if frac < train_frac:
                tr.extend(c)
            elif frac < train_frac + val_frac:
                va.extend(c)
            else:
                te.extend(c)
        return tr, va, te

    def greedy_split_by_rows(cluster_list, train_frac, val_frac):
        """Row-count greedy — fine for the large plain-cluster pool."""
        n_total   = sum(len(c) for c in cluster_list)
        tgt_train = int(n_total * train_frac)
        tgt_val   = int(n_total * val_frac)
        tr, va, te = [], [], []
        for c in cluster_list:
            if len(tr) < tgt_train:
                tr.extend(c)
            elif len(va) < tgt_val:
                va.extend(c)
            else:
                te.extend(c)
        return tr, va, te

    # Enriched clusters: split by cluster count to guarantee top binders in all splits
    e_tr, e_va, e_te = split_by_cluster_count(enriched_clusters, TRAIN_FRAC, VAL_FRAC)
    # Plain clusters: row-count greedy is fine (large pool, no scarcity problem)
    p_tr, p_va, p_te = greedy_split_by_rows(plain_clusters, TRAIN_FRAC, VAL_FRAC)

    train_idx = e_tr + p_tr
    val_idx   = e_va + p_va
    test_idx  = e_te + p_te

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    print(
        f"Train: {len(train_df)} rows  |  "
        f"Val: {len(val_df)} rows  |  "
        f"Test: {len(test_df)} rows"
    )

    # Sanity check: top-1% strong-binder coverage in every split
    smina_col    = 'smina_vinardo_affinity'
    p1_threshold = np.percentile(df[smina_col], 1)
    for name, split in [('train', train_df), ('val', val_df), ('test', test_df)]:
        frac = (split[smina_col] <= p1_threshold).mean()
        print(f"  Top-1% binder prevalence [{name}]: {frac:.2%}")
    print(f"  (global 1% SMINA threshold: {p1_threshold:.3f})")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,    index=False)
    test_df.to_csv(test_csv,  index=False)
    print(f"Saved {train_csv}, {val_csv}, {test_csv}")


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adds derived weights and creates cluster-based splits.")
    parser.add_argument("--input", type=str, default="final_master_dataset.csv", help="Input master dataset CSV")
    parser.add_argument("--train", type=str, default="train_set.csv", help="Output train CSV")
    parser.add_argument("--val", type=str, default="val_set.csv", help="Output val CSV")
    parser.add_argument("--test", type=str, default="test_set.csv", help="Output test CSV")
    args = parser.parse_args()
    
    main(args.input, args.train, args.val, args.test)
