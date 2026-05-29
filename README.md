# AbDistill

**Structure-Guided Knowledge Distillation for Antibody–Ligand Binding Prediction**

> **TL;DR:** AbDistill lets you screen millions of antibody sequences in seconds by learning to mimic a slow, 3D structure-aware model from raw sequence alone. This repository provides the complete framework to reproduce the pipeline on custom datasets.

AbDistill trains a fast, sequence-only antibody binding predictor by distilling 3D structural knowledge from a structure-aware teacher model. The teacher (BindNet) reads Boltz-2 predicted structures and encodes binding geometry via PAE/PDE-biased cross-attention; the student (AbLang2 + CDR cross-attention) learns to reproduce that geometric understanding from raw Heavy|Light sequence alone. At inference time the student runs in milliseconds with no structure prediction required.

The current target ligand is **testosterone** — all models and cached features are specific to this molecule, but the pipeline generalises to any fixed small-molecule target by rerunning from Phase 1.

---

## Project Structure

```
AbDistill/
├── abdistill/
│   ├── models.py          # TeacherModel (BindNet) and StudentModel definitions
│   └── distillation.py    # All loss functions and evaluation metrics
├── scripts/
│   ├── generate_antifold.py      # Phase 1 — seed sequence generation
│   ├── generate_random_dataset.py# Phase 1 — random diversity sequences
│   ├── generate_boltz.py         # Phase 1 — Boltz-2 structure prediction
│   ├── score_smina.py            # Phase 1 — SMINA Vinardo docking scores
│   ├── aggregate_results.py      # Phase 1 — merge all outputs to master CSV
│   ├── prepare_dataset.py        # Phase 1 — cluster splits + derived weights
│   ├── precompute_features.py    # Phase 2 — ESM-IF1 + UniMol2 + Boltz-2 features
│   ├── train_teacher.py          # Phase 2 — BindNet teacher training
│   ├── precompute_teacher.py     # Phase 2 — freeze teacher, cache 256-D embeddings
│   ├── train.py                  # Phase 2 — student distillation training
│   ├── evaluate.py               # Phase 2 — test-set evaluation (teacher or student)
│   ├── evolve.py                 # Phase 3 — AdaLead CDR-H3 evolutionary search
│   └── generate_top_ga_seq.py    # Phase 3 — extract top candidates from GA run
└── setup.sh                      # One-shot environment setup
```

---

## Installation

```bash
bash setup.sh
conda activate abdistill
```

`setup.sh` does the following:
1. Installs system packages (`smina` binary, build tools)
2. Creates a conda environment `abdistill` (Python 3.11) with `hmmer` and `anarci` for IMGT numbering
3. Installs all Python dependencies: `boltz`, `ablang2`, `unimol_tools`, `fair-esm`, `torch_geometric`, `torch_scatter`, `antifold`, and standard ML libraries
4. Pre-downloads Boltz-2 model weights to `boltz_weights/` to avoid re-downloading at runtime

**GPU requirement:** A CUDA-capable GPU is strongly recommended. Boltz-2 structure prediction is impractical on CPU. The teacher and student training scripts use mixed-precision (FP16) and `cudnn.benchmark=True` automatically when a GPU is available.

---

## Three-Phase Pipeline

### Phase 1 — Dataset Construction

**Goal:** Build a labelled dataset of antibody sequences with Boltz-2 predicted complex structures and SMINA Vinardo docking scores.

#### 1a. Generate seed sequences

```bash
# AntiFold: structurally-conditioned CDR-H3 design on a scaffold PDB
python scripts/generate_antifold.py \
    --scaffold 1i9j_imgt.pdb \
    --num_seqs 10000
# (Outputs to initial_dataset_10000.csv)

# Random baseline sequences for diversity / negative control
python scripts/generate_random_dataset.py \
    --num_seqs 500
# (Outputs to random_dataset_500.csv)
```

`generate_antifold.py` samples CDR-H3 loops via AntiFold at temperatures 0.5 / 1.0 / 1.5 for diversity, conditioned on the IMGT-numbered scaffold PDB. The scaffold must have chains H (heavy) and L (light).

#### 1b. Run Boltz-2 structure prediction

```bash
python scripts/generate_boltz.py \
    --input antifold_sequences.csv \
    --ligand_smiles "CC1CCC2C(C1)CCC3C2CCC4=CC(=O)CCC34C" \
    --out_dir boltz_out/ \
    --gpu 2
```

Writes a YAML per sequence (chains A=Heavy, B=Light, C=Ligand) and calls the `boltz predict` CLI. Outputs to `boltz_out/boltz_results_{seq_id}/predictions/{seq_id}/`:
- `{seq_id}_model_0.pdb` — predicted complex
- `plddt_{seq_id}_model_0.npz` — per-token pLDDT
- `pae_{seq_id}_model_0.npz` — PAE matrix
- `pde_{seq_id}_model_0.npz` — PDE matrix

#### 1c. Score docking affinity with SMINA

```bash
python scripts/score_smina.py \
    --boltz_dir boltz_out/ \
    --workers 8
```

Splits each Boltz-2 PDB into receptor (chains A+B) and ligand (chain C), then calls `smina` with the Vinardo scoring function. Scores are written to `smina_affinity_{seq_id}.json` alongside the PDB. Lower (more negative) = stronger predicted binder.

#### 1d. Aggregate all results

```bash
python scripts/aggregate_results.py \
    --input antifold_sequences.csv \
    --random random_sequences.csv \
    --boltz_dir boltz_out/ \
    --out final_master_dataset.csv
```

Merges sequences, Boltz-2 confidence metrics (`ligand_iptm`, `affinity_pred_value`), and SMINA scores into a single CSV. Rows missing either score are flagged.

#### 1e. Prepare train / val / test splits

```bash
python scripts/prepare_dataset.py \
    --input final_master_dataset.csv \
    --train train_set.csv \
    --val val_set.csv \
    --test test_set.csv
```

Adds two derived columns:

| Column | Formula | Used for |
|--------|---------|----------|
| `ranking_weight` | `ligand_iptm² / (1 + \|pred1 − pred2\|)` | ListMLE and Boltz loss weighting |
| `latent_weight` | same | Distillation and Boltz loss weighting |

Splits are **cluster-based**: sequences with Hamming distance ≤ 2 are kept in the same split, preventing sequence-level leakage. Default ratio: 70 / 15 / 15 %.

---

### Phase 2 — Teacher–Student Distillation

**Goal:** Train a structure-aware teacher, distil its geometric knowledge into a fast sequence-only student.

#### 2a. Precompute structural features

```bash
python scripts/precompute_features.py
# (Note: input CSV, boltz_out/ dir, and feature_cache/ dir are hardcoded by default)
```

For each complex, extracts and caches:

| Key | Shape | Source |
|-----|-------|--------|
| `pocket` | `[N_p, 512]` | ESM-IF1 (GVP encoder, frozen) per residue. *Run in structure-only mode (dummy alanine sequence) to extract geometry independent of sequence identity.* |
| `ligand` | `[21, 768]` | UniMol2 (84M, frozen) per atom |
| `plddt` | `[N_p + 21]` | Boltz-2 combined protein + ligand pLDDT |
| `pae_p2l` | `[N_p, 21]` | Pocket→ligand PAE (primary interface signal, ~1.74 Å) |
| `pae_l2p` | `[21, N_p]` | Ligand→pocket PAE (secondary) |
| `pae_to_lig` | `[N_p]` | Per-residue mean PAE to ligand — data-driven paratope map |
| `pde_p2l` | `[N_p, 21]` | Pocket→ligand PDE (distance error, orthogonal to PAE) |
| `pde_l2p` | `[21, N_p]` | Ligand→pocket PDE |

Saved to `feature_cache/{seq_id}.npz`. Already-computed entries are skipped (incremental updates supported).

#### 2b. Train the teacher (BindNet)

```bash
python scripts/train_teacher.py
# (Note: train/val paths, epochs=60, and batch_size=64 are hardcoded by default)
```

**BindNet architecture:**

```
h_p [N_p, 512]  +  pLDDT-MLP  +  PAE-to-lig-MLP  →  proj_p  →  [N_p, 256]
h_l [21,  768]  +  pLDDT-MLP                       →  proj_l  →  [21,  256]
        │
        ▼
3 × BindNetLayer (4 heads each)
  Step 1: Ligand (Q) → Pocket (K,V)  bias = w_l2p·PAE_l2p + w_pde_l2p·PDE_l2p  →  FFN
  Step 2: Pocket (Q) → Ligand (K,V)  bias = w_p2l·PAE_p2l + w_pde_p2l·PDE_p2l  →  FFN
        │
        ▼
mean-pool(ligand) + attn-weighted-pool(protein)  →  concat [512]
        │
Fusion MLP: Lin(512→256) → LN → GELU → Lin(256→256)  →  complex_embed [256]
        │
Ranking Head: Lin(256→64) → LN → GELU → Dropout → Lin(64→1)
        │
Loss: Asymmetric ListMLE vs SMINA Vinardo  (fn_penalty=2.0)
```

Two checkpoints are saved independently:
- `best_teacher.pt` — best EEF@5% on validation
- `best_teacher_spearman.pt` — best Spearman ρ on validation

The training sampler over-samples strong binders (inverse-rank weighting) so top-1 % SMINA binders appear in every batch.

#### 2c. Cache teacher embeddings

```bash
python scripts/precompute_teacher.py \
    --model best_teacher.pt \
    --input final_master_dataset.csv \
    --cache_dir feature_cache/ \
    --out_emb teacher_embeddings.npy \
    --out_idx teacher_embedding_index.json
```

Runs the frozen trained teacher on every sample and saves the 256-D `complex_embed` vectors to `teacher_embeddings.npy` with a `{seq_id → row_index}` JSON index. These are the regression targets for student distillation — the teacher never runs again after this step.

#### 2d. Train the student

```bash
python scripts/train.py
# (Note: train/val paths, teacher cache paths, epochs=40, batch_size=32 are hardcoded by default)
```

**Student architecture:**

```
"VH|VL"  →  AbLang2 ("ablang2-paired")  →  [L, 480]
         →  Linear(480→256)
         →  CDRPositionEncoding (IMGT CDR-H1/H2/H3, CDR-L1/L2/L3 additive bias)
         →  MockLigandEncoder (10 learned static tokens, [10, 256])
         →  FullCrossAttention (ligand Q → protein K,V, CDR-biased keys)
         →  complex_embed  [256]
         →  4 heads
```

| Head | Output | Target | Loss | λ |
|------|--------|--------|------|---|
| Distillation | `[B, 256]` | `teacher_embeddings.npy` | Weighted MSE | 1.0 |
| Affinity Rank | `[B]` | SMINA Vinardo | Asymmetric ListMLE (fn_penalty=2.0) | 1.0 |
| Boltz Reg. | `[B]` | `boltz_affinity_pred_value` | Weighted MSE | 0.5 |
| Pose Quality | `[B, 1]` | `ligand_iptm > 0.75` | BCE | 0.1 |

**Staged AbLang2 unfreezing:**

| Epochs | AbLang2 state | LR schedule |
|--------|--------------|-------------|
| 0 – 5 | Frozen | 1e-3 → 1e-4 (cosine) |
| 6 – 15 | Blocks 10–11 only | 1e-4 → 1e-5 (cosine) |
| 16 – 39 | Full backbone | 1e-5 → 1e-6 (cosine) |

Training uses mixed-precision (FP16 autocast + GradScaler) and gradient clipping (max_norm=1.0).

Checkpoints: `best_student_model.pt` (EEF@5%) and `best_student_model_spearman.pt` (Spearman ρ).

#### 2e. Evaluate

```bash
# Student (sequence-only — no feature cache needed)
python scripts/evaluate.py \
    --type student \
    --model best_student_model.pt \
    --input test_set.csv \
    --out test_predictions.csv \
    --plot test_scatter.png

# Teacher (requires feature cache)
python scripts/evaluate.py \
    --type teacher \
    --model best_teacher.pt \
    --input test_set.csv \
    --cache_dir feature_cache/ \
    --out teacher_test_predictions.csv
```

Reports Spearman ρ, EEF@1 %, and EEF@5 % on the held-out test set.

**Metrics:**
- **Spearman ρ** — rank correlation between predicted and true SMINA scores. Expected to be negative (high predicted score = strong binder = negative SMINA).
- **EEF@k %** — Early Enrichment Factor at top-k %. How many more true top-k binders the model recovers versus random selection. EEF > 1 = better than random.

---

### Phase 3 — Evolutionary CDR-H3 Optimisation

**Goal:** Use the trained student as a fast oracle to guide CDR-H3 sequence optimisation via AdaLead.

```bash
python scripts/evolve.py
# (Note: input paths, population size, mutation rate, and rounds=10 are hardcoded)
```

The algorithm:
1. **Initialisation** — starts from the best binder in the input CSV (by SMINA score)
2. **AdaLead GA** — maintains a population of candidates, applying mutation and crossover
3. **Oracle scoring** — student model scores all mutants in batches; no Boltz-2 inference required
4. **Repeat** for 10 rounds

To extract the top-N candidates from a completed run:

```bash
python scripts/generate_top_ga_seq.py \
    --input evolved_candidates.csv \
    --top_n 500 \
    --out top_candidates.fasta
```

---

## Results

### Dataset Composition
The pipeline uses a clustered split (70/15/15) to prevent sequence leakage across a total dataset of ~10,500 unique sequences. Sequences were sampled primarily using AntiFold at various temperatures to explore the CDR-H3 landscape:

| Source | Train % | Val % | Test % |
|--------|---------|-------|--------|
| AntiFold Temp 0.5 | 11.97% | 0.00% | 0.00% |
| AntiFold Temp 1.0 | 42.15% | 15.28% | 14.16% |
| AntiFold Temp 1.2 | 0.71% | 0.22% | 0.45% |
| AntiFold Temp 1.5 | 41.59% | 66.52% | 68.76% |
| Random CDR sequences | 3.59% | 17.98% | 16.63% |

### Model Benchmarks
Evaluation on the held-out test set demonstrates that the Student model successfully distills the structural priors, maintaining strong ranking correlation (Spearman ρ) and early enrichment (EEF) while running orders of magnitude faster without requiring structural inputs.

| Model | Spearman ρ | EEF@5% |
|-------|-----------|--------|
| Teacher (BindNet 3D) | -0.589 | 5.52× |
| Student (Sequence-only) | -0.449 | 6.44× |

*(Note: Spearman ρ is expected to be negative as a highly negative SMINA score indicates stronger binding).*

### Evolutionary Optimisation (AdaLead GA)
The trained Student model was used as a fast oracle to optimize a baseline binder using an AdaLead genetic algorithm. The search was conducted over 30 generations, evaluating 200 mutants per generation (with a 10% mutation rate) and maintaining a board of 500 elite sequences. To verify the results, the top 500 candidates from the search were fully validated through Boltz-2 structure prediction and SMINA docking. The original 1I9J baseline binder had a SMINA affinity of -3.92 kcal/mol.
- **Affinity Distribution:** The optimized output heavily skewed towards stronger predicted binding bins:
  - **Strong Binders** (≤ -8.0 kcal/mol): 192 candidates (~38%)
  - **Moderate Binders** (-6.0 to -8.0 kcal/mol): 230 candidates (46%)
  - **Weak Binders** (> -6.0 kcal/mol): 78 candidates (16%)
- **Peak Performance:** The best candidate achieved a highly confident SMINA affinity of -10.75 kcal/mol.

---

## Key Design Decisions

**Why a fixed ligand (testosterone)?** The dataset was built around a single small molecule to control for ligand variation and isolate the antibody sequence signal. The `MockLigandEncoder` in the student model encodes this via 10 learned static tokens — meaningful only for testosterone. For a different target, retrain from Phase 1.

**Why Boltz-2 PAE and PDE as attention biases?** PAE (predicted aligned error) from Boltz-2 quantifies how confidently the model knows the relative position of two tokens. The pocket→ligand slice `pae_p2l` is low (~1.74 Å) for residues that are confidently positioned relative to the ligand — i.e., the structural paratope. Using this as a soft additive bias in cross-attention (separate learned scalar `w_pae_p2l` per direction and per layer) lets BindNet up-weight true interface contacts without hard masking. PDE (predicted distance error) adds an orthogonal distance-geometry signal.

**Why asymmetric ListMLE?** Standard MSE against docking scores treats all errors equally. The asymmetric penalty (`fn_penalty=2.0`) exponentially up-weights false negatives at the top of the ranked list — missing a true strong binder is much costlier than mis-ranking mediocre ones. This directly optimises EEF@1 %.

**Why staged AbLang2 unfreezing?** Unfreezing the full backbone immediately with a learning rate sized for the randomly-initialised heads causes catastrophic forgetting. The three-phase schedule lets the heads stabilise on frozen representations first, then fine-tunes only the top transformer blocks (which encode the most abstract sequence features), then finally fine-tunes the whole model at a low learning rate.

**Cluster-based train/val/test splits:** Antibody sequences generated by CDR mutation are highly similar. A random split would have near-identical sequences in both train and test sets, inflating metrics. Cluster-based splitting ensures that sequences within Hamming distance ≤ 2 are always in the same partition.

## Future Improvements

While AbDistill provides a strong baseline for structural knowledge distillation, several areas could be improved in future iterations:

1. **Larger Dataset:** Expanding the dataset beyond the current size could improve both teacher and student generalization across more diverse CDR-H3 landscapes.
2. **True Sequence Inputs for ESM-IF1:** ESM-IF1 is currently forced to run in a structure-only mode by passing a dummy poly-alanine sequence. Upgrading the pipeline to feed the true amino acid sequence to ESM-IF1 would allow the Teacher model to leverage both backbone geometry and residue-specific chemical environments.
3. **Dynamic Ligand Encoding:** The Student's `MockLigandEncoder` uses 10 static, learned tokens specific to testosterone. Integrating a dynamic small-molecule encoder (e.g., passing SMILES to a lightweight GNN or UniMol2) would allow a single Student model to generalize across multiple different targets without retraining.
4. **Boltz-2 Ensemble Averaging:** The Teacher currently extracts PAE/PDE and pLDDT features from a single Boltz-2 structure (`model_0`). Averaging these structural confidence metrics across the entire predicted ensemble could provide a more robust signal, particularly for highly flexible CDR loops.
5. **Boltz-2 Structural Embeddings:** Extracting internal representations directly from Boltz-2 via hooks could provide the Teacher with richer, end-to-end learned structural features.
6. **Cross-Reactivity Optimization:** By expanding the dataset and loss functions to include multiple targets (or off-targets), this framework could be naturally extended to perform multi-objective generation that explicitly optimizes for—or penalizes—cross-reactivity.

---

## Data Format

### Input CSV (master dataset)

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Unique sequence identifier |
| `heavy_sequence` | str | VH amino acid sequence |
| `light_sequence` | str | VL amino acid sequence |
| `smina_vinardo_affinity` | float | SMINA Vinardo score (kcal/mol, lower = better binder) |
| `boltz_affinity_pred_value` | float | Boltz-2 ensemble affinity prediction |
| `boltz_conf_ligand_iptm` | float | Boltz-2 ligand ipTM confidence (0–1) |
| `boltz_affinity_pred_value1` | float | Boltz-2 affinity seed 1 (used for disagreement weight) |
| `boltz_affinity_pred_value2` | float | Boltz-2 affinity seed 2 |
| `ranking_weight` | float | Added by `prepare_dataset.py` |
| `latent_weight` | float | Added by `prepare_dataset.py` |
---

## Reproducing the Full Pipeline

```bash
# 0. Setup
bash setup.sh && conda activate abdistill

# 1. Dataset
python scripts/generate_antifold.py --scaffold 1i9j_imgt.pdb --num_seqs 10000
python scripts/generate_random_dataset.py --num_seqs 500
python scripts/generate_boltz.py --input initial_dataset_10000.csv --ligand_smiles "CC1CCC2C(C1)CCC3C2CCC4=CC(=O)CCC34C" --out_dir boltz_out/
python scripts/score_smina.py --boltz_dir boltz_out/ --workers 8
python scripts/aggregate_results.py --input initial_dataset_10000.csv --random random_dataset_500.csv --boltz_dir boltz_out/ --out final_master_dataset.csv
python scripts/prepare_dataset.py --input final_master_dataset.csv --train train_set.csv --val val_set.csv --test test_set.csv

# 2. Teacher
python scripts/precompute_features.py --input train_set.csv --boltz_dir boltz_out/ --out_dir feature_cache/
python scripts/train_teacher.py --train train_set.csv --val val_set.csv --epochs 60
python scripts/precompute_teacher.py --model best_teacher.pt --input final_master_dataset.csv --cache_dir feature_cache/

# 3. Student
python scripts/train.py --train train_set.csv --val val_set.csv --epochs 40
python scripts/evaluate.py --type student --model best_student_model.pt --input test_set.csv

# 4. Optimise
python scripts/evolve.py
python scripts/generate_top_ga_seq.py --input evolved_candidates.csv --top_n 500 --out top_candidates.fasta
```

---

## Dependencies

| Package | Role |
|---------|------|
| `boltz` | Boltz-2 structure prediction and confidence outputs |
| `ablang2` | Paired antibody language model (student backbone) |
| `unimol_tools` | UniMol2 atomic ligand encoder (teacher) |
| `fair-esm` | ESM-IF1 structural pocket encoder (teacher) |
| `antifold` | Structurally-conditioned CDR sequence design and PSSM |
| `torch_geometric` + `torch_scatter` | Required by ESM-IF1 GVP layers |
| `anarci` | IMGT CDR numbering (for CDRPositionEncoding) |
| `smina` | Vinardo docking score oracle |
| `rdkit` | Ligand handling |
| `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm` | Standard ML stack |
