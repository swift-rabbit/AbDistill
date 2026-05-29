"""
evolve.py
────────────
Phase 3: Structure-Aware AdaLead Evolutionary Screening

Uses:
  1. 1D Student Model (for binding affinity predictions)
  2. AntiFold PSSM (for true 3D structural suitability / foldability)
  3. AdaLead combinatorial search initialized by Deep Mutational Scanning (DMS)
"""

import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abdistill.models import StudentModel

DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMINO_ACIDS     = list("ACDEFGHIKLMNPQRSTVWY")

# ─────────────────────────────────────────────────────────────────────────────
# Core Logic
# ─────────────────────────────────────────────────────────────────────────────

def load_baseline(input_csv, override_start=None, override_end=None, baseline_fasta=None):
    """Load the baseline sequence and identify the CDR H3 bounds dynamically or manually."""
    
    if baseline_fasta and os.path.exists(baseline_fasta):
        print(f"Loading custom baseline from {baseline_fasta}...")
        seqs_dict = {}
        curr_header = ""
        with open(baseline_fasta, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    curr_header = line[1:].lower()
                    seqs_dict[curr_header] = ""
                elif curr_header:
                    seqs_dict[curr_header] += line
                    
        baseline_heavy = next((seq for h, seq in seqs_dict.items() if 'heavy' in h or 'h' in h), None)
        baseline_light = next((seq for h, seq in seqs_dict.items() if 'light' in h or 'l' in h), None)
        
        if not baseline_heavy or not baseline_light:
            raise ValueError(f"Could not find both heavy and light chains in {baseline_fasta}. Headers must contain 'heavy'/'light' or 'h'/'l'.")
            
        if override_start is None or override_end is None:
            raise ValueError("When providing a custom FASTA, you MUST provide --cdr_start and --cdr_end!")
            
        cdr_start = override_start
        cdr_end = override_end
        print(f"Using Custom Baseline Sequences with manual CDR bounds: {cdr_start}-{cdr_end}")
        return baseline_heavy, baseline_light, cdr_start, cdr_end

    # Default CSV logic
    df = pd.read_csv(input_csv)
    best_row = df.loc[df['smina_vinardo_affinity'].idxmin()]
    baseline_heavy = str(best_row['heavy_sequence'])
    baseline_light = str(best_row['light_sequence'])
    
    if override_start is not None and override_end is not None:
        cdr_start = override_start
        cdr_end = override_end
    else:
        seqs = df['heavy_sequence'].values
        chars = np.array([list(s) for s in seqs])
        varying_positions = np.where((chars[0] != chars).any(axis=0))[0]
        
        if len(varying_positions) > 0:
            cdr_start = varying_positions[0]
            cdr_end = varying_positions[-1] + 1
        else:
            cdr_start, cdr_end = 95, 105
        
    print(f"Detected CDR H3 region from index {cdr_start} to {cdr_end} (Length: {cdr_end - cdr_start})")
    return baseline_heavy, baseline_light, cdr_start, cdr_end

def load_antifold_pssm(csv_path):
    """
    Loads the AntiFold 3D suitability matrix into a blazingly fast dictionary lookup.
    """
    if not os.path.exists(csv_path):
        print(f"Error: AntiFold matrix {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    df_h = df[df['pdb_chain'] == 'H'].reset_index(drop=True)
    
    pssm = []
    for i, row in df_h.iterrows():
        probs = {aa: row[aa] for aa in AMINO_ACIDS}
        pssm.append(probs)
        
    print(f"Loaded AntiFold 3D PSSM for Heavy Chain (Length: {len(pssm)})")
    return pssm

def generate_dms(sequence, start, end):
    """Generate all single point mutations for the CDR H3 region."""
    mutants = []
    for pos in range(start, end):
        for aa in AMINO_ACIDS:
            if sequence[pos] != aa:
                seq_list = list(sequence)
                seq_list[pos] = aa
                mutants.append("".join(seq_list))
    return mutants

def mutate_and_recombine(board_seqs, start, end, num_mutants, mutation_rate):
    """AdaLead Recombination: Sample parents from board, crossover, and mutate."""
    mutants = set()
    attempts = 0
    while len(mutants) < num_mutants and attempts < num_mutants * 5:
        attempts += 1
        
        # Sample parents (bias towards top of the board)
        p1 = random.choices(board_seqs, weights=np.linspace(1.0, 0.1, len(board_seqs)))[0]
        p2 = random.choices(board_seqs, weights=np.linspace(1.0, 0.1, len(board_seqs)))[0]
        
        # Crossover
        child = list(p1)
        for i in range(start, end):
            if random.random() < 0.5:
                child[i] = p2[i]
                
        # Point Mutation
        for i in range(start, end):
            if random.random() < mutation_rate:
                child[i] = random.choice(AMINO_ACIDS)
                
        mutants.add("".join(child))
        
    return list(mutants)

class AdaLeadBoard:
    def __init__(self, max_size):
        self.board = {}
        self.max_size = max_size
        
    def add(self, seqs, fitnesses, affinities, boltzs, antifolds):
        for s, f, a, b, af in zip(seqs, fitnesses, affinities, boltzs, antifolds):
            if s not in self.board:
                self.board[s] = {'fitness': f, 'aff': a, 'boltz': b, 'antifold': af}
                
        # Trim to keep only the best
        sorted_items = sorted(self.board.items(), key=lambda x: x[1]['fitness'], reverse=True)
        self.board = dict(sorted_items[:self.max_size])
        
    def get_seqs(self):
        return list(self.board.keys())

    def get_best_stats(self):
        best_entry = list(self.board.values())[0]
        return best_entry['fitness'], best_entry['aff'], best_entry['antifold']


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_batch(student_model, antifold_pssm, population_heavy, baseline_light, cdr_start, cdr_end):
    """Score sequences using Student Model (Binding) and AntiFold (3D Foldability)."""
    
    # 1. Student Model Evaluation (Affinity)
    student_model.eval()
    all_aff = []
    all_boltz = []
    batch_size = 128
    
    with torch.no_grad():
        for i in range(0, len(population_heavy), batch_size):
            batch_heavy = population_heavy[i:i+batch_size]
            paired_seqs = [f"{h}|{baseline_light}" for h in batch_heavy]
            
            _, pred_affinity, pred_boltz, _ = student_model(paired_seqs)
            all_aff.extend(pred_affinity.cpu().numpy())
            all_boltz.extend(pred_boltz.cpu().numpy())
            
    aff = np.array(all_aff)
    boltz = np.array(all_boltz)
    
    # 2. AntiFold 3D PSSM Evaluation (Structure)
    # We calculate the average Probability of the mutated CDRH3 loop
    antifold_probs = []
    for seq in population_heavy:
        nlls = []
        for i in range(cdr_start, cdr_end):
            # Look up the log-probability of the amino acid at this specific position
            log_p = antifold_pssm[i][seq[i]]
            nlls.append(-log_p)
            
        avg_nll = np.mean(nlls)
        avg_prob = np.exp(-avg_nll) # Convert NLL back to a 0.0 - 1.0 probability
        antifold_probs.append(avg_prob)
        
    antifold_probs = np.array(antifold_probs)
    
    # 3. Multiplicative Fitness (The Gatekeeper Strategy)
    # Convert affinity to a positive reward scale (Higher is better!).
    # Max to protect against out-of-distribution hallucination.
    aff_reward = np.maximum(0.001, aff + 3.0)
    
    # Fitness = (Tighter Binding) * (Probability it actually folds)
    fitness = aff_reward * antifold_probs
    
    return fitness, aff, boltz, antifold_probs

# ─────────────────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evolution(args):
    print("Loading Baseline Sequence & Identifying CDR H3...")
    heavy_wt, light_wt, cdr_start, cdr_end = load_baseline(args.input, args.cdr_start, args.cdr_end, args.baseline_fasta)
    
    print("\nLoading AntiFold 3D Suitability Matrix...")
    antifold_pssm = load_antifold_pssm(args.antifold)
    
    print("\nLoading 1D Student Model...")
    student_model = StudentModel(hidden_dim=256, teacher_dim=256).to(DEVICE)
    student_model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=True))
            
    board = AdaLeadBoard(max_size=args.board_size)
    
    # ── Phase 1: Deep Mutational Scan (DMS) ──
    print("\n[Phase 1] Seeding Board with Deep Mutational Scan (DMS)...")
    dms_seqs = generate_dms(heavy_wt, cdr_start, cdr_end)
    dms_seqs.insert(0, heavy_wt) # include wildtype
    
    f, a, b, af = evaluate_batch(student_model, antifold_pssm, dms_seqs, light_wt, cdr_start, cdr_end)
    board.add(dms_seqs, f, a, b, af)
    
    best_f, best_a, best_af = board.get_best_stats()
    print(f"DMS Complete. Starting Board Top Fitness: {best_f:.2f} | Raw Aff: {best_a:.2f} | AntiFold Prob: {best_af:.3f}")

    # ── Phase 2: AdaLead Directed Evolution ──
    print(f"\n[Phase 2] AdaLead Search ({args.generations} Generations)...")
    
    for gen in range(args.generations):
        # Generate new variants by recombining the best ones on the board
        board_seqs = board.get_seqs()
        new_variants = mutate_and_recombine(board_seqs, cdr_start, cdr_end, args.mutants_per_gen, args.mutation_rate)
        
        # Evaluate
        f, a, b, af = evaluate_batch(student_model, antifold_pssm, new_variants, light_wt, cdr_start, cdr_end)
        
        # Update board (it automatically keeps only the global top N)
        board.add(new_variants, f, a, b, af)
        
        best_f, best_a, best_af = board.get_best_stats()
        print(f"Gen {gen+1:02d} | Max Fit: {best_f:5.2f} | Best Aff Pred: {best_a:5.2f} | Best AntiFold: {best_af:4.3f}")
        
    # ── Save Results ──
    print("\nEvolution Complete! Saving Top Candidates...")
    
    results = []
    for seq, data in board.board.items():
        results.append({
            'heavy_sequence': seq,
            'light_sequence': light_wt,
            'fitness_score': data['fitness'],
            'pred_smina_rank': data['aff'],
            'pred_boltz_delta_g': data['boltz'],
            'antifold_prob': data['antifold']
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.out, index=False)
    print(f"Saved {len(df_res)} candidates to {args.out}!")
    print("\nNext step: Validate the best hits with Boltz-1!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Structure-Aware AdaLead Evolutionary Screening")
    
    # Files
    parser.add_argument("--input", type=str, default="final_master_dataset.csv", help="Dataset used to find the baseline sequence")
    parser.add_argument("--baseline_fasta", type=str, default=None, help="Optional FASTA file (e.g. ga_baseline.fasta) to override the baseline")
    parser.add_argument("--antifold", type=str, required=True, help="AntiFold likelihood CSV (e.g. antifold_1i9j_likelihood.csv)")
    parser.add_argument("--model", type=str, default="best_student_model_spearman.pt", help="Trained Student Model weights")
    parser.add_argument("--out", type=str, default="evolved_candidates.csv", help="Output file for generated variants")
    
    # Genetic Algorithm Params
    parser.add_argument("--board_size", type=int, default=500, help="How many elite sequences AdaLead remembers")
    parser.add_argument("--mutants_per_gen", type=int, default=200, help="How many new sequences to evaluate per step")
    parser.add_argument("--generations", type=int, default=30, help="Number of AdaLead generations")
    parser.add_argument("--mutation_rate", type=float, default=0.10, help="Chance to mutate a position when recombining")
    
    # Sequence Bounds
    parser.add_argument("--cdr_start", type=int, default=None, help="Start index of CDRH3 region (inclusive). Automatically detected if None.")
    parser.add_argument("--cdr_end", type=int, default=None, help="End index of CDRH3 region (exclusive). Automatically detected if None.")
    
    args = parser.parse_args()
    run_evolution(args)
