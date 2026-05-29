import os
import subprocess
import pandas as pd
import glob
import sys
import urllib.request
from tqdm import tqdm

def generate_initial_sequences(num_sequences=10, scaffold_pdb="1i9j_imgt.pdb"):
    """
    Uses AntiFold to generate diverse, structurally plausible CDRH3 loops by 
    structurally conditioning on the rest of the antibody scaffold.
    """
    if not os.path.exists(scaffold_pdb):
        print(f"Error: Could not find {scaffold_pdb} in the current directory.")
        print("Please ensure you have placed the IMGT-numbered scaffold PDB in the workspace.")
        sys.exit(1)

    fv_pdb = scaffold_pdb

    print(f"Generating {num_sequences} sequences using AntiFold on {fv_pdb}...")
    
    out_dir = "antifold_out"
    os.makedirs(out_dir, exist_ok=True)
    
    temperatures = [0.5, 1.0, 1.5]
    seqs_per_temp = (num_sequences // len(temperatures)) + 1
    
    unique_heavy_seqs = set()
    df_data = []
    
    for temp in temperatures:
        if len(df_data) >= num_sequences:
            break
            
        ask_count = int(seqs_per_temp * 1.5)
        print(f"--- Sampling at Temperature {temp} ---")
        
        cmd = [
            sys.executable, "-m", "antifold.main",
            "--pdb_file", os.path.abspath(fv_pdb),
            "--heavy_chain", "H",
            "--light_chain", "L",
            "--num_seq_per_target", str(ask_count),
            "--sampling_temp", str(temp),
            "--regions", "CDRH3",
            "--out_dir", os.path.abspath(out_dir)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"AntiFold failed at temp {temp}. Error: {e.stderr.decode('utf-8') if e.stderr else e}")
            continue
            
        pdb_basename = fv_pdb.split('.')[0]
        fasta_files = glob.glob(f"{out_dir}/{pdb_basename}*/*.fasta") + glob.glob(f"{out_dir}/*.fasta")
        
        if not fasta_files:
            continue
            
        fasta_files.sort(key=os.path.getmtime, reverse=True)
        fasta_file = fasta_files[0]
        
        sequences = []
        headers = []
        current_seq = ""
        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    headers.append(line)
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
                
        added_this_temp = 0
        for header, seq in zip(headers, sequences):
            if "WT" in header.upper() or "WILD" in header.upper():
                continue
                
            # AntiFold natively outputs chains separated by '/' (e.g. HEAVY/LIGHT)
            if '/' in seq:
                heavy_seq, light_seq = seq.split('/')
            else:
                # Fallback if no separator
                heavy_seq = seq[:122].replace(':', '')
                light_seq = seq[122:].replace(':', '')
                
            # Clean up any leftover separator chars
            heavy_seq = heavy_seq.replace(':', '')
            light_seq = light_seq.replace(':', '')
            
            if heavy_seq not in unique_heavy_seqs:
                unique_heavy_seqs.add(heavy_seq)
                df_data.append({
                    'id': f"seq_{len(unique_heavy_seqs)-1}_T{temp}",
                    'heavy_sequence': heavy_seq,
                    'light_sequence': light_seq
                })
                added_this_temp += 1
                if len(df_data) >= num_sequences:
                    break
                    
        print(f"Added {added_this_temp} unique sequences from Temperature {temp}.")
        
    while len(df_data) < num_sequences:
        remaining = num_sequences - len(df_data)
        temp = 1.2 
        ask_count = int(remaining * 2) + 50
        print(f"--- Fallback Sampling at Temp {temp} for {remaining} sequences ---")
        
        cmd = [
            sys.executable, "-m", "antifold.main",
            "--pdb_file", os.path.abspath(fv_pdb),
            "--heavy_chain", "H",
            "--light_chain", "L",
            "--num_seq_per_target", str(ask_count),
            "--sampling_temp", str(temp),
            "--regions", "CDRH3",
            "--out_dir", os.path.abspath(out_dir)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            break
            
        pdb_basename = fv_pdb.split('.')[0]
        fasta_files = glob.glob(f"{out_dir}/{pdb_basename}*/*.fasta") + glob.glob(f"{out_dir}/*.fasta")
        if not fasta_files: break
            
        fasta_files.sort(key=os.path.getmtime, reverse=True)
        fasta_file = fasta_files[0]
        
        sequences, headers, current_seq = [], [], ""
        with open(fasta_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    headers.append(line)
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq: sequences.append(current_seq)
                
        added_this_temp = 0
        for header, seq in zip(headers, sequences):
            if "WT" in header.upper() or "WILD" in header.upper(): continue
            
            if '/' in seq:
                heavy_seq, light_seq = seq.split('/')
            else:
                heavy_seq = seq[:122].replace(':', '')
                light_seq = seq[122:].replace(':', '')
                
            heavy_seq = heavy_seq.replace(':', '')
            light_seq = light_seq.replace(':', '')
            
            if heavy_seq not in unique_heavy_seqs:
                unique_heavy_seqs.add(heavy_seq)
                df_data.append({
                    'id': f"seq_{len(unique_heavy_seqs)-1}_T{temp}",
                    'heavy_sequence': heavy_seq,
                    'light_sequence': light_seq
                })
                added_this_temp += 1
                if len(df_data) >= num_sequences: break
        print(f"Added {added_this_temp} unique sequences.")
        
    df = pd.DataFrame(df_data)
    print(f"Successfully generated exactly {len(df)} completely UNIQUE sequences.")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dataset sequences using AntiFold")
    parser.add_argument("--num_seqs", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--scaffold", type=str, required=True, help="Input IMGT-numbered Fv PDB scaffold")
    args = parser.parse_args()

    output_filename = f"initial_dataset_{args.num_seqs}.csv"
    df = generate_initial_sequences(args.num_seqs, args.scaffold)
    df.to_csv(output_filename, index=False)
    print(f"\nSaved raw sequences to {output_filename}")
