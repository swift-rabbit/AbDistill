import os
import glob
import json
import time
import subprocess
import concurrent.futures
import argparse

def process_complex(dir_path):
    """Splits the PDB, runs Smina in memory, and saves the score to JSON."""
    seq_id = os.path.basename(dir_path).replace("boltz_results_", "")
    pred_dir = os.path.join(dir_path, "predictions", seq_id)
    
    pdb_path = os.path.join(pred_dir, f"{seq_id}_model_0.pdb")
    json_out_path = os.path.join(pred_dir, f"smina_affinity_{seq_id}.json")
    
    # Skip if already calculated
    if os.path.exists(json_out_path):
        return f"[{seq_id}] Skipped (Already scored)"
        
    if not os.path.exists(pdb_path):
        return f"[{seq_id}] ERROR: PDB not found"

    # We need unique temp files for each parallel worker to prevent collisions
    rec_path = os.path.join(pred_dir, f"temp_rec_{seq_id}.pdb")
    lig_path = os.path.join(pred_dir, f"temp_lig_{seq_id}.pdb")

    # 1. Split PDB into Receptor (A, B) and Ligand (C)
    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    with open(rec_path, 'w') as rec, open(lig_path, 'w') as lig:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21]
                if chain in ['A', 'B']:
                    rec.write(line)
                elif chain == 'C':
                    lig.write(line)

    # 2. Run Smina
    affinity = None
    try:
        result = subprocess.run(
            ["smina", "--receptor", rec_path, "--ligand", lig_path, "--score_only", "--scoring", "vinardo", "--cpu", "1"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 3. Parse output
        for line in result.stdout.split('\n'):
            if "Affinity:" in line:
                # Extract the float value
                affinity = float(line.split()[1])
                break
                
    except Exception as e:
        return f"[{seq_id}] ERROR during Smina execution: {e}"
        
    finally:
        # 4. Cleanup temporary files immediately
        if os.path.exists(rec_path): os.remove(rec_path)
        if os.path.exists(lig_path): os.remove(lig_path)

    # 5. Save to JSON perfectly matching Boltz format
    if affinity is not None:
        with open(json_out_path, 'w') as f:
            json.dump({"smina_vinardo_affinity": affinity}, f, indent=4)
        return f"[{seq_id}] Success: {affinity} kcal/mol"
    else:
        return f"[{seq_id}] ERROR: Affinity not found in output"

def main(base_dir, cpus):
    print(f"Locating Boltz prediction folders in {base_dir}...")
    all_dirs = glob.glob(os.path.join(base_dir, "boltz_results_*"))
    print(f"Found {len(all_dirs)} total complexes to score.")
    
    start_time = time.time()
    
    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        results = executor.map(process_complex, all_dirs)
        
        for i, res in enumerate(results):
            if "Success" in res:
                success_count += 1
            if i % 500 == 0:
                print(f"Progress: {i}/{len(all_dirs)}... Latest: {res}")

    end_time = time.time()
    print(f"\n Finished! Successfully scored {success_count} complexes.")
    print(f" Total parallel execution time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Smina scoring on Boltz output folders")
    parser.add_argument("--dir", type=str, default="boltz_out", help="Directory containing boltz_results_* folders")
    parser.add_argument("--cpus", type=int, default=10, help="Number of concurrent CPU workers to use")
    args = parser.parse_args()
    
    main(args.dir, args.cpus)
