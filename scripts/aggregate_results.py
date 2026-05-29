import os
import glob
import json
import pandas as pd
import argparse

def main(base_dir, input_csv, random_dataset, out_file):
    print("Loading sequence CSVs for reference...")
    
    # 1. Compulsory initial dataset
    if not os.path.exists(input_csv):
        print(f"Error: Required file {input_csv} not found.")
        return
        
    df_master = pd.read_csv(input_csv)
    print(f"Loaded {len(df_master)} sequences from {input_csv}")
    
    # 2. Optional random dataset
    if random_dataset:
        if not os.path.exists(random_dataset):
            print(f"Error: Optional file {random_dataset} not found. Skipping.")
        else:
            df_hard = pd.read_csv(random_dataset)
            df_master = pd.concat([df_master, df_hard], ignore_index=True)
            print(f"Loaded {len(df_hard)} sequences from {random_dataset}")
            
    # Create a fast lookup dictionary: id -> {heavy_sequence, light_sequence}
    seq_lookup = {}
    for _, row in df_master.iterrows():
        seq_lookup[row['id']] = {
            "heavy_sequence": row.get("heavy_sequence", ""),
            "light_sequence": row.get("light_sequence", "")
        }
        
    print(f"Loaded {len(seq_lookup)} total unique sequences into lookup table.")
    
    # Use generic prefix to match our refactoring (no instA_)
    all_dirs = glob.glob(os.path.join(base_dir, "boltz_results_*"))
    
    print(f"Found {len(all_dirs)} prediction folders in '{base_dir}'. Aggregating data...")
    if len(all_dirs) == 0:
        print("No folders found. Check your --dir argument.")
        return
    
    aggregated_data = []
    
    for d in all_dirs:
        seq_id = os.path.basename(d).replace("boltz_results_", "")
        pred_dir = os.path.join(d, "predictions", seq_id)
        
        # Look up sequences
        seqs = seq_lookup.get(seq_id, {"heavy_sequence": "UNKNOWN", "light_sequence": "UNKNOWN"})
        
        row_data = {
            "id": seq_id,
            "heavy_sequence": seqs["heavy_sequence"],
            "light_sequence": seqs["light_sequence"]
        }
        
        # 1. Parse all keys from Boltz Affinity JSON
        boltz_aff_path = os.path.join(pred_dir, f"affinity_{seq_id}.json")
        if os.path.exists(boltz_aff_path):
            with open(boltz_aff_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    row_data[f"boltz_{k}"] = v
                
        # 2. Parse all keys from Smina Affinity JSON
        smina_aff_path = os.path.join(pred_dir, f"smina_affinity_{seq_id}.json")
        if os.path.exists(smina_aff_path):
            with open(smina_aff_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    row_data[k] = v
                
        # 3. Parse all keys from Boltz Confidence JSON
        conf_path = os.path.join(pred_dir, f"confidence_{seq_id}_model_0.json")
        if os.path.exists(conf_path):
            with open(conf_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    row_data[f"boltz_conf_{k}"] = v
                
        aggregated_data.append(row_data)
        
    # Save to a master CSV
    df_final = pd.DataFrame(aggregated_data)
    df_final.to_csv(out_file, index=False)
    
    print(f"\n Successfully aggregated {len(df_final)} rows!")
    print(f"Data saved to -> {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Boltz JSON outputs and Smina JSON outputs into a final CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Compulsory input CSV containing generated sequences (e.g. initial_dataset_10000.csv)")
    parser.add_argument("--random_dataset", type=str, default=None, help="Optional input CSV containing random sequences generated from generate_random_dataset.py")
    parser.add_argument("--dir", type=str, default="boltz_out", help="Directory containing boltz_results_* folders")
    parser.add_argument("--out", type=str, default="final_master_dataset.csv", help="Name of output CSV")
    
    args = parser.parse_args()
    
    main(args.dir, args.input_csv, args.random_dataset, args.out)
