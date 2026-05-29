import os
import subprocess
import pandas as pd
from tqdm import tqdm

def run_boltz_inference(sequence_id, heavy_sequence, light_sequence, ligand_smiles, gpu_id, num_samples=1, output_dir="boltz_out"):
    """
    Runs Boltz-2 for a given heavy and light chain pair with the ligand.
    """
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{sequence_id}.yaml")
    
    yaml_content = f"""sequences:
  - protein:
      id: "A"
      sequence: "{heavy_sequence}"
      msa: "empty"
  - protein:
      id: "B"
      sequence: "{light_sequence}"
      msa: "empty"
  - ligand:
      id: "C"
      smiles: "{ligand_smiles}"
properties:
  - affinity:
      binder: "C"
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    import sys
    boltz_bin = os.path.join(os.path.dirname(sys.executable), "boltz")
    cmd = [
        boltz_bin, "predict",
        yaml_path,
        "--out_dir", output_dir,
        "--output_format", "pdb",
        "--override",
        "--no_kernels", 
        "--diffusion_samples", str(num_samples)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    nccl_candidates = [
        "/venv/ablig/lib/python3.10/site-packages/nvidia/nccl/lib",
        "/venv/ablig/lib/python3.10/site-packages/torch/lib",
    ]
    nccl_prepend = ":".join(p for p in nccl_candidates if os.path.exists(p))
    if nccl_prepend:
        env["LD_LIBRARY_PATH"] = f"{nccl_prepend}:{env.get('LD_LIBRARY_PATH', '')}"

    env["BOLTZ_CACHE"] = os.path.abspath("boltz_weights")
    
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        expected_folder = os.path.join(output_dir, f"{sequence_id}")
        if not os.path.exists(expected_folder): return False
        return True
    except subprocess.CalledProcessError:
        return False

def massive_parallel_oracle(df, ligand_smiles, num_gpus=2, concurrent_per_gpu=4, output_dir="boltz_out"):
    """
    Distributes Boltz-2 inference across GPUs, with heavy concurrency per GPU.
    """
    total_workers = num_gpus * concurrent_per_gpu
    print(f"Starting parallel Boltz oracle with {total_workers} concurrent workers...")
    
    def worker_task(args):
        seq_id, heavy_sequence, light_sequence, worker_id = args
        gpu_id = worker_id % num_gpus
        
        import os, json, shutil
        boltz_score = 0.0
        
        boltz_prefix = f"{output_dir}/boltz_results_{seq_id}"
        exact_json_path = f"{boltz_prefix}/predictions/{seq_id}/affinity_{seq_id}.json"
        
        if os.path.exists(exact_json_path) and os.path.getsize(exact_json_path) > 0:
            try:
                with open(exact_json_path, "r") as f: data = json.load(f)
                if isinstance(data, dict): boltz_score = data.get("affinity_pred_value", 0.0)
            except Exception: pass
            return seq_id, boltz_score
        
        if os.path.exists(boltz_prefix):
            try: shutil.rmtree(boltz_prefix)
            except Exception: pass
                
        success = run_boltz_inference(seq_id, heavy_sequence, light_sequence, ligand_smiles, gpu_id, num_samples=1, output_dir=output_dir)
        if not success: return seq_id, boltz_score
            
        if os.path.exists(exact_json_path):
            try:
                with open(exact_json_path, "r") as f: data = json.load(f)
                if isinstance(data, dict): boltz_score = data.get("affinity_pred_value", 0.0)
            except Exception: pass

        return seq_id, boltz_score

    tasks = [(row['id'], row['heavy_sequence'], row['light_sequence'], i) for i, row in df.iterrows()]
    results = []
    
    from multiprocessing.pool import ThreadPool
    with ThreadPool(processes=total_workers) as pool:
        for res in tqdm(pool.imap_unordered(worker_task, tasks), total=len(tasks)):
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Boltz-2 3D generation on a CSV of sequences")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file from AntiFold")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent processes per GPU")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of sequences to process before saving")
    parser.add_argument("--out_dir", type=str, default="boltz_out", help="Directory to save Boltz-2 outputs")
    parser.add_argument("--start", type=int, default=None, help="Start sequence index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End sequence index (inclusive)")
    args = parser.parse_args()

    testosterone_smiles = "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O"
    df_initial = pd.read_csv(args.input_csv)
    
    if args.start is not None and args.end is not None:
        df_initial = df_initial.iloc[args.start : args.end + 1].reset_index(drop=True)
        print(f"RANGE ACTIVATED: Processing from sequence {args.start} to {args.end}")

    print("Ensuring Boltz-2 cache is ready...")
    try:
        from boltz.main import download_boltz2
        import pathlib
        cache_dir = os.path.abspath("boltz_weights")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["BOLTZ_CACHE"] = cache_dir
        download_boltz2(cache=pathlib.Path(cache_dir))
    except Exception as e:
        print(f"Warning during cache download: {e}")

    for start_idx in range(0, len(df_initial), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(df_initial))
        df_batch = df_initial.iloc[start_idx:end_idx].copy()
        
        print(f"\n--- Processing batch {start_idx} to {end_idx-1} ---")
        massive_parallel_oracle(df_batch, testosterone_smiles, num_gpus=args.num_gpus, concurrent_per_gpu=args.concurrent, output_dir=args.out_dir)
        
    print("\nComplete! Boltz-2 structures generated successfully in the output directory.")
