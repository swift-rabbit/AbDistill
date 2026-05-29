import pandas as pd
import random

# Fixed backbone sequences for 1I9J
HEAVY_PREFIX = "EVKLVESGGGLVKPGGSLKLSCAASGFTFSTYALSWVRQTADKRLEWVASIVSGGNTYYSGSVKGRFTISRDIARNILYLQMSSLRSEDTAMYYC"
HEAVY_SUFFIX = "WGQGTLVTVSA"
LIGHT_SEQUENCE = "DVVVTQTPLSLPVSLGDQASISCRSSQSIVHSNGNSYLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPPTFGGGTKLEIK"

# 20 standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def generate_random_cdrh3(length=12):
    """Generate a completely random string of amino acids of fixed length."""
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))

def main(num_seqs):
    print(f"Generating {num_seqs} completely random sequences...")
    data = []
    
    for i in range(num_seqs):
        # Generate a purely random, physically impossible CDRH3 loop
        random_loop = generate_random_cdrh3()
        
        # Insert it directly into the rigid 1I9J heavy chain backbone
        heavy_seq = HEAVY_PREFIX + random_loop + HEAVY_SUFFIX
        
        # Save it
        data.append({
            "id": f"random_{i}",
            "heavy_sequence": heavy_seq,
            "light_sequence": LIGHT_SEQUENCE
        })
        
    df = pd.DataFrame(data)
    out_file = f"random_dataset_{num_seqs}.csv"
    df.to_csv(out_file, index=False)
    
    print(f"Successfully generated {num_seqs} sequences and saved to '{out_file}'!")
    print(f"Example Heavy Sequence: {df.iloc[0]['heavy_sequence']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a dataset of random sequences.")
    parser.add_argument("--num_seqs", type=int, default=500, help="Number of sequences to generate (default: 500)")
    args = parser.parse_args()
    
    main(args.num_seqs)
