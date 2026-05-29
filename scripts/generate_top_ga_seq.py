import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract the top N candidates from the genetic algorithm output.")
    parser.add_argument("--input", type=str, default="evolved_candidates.csv", help="Input CSV from GA")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top sequences to extract")
    parser.add_argument("--out", type=str, default="top_ga_sequences.csv", help="Output CSV for Boltz-1 validation")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    # Sort and slice top K
    df = df.sort_values('fitness_score', ascending=False).head(args.top_k).reset_index(drop=True)

    # Add an ID column required by generate_boltz.py
    df['id'] = [f"ev_rank_{i+1}" for i in range(len(df))]

    # Reorder columns just to be clean
    cols = ['id', 'heavy_sequence', 'light_sequence', 'fitness_score', 'pred_smina_rank', 'antifold_prob']
    valid_cols = [c for c in cols if c in df.columns]
    df = df[valid_cols]

    df.to_csv(args.out, index=False)
    print(f"Saved top {len(df)} candidates to {args.out}")

if __name__ == "__main__":
    main()
