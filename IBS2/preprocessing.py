from Bio import SeqIO
import pandas as pd
import itertools

# ---------------------------
# Helper Functions
# ---------------------------

def fast_percent_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    return matches / min_len * 100

def count_gaps(seq):
    return seq.count('-')

# ---------------------------
# Main processing
# ---------------------------

tfa_file = "RV100/BBA0001.tfa"

sequences = list(SeqIO.parse(tfa_file, "fasta"))

# For testing, only use first 3 sequences
n=len(sequences)
print(f"sequences[:{n}]")

results = []

for seq1, seq2 in itertools.combinations(sequences, 2):
    seq1_str = str(seq1.seq)
    seq2_str = str(seq2.seq)
    
    pid = fast_percent_identity(seq1_str, seq2_str)
    len_diff = abs(len(seq1_str) - len(seq2_str))
    gaps_seq1 = count_gaps(seq1_str)
    gaps_seq2 = count_gaps(seq2_str)
    
    pair_id = f"{seq1.id}_{seq2.id}"
    
    results.append({
        "pair_id": pair_id,
        "seq1": seq1_str,
        "seq2": seq2_str,
        "percent_identity": pid,
        "length_diff": len_diff,
        "num_gaps_seq1": gaps_seq1,
        "num_gaps_seq2": gaps_seq2
    })

df = pd.DataFrame(results)
df.to_csv("BBA0001_pairwise_features_fast.csv", index=False)
print("CSV created with", len(df), "pairs")
