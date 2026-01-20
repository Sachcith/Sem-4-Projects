# File: count_sequences.py

import os
from Bio import SeqIO

# Path to folder
folder_path = "RV100"

# Dictionary to store counts per file
file_counts = {}
total_sequences = 0

# Loop over all files in folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tfa"):
        filepath = os.path.join(folder_path, filename)
        # Read sequences using Biopython
        seqs = list(SeqIO.parse(filepath, "fasta"))
        count = len(seqs)
        file_counts[filename] = count
        total_sequences += count

# Print counts per file
print("Sequences per file:")
for fname, cnt in sorted(file_counts.items()):
    if cnt==32:
        print(f"{fname}: {cnt}")

print("\nTotal sequences in folder:", total_sequences)