from Bio import SeqIO
from Bio import pairwise2
from Bio.Align import substitution_matrices
import pandas as pd
import itertools
import math
from collections import Counter
from Bio.SeqUtils import molecular_weight


# ---------------------------
# Helper Functions
# ---------------------------

def fast_percent_identity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    return matches / min_len * 100

def count_gaps(seq):
    return seq.count('-')

def needleman_wunsch_score(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    return alignments[0].score

# Load BLOSUM62
blosum62 = substitution_matrices.load("BLOSUM62")

def blosum62_score(seq1, seq2):
    score = 0
    min_len = min(len(seq1), len(seq2))
    for a, b in zip(seq1[:min_len], seq2[:min_len]):
        if (a, b) in blosum62:
            score += blosum62[(a, b)]
        elif (b, a) in blosum62:
            score += blosum62[(b, a)]
    return score

def kmer_jaccard(seq1, seq2, k=3):
    kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
    kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return intersection / union if union != 0 else 0

def smith_waterman_score(seq1, seq2):
    alignments = pairwise2.align.localxx(seq1, seq2, one_alignment_only=True)
    return alignments[0].score

def global_score_with_gaps(seq1, seq2):
    alignments = pairwise2.align.globalms(seq1, seq2, 1, -1, -2, -0.5, one_alignment_only=True)
    return alignments[0].score

def alignment_coverage(seq1, seq2):
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    aligned_seq1, aligned_seq2 = alignment.seqA, alignment.seqB
    aligned_positions = sum(
        1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' and b != '-'
    )
    return aligned_positions / min(len(seq1), len(seq2))

def shannon_entropy(seq):
    counts = Counter(seq)
    total = len(seq)
    entropy = 0
    for aa in counts.values():
        p = aa / total
        entropy -= p * math.log2(p)
    return entropy

def linguistic_complexity(seq, k=3):
    observed = len(set(seq[i:i+k] for i in range(len(seq) - k + 1)))
    possible = min(len(seq) - k + 1, 20**k)
    return observed / possible if possible > 0 else 0

def low_complexity_score(seq):
    counts = Counter(seq)
    most_common = counts.most_common(1)[0][1]
    return most_common / len(seq)

def repeat_ratio(seq, k=2):
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    total = len(kmers)
    unique = len(set(kmers))
    return 1 - (unique / total) if total > 0 else 0

hydro_scale = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def avg_hydrophobicity(seq):
    values = [hydro_scale.get(aa, 0) for aa in seq]
    return sum(values) / len(values)

positive = set("KRH")
negative = set("DE")

def net_charge(seq):
    pos = sum(1 for aa in seq if aa in positive)
    neg = sum(1 for aa in seq if aa in negative)
    return (pos - neg) / len(seq)

helix_set = set("AELM")
sheet_set = set("VIYFW")
coil_set  = set("GPSD")

def secondary_structure_proxy(seq):
    h = sum(1 for aa in seq if aa in helix_set) / len(seq)
    e = sum(1 for aa in seq if aa in sheet_set) / len(seq)
    c = sum(1 for aa in seq if aa in coil_set) / len(seq)
    return h, e, c

def seq_molecular_weight(seq):
    return molecular_weight(seq, seq_type="protein")

def indel_stats(seq1, seq2):
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    a1, a2 = aln.seqA, aln.seqB
    
    indel_runs = 0
    in_indel = False
    total_indels = 0

    for x, y in zip(a1, a2):
        if x == '-' or y == '-':
            total_indels += 1
            if not in_indel:
                indel_runs += 1
                in_indel = True
        else:
            in_indel = False

    indel_fraction = total_indels / len(a1)
    return indel_runs, total_indels, indel_fraction

def substitution_entropy(seq1, seq2):
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    pairs = [(a, b) for a, b in zip(aln.seqA, aln.seqB) if a != '-' and b != '-']
    
    if not pairs:
        return 0
    
    counts = Counter(pairs)
    total = sum(counts.values())
    
    entropy = 0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
        
    return entropy

aa_classes = {
    'hydrophobic': set("AVILMFWY"),
    'polar': set("STNQ"),
    'positive': set("KRH"),
    'negative': set("DE"),
    'special': set("CGP")
}

def aa_class(aa):
    for k, v in aa_classes.items():
        if aa in v:
            return k
    return "other"

def chemical_substitution_stats(seq1, seq2):
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    
    conservative = 0
    radical = 0
    total = 0

    for a, b in zip(aln.seqA, aln.seqB):
        if a == '-' or b == '-':
            continue
        total += 1
        if aa_class(a) == aa_class(b):
            conservative += 1
        else:
            radical += 1

    if total == 0:
        return 0, 0

    return conservative / total, radical / total



# ---------------------------
# Main processing
# ---------------------------

tfa_file = "../RV100/BBA0171.tfa"

sequences = list(SeqIO.parse(tfa_file, "fasta"))
n = len(sequences)
print(f"Loaded {n} sequences")
results = []

for seq1, seq2 in itertools.combinations(sequences, 2):
    seq1_str = str(seq1.seq)
    seq2_str = str(seq2.seq)
    
    pid = fast_percent_identity(seq1_str, seq2_str)
    len_diff = abs(len(seq1_str) - len(seq2_str))
    gaps_seq1 = count_gaps(seq1_str)
    gaps_seq2 = count_gaps(seq2_str)

    nw_score = needleman_wunsch_score(seq1_str, seq2_str)
    blosum_score = blosum62_score(seq1_str, seq2_str)
    kmer_sim = kmer_jaccard(seq1_str, seq2_str, k=3)
    sw_score = smith_waterman_score(seq1_str, seq2_str)
    global_gap_score = global_score_with_gaps(seq1_str, seq2_str)
    coverage = alignment_coverage(seq1_str, seq2_str)

    entropy1 = shannon_entropy(seq1_str)
    entropy2 = shannon_entropy(seq2_str)
    entropy_diff = abs(entropy1 - entropy2)

    complexity1 = linguistic_complexity(seq1_str)
    complexity2 = linguistic_complexity(seq2_str)
    complexity_diff = abs(complexity1 - complexity2)

    lowcomp1 = low_complexity_score(seq1_str)
    lowcomp2 = low_complexity_score(seq2_str)
    lowcomp_diff = abs(lowcomp1 - lowcomp2)

    repeat1 = repeat_ratio(seq1_str)
    repeat2 = repeat_ratio(seq2_str)
    repeat_diff = abs(repeat1 - repeat2)

    # Hydrophobicity
    hydro1 = avg_hydrophobicity(seq1_str)
    hydro2 = avg_hydrophobicity(seq2_str)
    hydro_diff = abs(hydro1 - hydro2)

    # Charge
    charge1 = net_charge(seq1_str)
    charge2 = net_charge(seq2_str)
    charge_diff = abs(charge1 - charge2)

    # Secondary structure proxy
    h1, e1, c1 = secondary_structure_proxy(seq1_str)
    h2, e2, c2 = secondary_structure_proxy(seq2_str)

    helix_diff = abs(h1 - h2)
    sheet_diff = abs(e1 - e2)
    coil_diff  = abs(c1 - c2)

    # Molecular weight
    mw1 = seq_molecular_weight(seq1_str)
    mw2 = seq_molecular_weight(seq2_str)
    mw_diff = abs(mw1 - mw2)

    # Indel stats
    indel_runs, total_indels, indel_fraction = indel_stats(seq1_str, seq2_str)

    # Substitution entropy
    sub_entropy = substitution_entropy(seq1_str, seq2_str)

    # Chemical substitution style
    conservative_frac, radical_frac = chemical_substitution_stats(seq1_str, seq2_str)


    pair_id = f"{seq1.id}_{seq2.id}"
    
    results.append({
        "pair_id": pair_id,
        "seq1": seq1_str,
        "seq2": seq2_str,
        "percent_identity": pid,
        "length_diff": len_diff,
        "num_gaps_seq1": gaps_seq1,
        "num_gaps_seq2": gaps_seq2,
        "nw_score": nw_score,
        "blosum62_sum": blosum_score,
        "kmer_jaccard_3": kmer_sim,"sw_score": sw_score,
        "global_gap_score": global_gap_score,
        "alignment_coverage": coverage,
        "entropy_diff": entropy_diff,
        "complexity_diff": complexity_diff,
        "low_complexity_diff": lowcomp_diff,
        "repeat_ratio_diff": repeat_diff,
        "hydrophobicity_diff": hydro_diff,
        "charge_diff": charge_diff,
        "helix_propensity_diff": helix_diff,
        "sheet_propensity_diff": sheet_diff,
        "coil_propensity_diff": coil_diff,
        "molecular_weight_diff": mw_diff,
        "indel_runs": indel_runs,
        "total_indels": total_indels,
        "indel_fraction": indel_fraction,
        "substitution_entropy": sub_entropy,
        "conservative_sub_frac": conservative_frac,
        "radical_sub_frac": radical_frac
    })

df = pd.DataFrame(results)
df.to_csv("BBA0171_pairwise_features.csv", index=False)
print("CSV created with", len(df), "pairs")
