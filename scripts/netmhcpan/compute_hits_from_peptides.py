import argparse
import os
import sys
from collections import defaultdict

sys.path.append(os.getcwd())

import numpy as np
from Bio import SeqIO
from tqdm.auto import tqdm

from src.constants import IMMUNOGENICITY_Q1, IMMUNOGENICITY_Q3

MHC_LIST = [
    "HLA-A01:01",
    "HLA-A02:01",
    "HLA-A03:01",
    "HLA-A24:02",
    "HLA-A26:01",
    "HLA-B07:02",
    "HLA-B08:01",
    "HLA-B27:05",
    "HLA-B39:01",
    "HLA-B40:01",
    "HLA-B58:01",
    "HLA-B15:01",
]


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Compute hits and scores from peptides files"
    )
    parser.add_argument("--sequences_filepath", type=str, required=True)
    parser.add_argument("--peptides_dir", type=str, required=True)
    parser.add_argument("--peptide_file_prefix", type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate_peptides_netMHCpan(peptides_dir, file_prefix):
    netmhcpan_peptides = defaultdict(lambda: [None] * len(MHC_LIST))

    for mhc_idx, mhc_name in enumerate(MHC_LIST):
        mhc_name_filtered = mhc_name.replace(":", "").replace("HLA-", "")

        filename = os.path.join(
            peptides_dir, f"{file_prefix}_{mhc_name_filtered}.pep.out"
        )
        with open(filename, "r") as file:
            line_nr = 1
            lines = file.readlines()

            for line in lines:
                # The first 48 lines are headers
                if line_nr >= 49 and line[5:8] == "HLA":
                    # The rank column
                    peptide = line[22 : (22 + 9)]
                    # The peptide column
                    rank_el = float(line[96 : (96 + 8)])

                    netmhcpan_peptides[peptide][mhc_idx] = rank_el
                line_nr += 1

    return dict(netmhcpan_peptides)


def score_sequence_nMp_with_dashes(seq, nMp_peptide_scores):
    # Replace sequences' unknown characters
    seq = seq.replace("-", "X")
    seq = seq.replace(">", "X")
    score = 0

    epitopes = set()
    for position in range(len(seq) - 9):
        epitope = seq[position : (position + 9)]
        entry = nMp_peptide_scores[epitope]
        for mhc_rank in entry:
            if mhc_rank < 2.0:
                score += 1
                epitopes.add(epitope)
    return score / len(MHC_LIST), epitopes


def main():
    args = argument_parser()

    hits_save_path = args.sequences_filepath.replace(".fasta", "hits.npy")
    scores_save_path = args.sequences_filepath.replace(".fasta", "scores.npy")

    seq_to_score = {}
    nMp_seq_hits = {}

    for record in SeqIO.parse(args.sequences_filepath, "fasta"):
        seq_to_score.update({str(record.seq): int(record.id)})

    print("Evaluating peptides for each alelle")
    nMp_peptide_scores = evaluate_peptides_netMHCpan(
        args.peptides_dir, args.peptide_file_prefix
    )

    for seq in tqdm(list(seq_to_score.keys())[:-1]):
        nMp_seq_hits[seq], epitopes = score_sequence_nMp_with_dashes(
            seq, nMp_peptide_scores
        )

    print(
        "Hits calculated, 1st and 3rd quantiles: ",
        np.quantile(list(nMp_seq_hits.values()), [0.25, 0.75]),
    )
    np.save(hits_save_path, nMp_seq_hits, allow_pickle=True)

    nMp_scores = {}
    for sequence_key, nMp_score in nMp_seq_hits.items():
        if nMp_score <= IMMUNOGENICITY_Q1:
            nMp_scores[sequence_key] = 0
        elif IMMUNOGENICITY_Q1 <= nMp_score <= IMMUNOGENICITY_Q3:
            nMp_scores[sequence_key] = 1
        elif nMp_score > IMMUNOGENICITY_Q3:
            nMp_scores[sequence_key] = 2

    np.save(scores_save_path, nMp_scores, allow_pickle=True)


if __name__ == "__main__":
    main()
