import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.append(os.getcwd())

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Bio import SeqIO
from scipy.stats import ttest_ind
from tqdm.auto import tqdm

from src.constants import IMMUNOGENICITY_Q1, IMMUNOGENICITY_Q3

matplotlib.rcParams.update({"font.size": 15})
matplotlib.rcParams.update({"patch.force_edgecolor": False})

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
KMER = 9


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Compute hits and scores from peptides files"
    )
    parser.add_argument("--sequences_filepath", type=str, default="")
    parser.add_argument("--low_antigenicity_sequences_filepath", type=str, default="")
    parser.add_argument(
        "--intermediate_antigenicity_sequences_filepath", type=str, default=""
    )
    parser.add_argument("--high_antigenicity_sequences_filepath", type=str, default="")
    parser.add_argument("--peptides_dir", type=str, required=True)
    parser.add_argument("--peptide_file_prefix", type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate_peptides_netMHCpan(peptides_dir, file_prefix):
    netmhcpan_peptides = defaultdict(lambda: [None] * len(MHC_LIST))

    for mhc_idx, mhc_name in enumerate(MHC_LIST):
        mhc_name_filtered = mhc_name.replace(":", "").replace("HLA-", "")

        filename = os.path.join(peptides_dir, f"{file_prefix}_{mhc_name_filtered}.out")
        with open(filename, "r") as file:
            line_nr = 1
            lines = file.readlines()

            for line in lines:
                # The first 48 lines are headers
                if line_nr >= 49 and line[5:8] == "HLA":
                    # The rank column
                    peptide = line[22 : (22 + KMER)]
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
    for position in range(len(seq) - KMER):
        epitope = seq[position : (position + KMER)]
        entry = nMp_peptide_scores[epitope]
        for mhc_rank in entry:
            if mhc_rank < 2.0:
                score += 1
                epitopes.add(epitope)
    return score / len(MHC_LIST), epitopes


def t_test_pair(greater_dist, lower_dist):
    stat, p_value = ttest_ind(greater_dist, lower_dist, alternative="greater")
    print(f"T-test statistics: {stat}")
    print(f"T-test p-value: {p_value}")


def main():
    args = argument_parser()

    seq_to_score = {}
    nMp_seq_hits = {}
    generated_seqs_new = {}

    if (
        len(args.low_antigenicity_sequences_filepath) > 0
        and len(args.intermediate_antigenicity_sequences_filepath) > 0
        and len(args.high_antigenicity_sequences_filepath) > 0
    ):
        print("Computing 3 different netMHCpan score distributions")
        generated_sequences_paths = [
            args.low_antigenicity_sequences_filepath,
            args.intermediate_antigenicity_sequences_filepath,
            args.high_antigenicity_sequences_filepath,
        ]
        for immunogenicity_score, generated_sequences_path in enumerate(
            generated_sequences_paths
        ):
            generated_seqs_new[immunogenicity_score] = {}
            for record in SeqIO.parse(generated_sequences_path, "fasta"):
                generated_seqs_new[immunogenicity_score].update(
                    {str(record.seq): int(record.id)}
                )
                seq_to_score.update({str(record.seq): int(record.id)})

    if len(args.sequences_filepath) > 0:
        print("Computing netMHCpan scores for 1 sequence file")
        hits_save_path = args.sequences_filepath.replace(".fasta", "hits.npy")
        scores_save_path = args.sequences_filepath.replace(".fasta", "scores.npy")

        if args.sequences_filepath.endswith("fasta"):
            for record in SeqIO.parse(args.sequences_filepath, "fasta"):
                seq_to_score.update({str(record.seq): int(record.id)})
        elif args.sequences_filepath.endswith(".txt"):
            with open(f"{args.sequences_filepath}") as sequences_file:
                sequences = sequences_file.readlines()
                for seq in sequences:
                    seq_to_score[seq.replace("\n", "")] = 1

    print("Evaluating peptides for each alelle")
    nMp_peptide_scores = evaluate_peptides_netMHCpan(
        args.peptides_dir, args.peptide_file_prefix
    )

    for seq in tqdm(list(seq_to_score.keys())):
        nMp_seq_hits[seq], epitopes = score_sequence_nMp_with_dashes(
            seq, nMp_peptide_scores
        )

    print(
        "Hits calculated, 1st and 3rd quantiles: ",
        np.quantile(list(nMp_seq_hits.values()), [0.25, 0.75]),
    )
    if len(args.sequences_filepath) > 0:
        print(f"Mean: {statistics.mean(list(nMp_seq_hits.values()))}")
        print(f"std: {statistics.stdev(list(nMp_seq_hits.values()))}")
        np.save(hits_save_path, nMp_seq_hits, allow_pickle=True)

    nMp_scores = {}
    for sequence_key, nMp_score in nMp_seq_hits.items():
        if nMp_score <= IMMUNOGENICITY_Q1:
            nMp_scores[sequence_key] = 0
        elif nMp_score > IMMUNOGENICITY_Q3:
            nMp_scores[sequence_key] = 2
        else:
            nMp_scores[sequence_key] = 1

    if len(args.sequences_filepath) > 0:
        np.save(scores_save_path, nMp_scores, allow_pickle=True)

    if generated_seqs_new:
        low_hits = []
        medium_hits = []
        high_hits = []
        for immunogenicity_score, generated_sequences in generated_seqs_new.items():
            sequence_hits = [nMp_seq_hits[seq] for seq in generated_sequences.keys()]

            if immunogenicity_score == 0:
                low_hits += sequence_hits
            if immunogenicity_score == 1:
                medium_hits += sequence_hits
            if immunogenicity_score == 2:
                high_hits += sequence_hits

            print(
                f"Statistics of sequences with immunogenicity of {immunogenicity_score}"
            )
            print(f"Mean: {statistics.mean(sequence_hits)}")
            print(f"Stddev: {statistics.stdev(sequence_hits)}")

            sns.histplot(sequence_hits, stat="density", kde=True, bins=15)

        print("Medium immunogenicity vs Low immunogenicity")
        t_test_pair(medium_hits, low_hits)
        print("High immunogenicity vs Medium immunogenicity")
        t_test_pair(high_hits, medium_hits)
        print("High immunogenicity vs Low immunogenicity")
        t_test_pair(high_hits, low_hits)

        plt.legend(labels=["low", "medium", "high"])
        # 50.66667 nMp_seq_scores_p75: 51.16667
        plt.plot(
            [IMMUNOGENICITY_Q1, IMMUNOGENICITY_Q1], [0, 1.0], c="black", linewidth=0.5
        )
        plt.plot(
            [IMMUNOGENICITY_Q3, IMMUNOGENICITY_Q3], [0, 1.0], c="black", linewidth=0.5
        )
        plt.xlabel("AS distribution generated samples")
        plt.savefig("AS distribution generated samples")


if __name__ == "__main__":
    main()
