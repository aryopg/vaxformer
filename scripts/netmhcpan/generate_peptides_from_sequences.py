import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Generate peptides from sequences")
    parser.add_argument("--sequences_filepath", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()

    with open(f"{args.sequences_filepath}") as sequences_file:
        if args.sequences_filepath.endswith(".txt"):
            sequences = sequences_file.readlines()
            sequences = [seq.replace("\n", "") for seq in sequences]
        elif args.sequences_filepath.endswith(".fasta"):
            fasta_sequences = sequences_file.readlines()
            sequences = []
            sequence = ""
            for seq_ in fasta_sequences:
                if seq_.startswith(">"):
                    if sequence:
                        sequences += [sequence]
                        sequence = ""
                    else:
                        continue
                else:
                    sequence = sequence + seq_.replace("\n", "")
            if sequence:
                sequences += [sequence]
        else:
            raise ValueError(
                f"Unknown file extension of {args.sequences_filepath}. Script only handles txt or fasta file"
            )

    with open(f"{args.sequences_filepath}.pep", "w") as peptides_file:
        for prot in sequences:
            prot = prot.replace("\n", "")
            for i in range(0, len(prot) - 9):
                pep = prot[i : i + 9]
                peptides_file.writelines([pep, "\n"])


if __name__ == "__main__":
    main()
