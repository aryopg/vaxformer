import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="Generate peptides from sequences")
    parser.add_argument("--sequences_filepath", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()

    sequences = open(f"{args.sequences_filepath}").readlines()
    peptides = open(f"{args.sequences_filepath}.pep", "w")
    for prot in sequences:
        prot = prot.replace("\n", "")
        for i in range(0, len(prot) - 9):
            pep = prot[i : i + 9]
            peptides.writelines([pep, "\n"])


if __name__ == "__main__":
    main()
