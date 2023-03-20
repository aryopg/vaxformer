from typing import Dict

import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch import Tensor


def generate_square_subsequent_mask(size: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)


def write_sequences_to_fasta(sequences: Dict[str, int], filename):
    """


    Args:
        sequences (Dict[str, int]): _description_
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    records = []

    for sequence, count in sequences.items():
        record = SeqRecord(Seq(sequence), id=f"", description=f"{count}")
        records.append(record)

    fd = open(filename, "w")
    SeqIO.write(records, fd, format="fasta")
    fd.close()

    return records
