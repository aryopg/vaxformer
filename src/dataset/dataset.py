import numpy as np
import torch

from ..configs import DatasetConfigs
from ..constants import (
    IMMUNOGENICITY_ONE_HOT,
    START_TOKEN,
)
from .tokenizer import Tokenizer


def load_sequences_file(filename):
    with open(filename, "r") as file:
        viral_seqs = file.readlines()
    return [viral_seq.replace("\n", "") for viral_seq in viral_seqs]


def load_immunogenicity_scores(
    immunogenicity_scores_filepath, sequences, one_hot: bool
):
    if immunogenicity_scores_filepath:
        immunogenicity_scores = np.load(immunogenicity_scores_filepath)
        if one_hot:
            immunogenicity_scores = np.array(
                [IMMUNOGENICITY_ONE_HOT[score] for score in immunogenicity_scores]
            )
    else:
        if one_hot:
            immunogenicity_scores = np.array(
                [IMMUNOGENICITY_ONE_HOT[1] for _ in range(len(sequences))]
            )
        else:
            immunogenicity_scores = np.array([1 for _ in range(len(sequences))])

    return torch.tensor(immunogenicity_scores)


class SequenceDataset:
    def __init__(
        self,
        dataset_configs: DatasetConfigs,
        split: str,
        max_seq_len: int,
        sequence_one_hot: bool = True,
        label_one_hot: bool = True,
        prepend_start_token: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.sequence_one_hot = sequence_one_hot
        self.label_one_hot = label_one_hot
        self.prepend_start_token = prepend_start_token
        self.tokenizer = Tokenizer(
            self.max_seq_len, self.sequence_one_hot, prepend_start_token
        )

        if split == "train":
            sequences_filepath = dataset_configs.train.sequences_path
            immunogenicity_scores_filepath = (
                dataset_configs.train.immunogenicity_scores_path
            )
        elif split == "val":
            sequences_filepath = dataset_configs.val.sequences_path
            immunogenicity_scores_filepath = (
                dataset_configs.val.immunogenicity_scores_path
            )
        elif split == "test":
            sequences_filepath = dataset_configs.test.sequences_path
            immunogenicity_scores_filepath = (
                dataset_configs.test.immunogenicity_scores_path
            )

        self.sequences = load_sequences_file(sequences_filepath)
        self.immunogenicity_scores = load_immunogenicity_scores(
            immunogenicity_scores_filepath, self.sequences, self.label_one_hot
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.prepend_start_token:
            sequence = START_TOKEN + self.sequences[idx]
        else:
            sequence = self.sequences[idx]

        return (
            self.tokenizer.encode(sequence),
            self.immunogenicity_scores[idx],
        )
