from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from ..configs import ModelConfigs
from ..constants import AMINO_ACID_INDICES, IMMUNOGENICITY_ONE_HOT


class VaxLSTM(nn.Module):
    def __init__(self, model_configs: ModelConfigs, device=None, **kwargs):
        super().__init__()
        self.max_seq_len = model_configs.hyperparameters.max_seq_len

        self.embedding_dim = model_configs.hyperparameters.embedding_dim
        self.hidden_dim = model_configs.hyperparameters.hidden_dim
        self.num_layers = model_configs.hyperparameters.num_layers

        self.vocab_size = len(AMINO_ACID_INDICES) + 1
        self.padding_idx = kwargs["padding_idx"]
        self.start_idx = kwargs["start_idx"]

        self.conditional = len(IMMUNOGENICITY_ONE_HOT)

        self.device = device

        self._build_model()

    def _build_model(self):
        self.immunogenicity_embedding = nn.Embedding(self.conditional, self.conditional)
        self.sequence_embeddings = nn.Embedding(
            self.vocab_size, self.hidden_dim, padding_idx=self.padding_idx
        )
        self.lstm = nn.LSTM(
            self.hidden_dim + self.conditional,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        input_sequences,
        hidden,
        input_immunogenicity,
        temperature=None,
        return_hidden=False,
    ):
        # batch_size * len * embedding_dim
        embedded_sequences = self.sequence_embeddings(input_sequences)
        if len(input_sequences.size()) == 1:
            # batch_size * 1 * embedding_dim
            embedded_sequences = embedded_sequences.unsqueeze(1)

        # batch_size * 1 * 3
        immunogenicity_embedded = self.immunogenicity_embedding(
            input_immunogenicity
        ).unsqueeze(1)
        # batch_size * len * 3
        immunogenicity_embedded = immunogenicity_embedded.repeat(
            1, embedded_sequences.size(1), 1
        )

        # batch_size * len * (embedding_dim + 3)
        embedded_sequences = torch.cat(
            (embedded_sequences, immunogenicity_embedded), dim=2
        )

        # batch_size * seq_len * hidden_dim
        encoded_sequences, hidden = self.lstm(embedded_sequences, hidden)
        # (batch_size * len) * hidden_dim
        encoded_sequences = encoded_sequences.contiguous().view(-1, self.hidden_dim)
        # (batch_size * seq_len) * vocab_size
        prediction = self.projection(encoded_sequences)

        if temperature is not None:
            prediction = prediction / temperature

        prediction = self.softmax(prediction)

        if return_hidden:
            return prediction, hidden
        else:
            return prediction

    def step(self, sequences, immunogenicity_scores):
        batch_size = immunogenicity_scores.size(0)

        input_sequences = sequences[:, :-1]
        output_sequences = sequences[:, 1:]

        hidden = self.init_hidden(batch_size)

        prediction = self.forward(input_sequences, hidden, immunogenicity_scores)
        loss = F.nll_loss(prediction, output_sequences.contiguous().view(-1))
        # loss = 0
        # for i in range(input_sequences.size(0)):
        #     output, hidden = self.forward(
        #         input_sequences[:, i], hidden, immunogenicity_scores
        #     )
        #     step_loss = F.nll_loss(output, output_sequences[:, i])
        #     loss += step_loss

        return {"loss": loss, "perplexity": torch.exp(loss)}

    def generate_sequences(
        self, num_sequences, immunogenicity_score, temperature=1.0, batch_size=None
    ):
        self.eval()
        # padding is all ones
        samples = torch.ones(num_sequences, self.max_seq_len).to(self.device)

        if batch_size is None:
            batch_size = num_sequences

        if batch_size > num_sequences:
            batch_size = num_sequences

        for idx in tqdm(range(0, num_sequences, batch_size)):
            hidden = self.generator.init_hidden(batch_size)
            input_sequences = torch.LongTensor([self.start_idx] * batch_size).unsqueeze(
                dim=1
            )
            input_sequences = input_sequences.to(self.device)

            immunogenicity_scores = torch.LongTensor(
                [immunogenicity_score] * batch_size
            ).to(self.device)

            for i in tqdm(range(self.max_seq_len)):
                out, hidden = self.generator.forward(
                    input_sequences,
                    hidden,
                    immunogenicity_scores,
                    temperature=0.8,
                    return_hidden=True,
                )

                new_input_sequences = torch.multinomial(torch.exp(out), num_samples=1)
                input_sequences = new_input_sequences.view(-1)
                samples[idx : idx + batch_size, i] = new_input_sequences.view(-1)

        return samples

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
        )
