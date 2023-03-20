import torch
from torch import nn
from torch.nn import TransformerDecoder
from torch.nn import functional as F
from tqdm import tqdm

from ..configs import ModelConfigs
from ..constants import AMINO_ACID_INDICES, IMMUNOGENICITY_SCORES
from ..utils.model_utils import generate_square_subsequent_mask


class Vaxformer(nn.Module):
    def __init__(self, model_configs: ModelConfigs, device=None, **kwargs):
        super().__init__()
        self.max_seq_len = model_configs.hyperparameters.max_seq_len
        self.embedding_dim = model_configs.hyperparameters.embedding_dim
        self.hidden_dim = model_configs.hyperparameters.hidden_dim
        self.nhead = model_configs.hyperparameters.nhead
        self.num_layers = model_configs.hyperparameters.num_layers
        self.dropout = model_configs.hyperparameters.dropout

        # include start token in the vocab size
        self.vocab_size = len(AMINO_ACID_INDICES) + 1
        self.immunogenicity_size = len(IMMUNOGENICITY_SCORES)

        self.padding_idx = kwargs["padding_idx"]
        self.start_idx = kwargs["start_idx"]

        self.device = device

        self._build_model()

    def _build_model(self):
        self.amino_acid_embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.immunogenicity_embedding = nn.Embedding(
            self.immunogenicity_size, self.hidden_dim
        )

        self.positional_embedding = PositionalEncoder(
            self.hidden_dim, self.dropout, self.max_seq_len, self.device
        )

        transformer_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=self.nhead, batch_first=True
        )
        layer_norm = nn.LayerNorm(self.hidden_dim)
        self.transformer = TransformerDecoder(
            transformer_layer, num_layers=self.num_layers, norm=layer_norm
        )

    def forward(self, input_sequence, input_immunogenicity, mask=None):
        amino_acid_embedded = self.amino_acid_embedding(input_sequence)
        amino_acid_positioned = self.positional_embedding(amino_acid_embedded)

        immunogenicity_embedded = self.immunogenicity_embedding(
            input_immunogenicity
        ).unsqueeze(1)

        amino_acid_decoded = self.transformer(
            amino_acid_positioned, memory=immunogenicity_embedded, tgt_mask=mask
        )
        out = amino_acid_decoded @ self.amino_acid_embedding.weight.T

        return out

    def step(self, input_sequence, input_immunogenicity_score):
        input = input_sequence[:, :-1]
        target = input_sequence[:, 1:].contiguous().view(-1)
        mask = generate_square_subsequent_mask(input.size(1)).to(self.device)

        generated_sequences = self.forward(input, input_immunogenicity_score, mask)
        generated_sequences = generated_sequences.view(-1, self.vocab_size)

        loss = F.cross_entropy(generated_sequences, target)
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
            input_sequences = torch.LongTensor([self.start_idx] * batch_size).unsqueeze(
                dim=1
            )
            input_sequences = input_sequences.to(self.device)

            immunogenicity_scores = torch.LongTensor(
                [immunogenicity_score] * batch_size
            ).to(self.device)

            for i in tqdm(range(self.max_seq_len)):
                out = self.forward(input_sequences, immunogenicity_scores)
                out = out[:, -1, :] / temperature
                out = F.softmax(out, dim=-1)

                new_input_sequences = torch.multinomial(out, num_samples=1)
                samples[idx : idx + batch_size, i] = new_input_sequences.squeeze()
                input_sequences = torch.cat(
                    (input_sequences, new_input_sequences), dim=1
                )

        return samples


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout, max_seq_len=1300, device=None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position = torch.arange(max_seq_len).unsqueeze(1)
        self.positional_encoding = torch.zeros(1, max_seq_len, hidden_dim)

        _2i = torch.arange(0, hidden_dim, step=2).float()
        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(
            self.position / (10000 ** (_2i / hidden_dim))
        )
        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(
            self.position / (10000 ** (_2i / hidden_dim))
        )

        self.device = device

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        position_encoding = self.positional_encoding[:batch_size, :seq_len, :].to(
            self.device
        )

        x += position_encoding

        return self.dropout(x)
