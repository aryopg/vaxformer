"""
Code adapted from:
https://github.com/hcgasser/SpikeVAE/blob/main/VAEmodel/SpikeOracle/models/VAE/fc.py
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..configs import ModelConfigs
from ..constants import AMINO_ACID_INDICES, IMMUNOGENICITY_ONE_HOT


class SpikeFCVAE(nn.Module):
    """Standard VAE with Gaussian Prior and approx posterior."""

    def __init__(self, model_configs: ModelConfigs, device=None):
        super().__init__()

        self.amino_acid_dim = len(AMINO_ACID_INDICES)
        self.sequence_len = model_configs.hyperparameters.max_seq_len
        self.hidden_dims = model_configs.hyperparameters.hidden_dims
        self.latent_dim = model_configs.hyperparameters.latent_dim

        self.conditional = len(IMMUNOGENICITY_ONE_HOT)
        self.dropout = model_configs.hyperparameters.dropout
        self.kl_target = model_configs.hyperparameters.kl_target

        self.beta = 0
        self.P = 0
        self.I = 0

        self.z = None

        self.device = device

        self.build_model()

    def build_model(self):
        enc_input_dim = self.amino_acid_dim * self.sequence_len + self.conditional
        self.encoder = FCNetwork(
            input_dim=enc_input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )

        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        dec_input_dim = self.latent_dim + self.conditional
        dec_output_dim = self.amino_acid_dim * self.sequence_len
        self.decoder = FCNetwork(
            input_dim=dec_input_dim,
            hidden_dims=self.hidden_dims[::-1],
            output_dim=dec_output_dim,
            dropout=self.dropout,
        )

    def encode(self, input_seq):
        encoded_seq = self.encoder(input_seq)
        encoded_seq = F.relu(encoded_seq)

        mu = self.fc_mu(encoded_seq)
        log_var = self.fc_var(encoded_seq)

        return mu, log_var

    def forward(self, x, y, sample=True):
        self.z, x_hat_logit, _, _ = self._run_step(x, y, sample)
        x_hat_logit = x_hat_logit.reshape(-1, self.sequence_len, self.amino_acid_dim)
        x_hat = F.softmax(x_hat_logit, -1)
        return x_hat

    def reparametrize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def _run_step(self, input_seq, label, sample=True):
        input_seq = torch.flatten(input_seq, start_dim=-2)
        if self.conditional != 0:
            input_seq = torch.cat([input_seq, label], dim=1)

        mu, log_var = self.encode(input_seq)
        if sample:
            p, q, z = self.reparametrize(mu, log_var)
        else:
            p, q, z = None, None, mu

        if self.conditional == 0:
            return z, self.decoder(z), p, q
        else:
            d = torch.cat([z, label], dim=1)
            return z, self.decoder(d), p, q

    def step(self, input_sequence, input_immunogenicity_score):
        _, reconstructed_sequence, p, q = self._run_step(
            input_sequence, input_immunogenicity_score
        )

        return self.loss_function(reconstructed_sequence, input_sequence, p, q)

    def loss_function(
        self, reconstructed_sequence, input_sequence, p, q
    ) -> Dict[str, float]:
        reconstruction_loss = F.cross_entropy(
            reconstructed_sequence.view(-1, self.amino_acid_dim),
            torch.max(input_sequence, dim=-1).indices.contiguous().view(-1),
        )

        kl = torch.distributions.kl_divergence(q, p)
        kl = kl.mean()

        kl_coeff = self.calc_beta(
            float(kl.detach()), self.kl_target, 1e-3, 5e-4, 1e-4, 1
        )

        loss = kl * kl_coeff + reconstruction_loss

        return {
            "combined_loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl,
        }

    def calc_beta(self, actual_kl, target_kl, Kp, Ki, beta_min, beta_max):
        error = target_kl - actual_kl
        self.P = Kp / (1 + np.exp(error))

        if beta_min < self.beta and self.beta < beta_max:
            self.I = self.I - Ki * error

        self.beta = min(max(self.P + self.I + beta_min, beta_min), beta_max)
        return self.beta

    def get_latents_from_sequences(self, sequences, immunogenicity_scores):
        mus = []
        log_vars = []
        latents = []

        self.eval()

        for sequence, immunogenicity_score in zip(sequences, immunogenicity_scores):
            x = sequence.unsqueeze(dim=0)
            x = torch.flatten(x, start_dim=-2)
            x = torch.cat([x, immunogenicity_score.unsqueeze(dim=0)], dim=1)

            encoded = self.encoder(x)
            mu = self.fc_mu(encoded)
            log_var = self.fc_var(encoded)

            p, q, z = self.reparametrize(mu, log_var)

            mus.append(mu.detach())
            log_vars.append(log_var.detach())
            latents.append(z.detach())

        return mus, log_vars, latents

    def get_sequences_from_latents(
        self, latents, condition, tokenizer
    ) -> Dict[str, int]:
        self.eval()
        seqs = defaultdict(lambda: 0)
        for z in latents:
            h = torch.tensor([[0, 0, 0]])
            h[0][condition] = 1
            z = torch.cat([z.unsqueeze(dim=0), h.to(self.device)], dim=1)

            seq = self.decoder(z)
            seq = tokenizer.decode(seq.reshape(1, self.sequence_len, -1))[0]
            seqs[seq] += 1

        return seqs


class FCNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        input_dim_ = input_dim
        self.network = nn.Sequential()
        for block_idx, hidden_dim in enumerate(hidden_dims):
            if block_idx == len(hidden_dims) - 1:
                dropout = 0

            self.network.add_module(
                f"linear_block_{block_idx}",
                self.get_fc_block(
                    input_dim_,
                    hidden_dim,
                    dropout,
                    batch_norm=True,
                    activation=nn.LeakyReLU(0.1),
                ),
            )
            input_dim_ = hidden_dim
        if self.output_dim is not None:
            self.network.add_module(
                f"linear_block_{block_idx + 1}",
                self.get_fc_block(
                    input_dim_,
                    self.output_dim,
                    0,
                    batch_norm=False,
                    activation=None,
                ),
            )

    def forward(self, x):
        return self.network(x)

    def get_fc_block(
        self,
        input_dim,
        output_dim,
        dropout=0,
        batch_norm=True,
        activation=nn.LeakyReLU(0.1),
    ):
        block = nn.Sequential()
        block.add_module("linear", nn.Linear(input_dim, output_dim))
        if batch_norm:
            block.add_module("batch_norm", nn.BatchNorm1d(output_dim))
        if activation is not None:
            block.add_module("activation", activation)
        if dropout > 0:
            block.add_module(str(3), nn.Dropout(dropout))

        return block
