"""Sparse autoencoder for residual stream decomposition.

    h = ReLU(W_enc @ x + b_enc)
    x_hat = W_dec @ h + b_dec
    loss = MSE(x, x_hat) + λ * L1(h)
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Vanilla SAE: linear encoder with ReLU, linear decoder, L1 penalty."""

    def __init__(self, input_dim: int, n_features: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        # Initialize decoder columns to unit norm (Bricken et al. 2023)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        x_hat = self.decoder(hidden)
        return x_hat, hidden

    @property
    def n_features(self):
        return self.encoder.out_features

    @property
    def input_dim(self):
        return self.encoder.in_features

    def normalize_decoder_(self):
        """Normalize decoder columns to unit norm (in-place)."""
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    @classmethod
    def load(cls, weights_path, config):
        """Load a trained SAE from weights file and config dict."""
        sae = cls(config["input_dim"], config["n_features"])
        sae.load_state_dict(torch.load(weights_path, weights_only=True))
        sae.eval()
        return sae
