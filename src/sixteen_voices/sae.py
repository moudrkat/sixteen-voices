"""Sparse autoencoder for residual stream decomposition.

    h = ReLU(W_enc @ x + b_enc)
    x_hat = W_dec @ h + b_dec
    loss = MSE(x, x_hat) + λ * L1(h)
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """SAE: linear encoder with ReLU or TopK activation, linear decoder.

    activation="relu"  — standard ReLU, sparsity via L1 penalty in loss
    activation="topk"  — keep only top-k activations per token (Gao et al. 2024),
                         no L1 needed
    """

    def __init__(self, input_dim: int, n_features: int,
                 activation: str = "relu", k: int = 16):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)
        self.activation = activation
        self.k = k

        # Initialize decoder columns to unit norm (Bricken et al. 2023)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def forward(self, x):
        pre_act = self.encoder(x)
        if self.activation == "topk":
            topk = torch.topk(pre_act, k=self.k, dim=-1)
            hidden = torch.zeros_like(pre_act)
            hidden.scatter_(-1, topk.indices, torch.relu(topk.values))
        else:
            hidden = torch.relu(pre_act)
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
        sae = cls(
            config["input_dim"], config["n_features"],
            activation=config.get("activation", "relu"),
            k=config.get("k", 16),
        )
        sae.load_state_dict(torch.load(weights_path, weights_only=True))
        sae.eval()
        return sae
