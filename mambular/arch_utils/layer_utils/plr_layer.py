import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math


class Periodic(nn.Module):
    """Periodic transformation with learned frequency coefficients."""

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, but got {sigma=}")

        self._sigma = sigma
        self.weight = Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x):
        x = 2 * math.pi * self.weight * x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


class SNLinear(nn.Module):
    """Separate linear layers for each feature embedding."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "_NLinear requires a 3D input (batch, features, embedding)."
            )
        if x.shape[-(self.weight.ndim - 1) :] != self.weight.shape[:-1]:
            raise ValueError("Input shape mismatch with weight dimensions.")

        x = x.transpose(0, 1) @ self.weight
        return x.transpose(0, 1) + self.bias


class PeriodicEmbeddings(nn.Module):
    """Embeddings for continuous features using Periodic + Linear (+ ReLU) transformations.

    Supports PL, PLR, and PLR(lite) embedding types.

    Shape:
        - Input: (*, n_features)
        - Output: (*, n_features, d_embedding)
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool = False,
    ):
        """
        Args:
            n_features (int): Number of features.
            d_embedding (int): Size of each feature embedding.
            n_frequencies (int): Number of frequencies per feature.
            frequency_init_scale (float): Initialization scale for frequency coefficients.
            activation (bool): If True, applies ReLU, making it PLR; otherwise, PL.
            lite (bool): If True, uses shared linear layer (PLR lite); otherwise, separate layers.
        """
        super().__init__()
        self.periodic = Periodic(n_features, n_frequencies, frequency_init_scale)

        # Choose linear transformation: shared or separate
        if lite:
            if not activation:
                raise ValueError("lite=True requires activation=True")
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.linear = SNLinear(n_features, 2 * n_frequencies, d_embedding)

        self.activation = nn.ReLU() if activation else None

    def forward(self, x):
        """Forward pass."""
        x = self.periodic(x)
        x = self.linear(x)
        return self.activation(x) if self.activation else x