import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_utils.batch_ensemble_layer import (
    LinearBatchEnsembleLayer,
    MultiHeadAttentionBatchEnsemble,
)


def reglu(x):
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class ReGLU(nn.Module):
    def forward(self, x):
        return reglu(x)


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        assert x.size(-1) % 2 == 0, "Input dimension must be even"
        split_dim = x.size(-1) // 2
        return x[..., :split_dim] * torch.sigmoid(x[..., split_dim:])


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, config):
        super().__init__(
            d_model=getattr(config, "d_model", 128),
            nhead=getattr(config, "n_heads", 8),
            dim_feedforward=getattr(config, "transformer_dim_feedforward", 2048),
            dropout=getattr(config, "attn_dropout", 0.1),
            activation=getattr(config, "transformer_activation", F.relu),
            layer_norm_eps=getattr(config, "layer_norm_eps", 1e-5),
            norm_first=getattr(config, "norm_first", False),
        )
        self.bias = getattr(config, "bias", True)
        self.custom_activation = getattr(config, "transformer_activation", F.relu)

        # Additional setup based on the activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(
            self.custom_activation, (ReGLU, GLU)
        ):
            self.linear1 = nn.Linear(
                self.linear1.in_features,
                self.linear1.out_features * 2,
                bias=self.bias,
            )
            self.linear2 = nn.Linear(
                self.linear2.in_features,
                self.linear2.out_features,
                bias=self.bias,
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Use the provided activation function
        if self.custom_activation in [ReGLU, GLU] or isinstance(
            self.custom_activation, (ReGLU, GLU)
        ):
            src2 = self.linear2(self.custom_activation(self.linear1(src)))
        else:
            src2 = self.linear2(self.custom_activation(self.linear1(src)))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Literal
import copy


class BatchEnsembleTransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Batch Ensembling.

    This class implements a single layer of the Transformer encoder with batch ensembling applied to the
    multi-head attention and feedforward network as desired.

    Parameters
    ----------
    embed_dim : int
        The dimension of the embedding.
    num_heads : int
        Number of attention heads.
    ensemble_size : int
        Number of ensemble members.
    dim_feedforward : int, optional
        Dimension of the feedforward network model. Default is 2048.
    dropout : float, optional
        Dropout value. Default is 0.1.
    activation : {'relu', 'gelu'}, optional
        Activation function of the intermediate layer. Default is 'relu'.
    scaling_init : {'ones', 'random-signs', 'normal'}, optional
        Initialization method for the scaling factors in batch ensembling. Default is 'ones'.
    batch_ensemble_projections : list of str, optional
        List of projections to which batch ensembling should be applied in the attention layer.
        Default is ['query'].
    batch_ensemble_ffn : bool, optional
        Whether to apply batch ensembling to the feedforward network. Default is False.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ensemble_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
        batch_ensemble_projections: List[str] = ["query"],
        batch_ensemble_ffn: bool = False,
    ):
        super(BatchEnsembleTransformerEncoderLayer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ensemble_size = ensemble_size
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.batch_ensemble_ffn = batch_ensemble_ffn

        # Multi-head attention with batch ensembling
        self.self_attn = MultiHeadAttentionBatchEnsemble(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ensemble_size=ensemble_size,
            scaling_init=scaling_init,
            batch_ensemble_projections=batch_ensemble_projections,
        )

        # Feedforward network
        if batch_ensemble_ffn:
            # Apply batch ensembling to the feedforward network
            self.linear1 = BatchEnsembleLinear(
                embed_dim, dim_feedforward, ensemble_size, scaling_init
            )
            self.linear2 = BatchEnsembleLinear(
                dim_feedforward, embed_dim, ensemble_size, scaling_init
            )
        else:
            # Standard feedforward network
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "gelu":
            self.activation_fn = F.gelu
        else:
            raise ValueError(
                f"Invalid activation '{activation}'. Choose from 'relu' or 'gelu'."
            )

    def forward(self, src, src_mask: Optional[torch.Tensor] = None):
        """
        Pass the input through the encoder layer.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor of shape (N, S, E, D), where:
                - N: Batch size
                - S: Sequence length
                - E: Ensemble size
                - D: Embedding dimension
        src_mask : torch.Tensor, optional
            The source mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D).

        """
        # Self-attention
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward network
        if self.batch_ensemble_ffn:
            src2 = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        else:
            N, S, E, D = src.shape
            src_reshaped = src.view(N * E * S, D)
            src2 = self.linear1(src_reshaped)
            src2 = self.activation_fn(src2)
            src2 = self.dropout(src2)
            src2 = self.linear2(src2)
            src2 = src2.view(N, S, E, D)

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def dropout(self, x):
        """
        Apply dropout to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying dropout.

        """
        return F.dropout(x, p=self.dropout, training=self.training)


class BatchEnsembleTransformerEncoder(nn.Module):
    """
    Transformer Encoder with Batch Ensembling.

    This class implements the Transformer encoder consisting of multiple encoder layers with batch ensembling.

    Parameters
    ----------
    num_layers : int
        Number of encoder layers to stack.
    embed_dim : int
        The dimension of the embedding.
    num_heads : int
        Number of attention heads.
    ensemble_size : int
        Number of ensemble members.
    dim_feedforward : int, optional
        Dimension of the feedforward network model. Default is 2048.
    dropout : float, optional
        Dropout value. Default is 0.1.
    activation : {'relu', 'gelu'}, optional
        Activation function of the intermediate layer. Default is 'relu'.
    scaling_init : {'ones', 'random-signs', 'normal'}, optional
        Initialization method for the scaling factors in batch ensembling. Default is 'ones'.
    batch_ensemble_projections : list of str, optional
        List of projections to which batch ensembling should be applied in the attention layer.
        Default is ['query'].
    batch_ensemble_ffn : bool, optional
        Whether to apply batch ensembling to the feedforward network. Default is False.
    norm : nn.Module, optional
        Optional layer normalization module.

    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ensemble_size: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Literal["relu", "gelu"] = "relu",
        scaling_init: Literal["ones", "random-signs", "normal"] = "ones",
        batch_ensemble_projections: List[str] = ["query"],
        batch_ensemble_ffn: bool = False,
        norm: Optional[nn.Module] = None,
    ):
        super(BatchEnsembleTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                BatchEnsembleTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ensemble_size=ensemble_size,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    scaling_init=scaling_init,
                    batch_ensemble_projections=batch_ensemble_projections,
                    batch_ensemble_ffn=batch_ensemble_ffn,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm

    def forward(self, src, mask: Optional[torch.Tensor] = None):
        """
        Pass the input through the encoder layers in turn.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor of shape (N, S, E, D).
        mask : torch.Tensor, optional
            The source mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor of shape (N, S, E, D).
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
