from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class DefaultMambaTabConfig:
    """Configuration class for the Default MambaTab model with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which the learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 regularization) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=1
        Number of layers in the model.
    expand_factor : int, default=2
        Expansion factor for the feed-forward layers.
    bias : bool, default=False
        Whether to use bias in the linear layers.
    d_conv : int, default=16
        Dimensionality of the convolutional layers.
    conv_bias : bool, default=True
        Whether to use bias in the convolutional layers.
    dropout : float, default=0.05
        Dropout rate for regularization.
    dt_rank : str, default="auto"
        Rank of the decision tree used in the model.
    d_state : int, default=128
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for the decision tree.
    dt_init : str, default="random"
        Initialization method for the decision tree.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    activation : callable, default=nn.ReLU()
        Activation function for the model.
    axis : int, default=1
        Axis along which operations are applied, if applicable.
    num_embedding_activation : callable, default=nn.ReLU()
        Activation function for numerical embeddings.
    embedding_type : str, default="linear"
        Type of embedding to use ('linear', etc.).
    embedding_bias : bool, default=False
        Whether to use bias in the embedding layers.
    plr_lite : bool, default=False
        Whether to use a lightweight version of Piecewise Linear Regression (PLR).
    n_frequencies : int, default=48
        Number of frequencies for PLR embeddings.
    frequencies_init_scale : float, default=0.01
        Initial scale for frequency parameters in embeddings.
    head_layer_sizes : list, default=()
        Sizes of the fully connected layers in the model's head.
    head_dropout : float, default=0.0
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.ReLU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    norm : str, default="LayerNorm"
        Type of normalization to be used ('LayerNorm', 'RMSNorm', etc.).
    use_pscan : bool, default=False
        Whether to use PSCAN for the state-space model.
    mamba_version : str, default="mamba-torch"
        Version of the Mamba model to use ('mamba-torch', 'mamba1', 'mamba2').
    bidirectional : bool, default=False
        Whether to process data bidirectionally.
    """

    # Optimizer Parameters
    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1

    # Architecture Parameters
    d_model: int = 64
    n_layers: int = 1
    expand_factor: int = 2
    bias: bool = False
    d_conv: int = 16
    conv_bias: bool = True
    dropout: float = 0.05
    dt_rank: str = "auto"
    d_state: int = 128
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    activation: Callable = nn.ReLU()  # noqa: RUF009
    axis: int = 1

    # Embedding Parameters
    num_embedding_activation: Callable = nn.ReLU()  # noqa: RUF009
    embedding_type: str = "linear"
    embedding_bias: bool = False
    plr_lite: bool = False
    n_frequencies: int = 48
    frequencies_init_scale: float = 0.01

    # Head Parameters
    head_layer_sizes: list = field(default_factory=list)
    head_dropout: float = 0.0
    head_skip_layers: bool = False
    head_activation: Callable = nn.ReLU()  # noqa: RUF009
    head_use_batch_norm: bool = False

    # Additional Features
    norm: str = "LayerNorm"
    use_pscan: bool = False
    mamba_version: str = "mamba-torch"
    bidirectional: bool = False
