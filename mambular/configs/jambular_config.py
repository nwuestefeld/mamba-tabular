from dataclasses import dataclass
import torch.nn as nn


@dataclass
class DefaultJambularConfig:
    """
    Configuration class for the Default Jambular model with predefined hyperparameters.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=8
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
        Rank of the decision tree.
    d_state : int, default=32
        Dimensionality of the state in recurrent layers.
    dt_scale : float, default=1.0
        Scaling factor for decision tree.
    dt_init : str, default="random"
        Initialization method for decision tree.
    dt_max : float, default=0.1
        Maximum value for decision tree initialization.
    dt_min : float, default=1e-04
        Minimum value for decision tree initialization.
    dt_init_floor : float, default=1e-04
        Floor value for decision tree initialization.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the model.
    embedding_activation : callable, default=nn.Identity()
        Activation function for embeddings.
    head_layer_sizes : list, default=(128, 64, 32)
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    pooling_method : str, default="avg"
        Pooling method to be used ('avg', 'max', etc.).
    bidirectional : bool, default=False
        Whether to use bidirectional processing of the input sequences.
    use_learnable_interaction : bool, default=False
        Whether to use learnable feature interactions before passing through mamba blocks.
    use_cls : bool, default=True
        Whether to append a cls to the end of each 'sequence'.
    shuffle_embeddings : bool, default=False.
        Whether to shuffle the embeddings before being passed to the Mamba layers.
    layer_norm_eps : float, default=1e-05
        Epsilon value for layer normalization.
    AD_weight_decay : bool, default=True
        whether weight decay is also applied to A-D matrices.
    BC_layer_norm: bool, default=False
        whether to apply layer normalization to B-C matrices.
    cat_encoding : str, default="int"
        whether to use integer encoding or one-hot encoding for cat features.
    """

    lr: float = 1e-04
    lr_patience: int = 10
    weight_decay: float = 1e-06
    lr_factor: float = 0.1
    d_model: int = 64
    n_layers: int = 4
    expand_factor: int = 2
    bias: bool = False
    d_conv: int = 4
    conv_bias: bool = True
    dropout: float = 0.0
    dt_rank: str = "auto"
    d_state: int = 128
    dt_scale: float = 1.0
    dt_init: str = "random"
    dt_max: float = 0.1
    dt_min: float = 1e-04
    dt_init_floor: float = 1e-04
    norm: str = "LayerNorm"
    activation: callable = nn.SiLU()
    embedding_activation: callable = nn.Identity()
    head_layer_sizes: list = ()
    head_dropout: float = 0.5
    head_skip_layers: bool = False
    head_activation: callable = nn.SELU()
    head_use_batch_norm: bool = False
    layer_norm_after_embedding: bool = False
    pooling_method: str = "avg"
    bidirectional: bool = False
    use_learnable_interaction: bool = False
    use_cls: bool = False
    shuffle_embeddings: bool = False
    layer_norm_eps: float = 1e-05
    AD_weight_decay: bool = True
    BC_layer_norm: bool = False
    cat_encoding: str = "int"


    #Attention stuff
    num_attention_heads: int = 16
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.

    #MoE stuf
    num_experts: int = 16
    num_experts_per_tok: int = 2


    # structure
    #Mamba -> Expert -> Attn -> Expert
    # 3rd layer (l=2) is Attn
    attn_layer_offset: int = 2
    attn_layer_period: int = 4

    #2nd layer (l=1) and 4th layer (l=3) is Expert
    expert_layer_offset: int = 1
    expert_layer_period: int = 2