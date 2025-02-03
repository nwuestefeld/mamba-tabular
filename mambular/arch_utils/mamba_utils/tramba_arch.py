import torch.nn as nn
from ..get_norm_fn import get_normalization_layer
from .mamba_arch import ResidualBlock
from ..transformer_utils import CustomTransformerEncoderLayer


class Tramba(nn.Module):
    """Mamba model composed of  MambaBlocks and Attention layers.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of alternating ResidualBlock (Mamba layers) and
        attention layers constituting the model.
    """

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
   
        
         # transformer encoder
        self.norm_f = get_normalization_layer(config)
        encoder_layer = CustomTransformerEncoderLayer(config=config)
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=self.norm_f,
        )

        self.layers = nn.ModuleList([
            ResidualBlock(
                d_model=config.d_model,
                expand_factor= config.expand_factor,
                bias=config.bias,
                d_conv=config.d_conv,
                conv_bias=config.conv_bias,
                dropout=config.dropout,
                dt_rank=config.dt_rank,
                d_state=config.d_state,
                dt_scale=config.dt_scale,
                dt_init=config.dt_init,
                dt_max=config.dt_max,
                dt_min=config.dt_min,
                dt_init_floor=config.dt_init_floor,
                norm=config.norm,
                activation=config.activation,
                bidirectional=config.bidirectional,
                use_learnable_interaction=config.use_learnable_interaction,
                layer_norm_eps=config.layer_norm_eps,
                AD_weight_decay=config.AD_weight_decay,
                BC_layer_norm=config.BC_layer_norm,
                use_pscan=config.use_pscan,) for _ in range(config.n_layers)
        ])

    def forward(self, x):
        #Transformer followed by  n MambaBlocks
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return x
    

    

    