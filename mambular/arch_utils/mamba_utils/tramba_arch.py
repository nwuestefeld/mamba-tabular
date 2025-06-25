import torch.nn as nn
from ..get_norm_fn import get_normalization_layer
from .mamba_arch import ResidualBlock, Mamba
from .mamba_original import MambaOriginal
from ..transformer_utils import CustomTransformerEncoderLayer


class Tramba(nn.Module):
    """Tramba model composed of an Attention layer followed by Mamba Blocks.
    This model design introdcuces positional invariance by using a Transformer
    encoder before the Mamba blocks.
    For more details see the paper: "On Embeddings for Numerical Features in Tabular Deep Learning"

    Attributes:
        config (TrambularConfig): Configuration object for the Trambular Model.
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
            num_layers=1,
            norm=self.norm_f,
        )
        
        if config.mamba_version == "mamba-torch" or config.mamba_version == "mamba_triton":
            self.mamba = Mamba(config)
        else:
            self.mamba = MambaOriginal(config)
        
 

    def forward(self, x):
        #Transformer followed by  n MambaBlocks
        x = self.encoder(x)
        x = self.mamba(x)

       # for layer in self.layers:
       #     x = layer(x)
        return x
    
